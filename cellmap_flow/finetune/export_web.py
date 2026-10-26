"""Export a model finetuned on the server for the browser: ONNX files and the catalog's manifest.

A model finetuned here is served from the YAML its run wrote
(run_outputs.write_serving_yaml): a ``type: finetune`` entry, its base model
with a LoRA adapter or a full finetune's weights on top. This turns it into
a folder laid out as the browser's catalog lays out a model
(cellmap/onnx-web-models, and a model the browser publishes), so that it can
be published to Hugging Face and opened in the browser, or served with
``type: onnx``:

- ``<output>/<folder>/manifest.json``: the model's manifest in the catalog's
  format (web/src/models/manifest.ts parseManifest; the server reads it with
  models/configs/onnx.py parse_manifest, which checks it here before the
  export counts as done): its channels, voxel sizes, input normalization,
  what its output is, the ONNX tensors' names, and one row per variant with
  its block sizes, its file and how far it is from the PyTorch model;
- ``<output>/<folder>/<folder>_in<n>_<variant>.onnx``: one file per variant
  (fp32, fp16, int8w, int8w_fp16, or those ``--variants`` names);
- ``<output>/result.json``: what the export came to, ``{"folder",
  "manifest"}`` or ``{"error"}``, which it also prints as the last line of its
  output; ``<output>/export_log.txt`` is its log when it runs as a job.

How: the base network is built as the trainer builds it
(model_loading.load_trainable_model), the finetune folded into its weights
(export_merged.apply_finetune), and the plain network exported with
cellmap-models' onnx_web (``export_onnx_web``) at the block the model is
served at. onnx_web rewrites a 3D transposed convolution, which
onnxruntime-web's WebGPU cannot run, and every Conv3d, which it runs about
16x slower than Conv2d, into ops it runs fast, and checks each file with
onnxruntime against the PyTorch network. It checks them on a block
of the raw the model was trained on, read and normalized as the trainer reads
and normalizes its patches (data.loader.dataset_from_manifest), centred where
it draws one from what was annotated; the fp32 file must agree with the
network to FP32_REL_TOL of the output's scale or the export fails.

The dashboard starts an export and follows it (routes/finetune/publish.py)::

    export = start_export(yaml_path, corrections_dir, run_dir)
    export_status(export)
    # {"status": "running" | "done" | "failed",
    #  "error": str | None,       # failed: why
    #  "dir": str,                # the export's directory
    #  "folder": str | None,      # done: the model's folder in it
    #  "manifest": dict | None}   # done: its manifest.json

``start_export`` submits the export as a job (job_manager.submit.launch: on
LSF when there is bsub, else a process here) in the environment the model
finetunes in, into ``<run_dir>/web_export/<YAML's name>/``, and returns at
once; the files to upload are ``<dir>/<folder>/manifest.json`` and
``<dir>/<folder>/<file>`` for each of the manifest's variants. From the
command line::

    python -m cellmap_flow.finetune.export_web --yaml <serving yaml> \\
        --corrections <session>/corrections --output <dir> [--variants fp32,fp16]

What it cannot export, said rather than worked around:

- Cubes only, at the one block the model is served at: export_onnx_web takes
  cubic blocks. A 2D model's block is one slice thick, so 2D models (and
  Cellpose) are refused; one input channel only, as the browser feeds one.
- What onnx_web rewrites it asserts: Conv3d unpadded, undilated and
  ungrouped, ConvTranspose3d with kernel and stride 2. Valid-padding U-Nets
  are; anything else fails the export with onnx_web's message.
- At most 2 GB of float32 weights: onnx_web keeps them inside the .onnx
  file, which protobuf caps there. The large cellmap UNets (about 800M
  parameters, 3.2 GB, such as mito-aff-unet-setup-16) are refused; no
  browser would load them either.
- The export runs on the CPU (onnx_web's export and checks are CPU ones), in
  a job that holds the one GPU launch asks for, as every finetune job does.
- The manifest has no ``finetune`` block: the browser cannot finetune the
  exported model further (its head and full-network specs name graph nodes
  and a training graph that onnx_web does not write).
- ``output`` says what the network's output is as the trainer's probe reads
  it (probe_output_class): probabilities, a tanh's range, or unbounded; it
  cannot tell logits from distances.

Needs cellmap-models with ``model_export.onnx_web`` (the
onnx-web-export-and-pruning branch, which pixi.toml's finetune feature
installs) and its export extra, onnx and onnxruntime, where the export runs.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODULE = "cellmap_flow.finetune.export_web"
RESULT_NAME = "result.json"
LOG_NAME = "export_log.txt"
# An export of a model that fits in a browser takes minutes on the CPU.
WALLTIME = "01:00"
# fp32 is the network's own arithmetic, reordered by onnx_web's rewrites; it
# agrees to float rounding, far below this share of the output's largest value.
FP32_REL_TOL = 1e-3
# protobuf's cap on one serialized message, which an .onnx file is.
ONNX_FILE_LIMIT = 2**31
EXPORTER = "cellmap_models.model_export.onnx_web (torch.onnx.export, TorchScript, dynamo=False)"
# probe_output_class's classes (portable.label_it), as the manifest's
# ``output`` says them; the browser reads "probab" as [0, 1] and "unbounded"
# as no output activation (backend/routes/models.ts, advice).
OUTPUT_DESCRIPTIONS = {
    "unit": "probabilities in [0, 1] (the network ends in a sigmoid or a softmax)",
    "signed_unit": "values in [-1, 1] (the network ends in a tanh)",
    "unbounded": "unbounded (no output activation: logits, distances or affinities)",
}


@dataclass
class WebExport:
    """An export ``start_export`` started: the directory it writes and its job
    (a jobs.lsf.LSFJob, or a jobs.local.LocalJob where there is no bsub)."""

    output_dir: Path
    job: object

    @property
    def result_file(self) -> Path:
        return self.output_dir / RESULT_NAME

    @property
    def log_file(self) -> Path:
        return self.output_dir / LOG_NAME


def serving_entry(yaml_path):
    """The serving YAML at ``yaml_path`` and its finetuned model's entry, ``(config, entry)``.

    Raises:
        ValueError: the YAML serves no ``type: finetune`` model.
    """
    import yaml

    with open(yaml_path) as f:
        config = yaml.safe_load(f) or {}
    models = config.get("models") or []
    entry = models[0] if models else None
    if not isinstance(entry, dict) or entry.get("type") != "finetune":
        raise ValueError(f"{yaml_path} serves no finetuned model (a type: finetune entry)")
    return config, entry


def folder_name(model_name: str) -> str:
    """The model's folder: its name as the catalog's names are made (manifest.ts
    isLayerName), lower case, runs of anything but letters, digits, ``_`` and
    ``-`` made one ``_``."""
    return re.sub(r"[^a-z0-9_-]+", "_", model_name.lower()).strip("_-") or "model"


def start_export(yaml_path, corrections_dir, run_dir) -> WebExport:
    """Start exporting the model ``yaml_path`` serves, trained on the session
    ``corrections_dir``, into ``<run_dir>/web_export/<YAML's name>/``.

    The job runs in the environment the model finetunes in
    (submit.module_command) on the YAML's queue and charge group, which are
    its finetune job's. A result an earlier export of the same YAML left is
    removed first, so that ``export_status`` does not read it as this one's.

    Raises:
        ValueError: the YAML serves no finetuned model, or its environment
            cannot run the export.
        RuntimeError: the job could not be submitted.
    """
    from cellmap_flow.finetune.job_manager import submit
    from cellmap_flow.finetune.model_loading import model_config_from_entry

    config, entry = serving_entry(yaml_path)
    output_dir = Path(run_dir) / "web_export" / Path(yaml_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RESULT_NAME).unlink(missing_ok=True)
    args = ["--yaml", str(yaml_path), "--corrections", str(corrections_dir), "--output", str(output_dir)]
    command = submit.module_command(model_config_from_entry(entry), MODULE, args, output_dir / LOG_NAME)
    job = submit.launch(
        f"web_export_{folder_name(entry.get('name') or 'model')}", command,
        queue=config.get("queue"), charge_group=config.get("charge_group"), walltime=WALLTIME,
    )
    return WebExport(output_dir, job)


def export_status(export: WebExport) -> dict:
    """Where ``export`` stands (see the module's docstring for the answer's keys).

    "running" until the export has written its result, then "done" or
    "failed" as the result says; "failed" too, with the tail of its log as the
    error, when its job has ended without writing one (killed, out of time or
    memory).
    """
    from cellmap_flow.jobs.spec import JobStatus, exists_now, tail

    answer = {"status": "running", "error": None, "dir": str(export.output_dir), "folder": None, "manifest": None}
    if exists_now(export.result_file):
        result = json.loads(export.result_file.read_text())
        if result.get("error"):
            return {**answer, "status": "failed", "error": result["error"]}
        return {**answer, "status": "done", "folder": result["folder"], "manifest": result["manifest"]}
    if export.job.get_status() in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.KILLED):
        log = tail(export.log_file, 2000)
        return {**answer, "status": "failed",
                "error": "The export job ended without a result." + (f" Its log ends:\n{log}" if log else "")}
    return answer


def training_block(corrections_dir, input_block, output_block, input_norm):
    """A block of the raw the model was trained on, ``[1, 1, *input_block]``
    float32, normalized by ``input_norm``.

    Read as the trainer reads a patch of the session ``corrections_dir``, at
    the training manifest's voxel sizes, around a centre it draws from what
    was annotated: not a random anchor, which may lie anywhere in the
    dataset, resin included.

    Raises:
        ValueError: the session has no training manifest.
    """
    import numpy as np

    from cellmap_flow.finetune.data.loader import dataset_from_manifest
    from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME, read_manifest

    manifest = read_manifest(str(corrections_dir))
    if manifest is None:
        raise ValueError(f"{corrections_dir} has no {VIRTUAL_MANIFEST_FILENAME}, so the raw the model "
                         "was trained on is not known")
    dataset = dataset_from_manifest(
        {**manifest, "input_norm": input_norm, "anchor_fraction": 0.0},
        patch_voxels=(input_block, output_block),
    )
    centre, _, depth = dataset.sampler.draw(np.random.default_rng(0))
    centre = dataset.reader.snap(centre, depth)
    raw = dataset.reader.normalize(dataset.reader.raw(centre, depth))
    return np.ascontiguousarray(raw, dtype=np.float32)[None, None]


def _side(block, which) -> int:
    """The side of a cubic block of voxels; ValueError for any other."""
    if len(block) != 3 or len(set(block)) != 1:
        raise ValueError(f"The model's {which} block is {'x'.join(map(str, block))} voxels: export_onnx_web "
                         "exports cubes only (a 2D model's block is one slice thick)")
    return int(block[0])


def _task(channels) -> str:
    """What the model predicts, for the manifest's ``task``: a single channel's
    name, or the channels' common prefix up to a separator (mito_aff_z, _y,
    _x: mito_aff), else their names joined."""
    if len(channels) == 1:
        return channels[0]
    prefix = os.path.commonprefix(list(channels))
    cut = max(prefix.rfind(separator) for separator in "_-.")
    return prefix[:cut] if cut > 0 else "_".join(channels)


def export(yaml_path, corrections_dir, output_dir, variants=None) -> dict:
    """Export the model ``yaml_path`` serves into ``output_dir``; ``{"folder", "manifest"}``.

    ``variants``: onnx_web's precisions to write, all of them by default.

    Raises:
        ValueError: the model is one this cannot export (see the module's
            docstring), or its fp32 file does not agree with it.
    """
    import torch

    try:
        from cellmap_models.model_export.onnx_web import VARIANTS, export_onnx_web
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. The web export needs cellmap-models' onnx_web (its onnx-web-export-and-pruning branch, "
            "which pixi.toml's finetune feature installs) and its export extra, onnx and onnxruntime."
        ) from e

    from cellmap_flow.finetune.export_merged import apply_finetune
    from cellmap_flow.finetune.lora_wrapper import _replace_interpreter_modules
    from cellmap_flow.finetune.model_loading import load_trainable_model, model_config_from_entry, root_base_model_dict
    from cellmap_flow.finetune.trainer.supervised import probe_output_class
    from cellmap_flow.models.configs.onnx import parse_manifest
    from cellmap_flow.models.entry import numbers
    from cellmap_flow.pipeline_spec import normalize_steps

    config, entry = serving_entry(yaml_path)
    input_norm = list(normalize_steps((config.get("json_data") or {}).get("input_norm")))
    finetuned = model_config_from_entry(entry)
    base = finetuned.base_model_config
    geometry = base.geometry
    size = _side(geometry.input_shape, "input")
    _side(geometry.output_shape, "output")
    if geometry.input_channels != 1:
        raise ValueError(f"The model takes {geometry.input_channels} input channels; the browser feeds one")

    network = apply_finetune(load_trainable_model(base), finetuned.lora_adapter_path, finetuned.weights_path).cpu()
    params = sum(p.numel() for p in network.parameters())
    if 4 * params >= ONNX_FILE_LIMIT:
        raise ValueError(f"The network has {params:,} parameters, {4 * params / 1e9:.1f} GB in float32: "
                         "export_onnx_web keeps the weights in the .onnx file, which protobuf caps at 2 GB")
    # The leaves torch.export's unflatten left (a TorchScript base's) as the
    # real layers onnx_web looks for and rewrites; they share the weights.
    _replace_interpreter_modules(network, transposed=True)

    block = training_block(corrections_dir, geometry.input_shape, geometry.output_shape, input_norm)
    with torch.no_grad():
        output_class = probe_output_class(network, torch.from_numpy(block))
        scale = float(network(torch.from_numpy(block)).abs().max())

    folder = folder_name(finetuned.name)
    root = root_base_model_dict(finetuned)
    how = "LoRA, merged into the network" if finetuned.lora_adapter_path else "the whole network"
    base_name = root.get("repo") or root.get("name") or f"a {root.get('type')} model"
    channels = list(geometry.channel_names or (f"channel_{i}" for i in range(geometry.output_channels)))
    rows = export_onnx_web(
        network, os.path.join(output_dir, folder), folder, [size],
        variants=variants or VARIANTS, validation_input=block,
        manifest_extra={
            "model": folder,
            "task": _task(channels),
            "checkpoint": f"finetuned with cellmap-flow ({how}) from {base_name}",
            "channels": channels,
            # The voxel sizes it was trained at (FinetuneModelConfig's
            # trained_at_geometry), when they are not the base's.
            "input_voxel_size_nm": numbers(finetuned.input_voxel_size or geometry.input_voxel_size),
            "output_voxel_size_nm": numbers(finetuned.output_voxel_size or geometry.output_voxel_size),
            "output": OUTPUT_DESCRIPTIONS[output_class],
            "input_norm": input_norm,
            **({"source_repo": root["repo"]} if root.get("repo") else {}),
            "torch_version": torch.__version__,
            "exporter": EXPORTER,
        },
    )
    for row in rows:
        if row["variant"] == "fp32" and row["max_abs"] > FP32_REL_TOL * max(scale, 1e-6):
            raise ValueError(f"The fp32 export differs from the network by up to {row['max_abs']:.3g}, "
                             f"{row['max_abs'] / max(scale, 1e-6):.2e} of its output's scale ({scale:.3g}), "
                             f"over the {FP32_REL_TOL:.0e} tolerance")
    manifest_path = os.path.join(output_dir, folder, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    parse_manifest(manifest, manifest_path)
    return {"folder": folder, "manifest": manifest}


def _write_result(output_dir: Path, result: dict) -> None:
    """``result.json``, written whole before it appears: export_status may read it at any moment."""
    partial = output_dir / f".{RESULT_NAME}.partial"
    partial.write_text(json.dumps(result))
    os.replace(partial, output_dir / RESULT_NAME)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yaml", required=True, help="the finetuned model's serving YAML")
    parser.add_argument("--corrections", required=True,
                        help="the session's corrections directory, which the model was trained on")
    parser.add_argument("--output", required=True, help="the directory to export into")
    parser.add_argument("--variants", default=None,
                        help="comma-separated onnx_web precisions (fp32,fp16,int8w,int8w_fp16; all by default)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    try:
        result = export(args.yaml, args.corrections, output_dir, variants)
    except Exception as e:
        logger.exception("The web export failed")
        result = {"error": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"}
    _write_result(output_dir, result)
    print(json.dumps(result), flush=True)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
