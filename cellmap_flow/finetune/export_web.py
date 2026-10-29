"""Export a model finetuned on the server to publish it: a cellmap-models export for servers, and ONNX files and the catalog's manifest for the browser.

A model finetuned here is served from the YAML its run wrote
(run_outputs.write_serving_yaml): a ``type: finetune`` entry, its base model
with a LoRA adapter or a full finetune's weights on top. This turns it into
the files a publish uploads to Hugging Face (dashboard/routes/finetune/publish.py),
laid out in ``<output>`` as the repository lays them out, in two forms:

- for servers, always: the network as cellmap-models exports one, as
  cellmap's own model repositories hold it at their root, which a server
  opens by the repository's name (``type: huggingface``) and a folder of
  them serves (``type: cellmap``; export_merged.export_folder):
  ``<output>/model.ts``, the TorchScript that is served;
  ``<output>/model.pt2``, the torch.export program a LoRA finetune of it
  trains, where torch.export can make one; ``<output>/metadata.json``, its
  channels, voxel sizes and block in cellmap-models' format, and its input
  normalization in cellmap-flow's (``input_norm``, which a server takes
  when the model is added); and ``<output>/export_record.json``, how it
  was exported;
- for the browser, when export_onnx_web can make one (below): a folder laid
  out as the browser's catalog lays out a model (cellmap/onnx-web-models,
  and a model the browser publishes), which the browser opens and a server
  serves with ``type: onnx``. ``<output>/<folder>/manifest.json`` is the
  model's manifest in the catalog's format (web/src/models/manifest.ts
  parseManifest; the server reads it with models/configs/onnx.py
  parse_manifest, which checks it here before the export counts as done):
  its channels, voxel sizes, input normalization, what its output is, the
  ONNX tensors' names, and one row per variant with its block sizes, its
  file and how far it is from the PyTorch model; and
  ``<output>/<folder>/<folder>_in<n>_<variant>.onnx`` is one file per
  variant (fp32, fp16, int8w, int8w_fp16, or those ``--variants`` names).

``<output>/result.json`` says what the export came to, which it also prints
as the last line of its output, ``{"error"}`` or::

    {"folder": str,             # the model's folder: the browser form's, and its name
     "manifest": dict,          # the browser form's manifest.json; without one, what it
                                # would say of the model, with no variants
     "server": {"files": [str], # the server form's files, in <output>
                "input_size": [int], "output_size": [int]},  # the block model.ts is traced at
     "no_browser": str | None}  # why there is no browser form; None when there is one

``<output>/export_log.txt`` is its log when it runs as a job.

How: the base network is built as the trainer builds it
(model_loading.load_trainable_model), the finetune folded into its weights
(export_merged.apply_finetune), and the plain network exported at the block
the model is served at: traced to TorchScript (and exported with
torch.export) for servers, and exported with cellmap-models' onnx_web
(``export_onnx_web``) for the browser. onnx_web rewrites a 3D transposed
convolution, which onnxruntime-web's WebGPU cannot run, and every Conv3d,
which it runs about 16x slower than Conv2d, into ops it runs fast, and
checks each file with onnxruntime against the PyTorch network. It checks
them on a block of the raw the model was trained on, read and normalized as
the trainer reads and normalizes its patches (data.loader.dataset_from_manifest),
centred where it draws one from what was annotated; the fp32 file must agree
with the network to FP32_REL_TOL of the output's scale.

The dashboard starts an export and follows it (routes/finetune/publish.py)::

    export = start_export(yaml_path, corrections_dir, run_dir)
    export_status(export)
    # {"status": "running" | "done" | "failed",
    #  "error": str | None,       # failed: why
    #  "dir": str,                # the export's directory
    #  "job": {"id": str | None,  # its LSF job's id, None for a process here
    #          "queued": bool,    # whether it is still waiting for its node
    #          "log": str,        # its log
    #          "epoch": int | None, "epochs": int | None},  # a training's epoch, of how many
    #  and, done, result.json's "folder", "manifest", "server" and "no_browser"
    #  (None until then)}

The epoch is the last "Starting epoch N of M" line (markers.EPOCH_START_RE)
near the end of the log: distill prints them; an export prints none.

``start_export`` submits the export as a job (job_manager.submit.launch: on
LSF when there is bsub, else a process here) in the environment the model
finetunes in, into ``<run_dir>/web_export/<YAML's name>/``, and returns at
once; the files to upload are ``<dir>/<file>`` at the repository's root for
each of the server form's files, and ``<dir>/<folder>/manifest.json`` and
``<dir>/<folder>/<file>`` for each of the manifest's variants. From the
command line::

    python -m cellmap_flow.finetune.export_web --yaml <serving yaml> \\
        --corrections <session>/corrections --output <dir> [--variants fp32,fp16]

What it cannot export, said rather than worked around:

- A model served through its base's own prediction rather than its
  network's forward (Cellpose, which segments with its own eval): a
  TorchScript of the network would not predict what is served, so the
  export fails. Every other model has its server form.
- The browser form is exported at the block the model is served at, its
  (z, y, x) shape as it is: a cube, a 2D model's slice (1, y, x), or any
  other block. None for a model with more than one input channel, as the
  browser feeds one.
- What onnx_web rewrites it asserts: Conv3d unpadded, undilated and
  ungrouped, ConvTranspose3d with kernel equal to stride. Valid-padding
  U-Nets are; anything else has no browser form, for onnx_web's reason.
- At most 2 GB of float32 weights in the browser form: onnx_web keeps them
  inside the .onnx file, which protobuf caps there. The large cellmap UNets
  (about 800M parameters, 3.2 GB, such as mito-aff-unet-setup-16) have
  none; no browser would load them either.
- A model without a browser form is still exported, its server form alone,
  and ``no_browser`` says why: the publish uploads that, and its card says
  so.
- The network runs on the GPU the job holds when there is one (the one
  launch asks for, as every finetune job does), for its probes and the
  outputs the ONNX files are checked against; the server form is traced on
  the CPU, as a TorchScript traced on a GPU keeps that GPU in it, and
  onnx_web exports and checks its files on the CPU.
- The manifest has no ``finetune`` block: the browser cannot finetune the
  exported model further (its head and full-network specs name graph nodes
  and a training graph that onnx_web does not write).
- ``output`` says what the network's output is as the trainer's probe reads
  it (probe_output_class): probabilities, a tanh's range, or unbounded; it
  cannot tell logits from distances.

The browser form needs cellmap-models with ``model_export.onnx_web`` (the
onnx-web-export-and-pruning branch, which pixi.toml's finetune feature
installs) and its export extra, onnx and onnxruntime, where the export
runs; without them there is none, and ``no_browser`` says so.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cellmap_flow

logger = logging.getLogger(__name__)

MODULE = "cellmap_flow.finetune.export_web"
RESULT_NAME = "result.json"
LOG_NAME = "export_log.txt"
# An export takes minutes on the CPU: a few forwards of the network at its
# block and writing its files, 3 GB each for the largest cellmap UNets.
WALLTIME = "01:00"
# fp32 is the network's own arithmetic, reordered by onnx_web's rewrites; it
# agrees to float rounding, far below this share of the output's largest value.
FP32_REL_TOL = 1e-3
# protobuf's cap on one serialized message, which an .onnx file is.
ONNX_FILE_LIMIT = 2**31
EXPORTER = "cellmap_models.model_export.onnx_web (torch.onnx.export, TorchScript, dynamo=False)"
# The --variants flag's help, export_web's and distill's.
VARIANTS_HELP = "comma-separated onnx_web precisions (fp32,fp16,int8w,int8w_fp16; all by default)"
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
    ``corrections_dir``, into ``<run_dir>/web_export/<YAML's name>/``, as a
    job (start_job).

    Raises:
        ValueError: the YAML serves no finetuned model, or its environment
            cannot run the export.
        RuntimeError: the job could not be submitted.
    """
    return start_job(MODULE, "web_export", yaml_path, corrections_dir,
                     Path(run_dir) / "web_export" / Path(yaml_path).stem, WALLTIME)


def start_job(module, job_name, yaml_path, corrections_dir, output_dir, walltime, flags=()) -> WebExport:
    """Start ``python -m <module>`` on the model ``yaml_path`` serves,
    trained on the session ``corrections_dir``, writing into ``output_dir``
    (the export's job, and distill's), and return at once.

    The job, ``<job_name>_<model's folder>``, runs in the environment the
    model finetunes in (submit.module_command) on the YAML's queue and
    charge group, which are its finetune job's, with ``flags`` after the
    module's --yaml, --corrections and --output. A result an earlier job
    left in ``output_dir`` is removed first, so that ``export_status`` does
    not read it as this one's.

    Raises:
        ValueError: the YAML serves no finetuned model, or its environment
            cannot run the job.
        RuntimeError: the job could not be submitted.
    """
    from cellmap_flow.finetune.job_manager import submit
    from cellmap_flow.finetune.model_loading import model_config_from_entry

    config, entry = serving_entry(yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RESULT_NAME).unlink(missing_ok=True)
    args = ["--yaml", str(yaml_path), "--corrections", str(corrections_dir), "--output", str(output_dir), *flags]
    command = submit.module_command(model_config_from_entry(entry), module, args, output_dir / LOG_NAME)
    job = submit.launch(
        f"{job_name}_{folder_name(entry.get('name') or 'model')}", command,
        queue=config.get("queue"), charge_group=config.get("charge_group"), walltime=walltime,
    )
    return WebExport(output_dir, job)


def export_status(export: WebExport) -> dict:
    """Where ``export`` stands (see the module's docstring for the answer's keys).

    "running" until the export has written its result, then "done" or
    "failed" as the result says; "failed" too, with the tail of its log as the
    error, when its job has ended without writing one (killed, out of time or
    memory).
    """
    from cellmap_flow.finetune.markers import EPOCH_START_RE
    from cellmap_flow.jobs.spec import JobStatus, exists_now, tail

    job = {"id": getattr(export.job, "job_id", None), "queued": False, "log": str(export.log_file),
           "epoch": None, "epochs": None}
    answer = {"status": "running", "error": None, "dir": str(export.output_dir), "job": job,
              "folder": None, "manifest": None, "server": None, "no_browser": None}
    if exists_now(export.result_file):
        result = json.loads(export.result_file.read_text())
        if result.get("error"):
            return {**answer, "status": "failed", "error": result["error"]}
        # A distilled copy's result has a browser form alone: no server form, and no why not.
        return {**answer, "status": "done", **{key: result.get(key) for key in ("folder", "manifest", "server", "no_browser")}}
    status = export.job.get_status()
    if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.KILLED):
        log = tail(export.log_file, 2000)
        return {**answer, "status": "failed",
                "error": "The export job ended without a result." + (f" Its log ends:\n{log}" if log else "")}
    epochs = EPOCH_START_RE.findall(tail(export.log_file) or "")
    if epochs:
        job["epoch"], job["epochs"] = (int(n) for n in epochs[-1])
    job["queued"] = status == JobStatus.PENDING
    return answer


def training_manifest(corrections_dir) -> dict:
    """The training manifest of the session ``corrections_dir``
    (session.manifest.read_manifest): where the raw the model was trained
    on is, and how the trainer reads it.

    Raises:
        ValueError: the session has none.
    """
    from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME, read_manifest

    manifest = read_manifest(str(corrections_dir))
    if manifest is None:
        raise ValueError(f"{corrections_dir} has no {VIRTUAL_MANIFEST_FILENAME}, so the raw the model "
                         "was trained on is not known")
    return manifest


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

    dataset = dataset_from_manifest(
        {**training_manifest(corrections_dir), "input_norm": input_norm, "anchor_fraction": 0.0},
        patch_voxels=(input_block, output_block),
    )
    centre, _, depth = dataset.sampler.draw(np.random.default_rng(0))
    centre = dataset.reader.snap(centre, depth)
    raw = dataset.reader.normalize(dataset.reader.raw(centre, depth))
    return np.ascontiguousarray(raw, dtype=np.float32)[None, None]


def _task(channels) -> str:
    """What the model predicts, for the manifest's ``task``: a single channel's
    name, or the channels' common prefix up to a separator (mito_aff_z, _y,
    _x: mito_aff), else their names joined."""
    if len(channels) == 1:
        return channels[0]
    prefix = os.path.commonprefix(list(channels))
    cut = max(prefix.rfind(separator) for separator in "_-.")
    return prefix[:cut] if cut > 0 else "_".join(channels)


def finetuned_network(yaml_path):
    """The finetuned network ``yaml_path`` serves, on the CPU, as the export
    exports it and the distillation (finetune/distill.py) copies it: its base
    built as the trainer builds it (model_loading.load_trainable_model), the
    finetune folded into its weights (export_merged.apply_finetune), every
    layer a real one. Returns ``(network, finetuned, input_norm)``: the
    network, the finetuned model's config, and the input normalization the
    YAML serves it with, as steps.

    Raises:
        ValueError: the YAML serves no finetuned model, or one served other
            than by its network's forward (see the module's docstring).
    """
    from cellmap_flow.finetune.export_merged import apply_finetune
    from cellmap_flow.finetune.lora_wrapper import _replace_interpreter_modules
    from cellmap_flow.finetune.model_loading import load_trainable_model, model_config_from_entry
    from cellmap_flow.pipeline_spec import normalize_steps

    config, entry = serving_entry(yaml_path)
    input_norm = list(normalize_steps((config.get("json_data") or {}).get("input_norm")))
    finetuned = model_config_from_entry(entry)
    base = finetuned.base_model_config
    network = apply_finetune(load_trainable_model(base), finetuned.lora_adapter_path, finetuned.weights_path).cpu()
    _check_served_as_network(base, network)
    # The leaves torch.export's unflatten left (a TorchScript base's) as the
    # real layers onnx_web looks for and rewrites; they share the weights.
    _replace_interpreter_modules(network, transposed=True)
    return network, finetuned, input_norm


def describe(finetuned, geometry, folder, output_class, input_norm, checkpoint=None) -> dict:
    """What an export says of the model ``finetuned``, whose network has the
    ``geometry`` and an output of ``output_class`` (probe_output_class), in
    ``folder``: the browser form's manifest has it at its top, and the
    server form's metadata.json the same in its terms. ``checkpoint`` says
    where its weights came from; by default, how it was finetuned from what."""
    import torch

    from cellmap_flow.finetune.model_loading import root_base_model_dict
    from cellmap_flow.models.entry import numbers

    root = root_base_model_dict(finetuned)
    if checkpoint is None:
        how = "LoRA, merged into the network" if finetuned.lora_adapter_path else "the whole network"
        base_name = root.get("repo") or root.get("name") or f"a {root.get('type')} model"
        checkpoint = f"finetuned with cellmap-flow ({how}) from {base_name}"
    channels = list(geometry.channel_names or (f"channel_{i}" for i in range(geometry.output_channels)))
    return {
        "model": folder,
        "task": _task(channels),
        "checkpoint": checkpoint,
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
    }


def export(yaml_path, corrections_dir, output_dir, variants=None) -> dict:
    """Export the model ``yaml_path`` serves into ``output_dir``: its server
    form, and its browser form where export_onnx_web can make one. Returns
    result.json's ``{"folder", "manifest", "server", "no_browser"}`` (see the
    module's docstring).

    ``variants``: onnx_web's precisions to write, all of them by default.

    Raises:
        ValueError: the model is served other than by its network's forward
            (see the module's docstring), so it has no server form; or the
            server form does not open as cellmap-models opens an export.
    """
    import torch

    from cellmap_flow.finetune.model_loading import root_base_model_dict
    from cellmap_flow.finetune.trainer.supervised import probe_output_class
    from cellmap_flow.utils.device import best_device

    network, finetuned, input_norm = finetuned_network(yaml_path)
    geometry = finetuned.base_model_config.geometry
    block = training_block(corrections_dir, geometry.input_shape, geometry.output_shape, input_norm)
    # The network runs on the GPU when there is one, in float32 without
    # TF32 (which rounds a convolution to about the fp32 file's tolerance);
    # the server form is traced on the CPU (_export_server).
    device = best_device()
    network = network.to(device)
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        output_class = probe_output_class(network, torch.from_numpy(block).to(device))
        output = network(torch.from_numpy(block).to(device)).cpu()

    folder = folder_name(finetuned.name)
    root = root_base_model_dict(finetuned)
    description = describe(finetuned, geometry, folder, output_class, input_norm)

    input_size, output_size = list(geometry.input_shape), list(output.shape[2:])
    files = _export_server(
        network.cpu(), output_dir, input_size, output_size,
        _metadata(root, network, geometry, description),
        {
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "exporter": MODULE,
            "cellmap_flow": cellmap_flow.__version__,
            "torch": torch.__version__,
            # A path on this server's disks means nothing to the
            # repository's readers, so the base is said by its repo only.
            "finetuned_from": root.get("repo"),
            "base_type": root.get("type"),
            "finetune_mode": "lora" if finetuned.lora_adapter_path else "full",
            "input_shape": input_size,
            "output_shape": output_size,
        },
    )
    server = {"files": files, "input_size": input_size, "output_size": output_size}
    try:
        manifest = _export_browser(network.to(device), Path(output_dir) / folder, folder, geometry, block,
                                   float(output.abs().max()), description, variants)
    except Exception as e:
        logger.warning("The model has no browser form", exc_info=True)
        return {"folder": folder, "manifest": {**description, "variants": []}, "server": server, "no_browser": _said(e)}
    return {"folder": folder, "manifest": manifest, "server": server, "no_browser": None}


def _check_served_as_network(base, network) -> None:
    """Raise ValueError unless the finetuned model is served as ``network``'s
    forward: what a TorchScript of it reproduces. Asked as
    FinetuneModelConfig asks its base how to serve the trained module
    (``serve_trained``); a type that predicts through its own process_chunk
    (Cellpose's eval, which segments) says otherwise."""
    from cellmap_flow.models.configs.base import Config

    served = Config()
    base.serve_trained(served, network)
    if served.model is not network or getattr(served, "process_chunk", None):
        raise ValueError(f"The model is served through {type(base).__name__}'s own prediction, not its "
                         "network's forward, so a TorchScript of the network would not predict what is served")


def _export_server(network, output_dir, input_size, output_size, metadata, provenance):
    """The server form: export_merged.export_folder's files in ``output_dir``,
    opened as cellmap-models opens an export (CellmapModel, which checks
    metadata.json and that there is a model file) so that done means a
    server can open it. Returns the files' names."""
    from cellmap_models.model_export.cellmap_model import CellmapModel

    from cellmap_flow.finetune.export_merged import export_folder

    files = export_folder(network, str(output_dir), input_size, output_size, metadata, provenance)
    try:
        CellmapModel(str(output_dir))
    except ValueError as e:
        raise ValueError(f"The TorchScript export does not open as a cellmap-models export: {e}") from None
    return files


def _metadata(root, network, geometry, description) -> dict:
    """The server form's metadata.json: the base's own where the base is a
    cellmap-models export (its model type, framework, normalization...),
    else one made from the model's config, with what is this model's put in:
    its name, channels and voxel sizes, its input normalization as
    cellmap-flow's steps (``input_norm``, beside cellmap-models' fields,
    which its reader leaves be), and what it is. export_folder adds the
    block."""
    base = _base_metadata(root)
    # The base's training iterations and author are not this model's.
    for key in ("iteration", "author"):
        base.pop(key, None)
    what = f"{description['model']}: {description['checkpoint']}, exported by {MODULE}."
    return {
        "model_type": type(network).__name__,
        "framework": "torch",
        "spatial_dims": len(geometry.input_shape),
        **base,
        "model_name": description["model"],
        "in_channels": geometry.input_channels,
        "out_channels": geometry.output_channels,
        "channels_names": description["channels"],
        "input_voxel_size": description["input_voxel_size_nm"],
        "output_voxel_size": description["output_voxel_size_nm"],
        "input_norm": description["input_norm"],
        "description": f"{what} Its base: {base['description']}" if base.get("description") else what,
    }


def _base_metadata(root) -> dict:
    """The metadata.json of the model ``root`` (a model entry) when it is a
    cellmap-models export, ``type: cellmap``'s folder or ``type:
    huggingface``'s repository; {} for a model of another type."""
    if root.get("type") == "cellmap":
        path = os.path.join(root["folder_path"], "metadata.json")
    elif root.get("type") == "huggingface":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(root["repo"], "metadata.json", revision=root.get("revision"))
    else:
        return {}
    with open(path) as f:
        return json.load(f)


def _export_browser(network, folder_dir, folder, geometry, block, scale, description, variants, blocks=None) -> dict:
    """The browser form: export_onnx_web's files of ``network`` and their
    manifest, with ``description`` at its top, in ``folder_dir``, at each of
    ``blocks`` ((z, y, x) shapes; by default the one ``geometry`` gives),
    checked on the middle of ``block``, which holds the largest of them, and
    against ``scale``, its output's largest value. Returns the manifest.

    Raises:
        ValueError: the model is one export_onnx_web cannot export (see the
            module's docstring), or its fp32 file does not agree with it.
        ModuleNotFoundError: there is no onnx_web here.
    """
    try:
        from cellmap_models.model_export.onnx_web import VARIANTS, export_onnx_web
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"{e}. The web export needs cellmap-models' onnx_web (its onnx-web-export-and-pruning branch, "
            "which pixi.toml's finetune feature installs) and its export extra, onnx and onnxruntime."
        ) from e

    from cellmap_flow.models.configs.onnx import parse_manifest

    if geometry.input_channels != 1:
        raise ValueError(f"The model takes {geometry.input_channels} input channels; the browser feeds one")
    params = sum(p.numel() for p in network.parameters())
    if 4 * params >= ONNX_FILE_LIMIT:
        raise ValueError(f"The network has {params:,} parameters, {4 * params / 1e9:.1f} GB in float32: "
                         "export_onnx_web keeps the weights in the .onnx file, which protobuf caps at 2 GB")

    rows = export_onnx_web(network, str(folder_dir), folder, blocks or [tuple(geometry.input_shape)],
                           variants=variants or VARIANTS, validation_input=block, manifest_extra=description)
    for row in rows:
        if row["variant"] == "fp32" and row["max_abs"] > FP32_REL_TOL * max(scale, 1e-6):
            raise ValueError(f"The fp32 export differs from the network by up to {row['max_abs']:.3g}, "
                             f"{row['max_abs'] / max(scale, 1e-6):.2e} of its output's scale ({scale:.3g}), "
                             f"over the {FP32_REL_TOL:.0e} tolerance")
    manifest_path = folder_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parse_manifest(manifest, str(manifest_path))
    return manifest


def _said(e: Exception) -> str:
    """``e`` as the result says it: a ValueError's message, which says what
    to do, else its type too."""
    return str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"


def _write_result(output_dir: Path, result: dict) -> None:
    """``result.json``, written whole before it appears: export_status may read it at any moment."""
    partial = output_dir / f".{RESULT_NAME}.partial"
    partial.write_text(json.dumps(result))
    os.replace(partial, output_dir / RESULT_NAME)


def job_parser(description: str, output_help: str) -> argparse.ArgumentParser:
    """The command line start_job gives a job: ``--yaml``, ``--corrections``
    and ``--output`` (``output_help``); the module adds its own flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--yaml", required=True, help="the finetuned model's serving YAML")
    parser.add_argument("--corrections", required=True,
                        help="the session's corrections directory, which the model was trained on")
    parser.add_argument("--output", required=True, help=output_help)
    return parser


def run_job(args, work, log: logging.Logger, failed: str) -> int:
    """A job's body (export_web's and distill's ``main``): ``work(output_dir,
    variants)``'s result, or ``{"error"}`` when it raises (logged to ``log``
    after ``failed``), written to result.json (_write_result), which
    export_status reads, and printed as the output's last line. Returns
    the exit code: 1 on an error, else 0.

    ``args`` is job_parser's, with ``--variants`` (VARIANTS_HELP) besides.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    try:
        result = work(output_dir, variants)
    except Exception as e:
        log.exception(failed)
        result = {"error": _said(e)}
    _write_result(output_dir, result)
    print(json.dumps(result), flush=True)
    return 1 if "error" in result else 0


def main(argv=None) -> int:
    parser = job_parser(__doc__.split("\n\n")[0], "the directory to export into")
    parser.add_argument("--variants", default=None, help=VARIANTS_HELP)
    args = parser.parse_args(argv)
    return run_job(args, lambda output_dir, variants: export(args.yaml, args.corrections, output_dir, variants),
                   logger, "The export failed")


if __name__ == "__main__":
    sys.exit(main())
