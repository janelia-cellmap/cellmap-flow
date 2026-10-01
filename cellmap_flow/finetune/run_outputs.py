"""What each iteration of a finetune run leaves behind, and where.

For a run with output directory ``<run>`` (a dashboard run's is
``<session>/runs/<name>``):

- ``<run>/iterations/NNN_<ts>/lora_adapter/`` or ``.../full_finetune/``: the
  iteration's export (``iteration_dir``), so the YAML written for it keeps
  serving its weights after the next restart;
- ``<run>/lora_adapter`` or ``<run>/full_finetune``: a relative symlink to
  the newest export (``point_latest_export``);
- ``<models>/<name>_finetuned_<ts>.yaml``: the YAML that serves it
  (``write_serving_yaml``), in the session's ``models/`` (``models_dir``).

The job manager and the dashboard read these, and a job outlives a
dashboard upgrade, so the layout only ever changes compatibly.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from cellmap_flow.finetune.model_loading import root_base_model_dict

logger = logging.getLogger(__name__)


def finetuned_model_name(model_config, timestamp) -> str:
    """The name the model trained from ``model_config`` is served under."""
    return f"{model_config.name}_finetuned_{timestamp}"


def iteration_dir(output_dir, iteration: int, timestamp: str) -> Path:
    """The directory iteration ``iteration`` (from 1) exports into."""
    return Path(output_dir) / "iterations" / f"{iteration:03d}_{timestamp}"


def models_dir(args) -> Path:
    """Where the served-model YAMLs go: --models-dir, else the session's models/.

    A dashboard run's output directory is <session>/runs/<name>, so the
    session is two levels up: three put the YAMLs in <base>/models, outside
    the session and shared by all of them, and turned a headless
    --output-dir /nrs/x/run into /models, a permission error after training
    had succeeded. A run that is not under a runs/ directory keeps its YAMLs
    in its own models/.
    """
    if getattr(args, "models_dir", None):
        return Path(args.models_dir)
    output_dir = Path(args.output_dir)
    if output_dir.parent.name == "runs":
        return output_dir.parent.parent / "models"
    return output_dir / "models"


def point_latest_export(output_dir: Path, export_dir: Path, name: str) -> None:
    """Make <output_dir>/<name> the latest iteration's export.

    ``name`` is the strategy's export_name: lora_adapter or full_finetune.

    The old names stay, as relative symlinks to the newest export, for
    whatever reads them: finetune_export_kwargs, the completion check, and
    anyone who built <run>/lora_adapter by hand. A real directory there,
    from a run before per-iteration exports, is moved into iterations/
    first rather than deleted.
    """
    output_dir = Path(output_dir)
    link = output_dir / name
    target = Path(export_dir) / name
    if link.exists() and not link.is_symlink():
        keep = output_dir / "iterations" / f"000_before_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.move(str(link), str(keep / name))
        logger.info(f"Moved the earlier {name}/ to {keep / name}")
    tmp = output_dir / f".{name}.latest"
    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        os.symlink(os.path.relpath(target, output_dir), tmp, target_is_directory=True)
        os.replace(tmp, link)
    except OSError as e:
        # A filesystem without symlinks: keep the old name as a copy.
        logger.warning(f"Could not link {link} to {target} ({e}); copying it instead.")
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        shutil.copytree(target, link)


def write_serving_yaml(args, model_config, timestamp, *, is_lora: bool, export_dir):
    """Write the YAML that serves this iteration's export; its (model name, path).

    Args:
        args: the job's flags.
        model_config: the model the run trained.
        timestamp: the iteration's, as in its export directory.
        is_lora: whether the export is a LoRA adapter (else full weights).
            Pass what the model is: ``args.lora_r`` can be changed by a
            restart without changing the model.
        export_dir: the directory this iteration exported into
            (iteration_dir); the YAML points there.
    """
    from cellmap_flow.finetune.finetuned_model_templates import (
        generate_finetuned_model_yaml
    )

    name = finetuned_model_name(model_config, timestamp)
    yaml_dir = models_dir(args)
    yaml_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Generating model config for {name} in {yaml_dir}...")

    corrections_path = Path(args.corrections)
    manifest = {}
    try:
        from cellmap_flow.finetune.session.manifest import read_manifest

        manifest = read_manifest(str(corrections_path)) or {}
    except Exception as _e:
        logger.warning(f"Could not read the manifest in {corrections_path}: {_e}")

    # The raw data the model is served on: the manifest's, which is what it
    # was trained on; else the first correction zarr's attrs (which are also
    # a fallback source of normalization/postprocessing metadata below); else
    # the path the job serves. There is no placeholder: a YAML pointing at a
    # made-up path is worse than none, and generate_finetuned_model_yaml
    # refuses to write one.
    data_path = manifest.get("raw_dataset_path")
    zarr_dirs = sorted(corrections_path.glob("*.zarr"))
    zattrs_input_norm = None
    zattrs_postprocess = None
    if zarr_dirs:
        zattrs_file = zarr_dirs[0] / ".zattrs"
        if zattrs_file.exists():
            with open(zattrs_file) as f:
                metadata = json.load(f)
                data_path = data_path or metadata.get("dataset_path")
                zattrs_input_norm = metadata.get("input_norm")
                zattrs_postprocess = metadata.get("postprocess")

    if not data_path:
        logger.warning("Could not extract data_path from corrections, using serve_data_path")
        data_path = getattr(args, "serve_data_path", None)

    # Bake the training-time input_norm/postprocess into the generated yaml
    # so the served finetuned model gets queried with the same normalization
    # (and produces output through the same postprocessing, e.g.
    # SigmoidPostprocessor) the adapter was trained on. Without this,
    # training-vs-inference scale mismatch silently destroys finetuning
    # quality.
    #
    # Two correction workflows exist and store this differently:
    #  - the manifest-based workflow (_virtual_sources.json, written by
    #    yaml_crops.py) -- checked first, since it's kept fresh on restart.
    #  - the annotation-volume/MinIO workflow (stored directly on the
    #    correction zarr's own .zattrs, written by annotation_core.py /
    #    session.volume.create_volume_zarr) -- used as a fallback
    #    when no manifest exists.
    train_input_norm = manifest.get("input_norm") or zattrs_input_norm
    train_postprocess = manifest.get("postprocess") or zattrs_postprocess

    if train_input_norm or train_postprocess:
        json_data = {
            "input_norm": train_input_norm or {},
            "postprocess": train_postprocess or {},
        }
    else:
        json_data = None
        logger.warning(
            "Could not find training input_norm/postprocess in either the "
            "corrections manifest or the correction zarr's own attrs. "
            "Generated finetuned yaml will lack normalization metadata."
        )

    export_root = Path(export_dir)
    # No scale: the trainer read data_path at the model's input voxel size,
    # which picks the level of a multiscale group, and the server picks the
    # same one the same way. Naming a level could only disagree with it.
    #
    # The job's own queue and charge group, when the job manager passed them;
    # the template's defaults otherwise.
    scheduler = {
        key: getattr(args, key) for key in ("queue", "charge_group") if getattr(args, key, None)
    }
    yaml_path = generate_finetuned_model_yaml(
        **scheduler,
        lora_adapter_path=str(export_root / "lora_adapter") if is_lora else None,
        weights_path=None if is_lora else str(export_root / "full_finetune" / "model_state_dict.pt"),
        # A LoRA adapter was trained on top of the whole base, finetune
        # layers included, and is served on top of it. Full weights replace
        # every parameter, so they only need the base's module tree -- and on
        # a finetune base that tree would be a PeftModel their names no
        # longer match.
        base_model_dict=model_config.to_dict() if is_lora else root_base_model_dict(model_config),
        model_name=name,
        output_path=yaml_dir / f"{name}.yaml",
        data_path=data_path,
        json_data=json_data,
    )
    logger.info(f"Generated YAML: {yaml_path}")

    return name, yaml_path
