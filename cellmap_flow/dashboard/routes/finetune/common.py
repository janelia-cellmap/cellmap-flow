import json
import logging
import os
from pathlib import Path

import zarr

from cellmap_flow.dashboard.finetune_utils import get_or_create_session_path
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)

USER_PREFS_FILE = os.path.expanduser("~/.cellmap_flow/user_prefs.json")
LOG_FILTER_PATTERNS = [
    r"^\s+base_model\.\S+\.lora_",
    r"^INFO:werkzeug:",
    r"^Array metadata \(scale=",
    r"^Host name:",
    r"^DEBUG trainer:",
]
RESTART_PASSTHROUGH_KEYS = [
    "lora_r",
    "lora_alpha",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "loss_type",
    "label_smoothing",
    "distillation_lambda",
    "margin",
    "balance_classes",
    "mask_unannotated",
    "gradient_accumulation_steps",
    "num_workers",
    "no_augment",
    "no_mixed_precision",
    "patch_shape",
    "output_type",
    "select_channel",
    "offsets",
]


def find_model_config(model_name):
    for model_config in getattr(g, "models_config", []) or []:
        if model_config.name == model_name:
            return model_config
    return None


def viewer_position_and_scales():
    if not hasattr(g, "viewer") or g.viewer is None:
        raise ValueError("Viewer not initialized")

    # .state, not .txn(): this only reads. txn() calls set_state() on exit
    # unconditionally, so using it here pushed a full viewer state -- built
    # from a snapshot that may predate a browser-side tool selection -- on
    # every crop creation, for no reason.
    s = g.viewer.state
    position = s.position
    dimensions = s.dimensions
    scales_nm = None

    if dimensions and hasattr(dimensions, "scales"):
        scales_nm = list(dimensions.scales)
        if hasattr(dimensions, "units"):
            units = dimensions.units
            if isinstance(units, str):
                units = [units] * len(scales_nm)
            converted_scales = []
            for scale, unit in zip(scales_nm, units):
                if unit == "m":
                    converted_scales.append(scale * 1e9)
                elif unit == "nm":
                    converted_scales.append(scale)
                else:
                    logger.warning(f"Unknown unit: {unit}, assuming nm")
                    converted_scales.append(scale)
            scales_nm = converted_scales

    if hasattr(position, "tolist"):
        position = position.tolist()
    elif hasattr(position, "__iter__"):
        position = list(position)

    return position, scales_nm


def ensure_corrections_storage(output_path):
    if output_path:
        session_path = get_or_create_session_path(output_path)
        corrections_dir = os.path.join(session_path, "corrections")
        os.makedirs(corrections_dir, exist_ok=True)
        zarr.open_group(corrections_dir, mode="a")
        return session_path, corrections_dir

    corrections_dir = os.path.expanduser("~/.cellmap_flow/corrections")
    os.makedirs(corrections_dir, exist_ok=True)
    zarr.open_group(corrections_dir, mode="a")
    return None, corrections_dir


def load_user_prefs():
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_user_prefs(prefs):
    try:
        os.makedirs(os.path.dirname(USER_PREFS_FILE), exist_ok=True)
        with open(USER_PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save user prefs: {e}")


def resolve_finetune_session(corrections_path_str):
    base_corrections_path = Path(corrections_path_str)
    if base_corrections_path.name == "corrections" and base_corrections_path.exists():
        return base_corrections_path.parent, base_corrections_path

    session_path = Path(get_or_create_session_path(str(base_corrections_path)))
    return session_path, session_path / "corrections"


def detect_sparse_annotations(corrections_path):
    try:
        for path in corrections_path.iterdir():
            if path.suffix == ".zarr" and (path / ".zattrs").exists():
                attrs = json.loads((path / ".zattrs").read_text())
                if attrs.get("source") == "sparse_volume":
                    return True
    except Exception as e:
        logger.warning(f"Error checking for sparse annotations: {e}")
    return False


def autodetect_output_type(model_config, output_type, offsets):
    from cellmap_flow.finetune.finetune_cli import _read_offsets_from_script

    resolved_output_type = output_type
    resolved_offsets = offsets

    if resolved_output_type is None:
        if hasattr(model_config, "script_path"):
            script_offsets = _read_offsets_from_script(model_config.script_path)
            if script_offsets is not None:
                resolved_output_type = "affinities"
                resolved_offsets = json.dumps(script_offsets)
                logger.info(
                    f"Auto-detected output_type='affinities' with "
                    f"{len(script_offsets)} offsets from model script"
                )

        if resolved_output_type is None:
            channels = None
            try:
                if hasattr(model_config, "_load_metadata"):
                    meta = model_config._load_metadata()
                    channels = meta.get("channels_names")
                elif getattr(model_config, "_config", None) is not None and hasattr(
                    model_config._config, "channels"
                ):
                    channels = model_config._config.channels
            except Exception:
                pass

            if not channels:
                # _config is only populated once something has built the model
                # in this process. The dashboard deliberately no longer does
                # that -- it asks the running server, or the geometry cache --
                # so _config is normally None here and this check silently
                # fell through to "binary" for an affinity model. Ask the same
                # sources, which now carry the channel names.
                from cellmap_flow.utils.model_geometry import resolve_model_geometry

                try:
                    geometry = resolve_model_geometry(
                        getattr(model_config, "name", None), model_config
                    )
                    channels = getattr(geometry, "channels", None) or getattr(
                        geometry, "channels_names", None
                    )
                except Exception as e:
                    logger.debug(f"Could not resolve channels for autodetect: {e}")

            if channels and any("_aff" in channel for channel in channels):
                resolved_output_type = "affinities"
                n_aff = sum(1 for channel in channels if "_aff" in channel)
                default_offsets = [
                    [1 if axis == index else 0 for axis in range(3)]
                    for index in range(min(n_aff, 3))
                ]
                resolved_offsets = json.dumps(default_offsets)
                logger.info(
                    f"Auto-detected output_type='affinities' from "
                    f"channel names: {channels}, offsets: {default_offsets}"
                )

        if resolved_output_type is None:
            resolved_output_type = "binary"

    if resolved_output_type == "affinities" and resolved_offsets is None:
        if hasattr(model_config, "script_path"):
            resolved_offsets = _read_offsets_from_script(model_config.script_path)
            if resolved_offsets is not None:
                logger.info(f"Auto-detected {len(resolved_offsets)} offsets from model script")
                resolved_offsets = json.dumps(resolved_offsets)
        if resolved_offsets is None:
            raise ValueError(
                "output_type='affinities' requires offsets. "
                "Define 'offsets' in the model script or pass them in the request."
            )
    elif isinstance(resolved_offsets, list):
        resolved_offsets = json.dumps(resolved_offsets)

    return resolved_output_type, resolved_offsets


def build_restart_params(data):
    updated_params = {}
    for key in RESTART_PASSTHROUGH_KEYS:
        if key in data and data[key] is not None:
            updated_params[key] = data[key]

    if "distillation_scope" in data and data["distillation_scope"] is not None:
        scope = str(data["distillation_scope"]).lower()
        if scope in {"all", "unlabeled"}:
            updated_params["distillation_all_voxels"] = scope == "all"
        else:
            logger.warning(f"Ignoring invalid distillation_scope: {data['distillation_scope']}")

    return updated_params


def get_lsf_job_id(finetune_job):
    if finetune_job.lsf_job:
        if hasattr(finetune_job.lsf_job, "job_id"):
            return finetune_job.lsf_job.job_id
        if hasattr(finetune_job.lsf_job, "process"):
            return f"PID:{finetune_job.lsf_job.process.pid}"
    return None


# Geometry the trainer cannot guess and will not run without.
_MANIFEST_REQUIRED_FIELDS = (
    "zarr_path",
    "dataset_path",
    "input_size",
    "output_size",
    "input_voxel_size",
    "output_voxel_size",
)


def write_volume_manifest(volume):
    """Mark a browser-painted annotation volume as trainable by the new path.

    ``create_dataloader`` picks its dataset by looking for this manifest and
    nothing else: present means VirtualPatchDataset, which streams patches
    from the volume zarr and is the only dataset that honours good regions.
    Absent means the legacy CorrectionDataset, which reads whatever
    per-chunk ``_chunk_*.zarr`` extracts the MinIO sync happened to
    materialize -- typically a handful of samples, one batch per epoch, and
    no notion of a good region at all.

    Only the YAML crop importer used to write one, so every session where
    you painted scribbles in the browser trained on the legacy path and
    silently ignored the regions you marked. Both volume-creating routes now
    call this.

    Returns the manifest path, or None when the volume record is too
    incomplete to describe (a resumed session whose .zattrs predates these
    fields, say) -- in which case the legacy path still applies, as before.
    """
    from cellmap_flow.finetune.virtual_dataset import write_manifest
    from cellmap_flow.globals import (
        current_input_norm_config,
        current_postprocess_config,
    )

    missing = [f for f in _MANIFEST_REQUIRED_FIELDS if not volume.get(f)]
    corrections_dir = volume.get("corrections_dir")
    if missing or not corrections_dir:
        logger.warning(
            "Not writing a virtual-sources manifest: volume record is missing "
            f"{missing or ['corrections_dir']}. Training will fall back to the "
            "legacy correction-chunk dataset, which ignores good regions."
        )
        return None

    manifest = {
        "kind": "volume_zarr_v1",
        "volume_zarr_path": volume["zarr_path"],
        "raw_dataset_path": volume["dataset_path"],
        "input_size_voxels": list(volume["input_size"]),
        "output_size_voxels": list(volume["output_size"]),
        "input_voxel_size_nm": list(volume["input_voxel_size"]),
        "output_voxel_size_nm": list(volume["output_voxel_size"]),
        # None means "one patch per populated chunk" -- full coverage of what
        # the user actually painted, rather than a fixed count.
        "patches_per_epoch": None,
        "jitter_voxels": None,
        "seed": 0,
        # The trainer runs on LSF where g.input_norms is empty, so the
        # normalization has to travel in the manifest. Without it the trainer
        # feeds the model raw uint8 while inference feeds it [-1, 1], and the
        # adapter is nonsense at inference time.
        "input_norm": current_input_norm_config(),
        "postprocess": current_postprocess_config(),
        # None -> auto-balance the dense and sparse pools.
        "dense_to_sparse_ratio": None,
    }
    path = write_manifest(str(corrections_dir), manifest)
    logger.info(
        f"Wrote virtual-sources manifest for {volume['zarr_path']} -> {path}; "
        "training will use VirtualPatchDataset and honour good regions."
    )
    return path
