import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import zarr

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session.minio import MINIO_PROXY_URL_ENV, proxied_url
from cellmap_flow.finetune.session.store import SessionStore
from cellmap_flow.models.geometry import channel_names_of

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
    "augment",
    "mask_unannotated",
    "gradient_accumulation_steps",
    "num_workers",
    "no_augment",
    "no_mixed_precision",
    "output_type",
    "select_channel",
    "offsets",
]


def find_model_config(model_name):
    for model_config in get_session().models_config or []:
        if model_config.name == model_name:
            return model_config
    return None


def current_chain():
    """The dashboard's chains as step lists, ``(input_norm, postprocess)``.

    A new volume and a training manifest record them, so that the trainer
    normalizes its input as inference does, and the finetuned model's YAML
    postprocesses as the dashboard does.
    """
    spec = get_session().pipeline_spec
    return list(spec.input_norm), list(spec.postprocess)


def viewer_position_and_scales():
    viewer = get_session().viewer
    if viewer is None:
        raise ValueError("Viewer not initialized")

    # .state, not .txn(): this only reads. txn() calls set_state() on exit
    # unconditionally, so using it here pushed a full viewer state -- built
    # from a snapshot that may predate a browser-side tool selection -- on
    # every crop creation, for no reason.
    s = viewer.state
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


def session_store():
    """The dashboard's sessions and volume registry (its session's dicts)."""
    session = get_session()
    return SessionStore(session.output_sessions, session.annotation_volumes)


def ensure_corrections_storage(output_path):
    if output_path:
        session_path = session_store().get_or_create(output_path)
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

    # This dashboard's session for the base path, if it made one; else the
    # newest one on disk, which is what a dashboard restart forgot. Only for
    # training: creating volumes still starts a session of its own.
    store = session_store()
    base = os.path.expanduser(str(base_corrections_path))
    if base not in get_session().output_sessions:
        latest = store.latest_on_disk(base)
        if latest is not None:
            logger.info(f"No session for {base} in this dashboard; using the latest on disk: {latest}")
            return Path(latest), Path(latest) / "corrections"

    session_path = Path(store.get_or_create(str(base_corrections_path)))
    return session_path, session_path / "corrections"


def detect_sparse_annotations(corrections_path):
    """Whether the session trains on painted (sparse) annotations.

    Read from the volume the manifest points at: annotated voxels outside
    its imported crops are painted. This looked for per-chunk extracts
    marked source == "sparse_volume", which are only written when there is
    no manifest -- and every session has one now -- so it was always False:
    the margin + distillation switch and the distance-model scribble guard in
    submit never fired, and mask_unannotated was never set. A session
    without a manifest cannot be trained, so it is not sparse either.
    """
    from cellmap_flow.finetune.session.manifest import has_painted_annotations, read_manifest

    try:
        manifest = read_manifest(str(corrections_path))
        if manifest and manifest.get("volume_zarr_path"):
            return has_painted_annotations(manifest["volume_zarr_path"])
    except Exception as e:
        logger.warning(f"Error checking for sparse annotations: {e}")
    return False


def autodetect_output_type(model_config, output_type, offsets):
    from cellmap_flow.finetune.target_transforms import read_offsets_from_script

    resolved_output_type = output_type
    resolved_offsets = offsets

    if resolved_output_type is None:
        if hasattr(model_config, "script_path"):
            script_offsets = read_offsets_from_script(model_config.script_path)
            if script_offsets is not None:
                resolved_output_type = "affinities"
                resolved_offsets = json.dumps(script_offsets)
                logger.info(
                    f"Auto-detected output_type='affinities' with "
                    f"{len(script_offsets)} offsets from model script"
                )

        if resolved_output_type is None:
            # Read as channel_names_of reads them: a string is one name. They
            # were iterated as they came, so a single "x_aff" channel was the
            # letters "x", "_", "a"... and never an affinity model.
            channels = None
            try:
                if hasattr(model_config, "_load_metadata"):
                    meta = model_config._load_metadata()
                    channels = channel_names_of(SimpleNamespace(channels_names=meta.get("channels_names")))
                elif getattr(model_config, "_config", None) is not None:
                    channels = channel_names_of(model_config._config)
            except Exception:
                pass

            if not channels:
                # _config is only populated once something has built the model
                # in this process. The dashboard deliberately no longer does
                # that -- it asks the running server, or the geometry cache --
                # so _config is normally None here and this check silently
                # fell through to "binary" for an affinity model. Ask the same
                # sources, which now carry the channel names.
                from cellmap_flow.models.geometry_cache import resolve_model_geometry

                try:
                    geometry = resolve_model_geometry(
                        getattr(model_config, "name", None), model_config
                    )
                    channels = channel_names_of(geometry)
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
            # The cellmap distance models announce themselves only by name
            # (e.g. cellmap/salivary_..._nuc_mouse_distance_32nm_...): their
            # metadata has a single plain channel name. Trained on a soft
            # tanh-distance target, they need the matching target type, not
            # a hard binary one that would flatten the output.
            names = " ".join(
                str(getattr(model_config, attr, "") or "")
                for attr in ("repo", "name", "model_name", "script_path", "checkpoint_path")
            ).lower()
            if "distance" in names:
                resolved_output_type = "distance"
                logger.info(
                    "Auto-detected output_type='distance' from the model name"
                )

        if resolved_output_type is None:
            resolved_output_type = "binary"

    if resolved_output_type == "affinities" and resolved_offsets is None:
        if hasattr(model_config, "script_path"):
            resolved_offsets = read_offsets_from_script(model_config.script_path)
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


class TrainingSettings(NamedTuple):
    """The target and loss a job trains with (see ``training_settings``)."""

    output_type: str
    loss_type: str
    label_smoothing: float
    distillation_lambda: float
    mask_unannotated: bool
    # The sentence submit's answer carries when the loss was switched for
    # sparse annotations, else None.
    note: str = None


SPARSE_MSE_NOTE = "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"


def training_settings(*, output_type, loss_type, label_smoothing, distillation_lambda, sparse):
    """What a job trains with, from what was asked for and whether its session is sparse.

    Some combinations cannot train, or would train the wrong thing, so they
    are replaced here, where the user sees it in the answer, rather than
    failing on the cluster:
    - sparse annotations with mse: margin loss, with distillation to the
      base model at 0.5, which is how sparse annotations train (see the
      distance case below);
    - a distance target with sparse annotations: a binary target with margin
      loss instead (see below);
    - a distance target otherwise: bce, without label smoothing.
    A sparse session also masks its unannotated voxels out of the loss.
    """
    note = None
    if sparse and loss_type == "mse":
        loss_type = "margin"
        distillation_lambda = 0.5
        note = SPARSE_MSE_NOTE
        logger.info(SPARSE_MSE_NOTE)

    if output_type == "distance" and sparse:
        # A distance target needs the 3D object boundary. Scribbles are
        # strokes with unannotated voxels all around them, so the safe
        # radius of every painted voxel is ~1 and next to nothing would be
        # supervised. Fall back to what sparse annotations already use:
        # a per-voxel binary target with margin loss (only the side of 0.5
        # is enforced, so the model's gradual field survives) and
        # distillation to the base model elsewhere.
        logger.info(
            "output_type=distance with sparse annotations: using binary "
            "target + margin loss instead (a distance transform needs dense 3D labels)"
        )
        output_type = "binary"
        loss_type = "margin"
        if distillation_lambda is None or distillation_lambda <= 0:
            distillation_lambda = 0.5
    elif output_type == "distance":
        # The soft distance target is only defined against BCE-with-logits;
        # margin/dice assume hard labels and smoothing would blur a target
        # that is already soft. The CLI rejects anything else.
        if loss_type != "bce" or label_smoothing:
            logger.info(
                f"output_type=distance: using bce loss without label smoothing "
                f"(requested loss_type={loss_type}, label_smoothing={label_smoothing})"
            )
        loss_type = "bce"
        label_smoothing = 0.0

    return TrainingSettings(
        output_type=output_type,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        distillation_lambda=distillation_lambda,
        mask_unannotated=bool(sparse),
        note=note,
    )


def build_restart_params(data):
    updated_params = {}
    for key in RESTART_PASSTHROUGH_KEYS:
        if key in data and data[key] is not None:
            updated_params[key] = data[key]

    # The trainer's flag is --no-augment. Send it alongside "augment" so a job
    # started before the trainer learned to map "augment" -- which it used to
    # drop -- still gets the toggle.
    if "augment" in updated_params and "no_augment" not in updated_params:
        augment = updated_params["augment"]
        if isinstance(augment, str):
            augment = augment.strip().lower() in ("true", "1", "yes", "on")
        updated_params["no_augment"] = not bool(augment)

    # --offsets is a JSON string on the trainer's side; a list would make its
    # json.loads() fail and kill the restart.
    if isinstance(updated_params.get("offsets"), (list, tuple)):
        updated_params["offsets"] = json.dumps(updated_params["offsets"])

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


def rewrite_minio_url_for_proxy(minio_url, request=None):
    """``minio_url`` as the browser can reach it through a reverse proxy.

    ``session.minio.proxied_url``, for ``request`` or else the Flask request
    being handled. Unless CELLMAP_FLOW_MINIO_PROXY_URL is set and the
    request came through a proxy (X-Forwarded-Host), and outside a request
    (a background import, a script), the URL is returned unchanged.
    """
    if request is None and os.environ.get(MINIO_PROXY_URL_ENV, "").strip():
        from flask import has_request_context, request as current_request

        if has_request_context():
            request = current_request
    return proxied_url(minio_url, request)


def write_volume_manifest(volume):
    """Mark an annotation volume as trainable by writing its manifest.

    ``create_dataloader`` requires this manifest: it is what points the
    trainer at the volume zarr to stream patches from, and it carries the
    patch geometry, the dense/sparse ratio and the dashboard's chains. The
    good regions beside it are honoured through the same dataset. Without
    one, training raises rather than fall back to anything.

    Returns the manifest path, or None when the volume record is too
    incomplete to describe (a resumed session whose .zattrs predates these
    fields, say). Nothing is guessed: such a session cannot be trained, and
    submit refuses it.
    """
    from cellmap_flow.finetune.session.manifest import write_manifest
    from cellmap_flow.finetune.session.volume import build_manifest

    try:
        if not volume.get("corrections_dir"):
            raise ValueError(f"The record of volume {volume.get('zarr_path')} has no corrections_dir.")
        input_norm, postprocess = current_chain()
        manifest = build_manifest(volume, input_norm=input_norm, postprocess=postprocess)
    except ValueError as e:
        logger.warning(f"Not writing a virtual-sources manifest: {e} The session cannot be trained without one.")
        return None
    path = write_manifest(str(volume["corrections_dir"]), manifest)
    logger.info(f"Wrote virtual-sources manifest for {volume['zarr_path']} -> {path}")
    return path
