import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import worldarray as wa

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.options import OUTPUT_CLASSES, TrainingOptions
from cellmap_flow.finetune.session.minio import MINIO_PROXY_URL_ENV, proxied_url
from cellmap_flow.finetune.session.store import SessionStore
from cellmap_flow.models.geometry import channel_names_of
from cellmap_flow.portable import volume as portable_volume
# INSTANCE_TARGETS: the targets trained on instances, whose annotation
# volumes hold an id per object; imported from here by the routes.
from cellmap_flow.portable.volume import INSTANCE_TARGETS, annotation_dtype  # noqa: F401
from cellmap_flow.serving.client import fetch_model_info, running_job_host
from cellmap_flow.portable.label_it import affinity_offsets

logger = logging.getLogger(__name__)

USER_PREFS_FILE = os.path.expanduser("~/.cellmap_flow/user_prefs.json")
LOG_FILTER_PATTERNS = [
    r"^\s+base_model\.\S+\.lora_",
    r"^INFO:werkzeug:",
    r"^Array metadata \(scale=",
    r"^Host name:",
    r"^DEBUG trainer:",
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
    """The viewer's position, in its voxels, and each axis's scale in nm:
    z, y, x when its dimensions name all three (they are x, y, z there),
    else in the viewer's order."""
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

    names = list(getattr(dimensions, "names", None) or [])
    if all(axis in names for axis in "zyx"):
        order = [names.index(axis) for axis in "zyx"]
        if position is not None and len(position) == len(names):
            position = [position[i] for i in order]
        if scales_nm is not None and len(scales_nm) == len(names):
            scales_nm = [scales_nm[i] for i in order]
    return position, scales_nm


def session_store():
    """The dashboard's sessions and volume registry (its session's dicts)."""
    session = get_session()
    return SessionStore(session.output_sessions, session.annotation_volumes)


def ensure_corrections_storage(output_path):
    """``(session path, corrections directory)``: the corrections directory
    of ``output_path``'s session in this dashboard (made when it has none),
    or, with no ``output_path``, ``~/.cellmap_flow/corrections`` and no
    session. The directory is made, and made a zarr v2 group (its
    ``.zgroup``, written again each time with the same content), the
    volumes' parent."""
    if output_path:
        session_path = session_store().get_or_create(output_path)
        corrections_dir = os.path.join(session_path, "corrections")
    else:
        session_path, corrections_dir = None, os.path.expanduser("~/.cellmap_flow/corrections")
    os.makedirs(corrections_dir, exist_ok=True)
    wa.write.write_bare_group(corrections_dir, "zarr2")
    return session_path, corrections_dir


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


def declared_output_type(model_config):
    """The training target a model type declares (``finetune_output_type``), or None.

    Cellpose predicts flows, so its type says "flows". A finetuned model
    trains as its base does, so its ``base_model_config`` is asked in turn
    (building that config builds no network).
    """
    while model_config is not None:
        declared = getattr(type(model_config), "finetune_output_type", None)
        if declared:
            return declared
        try:
            model_config = getattr(model_config, "base_model_config", None)
        except Exception as e:  # an entry whose base cannot be built declares nothing
            logger.debug(f"Could not read the base model's training target: {e}")
            return None
    return None


def annotation_dtype_for(model_config):
    """The labels' dtype of a new annotation volume for ``model_config``:
    ``cellmap_flow.portable.volume.annotation_dtype`` (the browser's rule too)
    of what it trains on (``autodetect_output_type``), uint16 for the
    instance targets in INSTANCE_TARGETS, else uint8; uint8 when that cannot
    be told."""
    try:
        output_type, _ = autodetect_output_type(model_config, None, None)
    except Exception as e:
        logger.info(f"Could not tell what {getattr(model_config, 'name', 'the model')} trains on ({e}): uint8 labels")
        return "uint8"
    return annotation_dtype(output_type)


# The targets with a channel for each of the model's: a model trained on one
# trains every channel at once, never one at a time. binary_broadcast is the
# one binary target given to every channel, which is asking for all of them.
EVERY_CHANNEL_TARGETS = ("affinities", "flows", "binary_broadcast")


def model_target(model_config, channel_names):
    """The training target a model says it has without being built:
    ``(output_type, offsets)``, the offsets a list or None; ``(None, None)``
    when nothing says.

    In turn: the target its type declares (``declared_output_type``:
    Cellpose's flows); the offsets its script assigns, read by parsing it
    (an affinity model's); the affinity offsets its output channel names
    give (portable/label_it.affinity_offsets: the browser's Python applies
    the same rule to its manifest's names, for training and seeding, so
    both modes treat such a model the same way). ``channel_names`` is a
    function of no arguments giving those names (None when they are not
    known), called only when the type and the script say nothing: finding
    them can cost a download, or the model's geometry.

    The one rule for what a model trains on, which ``autodetect_output_type``
    (at submit) and ``channel_choice`` (the models list, every two seconds)
    both apply.
    """
    from cellmap_flow.finetune.target_transforms import read_offsets_from_script

    declared = declared_output_type(model_config)
    if declared is not None:
        return declared, None
    if hasattr(model_config, "script_path"):
        script_offsets = read_offsets_from_script(model_config.script_path)
        if script_offsets is not None:
            return "affinities", script_offsets
    names = channel_names()
    offsets = affinity_offsets(names) if names else None
    if offsets is not None:
        return "affinities", offsets
    return None, None


def trains_one_channel(output_type, output_channels) -> bool:
    """Whether a model with ``output_channels`` output channels, trained on
    ``output_type``, trains one channel at a time (``select_channel``): more
    than one channel, on a target that is not one of EVERY_CHANNEL_TARGETS.
    The paint is then about one channel: the binary target on every channel
    would teach the paint's class to each (er and nucleus would learn
    "mito")."""
    return int(output_channels or 1) > 1 and output_type not in EVERY_CHANNEL_TARGETS


def channel_choice(model_config, output_channels, channel_names) -> bool:
    """The models list's ``channel_choice``: whether ``model_config``, with
    ``output_channels`` output channels named ``channel_names`` (None when
    unknown), trains one channel at a time (``trains_one_channel``) on the
    target ``model_target`` says. ``model_config`` may be None (a job served
    without a config), and then only the names say. Never builds the model:
    the list is asked every two seconds."""
    output_type, _ = model_target(model_config, lambda: channel_names)
    return trains_one_channel(output_type, output_channels)


def _channel_names_for_autodetect(model_config):
    """The output channel names of ``model_config`` for autodetect_output_type,
    or None: from a Hugging Face model's metadata.json, a built config, else
    the model's geometry (resolve_model_geometry: its server's, the cache's,
    or the model built here).

    Read as channel_names_of reads them: a string is one name, so a single
    "x_aff" channel is not the letters "x", "_", "a"...
    """
    channels = None
    try:
        if hasattr(model_config, "_load_metadata"):
            meta = model_config._load_metadata()
            channels = channel_names_of(SimpleNamespace(channels_names=meta.get("channels_names")))
        elif getattr(model_config, "_config", None) is not None:
            channels = channel_names_of(model_config._config)
    except Exception:
        pass
    if channels:
        return channels
    # _config is only populated once something has built the model in this
    # process, which the dashboard does not do: it asks the running server,
    # or the geometry cache, so ask those, which carry the channel names.
    # Without them an affinity model would train as "binary".
    from cellmap_flow.models.geometry_cache import resolve_model_geometry

    try:
        return channel_names_of(resolve_model_geometry(getattr(model_config, "name", None), model_config))
    except Exception as e:
        logger.debug(f"Could not resolve channels for autodetect: {e}")
        return None


def autodetect_output_type(model_config, output_type, offsets):
    """The training target of ``model_config``: ``(output_type, offsets)``,
    the offsets as JSON or None.

    ``output_type`` and ``offsets`` when given. Else what the model says
    (``model_target``, with its channel names read as
    ``_channel_names_for_autodetect`` reads them); else a distance model's,
    by its name; else binary. An affinity target without offsets takes its
    script's, or raises ValueError.
    """
    from cellmap_flow.finetune.target_transforms import read_offsets_from_script

    resolved_output_type = output_type
    resolved_offsets = offsets

    if resolved_output_type is None:
        resolved_output_type, found_offsets = model_target(
            model_config, lambda: _channel_names_for_autodetect(model_config)
        )
        if found_offsets is not None:
            resolved_offsets = json.dumps(found_offsets)
        if resolved_output_type is not None:
            logger.info(
                f"Auto-detected output_type={resolved_output_type!r}"
                + (f" with offsets {found_offsets}" if found_offsets is not None else "")
                + " from the model (its type, its script or its channel names)"
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


SPARSE_MSE_NOTE = "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"


def settings_note(asked: TrainingOptions, used: TrainingOptions) -> Optional[str]:
    """The sentence submit's answer carries when the loss was switched for sparse
    annotations (TrainingOptions.for_session), else None: the form still shows
    what was entered, so say what was used."""
    if used.loss_type == "interval" and asked.loss_type != "interval":
        return f"Distance model on scribbles: trained on distance bounds, distillation {used.distillation_lambda:g}."
    if asked.loss_type == "mse" and used.loss_type == "margin":
        return SPARSE_MSE_NOTE
    return None


def served_output_class(model_name) -> Optional[str]:
    """What the model's running server says its output is (model_info's
    ``output_class``, serving.probe), the reading ranking and seeding share;
    None when no server for it is up, or it has not predicted yet, and the
    training job then probes the model itself."""
    host = running_job_host(model_name)
    if not host:
        return None
    output_class = fetch_model_info(host).get("output_class")
    return output_class if output_class in OUTPUT_CLASSES else None


def restart_params(body) -> dict:
    """The training settings a restart body (requests.FinetuneRestart) changes, as
    the job's trainer takes them (cli.RESTARTABLE_ARGS). A job started before
    this version knows three of them by other names, so those go beside the
    new: ``no_augment`` with ``augment`` (a restart could never switch
    augmentation on or off while the toggle was dropped), ``no_mixed_precision``
    with ``mixed_precision`` and ``distillation_all_voxels`` with
    ``distillation_scope``."""
    params = body.changes()
    if "augment" in params:
        params["no_augment"] = not params["augment"]
    if "mixed_precision" in params:
        params["no_mixed_precision"] = not params["mixed_precision"]
    if "distillation_scope" in params:
        params["distillation_all_voxels"] = params["distillation_scope"] == "all"
    return params


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
    """Make an annotation volume its session's, trainable, by writing its manifest.

    ``create_dataloader`` requires this manifest: it is what points the
    trainer at the volume zarr to stream patches from, and it carries the
    patch geometry, the dense/sparse ratio and the dashboard's chains. The
    good regions beside it are honoured through the same dataset. Without
    one, training raises rather than fall back to anything. It also says
    which volume is the session's (SessionStore.session_volume).

    Returns the manifest path, or None when the volume record is too
    incomplete to describe (a resumed session whose .zattrs predates these
    fields, say). Nothing is guessed: such a session cannot be trained, and
    submit refuses it.

    A volume painted before any model gets none (portable/volume.py
    training_refusal: it is on no model's grid), and any manifest in its
    corrections dir is removed: one naming the volume before it would keep
    that one the session's, so Rank and Use this model would read the old
    paint while the user paints the new, and a restart or a resumed copy
    would train on it.
    """
    from cellmap_flow.finetune.session.manifest import remove_manifest, write_manifest
    from cellmap_flow.finetune.session.volume import build_manifest

    if portable_volume.training_refusal(volume) and volume.get("corrections_dir"):
        if remove_manifest(str(volume["corrections_dir"])):
            logger.info(f"Removed the manifest in {volume['corrections_dir']}: its volume is now "
                        f"{volume.get('zarr_path')}, painted before any model, which is not trained on.")
        return None
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
