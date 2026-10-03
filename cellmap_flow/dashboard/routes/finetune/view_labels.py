"""Labelling the box on screen in one click: from the model's prediction, or all background.

Two buttons beside "Mark This View as Good", over the same box (one model
output patch centred where the viewer looks, ``good_regions.view_box_nm``,
or ``box_voxels`` annotation voxels when the request gives them):

- POST ``/api/finetune/view-labels/seed``: the model's prediction there,
  segmented into objects, each written with an id of its own (2 and up),
  and background 1. The user then cleans it up with the brush; it is far
  quicker than painting a whole object from nothing. How it is segmented
  is the request's ``method``, one of the ``post.segment`` segmenters that
  fits what the model outputs (``SEED_METHODS``); ``seed_sources`` says
  which fit each running model.
- POST ``/api/finetune/view-labels/background``: every unannotated voxel 1,
  for a region the model fills with false positives.

Both fill only unannotated voxels (``session.fill``): a voxel the user
painted is a decision, a seed is a guess. The write goes to MinIO, which
neuroglancer reads, and the changed chunks are pulled to disk at once.
Neuroglancer keeps the chunks it has already read, so the page reloads the
viewer afterwards to show the new labels.

A box over LARGE_BOX_VOXELS voxels is only labelled when the request says
``confirm: true``; without it the answer is a 409 with
``needs_confirmation``, for the page to ask first.
"""

import functools
import json
from collections import deque
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import zarr
from flask import jsonify, request
from scipy.special import expit

from cellmap_flow.dashboard.finetune_utils import sync_annotation_volume_from_minio
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    INSTANCE_TARGETS,
    autodetect_output_type,
    find_model_config,
    session_store,
)
from cellmap_flow.dashboard.routes.finetune.good_regions import view_box_nm
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotation_layer
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import fill
from cellmap_flow.finetune.session.volume import volume_corner_nm
from cellmap_flow.io.ome import ome_corner
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post import segment
from cellmap_flow.serving.client import fetch_model_info
from cellmap_flow.serving.probe import SIGNED_UNIT, UNIT, classify_output_range
from cellmap_flow.serving.protocol import ARGS_KEY
from cellmap_flow.viewer.layers import prediction_voxel_override

logger = logging.getLogger(__name__)

# Past this many voxels (128^3) the page asks before labelling: a seed of
# that size is a lot to clean up, and all-background a lot to undo. One
# output patch, the default box, is well under it for the models in use.
LARGE_BOX_VOXELS = 128**3

# A chunk can be the first the server computes, which waits on its warm-up.
PREDICTION_TIMEOUT_SECONDS = 120
# Metadata only, asked while the page polls the picker: a server that does
# not answer by then is left out of the methods until it does.
METADATA_TIMEOUT_SECONDS = 5

# How a seed makes objects of the prediction (``post.segment``), each fitting
# a kind of output; ``seed_methods`` says which a model offers.
SEED_METHODS = (
    # Threshold, then an id per connected object: any output with a
    # foreground threshold. The default.
    "components",
    # Affinity models: their offset channels, joined above the threshold.
    "mutex_watershed",
    # Distance models: touching objects cut apart where the distance dips.
    "distance_watershed",
    # A server that serves integer instance labels (Cellpose's masks).
    "instances",
)

# Training targets made of instances, where ids must differ between nearby
# objects (finetune.cli's output types).


class _Refused(Exception):
    """A request these routes will not act on: ``status`` and a message."""

    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def _error(message, status, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


def _target_box(data):
    """``(volume_id, volume, lo, hi)``: the session's volume and the box's voxels in it."""
    volume_id, volume = session_store().session_volume()
    if volume is None:
        raise _Refused("No annotation volume to label. Create or resume one first.")
    state = get_session().minio_state
    if not state.get("ip") or not state.get("port"):
        raise _Refused("MinIO is not serving the annotation volume. Create or resume one first.")
    try:
        size_nm = data.get("size_nm")
        if data.get("box_voxels") is not None:
            size_nm = _box_nm(data["box_voxels"], volume)
        centre_nm, size_nm = view_box_nm(size_nm)
    except ValueError as e:
        raise _Refused(str(e), 400)
    shape = zarr.open_array(f"{volume['zarr_path']}/annotation/s0", mode="r").shape
    box = fill.box_voxels(volume, centre_nm - size_nm / 2, size_nm, shape)
    if box is None:
        raise _Refused("The view is outside the annotation volume.", 400)
    return volume_id, volume, box[0], box[1]


def _box_nm(box_voxels, volume):
    """The size in nm of a box of ``box_voxels`` annotation voxels (z, y, x,
    or one number for every axis).

    Any size: the box only says which of the served prediction to copy, and
    the model reads its own input around it whatever its size. A box
    smaller than a patch is less to check and fix by hand, and the voxels
    left unpainted around it are left out of training."""
    refusal = f"The box is z, y, x voxels, or one number, each at least 1; got {box_voxels}"
    try:
        voxels = np.asarray(box_voxels, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(refusal) from None
    if voxels.size not in (1, 3) or not np.all(np.isfinite(voxels)) or np.any(voxels < 1):
        raise ValueError(refusal)
    return (np.repeat(voxels, 3 // voxels.size) * np.asarray(volume["output_voxel_size"], dtype=float)).tolist()


def _confirmation_needed(data, lo, hi):
    """The 409 asking the page to confirm a large box, or None."""
    voxels = int(np.prod(np.asarray(hi) - np.asarray(lo)))
    if voxels <= LARGE_BOX_VOXELS or data.get("confirm"):
        return None
    return _error(
        f"The box is {voxels:,} voxels. Label it all?", 409, needs_confirmation=True, voxels=voxels
    )


# How many one-click label actions per volume Undo can take back.
UNDO_DEPTH = 10


def _undo_stack(volume_id):
    return get_session().label_undo.setdefault(volume_id, deque(maxlen=UNDO_DEPTH))


def _after_write(volume_id, what):
    """Pull the written box to disk and re-read the paint layer; whether the layer was refreshed."""
    # Now rather than at the next periodic sync, so training submitted
    # straight after sees the box. A failure is only late: the periodic
    # sync pulls the same chunks.
    try:
        sync_annotation_volume_from_minio(volume_id)
    except Exception as e:
        logger.warning(f"Could not pull the {what} box of {volume_id} to disk yet: {e}")
    return _refresh_layer(volume_id)


def _fill(volume_id, volume, lo, hi, labels_for):
    """Fill the box in MinIO and pull it to disk; returns the answer's counts."""
    n_foreground, n_background = fill.fill_unpainted(
        get_session().minio_state, volume_id, lo, hi, labels_for, volume.get("zarr_path"),
        undo=_undo_stack(volume_id),
    )
    layer_refreshed = False
    if n_foreground or n_background:
        layer_refreshed = _after_write(volume_id, "labelled")
    logger.info(
        f"Labelled box {lo.tolist()}..{hi.tolist()} of {volume_id}: "
        f"{n_foreground} foreground and {n_background} background voxels filled"
    )
    return {
        "success": True,
        "offset_voxels": lo.tolist(),
        "shape_voxels": (hi - lo).tolist(),
        "filled_foreground": n_foreground,
        "filled_background": n_background,
        # Neuroglancer shows the chunks it already read until the paint layer
        # is re-read: done here, under a new URL, when the layer was found;
        # else the page reloads the viewer.
        "reload_viewer": bool(n_foreground or n_background),
        "layer_refreshed": layer_refreshed,
        "can_undo": bool(_undo_stack(volume_id)),
    }


def _refresh_layer(volume_id) -> bool:
    try:
        return refresh_annotation_layer(get_session().viewer, volume_id)
    except Exception as e:
        logger.warning(f"Could not refresh the paint layer of {volume_id}: {e}")
        return False


# ---------------------------------------------------------------------------
# The prediction
# ---------------------------------------------------------------------------

def _prediction_sources():
    """``[(model_name, host)]``: every running server, oldest first.

    Any of them can seed: a prediction is read where the annotation voxels
    lie, whatever the server's voxel size. Only listing the volume's model
    and its finetunes left the picker empty before a volume existed, and
    whenever a job's name differed from the one the volume recorded.
    """
    return [
        (job.model_name, job.host)
        for job in get_session().jobs
        if getattr(job, "model_name", None) and getattr(job, "host", None)
    ]


def _prediction_server(base_model, chosen=None):
    """``(model_name, host)`` of the server whose prediction a seed reads.

    ``chosen``, when the page names a running one. Else the latest
    finetuned iteration of the model the volume is for (finetune_layers
    names each iteration's job ``<base>_finetuned...``), when its server is
    up: it is the model as it is now. Else the model's own server, else
    the only server running. (None, None) when there is no pick to make.
    """
    sources = _prediction_sources()
    if chosen:
        for name, host in sources:
            if name == chosen:
                return name, host
        raise _Refused(f"No running server for {chosen}. Pick another model to seed from.")
    if base_model:
        finetuned = [s for s in sources if s[0].startswith(f"{base_model}_finetuned")]
        own = [s for s in sources if s[0] == base_model]
        if finetuned or own:
            return (finetuned or own)[-1]
    return sources[0] if len(sources) == 1 else (None, None)


def _serves_cellpose_flows(model_name):
    """Whether ``model_name`` is a Cellpose model served with ``output: flows``
    (channels flow_y, flow_x, cell): its instances are Cellpose's masks,
    which its server makes on asking (CellposeMasksPostprocessor)."""
    model_config = find_model_config(model_name)
    return getattr(type(model_config), "cli_name", None) == "cellpose" and getattr(model_config, "output", None) == "flows"


# Cellpose's flows output: the cell probability is its third channel.
CELLPOSE_CELL_CHANNEL = 2


def _model_read_as(model_name, base_model):
    """The model whose config says what ``model_name`` outputs: a finetune's is its base's."""
    if base_model and model_name.startswith(f"{base_model}_finetuned"):
        return base_model
    return model_name


def model_output(model_name):
    """``(output_type, offsets)``: what the model is trained on, as submit decides it.

    ``autodetect_output_type``: "affinities" (with the offsets, a JSON
    list) when the model's script or channel names say so, "distance" for
    a distance model, a type's declared target ("flows" for Cellpose), else
    "binary". A model with no config here, or affinities without offsets,
    is "binary": a threshold is all a seed can do with it.
    """
    model_config = find_model_config(model_name)
    if model_config is None:
        return "binary", None
    try:
        return autodetect_output_type(model_config, None, None)
    except ValueError as e:
        logger.info(f"Seeding from {model_name} as a binary model: {e}")
        return "binary", None


def methods_for(output_type, serves_integers):
    """The seed methods that fit a model's output, the best fit first.

    ``components`` fits every output (an instance label is foreground where
    it is not 0), so it is always offered, last unless nothing else fits.
    """
    methods = []
    if serves_integers:
        methods.append("instances")
    if output_type == "affinities":
        methods.append("mutex_watershed")
    if output_type == "distance":
        methods.append("distance_watershed")
    return methods + ["components"]


def _served_base(host, model_name, postprocess=()):
    """The URL of the model's served array, with the dashboard's input
    normalization and ``postprocess`` (none by default)."""
    blob = PipelineSpec.from_steps(get_session().input_norms, postprocess).to_url_blob()
    return f"{host.rstrip('/')}/{model_name}{ARGS_KEY}{blob}{ARGS_KEY}"


def _serves_integers(host, model_name):
    """Whether the model's raw output is integer labels, from the served array's dtype; None when unknown."""
    try:
        meta = _get(f"{_served_base(host, model_name)}/s0/.zarray", METADATA_TIMEOUT_SECONDS).json()
        return np.dtype(meta["dtype"]).kind in "iu"
    except Exception as e:
        logger.debug(f"Could not read {model_name}'s served dtype: {e}")
        return None


# {(model name, host, base model): (job, methods)}, so the page's polling
# asks each server once. A restarted job is a new object, and is asked again.
_methods_seen = {}
# {(model name, host, base model): when it last failed to answer}: one that
# did not is asked again only after this long, rather than on every poll of
# the page, each of which waited METADATA_TIMEOUT_SECONDS on it.
_unanswered = {}
UNANSWERED_RETRY_SECONDS = 60


def seed_methods(model_name, host, base_model=None):
    """The methods a seed from ``model_name``'s server can use (``methods_for``)."""
    job = next(
        (j for j in get_session().jobs if getattr(j, "model_name", None) == model_name
         and getattr(j, "host", None) == host),
        None,
    )
    key = (model_name, host, base_model)
    seen = _methods_seen.get(key)
    if seen is not None and job is not None and seen[0] is job:
        return seen[1]
    failed_at = _unanswered.get(key)
    if failed_at is not None and time.monotonic() - failed_at < UNANSWERED_RETRY_SECONDS:
        integers = None
    else:
        integers = _serves_integers(host, model_name)
        if integers is None:
            _unanswered[key] = time.monotonic()
        else:
            _unanswered.pop(key, None)
    read_as = _model_read_as(model_name, base_model)
    methods = methods_for(model_output(read_as)[0], bool(integers) or _serves_cellpose_flows(read_as))
    if integers is not None and job is not None:
        _methods_seen[key] = (job, methods)
    return methods


@finetune_bp.route("/api/finetune/view-labels/sources", methods=["GET"])
def seed_sources():
    """The models a seed can read, for the page's picker: ``models`` (every
    running server, oldest first), ``default``, the one a seed reads when
    none is chosen (None when that is a choice for the user), and
    ``methods``, {model: the seed methods that fit its output, best first};
    and ``box_voxels``, the volume's output patch (z, y, x), the box a label
    action covers when it gives no other."""
    volume_id, volume = session_store().session_volume()
    base_model = (volume or {}).get("model_name")
    sources = _prediction_sources()
    default = _prediction_server(base_model)[0]
    methods = {name: seed_methods(name, host, base_model) for name, host in sources}
    # The page polls this, so it also says whether Undo has anything to take back.
    can_undo = bool(volume_id and get_session().label_undo.get(volume_id))
    return jsonify({
        "success": True, "models": [name for name, _ in sources], "default": default,
        "methods": methods, "can_undo": can_undo, "box_voxels": (volume or {}).get("output_size"),
    })


def _get(url, timeout=PREDICTION_TIMEOUT_SECONDS):
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"{url.split(ARGS_KEY)[0]}... answered HTTP {response.status_code}")
    return response


def read_prediction(host, model_name, volume, lo, hi, info, postprocess=()):
    """The model's raw output over annotation voxels ``[lo, hi)``.

    Read from its server as a layer is, with the dashboard's input
    normalization and no postprocessing: the trainer's target is set on the
    raw output, and that is where its decision boundary is. (``postprocess``
    asks for one anyway: Cellpose's masks of a flows server's output, for a
    seed of its instances.) Each annotation
    voxel takes the served voxel its centre lies in, placed where
    neuroglancer draws the layer (``prediction_voxel_override``), so the
    seed lies under the prediction the user sees.

    Returns ``(prediction, lo, hi)``: a (channels, z, y, x) array, float32
    or, when the server serves integer labels, as served (an id past 2^24
    is not exact in float32); and the box, shrunk to what the server covers.
    """
    base = _served_base(host, model_name, postprocess)
    multiscale = _get(f"{base}/.zattrs").json()["multiscales"][0]
    if [axis["name"] for axis in multiscale["axes"][:3]] != ["z", "y", "x"]:
        raise RuntimeError(f"{model_name}'s server serves axes {multiscale['axes']}, not z, y, x first")
    transforms = {t["type"]: t for t in multiscale["datasets"][0]["coordinateTransformations"]}
    declared = np.asarray(transforms["scale"]["scale"][:3], dtype=float)
    corner_nm = np.asarray(ome_corner(transforms["translation"]["translation"][:3], declared))
    drawn = np.asarray(
        prediction_voxel_override(host, get_session().dataset_path, info) or declared, dtype=float
    )
    # Neuroglancer keeps the served offset in voxels when it redraws the
    # layer at another voxel size.
    served_corner_nm = corner_nm / declared * drawn
    meta = _get(f"{base}/s0/.zarray").json()

    voxel_size = np.asarray(volume["output_voxel_size"], dtype=float)
    volume_corner = volume_corner_nm(volume.get("dataset_offset_nm"), voxel_size)
    indices, new_lo, new_hi = [], lo.copy(), hi.copy()
    for axis in range(3):
        centres = volume_corner[axis] + (np.arange(lo[axis], hi[axis]) + 0.5) * voxel_size[axis]
        index = np.floor((centres - served_corner_nm[axis]) / drawn[axis]).astype(int)
        inside = np.flatnonzero((index >= 0) & (index < meta["shape"][axis]))
        if not inside.size:
            raise _Refused("The view is outside what the model's server predicts.", 400)
        new_lo[axis], new_hi[axis] = lo[axis] + inside[0], lo[axis] + inside[-1] + 1
        indices.append(index[inside])

    # Fetched together, then read through zarr as the array they belong to.
    chunk = np.asarray(meta["chunks"][:3])
    first = np.array([i[0] for i in indices]) // chunk
    last = np.array([i[-1] for i in indices]) // chunk
    channel_key = ".0" if len(meta["shape"]) == 4 else ""
    keys = [
        ".".join(str(int(v)) for v in first + np.asarray(offset)) + channel_key
        for offset in np.ndindex(*(last - first + 1))
    ]
    with ThreadPoolExecutor(max_workers=min(len(keys), 8)) as pool:
        contents = list(pool.map(lambda key: _get(f"{base}/s0/{key}").content, keys))
    store = {".zarray": json.dumps(meta).encode(), **dict(zip(keys, contents))}
    served = zarr.open_array(store=store, mode="r")
    region = served[tuple(slice(i[0], i[-1] + 1) for i in indices)]
    prediction = region[np.ix_(*(i - i[0] for i in indices))]
    if prediction.dtype.kind not in "iu":
        prediction = prediction.astype(np.float32)
    prediction = np.moveaxis(prediction, -1, 0) if channel_key else prediction[None]
    return prediction, new_lo, new_hi


def probability(prediction, output_class=None, channels=None):
    """The prediction as a probability, (channels, z, y, x), 0.5 at the model's decision boundary.

    The boundary depends on the output's activation: 0.5 on [0, 1] output
    (a sigmoid already applied; the cellmap distance models end in one, and
    0.5 is their object boundary), 0 on tanh's [-1, 1], and 0 on unbounded
    output, logits or a signed distance alike. Each is turned into a
    probability, so ``channels`` -- the affinity channels, say -- can be
    averaged and one threshold means the same for every model.
    ``output_class`` is the server's (serving.probe), or read off the
    values when it has none.
    """
    if channels is not None:
        prediction = prediction[channels]
        if not prediction.shape[0]:
            raise ValueError(f"The model has no channel {channels.start}")
    prediction = prediction.astype(np.float32, copy=False)
    output_class = output_class or classify_output_range(float(prediction.min()), float(prediction.max()))
    if output_class == UNIT:
        return prediction
    if output_class == SIGNED_UNIT:
        return (prediction + 1) / 2
    return expit(prediction)


def foreground_mask(prediction, output_class=None, channels=None, threshold=0.5):
    """Where the model says foreground: its ``probability``, averaged over ``channels``, over ``threshold``.

    ``threshold`` is in probability whatever the activation: 0.5 is the
    model's own boundary.
    """
    return probability(prediction, output_class, channels).mean(axis=0) > threshold


def _seed_plan(base_model, select_channel):
    """``(channels, output_type, offsets)``: the channels a seed reads, and what the model outputs.

    As the trainer targets them: the selected channel; else an affinity
    model's offset channels (the rest, LSDs say, are masked out of its
    loss); else every channel, which a binary target is broadcast over.
    ``output_type`` and ``offsets`` are ``model_output``'s.
    """
    output_type, offsets = model_output(base_model)
    if select_channel is not None:
        channel = int(select_channel)
        if channel < 0:
            raise ValueError(f"select_channel must be 0 or more, got {channel}")
        return slice(channel, channel + 1), output_type, offsets
    if output_type == "affinities":
        return slice(0, len(json.loads(offsets))), output_type, offsets
    if _serves_cellpose_flows(base_model):
        # Thresholding reads the probability, not the flows beside it.
        return slice(CELLPOSE_CELL_CHANNEL, CELLPOSE_CELL_CHANNEL + 1), output_type, offsets
    return None, output_type, offsets


def segment_prediction(method, prediction, output_class, channels, offsets, threshold=0.5, min_size=0,
                       connectivity=1, per_slice=False):
    """The objects ``method`` makes of the prediction: 0 background, 1..n (``post.segment``).

    ``threshold`` is a probability (``probability``) for every method that
    thresholds: the foreground boundary for components and the distance
    watershed, and the mutex watershed's bias, the affinity above which
    two voxels join. ``connectivity`` and ``per_slice`` are the connected
    components'; the watersheds take ``connectivity`` too. ``instances``
    splits an id's disconnected parts: a server numbers each chunk's
    objects on its own (Cellpose's from 1), and the box can span chunks.
    """
    if method == "components":
        mask = foreground_mask(prediction, output_class, channels, threshold)
        return segment.connected_components(mask, connectivity, min_size, per_slice)
    if method == "mutex_watershed":
        offsets = json.loads(offsets)
        affinities = probability(prediction, output_class, slice(0, len(offsets)))
        return segment.mutex_watershed(affinities, offsets, bias=threshold, min_size=min_size)
    if method == "distance_watershed":
        distance = probability(prediction, output_class, channels).mean(axis=0)
        return segment.distance_watershed(distance, threshold, connectivity=connectivity, min_size=min_size)
    if method == "instances":
        channel = channels.start if channels is not None else 0
        if prediction.dtype.kind not in "iu":
            raise ValueError(f"The model serves {prediction.dtype}, not instance labels")
        return segment.relabel_instances(
            prediction[channel], split_disconnected=True, connectivity=connectivity, min_size=min_size
        )
    raise ValueError(f"method must be one of {', '.join(SEED_METHODS)}, got {method!r}")


@finetune_bp.route("/api/finetune/view-labels/seed", methods=["POST"])
def seed_view_from_prediction():
    """Fill the unannotated voxels of the box on screen from the model's prediction.

    JSON body, all optional: ``size_nm`` (as for mark-view), ``confirm``
    (label a large box), ``select_channel`` (the channel the session
    trains, when it trains one), ``model`` (a name from ``seed_sources``;
    the latest finetune, else the base, when absent), ``method`` (one of
    the model's ``seed_sources`` methods; "components" when absent),
    ``threshold`` (probability, 0 to 1; the model's own boundary, 0.5, when
    absent: see ``segment_prediction``), ``min_size`` (objects of fewer
    voxels are background), ``connectivity`` (1 faces, the default; 2 and
    edges; 3 all neighbours) and ``per_slice`` (components per z slice).
    """
    data = request.get_json(silent=True) or {}
    try:
        threshold = data.get("threshold")
        if threshold is not None and not 0 <= float(threshold) <= 1:
            raise _Refused(f"threshold must be between 0 and 1, got {threshold}", 400)
        min_size = int(data.get("min_size") or 0)
        if min_size < 0:
            raise _Refused(f"min_size cannot be negative, got {min_size}", 400)
        method = data.get("method") or "components"
        if method not in SEED_METHODS:
            raise _Refused(f"method must be one of {', '.join(SEED_METHODS)}, got {method!r}", 400)
        connectivity = segment.as_connectivity(data.get("connectivity", 1))
        per_slice = segment.as_bool(data.get("per_slice", False))
        volume_id, volume, lo, hi = _target_box(data)
        refused = _confirmation_needed(data, lo, hi)
        if refused:
            return refused
        base_model = volume.get("model_name")
        if not base_model:
            raise _Refused("The annotation volume does not say which model it is for.")
        model_name, host = _prediction_server(base_model, data.get("model"))
        if host is None:
            raise _Refused(f"No running server for {base_model}. Start it from the Models tab, or pick a model.")
        # The model read decides the channels: a finetune's are its base's.
        channels, output_type, offsets = _seed_plan(
            _model_read_as(model_name, base_model), data.get("select_channel")
        )
        offered = seed_methods(model_name, host, base_model)
        if method not in offered:
            raise _Refused(f"{model_name} cannot seed by {method}; it offers {', '.join(offered)}", 400)
        info = fetch_model_info(host)
        read_as = _model_read_as(model_name, base_model)
        cellpose_masks = method == "instances" and _serves_cellpose_flows(read_as)
        if cellpose_masks:
            # Its server makes Cellpose's masks of its flows on asking.
            from cellmap_flow.post.postprocessors import CellposeMasksPostprocessor

            channels = None
        prediction, lo, hi = read_prediction(
            host, model_name, volume, lo, hi, info,
            postprocess=(CellposeMasksPostprocessor(),) if cellpose_masks else (),
        )
        objects = segment_prediction(
            method, prediction, info.get("output_class"), channels, offsets,
            0.5 if threshold is None else float(threshold), min_size, connectivity, per_slice,
        )
        # An instance target needs ids apart from its neighbours'; a selected
        # channel trains as a binary target.
        instance_target = output_type in INSTANCE_TARGETS and data.get("select_channel") is None

        def labels_for(existing):
            # Ids counting up need room: an instance target on a uint16/uint32
            # volume gets them; a uint8 volume reuses free ids.
            return fill.seed_labels(objects, existing, count_up=instance_target and existing.dtype.itemsize > 1)

        answer = _fill(volume_id, volume, lo, hi, labels_for)
        return jsonify({**answer, "model": model_name, "threshold": threshold, "method": method})
    except _Refused as e:
        return _error(str(e), e.status)
    except FileNotFoundError as e:
        return _error(str(e), 409)
    except ValueError as e:
        return _error(str(e), 400)
    except (requests.RequestException, RuntimeError) as e:
        logger.warning(f"Could not read the prediction to seed from: {e}")
        return _error(f"Could not read the model's prediction: {e}", 502)
    except Exception as e:
        logger.error(f"Error seeding the view from the prediction: {e}", exc_info=True)
        return _error(str(e), 500)


@finetune_bp.route("/api/finetune/view-labels/background", methods=["POST"])
def label_view_background():
    """Label every unannotated voxel of the box on screen background (1).

    JSON body, all optional: ``size_nm`` and ``confirm``, as for seed.
    """
    data = request.get_json(silent=True) or {}
    try:
        volume_id, volume, lo, hi = _target_box(data)
        refused = _confirmation_needed(data, lo, hi)
        if refused:
            return refused
        return jsonify(_fill(volume_id, volume, lo, hi, np.ones_like))
    except _Refused as e:
        return _error(str(e), e.status)
    except FileNotFoundError as e:
        return _error(str(e), 409)
    except Exception as e:
        logger.error(f"Error labelling the view background: {e}", exc_info=True)
        return _error(str(e), 500)


@finetune_bp.route("/api/finetune/view-labels/split", methods=["POST"])
def split_view_objects():
    """Relabel the box on screen's foreground by connected component.

    For a merge the model made: paint a background wall through the object,
    in every slice it spans, and click; the two sides get ids of their own
    (``fill.relabel_objects``). A stroke joining two objects merges them.
    JSON body, all optional: ``size_nm`` and ``confirm``, as for seed;
    ``connectivity`` and ``per_slice``, as for a seed's components (with
    ``per_slice``, a wall in one slice is enough, and each slice of an
    object gets an id of its own).
    """
    data = request.get_json(silent=True) or {}
    try:
        relabel = functools.partial(
            fill.relabel_objects,
            connectivity=segment.as_connectivity(data.get("connectivity", 1)),
            per_slice=segment.as_bool(data.get("per_slice", False)),
        )
        volume_id, volume, lo, hi = _target_box(data)
        refused = _confirmation_needed(data, lo, hi)
        if refused:
            return refused
        n_changed, counts = fill.rewrite_foreground(
            get_session().minio_state, volume_id, lo, hi, relabel, volume.get("zarr_path"),
            undo=_undo_stack(volume_id),
        )
        layer_refreshed = False
        if n_changed:
            layer_refreshed = _after_write(volume_id, "relabelled")
        logger.info(
            f"Relabelled box {lo.tolist()}..{hi.tolist()} of {volume_id}: {counts['objects']} objects, "
            f"{counts['split']} split off, {counts['merged']} merged, {n_changed} voxels changed"
        )
        return jsonify({
            "success": True,
            "offset_voxels": lo.tolist(),
            "shape_voxels": (hi - lo).tolist(),
            "changed": n_changed,
            "reload_viewer": bool(n_changed),
            "layer_refreshed": layer_refreshed,
            "can_undo": bool(_undo_stack(volume_id)),
            **counts,
        })
    except _Refused as e:
        return _error(str(e), e.status)
    except FileNotFoundError as e:
        return _error(str(e), 409)
    except ValueError as e:
        return _error(str(e), 400)
    except Exception as e:
        logger.error(f"Error splitting the view's objects: {e}", exc_info=True)
        return _error(str(e), 500)


@finetune_bp.route("/api/finetune/view-labels/undo", methods=["POST"])
def undo_view_labels():
    """Take back the last Seed, All Background or Split of the session's volume.

    The box goes back to what it held before, on the voxels still as that
    action left them: a stroke painted there since is kept. 409 when there
    is nothing to undo.
    """
    try:
        volume_id, volume = session_store().session_volume()
        if volume is None:
            raise _Refused("No annotation volume to undo in.")
        stack = _undo_stack(volume_id)
        if not stack:
            raise _Refused("Nothing to undo.")
        lo, hi, before, after = stack.pop()
        restored = fill.restore_box(get_session().minio_state, volume_id, lo, hi, before, after)
        layer_refreshed = _after_write(volume_id, "restored") if restored else False
        logger.info(f"Undid the label action on box {lo.tolist()}..{hi.tolist()} of {volume_id}: "
                    f"{restored} voxels restored")
        return jsonify({
            "success": True,
            "restored": restored,
            "reload_viewer": bool(restored),
            "layer_refreshed": layer_refreshed,
            "can_undo": bool(stack),
        })
    except _Refused as e:
        return _error(str(e), e.status)
    except FileNotFoundError as e:
        return _error(str(e), 409)
    except Exception as e:
        logger.error(f"Error undoing the label action: {e}", exc_info=True)
        return _error(str(e), 500)
