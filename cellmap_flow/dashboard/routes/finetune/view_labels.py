"""Labelling the box on screen in one click: from the model's prediction, or all background.

Two buttons beside "Mark This View as Good", over the same box (one model
output patch centred where the viewer looks, ``good_regions.view_box_nm``):

- POST ``/api/finetune/view-labels/seed``: the model's prediction there,
  thresholded at its decision boundary, written as foreground 2 (an id per
  object, 2 and up, for an affinity model on an instance volume) and
  background 1. The user then cleans it up with the brush; it is far
  quicker than painting a whole object from nothing.
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

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import zarr
from flask import jsonify, request
from scipy.special import expit

from cellmap_flow.dashboard.finetune_utils import sync_annotation_volume_from_minio
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
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
        centre_nm, size_nm = view_box_nm(data.get("size_nm"))
    except ValueError as e:
        raise _Refused(str(e), 400)
    shape = zarr.open_array(f"{volume['zarr_path']}/annotation/s0", mode="r").shape
    box = fill.box_voxels(volume, centre_nm - size_nm / 2, size_nm, shape)
    if box is None:
        raise _Refused("The view is outside the annotation volume.", 400)
    return volume_id, volume, box[0], box[1]


def _confirmation_needed(data, lo, hi):
    """The 409 asking the page to confirm a large box, or None."""
    voxels = int(np.prod(np.asarray(hi) - np.asarray(lo)))
    if voxels <= LARGE_BOX_VOXELS or data.get("confirm"):
        return None
    return _error(
        f"The box is {voxels:,} voxels. Label it all?", 409, needs_confirmation=True, voxels=voxels
    )


def _fill(volume_id, volume, lo, hi, labels_for):
    """Fill the box in MinIO and pull it to disk; returns the answer's counts."""
    n_foreground, n_background = fill.fill_unpainted(
        get_session().minio_state, volume_id, lo, hi, labels_for, volume.get("zarr_path")
    )
    layer_refreshed = False
    if n_foreground or n_background:
        # Now rather than at the next periodic sync, so training submitted
        # straight after sees the box. A failure is only late: the periodic
        # sync pulls the same chunks.
        try:
            sync_annotation_volume_from_minio(volume_id)
        except Exception as e:
            logger.warning(f"Could not pull the labelled box of {volume_id} to disk yet: {e}")
        layer_refreshed = _refresh_layer(volume_id)
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


@finetune_bp.route("/api/finetune/view-labels/sources", methods=["GET"])
def seed_sources():
    """The models a seed can read, for the page's picker: ``models`` (every
    running server, oldest first) and ``default``, the one a seed reads when
    none is chosen (None when that is a choice for the user)."""
    _, volume = session_store().session_volume()
    names = [name for name, _ in _prediction_sources()]
    default = _prediction_server((volume or {}).get("model_name"))[0]
    return jsonify({"success": True, "models": names, "default": default})


def _get(url):
    response = requests.get(url, timeout=PREDICTION_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"{url.split(ARGS_KEY)[0]}... answered HTTP {response.status_code}")
    return response


def read_prediction(host, model_name, volume, lo, hi, info):
    """The model's raw output over annotation voxels ``[lo, hi)``.

    Read from its server as a layer is, with the dashboard's input
    normalization and no postprocessing: the trainer's target is set on the
    raw output, and that is where its decision boundary is. Each annotation
    voxel takes the served voxel its centre lies in, placed where
    neuroglancer draws the layer (``prediction_voxel_override``), so the
    seed lies under the prediction the user sees.

    Returns ``(prediction, lo, hi)``: a (channels, z, y, x) float32 array,
    and the box, shrunk to what the server covers.
    """
    blob = PipelineSpec.from_steps(get_session().input_norms, ()).to_url_blob()
    base = f"{host.rstrip('/')}/{model_name}{ARGS_KEY}{blob}{ARGS_KEY}"
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
    prediction = region[np.ix_(*(i - i[0] for i in indices))].astype(np.float32)
    prediction = np.moveaxis(prediction, -1, 0) if channel_key else prediction[None]
    return prediction, new_lo, new_hi


def foreground_mask(prediction, output_class=None, channels=None, threshold=0.5):
    """Where the model says foreground: its probability over ``threshold``.

    The decision boundary depends on the output's activation: 0.5 on [0, 1]
    output (a sigmoid already applied; the cellmap distance models end in
    one, and 0.5 is their object boundary), 0 on tanh's [-1, 1], and 0 on
    unbounded output, logits or a signed distance alike. Each is turned into
    a probability first, so ``channels`` -- the affinity channels, say --
    can be averaged. ``output_class`` is the server's (serving.probe), or
    read off the values when it has none. ``threshold`` is in probability
    whatever the activation: 0.5 is the model's own boundary.
    """
    if channels is not None:
        prediction = prediction[channels]
        if not prediction.shape[0]:
            raise ValueError(f"The model has no channel {channels.start}")
    output_class = output_class or classify_output_range(float(prediction.min()), float(prediction.max()))
    if output_class == UNIT:
        probability = prediction
    elif output_class == SIGNED_UNIT:
        probability = (prediction + 1) / 2
    else:
        probability = expit(prediction)
    return probability.mean(axis=0) > threshold


def _seed_plan(base_model, select_channel):
    """``(channels, affinities)``: the channels a seed reads, and whether the model predicts affinities.

    As the trainer targets them: the selected channel; else an affinity
    model's offset channels (the rest, LSDs say, are masked out of its
    loss); else every channel, which a binary target is broadcast over.
    """
    if select_channel is not None:
        channel = int(select_channel)
        if channel < 0:
            raise ValueError(f"select_channel must be 0 or more, got {channel}")
        return slice(channel, channel + 1), False
    model_config = find_model_config(base_model)
    output_type, offsets = (
        autodetect_output_type(model_config, None, None) if model_config else ("binary", None)
    )
    if output_type == "affinities":
        return slice(0, len(json.loads(offsets))), True
    return None, False


@finetune_bp.route("/api/finetune/view-labels/seed", methods=["POST"])
def seed_view_from_prediction():
    """Fill the unannotated voxels of the box on screen from the model's prediction.

    JSON body, all optional: ``size_nm`` (as for mark-view), ``confirm``
    (label a large box), ``select_channel`` (the channel the session
    trains, when it trains one), ``model`` (a name from ``seed_sources``;
    the latest finetune, else the base, when absent), ``threshold`` (probability, 0 to 1; the
    model's own boundary, 0.5, when absent) and ``min_size`` (objects of
    fewer voxels are background).
    """
    data = request.get_json(silent=True) or {}
    try:
        threshold = data.get("threshold")
        if threshold is not None and not 0 <= float(threshold) <= 1:
            raise _Refused(f"threshold must be between 0 and 1, got {threshold}", 400)
        min_size = int(data.get("min_size") or 0)
        if min_size < 0:
            raise _Refused(f"min_size cannot be negative, got {min_size}", 400)
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
        plan_for = base_model if model_name.startswith(f"{base_model}_finetuned") else model_name
        channels, affinities = _seed_plan(plan_for, data.get("select_channel"))
        info = fetch_model_info(host)
        prediction, lo, hi = read_prediction(host, model_name, volume, lo, hi, info)
        foreground = foreground_mask(
            prediction, info.get("output_class"), channels, 0.5 if threshold is None else float(threshold)
        )

        def labels_for(existing):
            # Ids counting up need room: an affinity target on a uint16/uint32
            # instance volume gets them; a uint8 volume reuses free ids.
            return fill.seed_labels(
                foreground, existing, count_up=affinities and existing.dtype.itemsize > 1, min_size=min_size
            )

        answer = _fill(volume_id, volume, lo, hi, labels_for)
        return jsonify({**answer, "model": model_name, "threshold": threshold})
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
    JSON body, all optional: ``size_nm`` and ``confirm``, as for seed.
    """
    data = request.get_json(silent=True) or {}
    try:
        volume_id, volume, lo, hi = _target_box(data)
        refused = _confirmation_needed(data, lo, hi)
        if refused:
            return refused
        n_changed, counts = fill.rewrite_foreground(
            get_session().minio_state, volume_id, lo, hi, fill.relabel_objects, volume.get("zarr_path")
        )
        layer_refreshed = False
        if n_changed:
            try:
                sync_annotation_volume_from_minio(volume_id)
            except Exception as e:
                logger.warning(f"Could not pull the relabelled box of {volume_id} to disk yet: {e}")
            layer_refreshed = _refresh_layer(volume_id)
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
