"""Label it's segmentation of a prediction, written once for both modes.

What the Finetune tab's *Seed from Prediction* makes of a model's output over
the box on screen: the request's options (``seed_options``), the channels a
seed reads (``seed_channels``; ``affinity_offsets`` for a model known by its
channel names alone), which served voxel lies under each annotation voxel
(``placement``, ``gather``), the output as a probability (``probability``)
and the objects of the *components* method (``components``: threshold,
connected components, smallest size, per slice). The server calls these
from ``dashboard/routes/finetune/view_labels.py`` around its read of the
model server's array; the browser from ``web/src/backend/routes/view-labels.ts``
around the engine's ``readOutput``, through ``label_it_calls.py``.

The mutex watershed is the one method not here: ``mwatershed`` is Rust, with
no Pyodide wheel, so each mode runs its own on what ``affinities`` gives
(``post.segment.mutex_watershed`` on the server, ``web/src/infer/affinity.ts``
in the browser, parity-tested against it). The distance watershed and a
model's own instances are the server's alone (scikit-image, Cellpose); no
model the browser runs outputs either.

scipy is imported inside the functions that need it, never at the top: the
browser's Python starts with numpy alone, and chunkmirage's worker loads a
package the first time a call fails to import it, then calls again, so the
first seed pays for scipy and opening a dataset does not.
"""

import numpy as np

# How a seed makes objects of the prediction (``post.segment``), each fitting
# a kind of output; which a model offers is each mode's to say
# (view_labels.methods_for, view-labels.ts methodsFor).
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

# A model's output activation, as serving/probe.py observes it (and
# re-exports these from here, so the browser's Python has them too).
UNIT = "unit"  # [0, 1] -- sigmoid/softmax already applied
SIGNED_UNIT = "signed_unit"  # [-1, 1] -- tanh already applied
UNBOUNDED = "unbounded"  # no output activation: logits, distances, affinities...

# The connectivities in the order the forms list them, with their names.
CONNECTIVITIES = {1: "faces", 2: "faces and edges", 3: "all neighbours"}


def classify_output_range(lo: float, hi: float) -> str:
    """Bucket an observed (min, max) output range into an activation class."""
    # Small tolerance: a saturating activation can land a hair outside its
    # asymptote in float32.
    eps = 1e-4
    if lo >= -eps and hi <= 1.0 + eps:
        return UNIT
    if lo >= -1.0 - eps and hi <= 1.0 + eps:
        return SIGNED_UNIT
    return UNBOUNDED


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def as_connectivity(value, ndim=3) -> int:
    """``value`` as scipy's connectivity for an ``ndim``-D array: 1 up to ``ndim``.

    Forms send strings, so "2" is 2. A value past ``ndim`` (3 for one 2D
    slice) is ``ndim``: every neighbour, which is what it asked for.
    Raises ValueError for anything that is not 1, 2 or 3, None among them.
    """
    try:
        connectivity = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"connectivity must be 1, 2 or 3, got {value!r}")
    if connectivity not in CONNECTIVITIES:
        raise ValueError(f"connectivity must be 1, 2 or 3, got {value!r}")
    return min(connectivity, ndim)


def as_bool(value) -> bool:
    """A form's "true"/"false" as a bool; anything else by Python's truth."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def seed_options(data) -> dict:
    """A seed request's options, from its JSON body, with the defaults the
    routes give an absent one: ``method`` ("components"), ``threshold`` (a
    probability, 0.5, the model's own boundary), ``min_size`` (0),
    ``connectivity`` (1, faces) and ``per_slice`` (false).

    Raises ValueError, which both modes answer 400, for a threshold outside
    0 to 1, a negative or non-integer size, a method not in SEED_METHODS
    and a connectivity that is not 1, 2 or 3. A ``connectivity`` given as
    null is refused with the rest, not read as absent: the key is there.
    """
    threshold = data.get("threshold")
    if threshold is not None and not 0 <= float(threshold) <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    min_size = int(data.get("min_size") or 0)
    if min_size < 0:
        raise ValueError(f"min_size cannot be negative, got {min_size}")
    method = data.get("method") or "components"
    if method not in SEED_METHODS:
        raise ValueError(f"method must be one of {', '.join(SEED_METHODS)}, got {method!r}")
    return {
        "method": method,
        "threshold": 0.5 if threshold is None else float(threshold),
        "min_size": min_size,
        "connectivity": as_connectivity(data.get("connectivity", 1)),
        "per_slice": as_bool(data.get("per_slice", False)),
    }


def affinity_offsets(channel_names):
    """The offsets of a model known by its channel names alone, or None
    when they name no affinity channel.

    The server's rule for a model whose script says none
    (``dashboard/routes/finetune/common.autodetect_output_type``): a model
    with a channel named ``*_aff*`` is an affinity model, and its first
    affinity channels, up to three, compare each voxel with the next one
    along z, y and x in turn, whatever their names say after ``_aff``.
    """
    n_aff = sum(1 for channel in channel_names or () if "_aff" in channel)
    if not n_aff:
        return None
    return [[1 if axis == index else 0 for axis in range(3)] for index in range(min(n_aff, 3))]


def seed_channels(output_type, offsets, select_channel=None):
    """The channels a seed reads, as a slice, or None for every one.

    As the trainer targets them: the selected channel; else an affinity
    model's offset channels, the first ``len(offsets)`` (the rest, LSDs
    say, are masked out of its loss); else every channel, which a binary
    target is broadcast over. ``offsets`` is a list of the model's offsets.
    A slice past the model's channels reads those there are
    (``probability``).
    """
    if select_channel is not None:
        channel = int(select_channel)
        if channel < 0:
            raise ValueError(f"select_channel must be 0 or more, got {channel}")
        return slice(channel, channel + 1)
    if output_type == "affinities":
        return slice(0, len(offsets))
    return None


# ---------------------------------------------------------------------------
# Where the prediction lies on the paint's grid
# ---------------------------------------------------------------------------


def placement(lo, hi, voxel_size, dataset_offset_nm, translation, declared, shape, drawn=None):
    """Which voxel of a served array each annotation voxel of ``[lo, hi)``
    reads: the one its centre lies in, so a seed from a model served at
    another voxel size (any served model can seed, not only the one the
    volume is for) lies under that model's layer.

    The volume's grid is its ``voxel_size`` from ``dataset_offset_nm``, the
    centre of its voxel 0 (its OME translation, as
    ``finetune/session/volume.volume_corner_nm`` reads it; None is 0). The
    served array's is its OME ``translation`` (voxel 0's centre too) and
    ``declared`` voxel size, over ``shape`` voxels; ``drawn`` is the voxel
    size Neuroglancer draws it at when that is another
    (``viewer.layers.prediction_voxel_override``), which keeps the served
    offset in voxels, so the array's corner moves with it.

    Returns ``(indices, lo, hi)``: per axis, the served index of each
    annotation voxel the array covers, and the box shrunk to those voxels;
    None when it covers none along some axis.
    """
    voxel = np.asarray(voxel_size, dtype=float)
    offset = np.zeros(3) if dataset_offset_nm is None else np.asarray(dataset_offset_nm, dtype=float)
    volume_corner = offset - voxel / 2
    declared = np.asarray(declared, dtype=float)
    drawn = declared if drawn is None else np.asarray(drawn, dtype=float)
    served_corner = (np.asarray(translation, dtype=float) - declared / 2) / declared * drawn
    indices, new_lo, new_hi = [], [int(v) for v in lo], [int(v) for v in hi]
    for axis in range(3):
        centres = volume_corner[axis] + (np.arange(new_lo[axis], new_hi[axis]) + 0.5) * voxel[axis]
        index = np.floor((centres - served_corner[axis]) / drawn[axis]).astype(int)
        # One run: the centres rise along the axis.
        inside = np.flatnonzero((index >= 0) & (index < int(shape[axis])))
        if not inside.size:
            return None
        new_lo[axis], new_hi[axis] = new_lo[axis] + int(inside[0]), new_lo[axis] + int(inside[-1]) + 1
        indices.append(index[inside])
    return indices, new_lo, new_hi


def read_box(indices):
    """``(lo, hi)``: the box of the served array the indices span, the one to read for ``gather``."""
    return [int(i[0]) for i in indices], [int(i[-1]) + 1 for i in indices]


def gather(region, indices):
    """The prediction on the annotation grid: each annotation voxel the
    served voxel ``placement`` gave it. ``region`` is (channels, z, y, x),
    the served box ``read_box`` gives; with the two grids equal, all of it
    in order."""
    return region[(slice(None),) + np.ix_(*(np.asarray(i) - i[0] for i in indices))]


# ---------------------------------------------------------------------------
# The prediction as a probability
# ---------------------------------------------------------------------------


def _sigmoid(x):
    # 1 / (1 + e^-x) in the input's float32. numpy's exp overflows to inf
    # below about -88, and 1 / inf is the 0 the sigmoid tends to there.
    # numpy's own, not scipy's expit: a seed by the mutex watershed and the
    # browser's rank read probabilities, and need no scipy for it.
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


def probability(prediction, output_class=None, channels=None):
    """The prediction as a probability, (channels, z, y, x), 0.5 at the model's decision boundary.

    The boundary depends on the output's activation: 0.5 on [0, 1] output
    (a sigmoid already applied; the cellmap distance models end in one, and
    0.5 is their object boundary), 0 on tanh's [-1, 1], and 0 on unbounded
    output, logits or a signed distance alike. Each is turned into a
    probability, so ``channels`` (a slice: the affinity channels, say) can
    be averaged and one threshold means the same for every model. A slice
    running past the model's channels reads those there are; one starting
    past them is refused. ``output_class`` is the server's
    (serving.probe), or read off the values when it has none.
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
    return _sigmoid(prediction)


def foreground_mask(prediction, output_class=None, channels=None, threshold=0.5):
    """Where the model says foreground: its ``probability``, averaged over ``channels``, over ``threshold``.

    ``threshold`` is in probability whatever the activation: 0.5 is the
    model's own boundary. Strictly over: a voxel at it is background.
    """
    return probability(prediction, output_class, channels).mean(axis=0) > threshold


def affinities(prediction, output_class, offsets):
    """The mutex watershed's input: the probability of an affinity model's
    offset channels, the first ``len(offsets)`` (as ``seed_channels``
    takes them; fewer when the model has fewer)."""
    return probability(prediction, output_class, slice(0, len(offsets)))


# ---------------------------------------------------------------------------
# The objects
# ---------------------------------------------------------------------------


def _structure(ndim, connectivity):
    from scipy.ndimage import generate_binary_structure

    return generate_binary_structure(ndim, as_connectivity(connectivity, ndim))


def remove_small(labels, min_size=0):
    """``labels`` with objects of fewer than ``min_size`` voxels made background.

    The rest are numbered 1..n again, in the order they had. ``labels``
    must already be numbered without large gaps (any segmenter's output):
    the sizes are counted per id.
    """
    labels = np.asarray(labels)
    if min_size is None or int(min_size) <= 1 or not labels.any():
        return labels.astype(np.uint64, copy=False)
    sizes = np.bincount(labels.ravel().astype(np.intp))
    keep = sizes >= int(min_size)
    keep[0] = False
    new_ids = np.zeros(sizes.size, dtype=np.uint64)
    new_ids[keep] = np.arange(1, np.count_nonzero(keep) + 1, dtype=np.uint64)
    return new_ids[labels.astype(np.intp)]


def connected_components(mask, connectivity=1, min_size=0, per_slice=False):
    """An id per connected object of ``mask`` (its nonzero voxels): uint64,
    0 for background and 1..n with no id skipped, numbered as
    ``scipy.ndimage.label`` numbers them.

    ``per_slice``: each z slice of a (z, y, x) mask is labelled on its own,
    in 2D, its ids following the slice before's. For data whose objects are
    2D (what Cellpose sees), or annotations painted slice by slice: an
    object touching itself across slices is still one per slice.
    ``min_size``: objects of fewer voxels are background (specks a
    threshold let through); per slice, each slice's piece is counted alone.
    """
    from scipy.ndimage import label

    mask = np.asarray(mask) != 0
    if per_slice:
        if mask.ndim != 3:
            raise ValueError(f"per_slice labels (z, y, x) masks, got shape {mask.shape}")
        structure = _structure(2, connectivity)
        labels = np.zeros(mask.shape, dtype=np.uint64)
        count = 0
        for z, plane in enumerate(mask):
            plane_labels, n = label(plane, structure)
            labels[z] = np.where(plane_labels > 0, plane_labels.astype(np.uint64) + np.uint64(count), 0)
            count += n
    else:
        labels, _ = label(mask, _structure(mask.ndim, connectivity))
    return remove_small(labels, min_size)


def components(prediction, output_class, channels, threshold=0.5, min_size=0, connectivity=1, per_slice=False):
    """The objects of the *components* method: the ``foreground_mask`` of
    the prediction over ``channels``, an id per connected object
    (``connected_components``)."""
    return connected_components(foreground_mask(prediction, output_class, channels, threshold), connectivity, min_size, per_slice)
