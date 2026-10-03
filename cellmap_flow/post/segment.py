"""Instance segmenters: from what a model predicts to an id per object.

Plain functions on numpy arrays, shared by the postprocessors a served layer
runs (``post.postprocessors``) and by the Finetune tab's "Seed from
Prediction" (``dashboard/routes/finetune/view_labels.py``), so a model's
objects come out the same in both. Which one fits follows from what the
model outputs, not from its name:

- a probability, or any map with a foreground threshold:
  ``connected_components`` of the thresholded mask. Touching objects stay
  one object.
- affinities to the neighbours at given offsets: ``mutex_watershed``. The
  model says where two voxels belong apart, so touching objects split.
- a distance to the object boundary: ``distance_watershed``, which cuts
  touching objects apart where the distance dips between their centres.
- integer instance labels of the model's own (Cellpose's masks):
  ``relabel_instances``.

Each returns uint64 labels, 0 for background and 1..n for the objects with
no id skipped, so ``labels.max()`` is how many there are.

``connectivity`` is scipy's (``ndimage.generate_binary_structure``): which
neighbours touch. 1, faces only (6 in 3D, 4 in a 2D slice), is the default
everywhere: a one-voxel background wall painted with the brush cuts an
object under it, where with 3 (all 26 neighbours) voxels across a diagonal
wall still touch. 2 adds the edge neighbours (18 in 3D, 8 in 2D).

The segmentation libraries are imported inside the functions: the
postprocessors import this module, and every chain is read through them,
in processes that never segment anything (``tests/utils/test_import_hygiene``).
"""

import numpy as np

# The connectivities in the order the forms list them, with their names.
CONNECTIVITIES = {1: "faces", 2: "faces and edges", 3: "all neighbours"}


def as_connectivity(value, ndim=3) -> int:
    """``value`` as scipy's connectivity for an ``ndim``-D array: 1 up to ``ndim``.

    Forms send strings, so "2" is 2. A value past ``ndim`` (3 for one 2D
    slice) is ``ndim``: every neighbour, which is what it asked for.
    Raises ValueError for anything that is not 1, 2 or 3.
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
    """An id per connected object of ``mask`` (its nonzero voxels).

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


def mutex_watershed(affinities, offsets, bias=0.0, min_size=0):
    """Objects from affinities: ``(channels, z, y, x)``, one channel per offset, in [0, 1].

    ``mwatershed.agglom`` joins voxels along edges whose affinity is above
    ``bias`` and keeps them apart along those below it, strongest edges
    first. Every voxel ends up in some fragment, background included, so a
    fragment whose mean affinity (over all the channels) is below ``bias``
    is background: it is where the model sees no object. ``offsets``: the
    voxel offset each channel compares, as the model was trained with; a
    list longer than the channels is cut to them. ``min_size``: as for
    ``connected_components``.
    """
    import fastremap
    import mwatershed as mws
    from scipy import ndimage

    affinities = np.asarray(affinities, dtype=np.float64)
    offsets = [list(offset) for offset in offsets][: affinities.shape[0]]
    segmentation = mws.agglom(affinities - float(bias), offsets)

    mean_affinity = np.mean(affinities, axis=0)
    fragment_ids = fastremap.unique(segmentation[segmentation > 0])
    kept = [
        fragment
        for fragment, mean in zip(fragment_ids, ndimage.mean(mean_affinity, segmentation, fragment_ids))
        if mean >= bias
    ]
    fastremap.mask_except(segmentation, kept, in_place=True)
    fastremap.renumber(segmentation, in_place=True)
    return remove_small(segmentation.astype(np.uint64, copy=False), min_size)


def distance_watershed(
    distance,
    threshold=0.5,
    markers="peaks",
    peak_depth=0.05,
    marker_threshold=0.9,
    connectivity=1,
    min_size=0,
):
    """Objects from a distance map: higher deeper inside an object, ``threshold`` at its boundary.

    Foreground is ``distance > threshold``. Each object grows from a marker,
    by a watershed of ``-distance`` inside the foreground, so two touching
    objects meet where the distance dips between their centres: the neck a
    threshold alone would leave joined. ``markers``:

    - "peaks" (default): the distance's maxima that stand at least
      ``peak_depth`` above the lowest point on any path to a higher one
      (h-maxima). A noisy plateau inside one object is one marker, and a
      neck shallower than ``peak_depth`` does not split.
    - "threshold": the connected parts of ``distance > marker_threshold``,
      the objects' cores. For a model whose distance saturates inside
      objects, where the peaks are flat.

    The defaults are for a probability-like map (0.5 at the boundary, as
    the cellmap distance models' sigmoid gives, and Seed from Prediction
    makes of any output); for a distance in voxels, give the threshold and
    depth in voxels. A foreground part no marker reaches is an object of
    its own, so nothing the threshold calls foreground is lost.
    """
    from scipy.ndimage import label
    from skimage.segmentation import watershed

    distance = np.asarray(distance, dtype=np.float32)
    foreground = distance > threshold
    if not foreground.any():
        return np.zeros(distance.shape, dtype=np.uint64)
    structure = _structure(distance.ndim, connectivity)
    if markers == "peaks":
        from skimage.morphology import h_maxima

        # Background is never a marker: set it below every object's voxels.
        inside = np.where(foreground, distance, float(threshold))
        peaks = h_maxima(inside, float(peak_depth), footprint=structure).astype(bool) & foreground
    elif markers == "threshold":
        peaks = distance > marker_threshold
    else:
        raise ValueError(f'markers must be "peaks" or "threshold", got {markers!r}')
    seeds, n = label(peaks, structure)
    labels = watershed(-distance, seeds, mask=foreground, connectivity=structure).astype(np.uint64)
    unreached = foreground & (labels == 0)
    if unreached.any():
        rest, _ = label(unreached, structure)
        labels[unreached] = rest[unreached].astype(np.uint64) + np.uint64(n)
    return remove_small(labels, min_size)


def relabel_instances(labels, split_disconnected=False, connectivity=1, min_size=0):
    """A model's own integer instance labels, numbered 1..n (0 stays background).

    ``split_disconnected``: each id's connected parts become objects of
    their own. For labels whose ids are unique only within a block (a
    Cellpose server numbers each chunk from 1), read across several blocks:
    the same id in two blocks is two objects, and they do not touch.
    """
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"instance labels are integers, got {labels.dtype}")
    if split_disconnected:
        from skimage.measure import label

        out = label(labels, background=0, connectivity=as_connectivity(connectivity, labels.ndim))
        return remove_small(out, min_size)
    ids, inverse = np.unique(labels, return_inverse=True)
    out = inverse.reshape(labels.shape).astype(np.uint64)
    if ids.size and ids[0] != 0:
        out += np.uint64(1)
    return remove_small(out, min_size)
