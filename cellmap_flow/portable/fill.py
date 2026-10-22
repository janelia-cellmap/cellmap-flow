"""Labelling a box of an annotation volume: the decisions behind the Finetune
tab's Seed, All Background and Split Objects and an accepted AI annotation
(``finetune/session/fill.py`` says what each is for), on arrays alone, so the
server makes them over MinIO's zarr and the browser over its paint store's
chunks (``portable/fill_calls.py`` for the Label it tools,
``portable/ai_annotate_calls.py`` for an accept).

- ``box_voxels``: the annotation voxels a box in world nm covers.
- ``seed_labels``: the ids a box gets from a segmentation: the id the user
  painted on an object kept, fresh ones for the rest (``fresh_ids``).
- ``relabel_objects``: the box's foreground relabelled by connected component.
- ``paint``, ``rewrite`` and ``restore``: which voxels a write changes and
  an undo puts back, with the counts the routes answer.

Nothing here reads or writes a volume: the callers read the box, hand it in
and write back what comes out (``finetune/session/fill.py`` through zarr
over MinIO, ``web/src/backend/routes/view-labels.ts`` and
``web/src/backend/ai-annotate-job.ts`` through the paint store's
``web/src/ai-annotate/fill.ts``).
"""

import numpy as np


def volume_corner_nm(dataset_offset_nm, output_voxel_size) -> np.ndarray:
    """The world position of an annotation volume's voxel-0 lower corner, in nm.

    ``dataset_offset_nm`` is voxel 0's *centre*, the OME translation, as
    ``finetune.session.volume.volume_corner_nm`` reads it (its docstring says
    why), so the corner is half a voxel before it; voxel 0's corner is at
    minus half a voxel when the volume has no offset.
    """
    voxel_size = np.asarray(output_voxel_size, dtype=float)
    offset = np.zeros(3) if dataset_offset_nm is None else np.asarray(dataset_offset_nm, dtype=float)
    return offset - voxel_size / 2


def box_voxels(volume, offset_nm, shape_nm, volume_shape):
    """``(lo, hi)``: the annotation voxels whose centres lie in a box, clipped to the volume.

    ``offset_nm`` and ``shape_nm`` are the box's lower corner and size in
    world nm, as a good region records them; ``volume`` is the volume's
    record, for its grid (``output_voxel_size``, ``dataset_offset_nm``).
    None when no voxel of the volume is in the box.
    """
    voxel_size = np.asarray(volume["output_voxel_size"], dtype=float)
    corner = volume_corner_nm(volume.get("dataset_offset_nm"), voxel_size)
    start = (np.asarray(offset_nm, dtype=float) - corner) / voxel_size
    end = start + np.asarray(shape_nm, dtype=float) / voxel_size
    # Voxel i's centre is at i + 0.5 voxels from the corner.
    lo = np.maximum(np.ceil(start - 0.5), 0).astype(int)
    hi = np.minimum(np.ceil(end - 0.5), np.asarray(volume_shape)).astype(int)
    if np.any(hi <= lo):
        return None
    return lo, hi


def fresh_ids(existing, count, count_up):
    """``count`` ids for new objects in a box that holds ``existing``.

    ``count_up``: from one past the box's largest label, so an affinity
    target's objects stay apart from every object nearby (ids elsewhere in
    the volume are not looked at: affinities only compare nearby voxels).
    Else the lowest ids the box does not use, so a uint8 volume's 254 ids
    are never run through: a repeat in another box is only a repeated
    colour, which a binary or distance target does not see. Raises
    ValueError when the dtype has no room.
    """
    biggest = np.iinfo(existing.dtype).max
    if count_up:
        first = max(int(existing.max()), 1) + 1
        ids = np.arange(first, first + count)
    else:
        used = np.unique(existing[existing >= 2]).astype(np.int64)
        candidates = np.arange(2, min(biggest, len(used) + count + 2) + 1)
        ids = np.setdiff1d(candidates, used, assume_unique=True)[:count]
    if ids.size < count or (ids.size and ids[-1] > biggest):
        raise ValueError(
            f"{count} new objects do not fit in the volume's {existing.dtype} labels: make a new annotation "
            "volume, which for an instance model (Cellpose, affinities) now holds uint16 ids, or raise Min "
            "size or the threshold so the seed makes fewer objects"
        )
    return ids


def counts_up(instance_target, dtype) -> bool:
    """Whether new objects' ids count up from the box's largest (``fresh_ids``):
    for a model trained on instances (``instance_target``: affinities, or
    Cellpose's flows) whose volume holds more than 8-bit ids. A uint8 volume
    reuses the ids its box leaves free, whatever the model, or its 254 would
    soon run out."""
    return bool(instance_target) and np.dtype(dtype).itemsize > 1


def _pair_counts(a, b):
    """The distinct pairs ``(a[i], b[i])`` of two arrays of non-negative
    int64 values, as the columns of a (2, n) array in order of ``a`` and then
    ``b``, and how often each occurs: ``np.unique(np.stack([a, b]), axis=1,
    return_counts=True)``, by one key per pair, ``a * (max(b) + 1) + b``,
    which sorts in the same order. Sorting one array of keys is many times
    quicker than sorting the pairs as rows (about 1 ms against 75 for the
    foreground of a 64^3 box, natively, and the browser's numpy is slower
    still). The keys stay within int64 for any label dtype up to uint32 in a
    box of fewer than 2^31 voxels (``a`` counts objects, at most one a voxel).
    """
    base = int(b.max(initial=0)) + 1
    keys, counts = np.unique(a * base + b, return_counts=True)
    return np.stack([keys // base, keys % base]), counts


def seed_labels(objects, existing, count_up=False):
    """Labels for a box from a segmentation: an id per object inside, 1 outside.

    ``objects`` is a segmenter's output over the box (0 background, objects
    1..n), ``existing`` what the box holds now. Each object gets an id of
    its own (``fresh_ids``; ``count_up`` for an instance target), so
    neighbouring objects are told apart, and a merge the model made shows
    as one colour to split; but an object the user has already painted part
    of takes the id painted there most, or the unpainted rest of it would be
    taught as a different object from its painted part. A binary or distance
    target reads every id as foreground, so the ids cost it nothing.
    """
    objects = np.asarray(objects).astype(np.int64, copy=False)
    foreground = objects > 0
    labels = np.where(foreground, 2, 1).astype(existing.dtype)
    if not foreground.any():
        return labels
    n = int(objects.max())
    ids = np.zeros(n + 1, dtype=np.int64)
    painted = (existing >= 2) & foreground
    if painted.any():
        pairs, counts = _pair_counts(objects[painted], existing[painted].astype(np.int64))
        # Most-painted last, so its id is the one that stays.
        order = np.argsort(counts, kind="stable")
        ids[pairs[0, order]] = pairs[1, order]
    present = np.zeros(n + 1, dtype=bool)
    present[np.unique(objects[foreground])] = True
    fresh = np.flatnonzero(present & (ids == 0))
    ids[fresh] = fresh_ids(existing, fresh.size, count_up)
    labels[foreground] = ids[objects[foreground]].astype(existing.dtype)
    return labels


def connected_components(mask, connectivity=1, per_slice=False):
    """An id per connected object of ``mask`` (its nonzero voxels), 1..n, 0 elsewhere.

    ``scipy.ndimage.label`` with the structure of ``connectivity`` (1 face
    neighbours, 2 and edges, 3 every neighbour, at most the array's
    dimensions), numbered as it numbers them: by the raster order of each
    object's first voxel. ``per_slice``: each z slice of a (z, y, x) mask is
    labelled on its own, in 2D, its ids following the slice before's. As
    ``post.segment.connected_components`` labels a seed's objects, without
    its minimum size.
    """
    from scipy.ndimage import generate_binary_structure, label

    mask = np.asarray(mask) != 0
    if not per_slice:
        labels, _ = label(mask, generate_binary_structure(mask.ndim, min(int(connectivity), mask.ndim)))
        return labels.astype(np.int64)
    if mask.ndim != 3:
        raise ValueError(f"per_slice labels (z, y, x) masks, got shape {mask.shape}")
    structure = generate_binary_structure(2, min(int(connectivity), 2))
    labels = np.zeros(mask.shape, dtype=np.int64)
    count = 0
    for z, plane in enumerate(mask):
        plane_labels, n = label(plane, structure)
        labels[z] = np.where(plane_labels > 0, plane_labels + count, 0)
        count += n
    return labels


def relabel_objects(existing, count_up=False, connectivity=1, per_slice=False):
    """The box's foreground (labels 2 and up) relabelled by connected component.

    Each component takes the id most of its voxels hold, so a stroke that
    joins two objects merges them into the id of the bigger. An id left on
    several components, an object a background wall cut in two, stays on the
    largest and the others get fresh ids (``fresh_ids``). Background (1)
    and unpainted (0) voxels are left alone. Returns ``(labels, counts)``:
    ``counts`` has ``objects``, ``split`` (components given a fresh id) and
    ``merged`` (components that held more than one id).

    A cut has to go through every slice the object spans: the brush paints
    one slice, and an object cut in one slice is still one object above and
    below it. ``split`` is 0 then, which the page says. With ``per_slice``
    each z slice's pieces are components on their own, so a wall in one
    slice splits that slice, and every slice of an object gets its own id.
    ``connectivity``: as for ``connected_components``.
    """
    labels = existing.copy()
    foreground = existing >= 2
    counts = {"objects": 0, "split": 0, "merged": 0}
    if not foreground.any():
        return labels, counts
    components = connected_components(foreground, connectivity, per_slice=per_slice)
    n = int(components.max())
    comp, old = components[foreground], existing[foreground].astype(np.int64)
    pairs, pair_counts = _pair_counts(comp, old)
    order = np.argsort(pair_counts, kind="stable")
    majority = np.zeros(n + 1, dtype=np.int64)
    majority[pairs[0, order]] = pairs[1, order]  # the most voxels last, so it stays
    counts["objects"] = int(n)
    counts["merged"] = int(np.count_nonzero(np.bincount(pairs[0], minlength=n + 1) > 1))

    sizes = np.bincount(comp, minlength=n + 1)
    fresh = []
    for old_id in np.unique(majority[1:]):
        claimants = np.flatnonzero(majority == old_id)
        if claimants.size > 1:
            keep = claimants[np.argmax(sizes[claimants])]
            fresh.extend(int(c) for c in claimants if c != keep)
    new = majority.copy()
    if fresh:
        new[fresh] = fresh_ids(existing, len(fresh), count_up)
    counts["split"] = len(fresh)
    labels[foreground] = new[comp].astype(existing.dtype)
    return labels, counts


def paint(existing, labels, overwrite=False):
    """The box after ``labels`` are written over ``existing``, and what that wrote.

    A label of 0 leaves its voxel as it is. Without ``overwrite`` only
    unannotated voxels (0) are written, as for a seed: what the user painted
    stays. With it every voxel the labels cover is written, painted or not,
    for a result the user has looked at and prefers to what is there.
    Returns ``(after, (n_foreground, n_background, n_overwritten))``: the
    box as it is then, of ``existing``'s dtype, or None when nothing would
    change; how many voxels were written with foreground (2 and up) and with
    background, and how many of those held a label before.
    """
    labels = np.asarray(labels, dtype=existing.dtype)
    if labels.shape != existing.shape:
        raise ValueError(f"The labels are {labels.shape}; the box holds {existing.shape}")
    if overwrite:
        write = (labels > 0) & (labels != existing)
    else:
        write = (existing == 0) & (labels > 0)
    n_overwritten = int(np.count_nonzero(write & (existing != 0)))
    n_foreground = int(np.count_nonzero(write & (labels >= 2)))
    counts = (n_foreground, int(np.count_nonzero(write)) - n_foreground, n_overwritten)
    if not write.any():
        return None, counts
    after = existing.copy()
    after[write] = labels[write]
    return after, counts


def rewrite(existing, labels):
    """The box after ``labels`` are written over its foreground voxels, and how many changed.

    Only foreground voxels (2 and up) whose label changed are written, so a
    stroke painted meanwhile elsewhere in the box is kept (Split Objects
    relabels the box it read). Returns ``(after, n_changed)``, ``after``
    None when none changed.
    """
    labels = np.asarray(labels, dtype=existing.dtype)
    changed = (existing >= 2) & (labels != existing)
    if not changed.any():
        return None, 0
    after = existing.copy()
    after[changed] = labels[changed]
    return after, int(np.count_nonzero(changed))


def restore(current, before, after):
    """The box put back to ``before`` where it still holds ``after``: an undo.

    Only voxels the action changed and nobody has touched since go back, so
    a stroke painted in the box after the action is kept. Returns
    ``(restored, n)``: the box then, or None when nothing goes back, and how
    many voxels did.
    """
    back = (before != after) & (current == after)
    if not back.any():
        return None, 0
    restored = current.copy()
    restored[back] = before[back]
    return restored, int(np.count_nonzero(back))
