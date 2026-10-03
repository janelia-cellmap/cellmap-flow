"""Labelling a whole box of an annotation volume at once.

The Finetune tab's "Seed" and "All background" buttons label the box on
screen in one click: from the model's own prediction, for the user to clean
up with the brush, or all background (1), for a region of false positives.
Either way only unannotated voxels (0) are filled. A stroke the user painted
is a decision about that voxel; a seed is a guess. "Split objects" relabels
the box's foreground by connected component (``relabel_objects``), so a
background wall painted through a merged object gives it two ids again.

The objects a seed writes come from ``post.segment``, the segmenter that
fits the model (``seed_labels`` takes its labels). Split Objects is
6-connected by default (face neighbours, scipy's default): a one-voxel wall
the brush paints cuts an object under it, where under 26-connectivity the
voxels across a diagonal wall would still touch.

The write goes to MinIO, as cc3d_relabel's does, not to the volume on disk:
MinIO holds strokes the periodic sync has not pulled yet, which is what "only
fill 0s" must see, and it is what neuroglancer reads. The caller then pulls
the changed chunks to disk with ``sync.sync_volume``, so disk and MinIO agree
and the periodic sync has nothing to revert.
"""

import logging
from pathlib import Path

import numpy as np
import s3fs
import zarr

from cellmap_flow.finetune.session import minio
from cellmap_flow.finetune.session.volume import volume_corner_nm

logger = logging.getLogger(__name__)


def box_voxels(volume, offset_nm, shape_nm, volume_shape):
    """``(lo, hi)``: the annotation voxels whose centres lie in a box, clipped to the volume.

    ``offset_nm`` and ``shape_nm`` are the box's lower corner and size in
    world nm, as a good region records them; ``volume`` is the volume's
    record, for its grid. None when no voxel of the volume is in the box.
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


def _fresh_ids(existing, count, count_up):
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


def seed_labels(objects, existing, count_up=False):
    """Labels for a box from a segmentation: an id per object inside, 1 outside.

    ``objects`` is a ``post.segment`` segmenter's output over the box (0
    background, objects 1..n), ``existing`` what the box holds now. Each
    object gets an id of its own (``_fresh_ids``; ``count_up`` for an
    instance target), so neighbouring objects are told apart, and a merge
    the model made shows as one colour to split; but an object the user has
    already painted part of takes the id painted there most, or the
    unpainted rest of it would be taught as a different object from its
    painted part. A binary or distance target reads every id as foreground,
    so the ids cost it nothing.
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
        pairs, counts = np.unique(
            np.stack([objects[painted], existing[painted].astype(np.int64)]), axis=1, return_counts=True
        )
        # Most-painted last, so its id is the one that stays.
        order = np.argsort(counts, kind="stable")
        ids[pairs[0, order]] = pairs[1, order]
    present = np.zeros(n + 1, dtype=bool)
    present[np.unique(objects[foreground])] = True
    fresh = np.flatnonzero(present & (ids == 0))
    ids[fresh] = _fresh_ids(existing, fresh.size, count_up)
    labels[foreground] = ids[objects[foreground]].astype(existing.dtype)
    return labels


def relabel_objects(existing, count_up=False, connectivity=1, per_slice=False):
    """The box's foreground (labels 2 and up) relabelled by connected component.

    Each component takes the id most of its voxels hold, so a stroke that
    joins two objects merges them into the id of the bigger. An id left on
    several components, an object a background wall cut in two, stays on the
    largest and the others get fresh ids (``_fresh_ids``). Background (1)
    and unpainted (0) voxels are left alone. Returns ``(labels, counts)``:
    ``counts`` has ``objects``, ``split`` (components given a fresh id) and
    ``merged`` (components that held more than one id).

    A cut has to go through every slice the object spans: the brush paints
    one slice, and an object cut in one slice is still one object above and
    below it. ``split`` is 0 then, which the page says. With ``per_slice``
    each z slice's pieces are components on their own, so a wall in one
    slice splits that slice, and every slice of an object gets its own id.
    ``connectivity``: as for ``post.segment.connected_components``.
    """
    from cellmap_flow.post.segment import connected_components

    labels = existing.copy()
    foreground = existing >= 2
    counts = {"objects": 0, "split": 0, "merged": 0}
    if not foreground.any():
        return labels, counts
    components = connected_components(foreground, connectivity, per_slice=per_slice).astype(np.int64)
    n = int(components.max())
    comp, old = components[foreground], existing[foreground].astype(np.int64)
    pairs, pair_counts = np.unique(np.stack([comp, old]), axis=1, return_counts=True)
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
        new[fresh] = _fresh_ids(existing, len(fresh), count_up)
    counts["split"] = len(fresh)
    labels[foreground] = new[comp].astype(existing.dtype)
    return labels, counts


def open_served_labels(state, volume_id):
    """The volume's ``annotation/s0`` as MinIO serves it, open for writing.

    The bucket key is ``<volume_id>.zarr``, as for the sync.
    """
    s3 = minio.make_s3_filesystem(state)
    root = f"{state['bucket']}/{volume_id}.zarr"
    if not s3.exists(f"{root}/annotation/s0/.zarray"):
        raise FileNotFoundError(f"MinIO has no {root}/annotation/s0; serve the volume first")
    # check=False skips S3Map's bucket probe: the array was just found.
    return s3, root, zarr.open(s3fs.S3Map(root=root, s3=s3, check=False), mode="r+")["annotation/s0"]


def _upload_chunks_only_on_disk(s3, root, arr, lo, hi, local_zarr_path):
    """Put the box's chunks that are on disk but not in MinIO up first.

    A chunk the box only partly covers is rewritten whole, from MinIO's copy
    or from zeros when MinIO has none, and the pull afterwards copies it over
    the one on disk. So a chunk only disk has -- an import whose mirror
    failed -- would lose what the box does not cover. Normally every chunk
    on disk is in MinIO and this only looks.
    """
    chunks = np.asarray(arr.chunks)
    first, last = np.asarray(lo) // chunks, (np.asarray(hi) - 1) // chunks
    local_s0 = Path(local_zarr_path) / "annotation" / "s0"
    for index in np.ndindex(*(last - first + 1)):
        key = ".".join(str(int(i)) for i in first + np.asarray(index))
        local = local_s0 / key
        remote = f"{root}/annotation/s0/{key}"
        if local.is_file() and not s3.exists(remote):
            logger.info(f"Chunk {key} of {root} is on disk only; uploading it before filling the box")
            s3.put(str(local), remote)


def fill_unpainted(state, volume_id, lo, hi, labels_for, local_zarr_path=None, undo=None):
    """Write ``labels_for(existing)`` into the box ``[lo, hi)`` where the volume holds 0.

    ``labels_for`` gets the box as MinIO holds it and returns labels of the
    same shape (0 for "leave as it is"). Returns ``(n_foreground,
    n_background)``: how many voxels were filled with each. Nothing is
    written when nothing would change. ``undo``, a list, gets ``(lo, hi,
    before, after)`` appended when something was written (``restore_box``).
    """
    s3, root, arr = open_served_labels(state, volume_id)
    if local_zarr_path:
        _upload_chunks_only_on_disk(s3, root, arr, lo, hi, local_zarr_path)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    existing = arr[box]
    labels = np.asarray(labels_for(existing), dtype=arr.dtype)
    fill = (existing == 0) & (labels > 0)
    if fill.any():
        before = existing.copy()
        existing[fill] = labels[fill]
        arr[box] = existing
        if undo is not None:
            undo.append((np.asarray(lo), np.asarray(hi), before, existing.copy()))
    n_foreground = int(np.count_nonzero(fill & (labels >= 2)))
    return n_foreground, int(np.count_nonzero(fill)) - n_foreground


def rewrite_foreground(state, volume_id, lo, hi, relabel, local_zarr_path=None, undo=None):
    """Write ``relabel(existing)``'s labels over the box's foreground voxels.

    ``relabel`` gets the box as MinIO holds it and returns ``(labels,
    counts)`` (``relabel_objects``); only foreground voxels (2 and up) whose
    label changed are written, so a stroke painted meanwhile elsewhere in
    the box is kept. Returns ``(n_changed, counts)``. ``undo``: as for
    ``fill_unpainted``.
    """
    s3, root, arr = open_served_labels(state, volume_id)
    if local_zarr_path:
        _upload_chunks_only_on_disk(s3, root, arr, lo, hi, local_zarr_path)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    existing = arr[box]
    labels, counts = relabel(existing)
    changed = (existing >= 2) & (np.asarray(labels, dtype=arr.dtype) != existing)
    if changed.any():
        before = existing.copy()
        existing[changed] = np.asarray(labels, dtype=arr.dtype)[changed]
        arr[box] = existing
        if undo is not None:
            undo.append((np.asarray(lo), np.asarray(hi), before, existing.copy()))
    return int(np.count_nonzero(changed)), counts


def restore_box(state, volume_id, lo, hi, before, after):
    """Put the box back to ``before`` where it still holds ``after``: an undo.

    Only voxels the action changed and nobody has touched since go back, so
    a stroke painted in the box after the action is kept. Returns how many
    voxels were restored.
    """
    _, _, arr = open_served_labels(state, volume_id)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    current = arr[box]
    restore = (before != after) & (current == after)
    if restore.any():
        current[restore] = before[restore]
        arr[box] = current
    return int(np.count_nonzero(restore))
