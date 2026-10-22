"""Labelling a whole box of an annotation volume at once.

The Finetune tab's "Seed" and "All background" buttons label the box on
screen in one click: from the model's own prediction, for the user to clean
up with the brush, or all background (1), for a region of false positives.
Either way only unannotated voxels (0) are filled. A stroke the user painted
is a decision about that voxel; a seed is a guess. ``paint_box`` is the
write under both, and can also write over painted voxels, for an AI
annotation the user has reviewed and chosen to keep over what is there. "Split objects" relabels
the box's foreground by connected component (``relabel_objects``), so a
background wall painted through a merged object gives it two ids again.

The objects a seed writes come from ``post.segment``, the segmenter that
fits the model (``seed_labels`` takes its labels). Split Objects is
6-connected by default (face neighbours, scipy's default): a one-voxel wall
the brush paints cuts an object under it, where under 26-connectivity the
voxels across a diagonal wall would still touch.

The decisions (which voxels a box covers, the ids a seed or a split gives,
which voxels each write changes and an undo puts back) are
``cellmap_flow.portable.fill``'s, which the browser makes too, over its paint
store; they are re-exported here, beside the reads and writes around them.

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
from cellmap_flow.portable.fill import (  # noqa: F401  (re-exported: the routes label boxes through them)
    box_voxels,
    paint,
    relabel_objects,
    restore,
    rewrite,
    seed_labels,
)

logger = logging.getLogger(__name__)


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


def paint_box(state, volume_id, lo, hi, labels_for, local_zarr_path=None, undo=None, overwrite=False):
    """Write ``labels_for(existing)`` into the box ``[lo, hi)``.

    ``labels_for`` gets the box as MinIO holds it and returns labels of the
    same shape (0 for "leave as it is"). Without ``overwrite`` only
    unannotated voxels (0) are written, as for a seed: what the user painted
    stays. With it every voxel the labels cover is written, painted or not,
    for a result the user has looked at and prefers to what is there.
    Returns ``(n_foreground, n_background, n_overwritten)``: how many voxels
    were written with foreground and with background, and how many of those
    held a label before. Nothing is written when nothing would change.
    ``undo``, a list, gets ``(lo, hi, before, after)`` appended when
    something was written (``restore_box``).
    """
    s3, root, arr = open_served_labels(state, volume_id)
    if local_zarr_path:
        _upload_chunks_only_on_disk(s3, root, arr, lo, hi, local_zarr_path)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    existing = arr[box]
    after, counts = paint(existing, labels_for(existing), overwrite=overwrite)
    if after is not None:
        arr[box] = after
        if undo is not None:
            undo.append((np.asarray(lo), np.asarray(hi), existing, after.copy()))
    return counts


def fill_unpainted(state, volume_id, lo, hi, labels_for, local_zarr_path=None, undo=None):
    """Write ``labels_for(existing)`` into the box ``[lo, hi)`` where the volume holds 0.

    ``labels_for`` gets the box as MinIO holds it and returns labels of the
    same shape (0 for "leave as it is"). Returns ``(n_foreground,
    n_background)``: how many voxels were filled with each. Nothing is
    written when nothing would change. ``undo``, a list, gets ``(lo, hi,
    before, after)`` appended when something was written (``restore_box``).
    """
    n_foreground, n_background, _ = paint_box(
        state, volume_id, lo, hi, labels_for, local_zarr_path=local_zarr_path, undo=undo
    )
    return n_foreground, n_background


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
    after, n_changed = rewrite(existing, labels)
    if after is not None:
        arr[box] = after
        if undo is not None:
            undo.append((np.asarray(lo), np.asarray(hi), existing, after.copy()))
    return n_changed, counts


def restore_box(state, volume_id, lo, hi, before, after):
    """Put the box back to ``before`` where it still holds ``after``: an undo.

    Only voxels the action changed and nobody has touched since go back, so
    a stroke painted in the box after the action is kept. Returns how many
    voxels were restored.
    """
    _, _, arr = open_served_labels(state, volume_id)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    restored, n = restore(arr[box], before, after)
    if restored is not None:
        arr[box] = restored
    return n
