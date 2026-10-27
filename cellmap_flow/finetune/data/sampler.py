"""Where a finetune patch is centred: the dense, sparse and rehearsal pools.

Two-pool stratified sampling. The annotated voxels of a session's volume are
split by its ``imported_crops`` boxes (recorded in the volume's attrs when
YAML crops are imported):

- **dense pool**: foreground voxels inside any imported crop (abundant
  ground truth);
- **sparse pool**: annotated voxels outside every crop, background as well
  as foreground (painted scribbles: by construction sparse and informative,
  since the user paints where the base model failed).

A patch picks a pool by ``dense_to_sparse_ratio`` (0.5/0.5 when both pools
have voxels, the non-empty one when only one has), a random voxel of that
pool, and a jitter of the centre. Without the split, voxel-uniform sampling
buries the scribbles: a typical session has ~40M dense voxels against ~10K
painted, so 999 patches in 1000 would be dense and the corrections barely
move the gradient. The split gives them a set share of every epoch,
whatever the voxel counts.

A third pool, the good regions, is drawn from directly (see PatchSampler).

The index reads only **populated** chunks, the ``z.y.x`` files under
``annotation/s0/``: an empty volume costs nothing, a painted region its
annotated voxels.

A one-slice patch (a 2D network's) can be cut in the plane its paint was
drawn in rather than always across z (``slice_planes``): paint drawn in a
z–y slice is one voxel thick in x, and an x–y slice through it holds only a
line, a voxel wide, which teaches a 2D network next to nothing (Cellpose
learned flows along y alone from it, and segmented strips).
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import worldarray as wa

from cellmap_flow.finetune.data.reader import across
from cellmap_flow.finetune.session.manifest import read_chunk, voxels_inside_any_bbox
from cellmap_flow.finetune.session.volume import volume_corner_nm
from cellmap_flow.portable.regions import region_centre_voxels

logger = logging.getLogger(__name__)

# What each axis a slice is cut across makes of it, by axis (z, y, x).
PLANE_NAMES = ("x–y", "z–x", "z–y")


def paint_depth_axes(voxels: np.ndarray) -> np.ndarray:
    """The axis each painted voxel's paint is thinnest across, as int8: 0 (z,
    an x–y slice) unless y or x is clearly thinner.

    ``voxels`` (N, 3) are the painted voxels, each once. A voxel's paint is
    measured by the run of painted voxels through it along each axis: paint
    drawn in a z–y slice has runs of 1 along x and long ones along z and y.
    y or x is taken when its run is at most half z's, so a blob painted in
    3D, or a lone voxel, stays an x–y slice as every patch was before, and a
    line along z, which lies in both side planes, goes to y.
    """
    voxels = np.asarray(voxels, dtype=np.int64)
    n = voxels.shape[0]
    depth = np.zeros(n, dtype=np.int8)
    if n == 0:
        return depth
    runs = np.empty((3, n), dtype=np.int32)
    for axis in range(3):
        a, b = (other for other in range(3) if other != axis)
        # Sorted into lines along `axis`, in order along it; a run ends where
        # the line changes or a voxel is skipped.
        order = np.lexsort((voxels[:, axis], voxels[:, b], voxels[:, a]))
        line = voxels[order]
        starts = np.ones(n, dtype=bool)
        starts[1:] = (
            (line[1:, a] != line[:-1, a])
            | (line[1:, b] != line[:-1, b])
            | (line[1:, axis] != line[:-1, axis] + 1)
        )
        run = np.cumsum(starts) - 1
        runs[axis, order] = np.bincount(run)[run]
    side = 1 + np.argmin(runs[1:], axis=0)
    thinner = 2 * runs[side, np.arange(n)] <= runs[0]
    depth[thinner] = side[thinner]
    return depth

# The pools' dtype. They are the bulk of the dataset, and every loader worker
# holds its own copy: a 40M-voxel crop is 480 MB of int32 per worker, 960 MB
# of int64. A volume is far below 2**31 voxels along any axis.
POOL_DTYPE = np.int32

# The share of painted patches centred on a painted foreground voxel; the
# rest are drawn by chunk, background fixes included. nnU-Net's
# oversample_foreground_percent (Isensee et al., 2021): the usual floor for
# sparse labels, so that a session of mostly background fixes still trains
# on the objects it painted, while the class-balanced loss weighs what each
# patch holds.
FOREGROUND_SHARE = 1 / 3

# The share of patches a painted session centres at a random point of the
# volume, by default. Their unpainted voxels are held to the teacher (as a
# good region's are), so the model is kept where it was everywhere nothing
# was painted. Without them every patch sat beside a stroke: a run that
# learned its strokes completely (supervised loss to 0.0002) shifted the
# whole network, and areas that had sat just below 0.5 turned into false
# positives across the volume, where no patch ever looked.
ANCHOR_SHARE = 0.25


class PatchSampler:
    """The patch centres of one annotation volume, and the draw among them.

    Rehearsal ("good") regions are boxes the user looked at and certified
    the model already handles. The supervised loss only touches voxels that
    were labelled, so nothing stops the adapter drifting everywhere else; a
    rehearsal patch pins the student to the teacher inside a region the user
    vouched for, chosen deliberately rather than "wherever happens to sit
    next to a scribble", which is where the model is least likely to be
    right. A good region carries no annotations, so neither of the other
    pools, which index annotated voxels, can reach it: it is a pool of its
    own, of box centres.

    Built once, where the dataset is made; the pools travel to each loader
    worker in the dataset's pickle.

    Attributes:
        dense, sparse: (N, 3) annotation-voxel indices of the two pools
            (POOL_DTYPE), in the order the chunk files are listed and
            scanned. Either may be empty, never both.
        sparse_chunk_starts: where each chunk's voxels begin in ``sparse``,
            plus its length at the end: chunk i holds rows
            ``sparse_chunk_starts[i]:sparse_chunk_starts[i + 1]``.
        sparse_foreground_rows: the rows of ``sparse`` that are foreground;
            the rest are painted background.
        effective_dense_ratio: the share of non-rehearsal patches taken from
            ``dense``; ``dense_to_sparse_ratio`` as asked (None: auto),
            clamped away from an empty pool.
        rehearsal_centres: (M, 3) good-region centres in annotation voxels,
            or None when no region is usable.
        effective_rehearsal_fraction: the share of patches centred on one;
            0 without a usable region.
        effective_anchor_fraction: the share of the other patches centred at a
            random point of the volume, held to the teacher where unpainted;
            ANCHOR_SHARE by default for a painted session, else 0.
        annotated_chunks: how many chunks hold a voxel of a pool, the
            default epoch length.
        class_voxels: (background, foreground), how many voxels of the
            volume are annotated as each, the crops' and the paint's alike.
        corner_nm, shape_voxels: where annotation voxel 0's lower corner is
            (see volume_corner_nm), and the volume's shape in voxels.
        paint_planes: how many painted voxels (of ``sparse``) lie in an x–y,
            a z–x and a z–y slice (``paint_depth_axes``), or None when
            ``slice_planes`` is None.
        sparse_depth: the axis each row of ``sparse`` is cut across, when
            ``slice_planes`` is "any" and some paint is in a side plane;
            else None, and every patch is cut across z.

    ``slice_planes`` is for a one-slice patch: None, a 3D patch, which
    takes paint drawn in any plane as it is; "xy", one slice that is always
    an x–y one, whose paint is only counted (``paint_planes``) so that paint
    in the side planes can be warned about; "any", one slice in whichever
    plane the paint around its centre lies in. ``jitter_voxels`` is then in
    the patch's own order (``reader.slice_axes``), its first entry across
    the slice.
    """

    def __init__(
        self,
        volume_zarr_path: str,
        output_voxel_size_nm,
        jitter_voxels,
        dense_to_sparse_ratio: Optional[float] = None,
        good_regions: Optional[list] = None,
        rehearsal_fraction: Optional[float] = None,
        anchor_fraction: Optional[float] = None,
        slice_planes: Optional[str] = None,
    ):
        if slice_planes not in (None, "xy", "any"):
            raise ValueError(f"slice_planes must be None, 'xy' or 'any', got {slice_planes!r}")
        self.volume_zarr_path = volume_zarr_path
        self.output_voxel_size = np.array(output_voxel_size_nm, dtype=float)
        self.jitter = np.array(jitter_voxels, dtype=int)
        self.dense_to_sparse_ratio = (
            float(dense_to_sparse_ratio) if dense_to_sparse_ratio is not None else None
        )
        self.good_regions = list(good_regions or [])
        self.rehearsal_fraction = (
            float(rehearsal_fraction) if rehearsal_fraction is not None else None
        )
        self._index_pools()
        self._index_planes(slice_planes)
        self._index_rehearsal()
        # Painted sessions only, by default: imported crops are dense, and a
        # run on them is unchanged unless asked.
        if anchor_fraction is not None:
            self.effective_anchor_fraction = max(0.0, min(1.0, float(anchor_fraction)))
        else:
            self.effective_anchor_fraction = ANCHOR_SHARE if self.sparse.shape[0] else 0.0

    # ------------------------------------------------------------------
    # The draw
    # ------------------------------------------------------------------

    def draw(self, rng: np.random.Generator) -> Tuple[np.ndarray, bool, int]:
        """A patch centre in annotation voxels, whether it is a rehearsal
        patch, and the axis the patch is cut across (0, z, unless
        ``slice_planes`` is "any").

        The centre is not whole voxels yet (see PatchReader.snap). What is
        drawn from ``rng``, in order: rehearsal or not (only when there is a
        usable good region); then which region, or else a random anchor or
        not (only when anchors are on), where, and its plane (only when
        some paint is in a side plane); or else which pool (only when
        both can be chosen); in the painted pool, whether to centre on
        foreground (only when some is painted), then which chunk, or which
        foreground voxel; which voxel, and the jitter.
        """
        if (
            self.effective_rehearsal_fraction > 0.0
            and rng.random() < self.effective_rehearsal_fraction
        ):
            # One region is one patch, centred exactly: no jitter, or the
            # loss would spill outside the area that was actually judged.
            # Across z: a region is a model output, a few slices deep where
            # it is a 2D model's, and a side slice would run far past it.
            centres = self.rehearsal_centres
            return centres[rng.integers(0, centres.shape[0])].copy(), True, 0

        if self.effective_anchor_fraction > 0.0 and rng.random() < self.effective_anchor_fraction:
            # Anywhere in the volume, whole voxels, no jitter needed.
            centre = rng.integers(0, self.shape_voxels).astype(np.float64)
            # In the planes the paint is in, in the same proportions: the
            # model is held to its teacher in each orientation it trains in.
            depth = 0
            if self.sparse_depth is not None:
                depth = int(self.sparse_depth[rng.integers(0, self.sparse_depth.shape[0])])
            return centre, True, depth

        use_dense = self.effective_dense_ratio >= 1.0 or (
            self.effective_dense_ratio > 0.0 and rng.random() < self.effective_dense_ratio
        )
        foreground = self.sparse_foreground_rows
        row = None  # of the painted pool
        if use_dense:
            anchor = self.dense[rng.integers(0, self.dense.shape[0])]
        elif foreground.size and rng.random() < FOREGROUND_SHARE:
            row = foreground[rng.integers(0, foreground.size)]
        else:
            # A chunk first, then a voxel in it: each place that was painted
            # gets about the same share of patches, however many voxels its
            # strokes cover. Drawn by voxel, one wide background stroke
            # (2,000 voxels where a nucleus had 300) took most of an epoch,
            # and a session's few foreground fixes were rarely seen.
            starts = self.sparse_chunk_starts
            chunk = rng.integers(0, starts.shape[0] - 1)
            row = rng.integers(starts[chunk], starts[chunk + 1])
        depth = 0
        if row is not None:
            anchor = self.sparse[row]
            if self.sparse_depth is not None:
                depth = int(self.sparse_depth[row])
        anchor = anchor.astype(np.float64)
        # Never off the slice: the jitter across it is the patch's first entry.
        jitter_range = across(self.jitter, depth)
        jitter = rng.integers(low=-jitter_range, high=jitter_range + 1, size=3).astype(np.float64)
        return anchor + jitter, False, depth

    # ------------------------------------------------------------------
    # The dense and sparse pools
    # ------------------------------------------------------------------

    def _index_pools(self) -> None:
        """Walk the volume's populated chunks into the dense and sparse pools.

        zarr v2 stores one file per chunk, named ``z.y.x``, so only the chunks
        that were written are read, and each annotated voxel goes to the dense
        pool (foreground inside an imported crop) or the sparse one (anything
        annotated outside every crop).
        """
        s0_path = os.path.join(self.volume_zarr_path, "annotation", "s0")
        if not os.path.isdir(s0_path):
            raise ValueError(
                f"Volume zarr at {self.volume_zarr_path} has no annotation/s0/ "
                "directory; was it created?"
            )

        # Volume-level metadata, read once: where voxel 0 is, and the crops'
        # boxes that decide dense from sparse.
        with open(os.path.join(self.volume_zarr_path, ".zattrs")) as f:
            root_attrs = json.load(f)
        self.corner_nm = volume_corner_nm(
            root_attrs.get("dataset_offset_nm"), self.output_voxel_size
        )
        imported = root_attrs.get("imported_crops", []) or []
        # The boxes as two stacked (M, 3) arrays, for vectorized membership
        # tests. Empty when no YAML crops were imported.
        if imported:
            bbox_offsets = np.array(
                [c["annotation_offset_voxels"] for c in imported], dtype=np.int64
            )
            bbox_shapes = np.array(
                [c["annotation_shape_voxels"] for c in imported], dtype=np.int64
            )
            bbox_ends = bbox_offsets + bbox_shapes
        else:
            bbox_offsets = np.zeros((0, 3), dtype=np.int64)
            bbox_ends = np.zeros((0, 3), dtype=np.int64)

        arr = wa.open(s0_path, units=None)  # index reads only
        self.shape_voxels = np.array(arr.shape, dtype=int)
        chunk_shape = np.array(arr.meta.chunk_shape, dtype=int)

        # The chunks that have a z.y.x file, in chunk-index order rather than
        # the filesystem's (ext4 lists names by a per-filesystem hash), so
        # that a seed draws the same patches on any machine and in a copied
        # session.
        chunks = wa.list_chunk_keys(s0_path, separator=".", ndim=3)
        if not chunks:
            raise ValueError(
                f"Volume zarr at {self.volume_zarr_path} has no populated chunks. "
                "Paint annotations or import crops first."
            )
        self.chunk_files = len(chunks)

        # The sparse (painted) pool holds every annotated voxel outside the
        # crops, background as well as foreground: a background-only
        # correction -- painting 1 where the model hallucinates -- further
        # than about half a patch from any foreground must be drawn too, or
        # the false-positive fix silently does nothing, and a session that
        # painted only background cannot train at all. The dense pool stays
        # foreground-centred: crops are mostly background, and centring on
        # it would change what they teach.
        dense_rows: List[np.ndarray] = []
        sparse_rows: List[np.ndarray] = []
        n_fg_chunks = 0  # chunks that contributed voxels to a pool
        sparse_is_fg: List[np.ndarray] = []  # per sparse_rows entry, which rows are foreground
        n_foreground = n_annotated = 0
        for index in chunks:
            chunk_origin = np.array(index, dtype=np.int64) * chunk_shape
            chunk_data = read_chunk(arr, index)
            annotated_local = np.argwhere(chunk_data >= 1).astype(np.int64)
            if not annotated_local.size:
                # The file exists (zarr writes fill chunks during slab
                # writes) but holds no annotation; skip it.
                continue
            is_fg = chunk_data[tuple(annotated_local.T)] >= 2
            n_foreground += int(is_fg.sum())
            n_annotated += is_fg.size
            if bbox_offsets.shape[0] == 0:
                # No imported crops: everything is sparse (painted).
                n_fg_chunks += 1
                sparse_rows.append((annotated_local + chunk_origin).astype(POOL_DTYPE))
                sparse_is_fg.append(is_fg)
                continue
            annotated_global = annotated_local + chunk_origin
            in_dense = voxels_inside_any_bbox(annotated_global, bbox_offsets, bbox_ends)
            contributed = False
            if (in_dense & is_fg).any():
                dense_rows.append(annotated_global[in_dense & is_fg].astype(POOL_DTYPE))
                contributed = True
            if (~in_dense).any():
                sparse_rows.append(annotated_global[~in_dense].astype(POOL_DTYPE))
                sparse_is_fg.append(is_fg[~in_dense])
                contributed = True
            n_fg_chunks += int(contributed)

        self.dense = (
            np.concatenate(dense_rows, axis=0) if dense_rows else np.zeros((0, 3), dtype=POOL_DTYPE)
        )
        self.sparse = (
            np.concatenate(sparse_rows, axis=0) if sparse_rows else np.zeros((0, 3), dtype=POOL_DTYPE)
        )
        # One block of rows per chunk: a chunk adds at most one to sparse_rows.
        self.sparse_chunk_starts = np.concatenate(
            [[0], np.cumsum([rows.shape[0] for rows in sparse_rows], dtype=np.int64)]
        ).astype(np.int64)
        self.sparse_foreground_rows = (
            np.flatnonzero(np.concatenate(sparse_is_fg)) if sparse_is_fg else np.zeros(0, dtype=np.int64)
        )
        if self.dense.shape[0] == 0 and self.sparse.shape[0] == 0 and bbox_offsets.shape[0]:
            # Imported crops that are all background, and nothing painted:
            # centre on the crops' annotated voxels rather than refuse. Their
            # chunks count as any other pool's do: a crop over 8 chunks gets
            # 8 patches an epoch, not 1.
            self.dense, n_fg_chunks = _annotated_voxels_in_crops(
                arr, chunks, chunk_shape, bbox_offsets, bbox_ends
            )
        self.annotated_chunks = n_fg_chunks
        self.class_voxels = (n_annotated - n_foreground, n_foreground)

        n_dense, n_sparse = int(self.dense.shape[0]), int(self.sparse.shape[0])
        if n_dense == 0 and n_sparse == 0:
            raise ValueError(
                f"Volume zarr at {self.volume_zarr_path} has populated chunks "
                "but no annotated voxels. Paint annotations or import crops first."
            )

        # An explicit ratio wins, else 0.5 when both pools have voxels, else
        # whichever pool is non-empty, so there are still patches to draw.
        if self.dense_to_sparse_ratio is None:
            if n_dense > 0 and n_sparse > 0:
                self.effective_dense_ratio = 0.5
            elif n_dense > 0:
                self.effective_dense_ratio = 1.0
            else:
                self.effective_dense_ratio = 0.0
        else:
            ratio = max(0.0, min(1.0, self.dense_to_sparse_ratio))
            # Clamped away from an empty pool, so draw() never picks one.
            if n_dense == 0:
                self.effective_dense_ratio = 0.0
            elif n_sparse == 0:
                self.effective_dense_ratio = 1.0
            else:
                self.effective_dense_ratio = ratio

    # ------------------------------------------------------------------
    # The planes the paint lies in
    # ------------------------------------------------------------------

    def _index_planes(self, slice_planes: Optional[str]) -> None:
        """Find the plane each painted voxel lies in, for a one-slice patch.

        The painted pool only: an imported crop is labelled through and
        through, and stays in x–y slices, as before. So do the random anchors
        of a session painted in x–y alone: ``sparse_depth`` is None then, and
        the draws are the ones they were.
        """
        self.paint_planes = None
        self.sparse_depth = None
        if slice_planes is None:
            return
        depth = paint_depth_axes(self.sparse)
        self.paint_planes = np.bincount(depth, minlength=3)
        if slice_planes == "any" and depth.any():
            self.sparse_depth = depth

    # ------------------------------------------------------------------
    # The rehearsal pool
    # ------------------------------------------------------------------

    def _index_rehearsal(self) -> None:
        """Turn the good-region boxes (nm) into patch centres (annotation voxels).

        A region is sized to one model output patch, so one region is one
        patch: centre on it and the loss covers exactly the area that was
        looked at and judged. No jitter -- jitter would slide the patch out
        of the region the user actually vouched for.
        """
        centres = []
        for region in self.good_regions:
            # What counts as a region, and where its centre lies, is the rule
            # of cellmap_flow.portable.regions, by which the browser's
            # trainer places regions too.
            where, centre_voxels = region_centre_voxels(
                region, self.corner_nm, self.output_voxel_size, self.shape_voxels
            )
            if where == "malformed":
                logger.warning(f"Skipping malformed good region: {region!r}")
                continue
            # A region marked against a different volume would sample pure
            # out-of-bounds zeros and quietly anchor the model to nothing.
            if where == "outside":
                logger.warning(
                    f"Good region {region.get('id', '?')} centres outside the "
                    f"volume at voxel {centre_voxels.astype(int).tolist()}; skipping."
                )
                continue
            centres.append(centre_voxels)

        if not centres:
            self.rehearsal_centres = None
            self.effective_rehearsal_fraction = 0.0
            return

        self.rehearsal_centres = np.array(centres, dtype=np.float64)
        if self.rehearsal_fraction is None:
            # One patch in four. Enough to hold the model without drowning
            # out the corrections: rehearsal patches carry dense teacher
            # targets over the whole output, whereas a scribble patch may
            # label only a few hundred voxels, so parity by patch count is
            # already generous to the anchor.
            self.effective_rehearsal_fraction = 0.25
        else:
            self.effective_rehearsal_fraction = max(0.0, min(1.0, self.rehearsal_fraction))

    def log_rehearsal_status(self) -> None:
        """Say what is happening with the good regions, if there are any, and
        with the random anchors.

        Rehearsal switched off -- a deliberate choice -- and good regions that
        all fell outside the volume are different states, and are told apart.
        """
        if self.effective_anchor_fraction > 0:
            logger.info(
                f"VirtualPatchDataset: {self.effective_anchor_fraction:.0%} of the other patches "
                "are anchors at random points of the volume, held to the teacher where unpainted"
            )
        if self.effective_rehearsal_fraction > 0:
            logger.info(
                f"VirtualPatchDataset: {len(self.rehearsal_centres)} good "
                f"region(s); {self.effective_rehearsal_fraction:.0%} of patches "
                f"will be rehearsal anchors "
                f"({'auto' if self.rehearsal_fraction is None else 'explicit'})"
            )
        elif not self.good_regions:
            return
        elif self.rehearsal_fraction is not None and self.rehearsal_fraction <= 0:
            logger.info(
                f"{len(self.good_regions)} good region(s) present, but "
                f"rehearsal is set to 0, so training will not anchor on them."
            )
        else:
            logger.warning(
                f"{len(self.good_regions)} good region(s) configured but none "
                "of them landed inside the annotation volume; training will "
                "not anchor on them. See the 'outside the volume' lines above."
            )


def _annotated_voxels_in_crops(arr, chunks, chunk_shape, bbox_offsets, bbox_ends):
    """Every annotated voxel inside the imported crops, as (N, 3) global
    indices, and how many chunks hold one."""
    rows = []
    for index in chunks:
        chunk_origin = np.array(index, dtype=np.int64) * chunk_shape
        annotated = np.argwhere(read_chunk(arr, index) >= 1).astype(np.int64) + chunk_origin
        if annotated.size:
            inside = voxels_inside_any_bbox(annotated, bbox_offsets, bbox_ends)
            if inside.any():
                rows.append(annotated[inside].astype(POOL_DTYPE))
    voxels = np.concatenate(rows, axis=0) if rows else np.zeros((0, 3), dtype=POOL_DTYPE)
    return voxels, len(rows)
