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
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import zarr

from cellmap_flow.finetune.session.manifest import voxels_inside_any_bbox
from cellmap_flow.finetune.session.volume import volume_corner_nm
from cellmap_flow.io.geometry import list_populated_chunks

logger = logging.getLogger(__name__)

# The pools' dtype. They are the bulk of the dataset, and every loader worker
# holds its own copy: a 40M-voxel crop is 480 MB of int32 per worker, 960 MB
# of int64. A volume is far below 2**31 voxels along any axis.
POOL_DTYPE = np.int32


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
        effective_dense_ratio: the share of non-rehearsal patches taken from
            ``dense``; ``dense_to_sparse_ratio`` as asked (None: auto),
            clamped away from an empty pool.
        rehearsal_centres: (M, 3) good-region centres in annotation voxels,
            or None when no region is usable.
        effective_rehearsal_fraction: the share of patches centred on one;
            0 without a usable region.
        annotated_chunks: how many chunks hold a voxel of a pool, the
            default epoch length.
        corner_nm, shape_voxels: where annotation voxel 0's lower corner is
            (see volume_corner_nm), and the volume's shape in voxels.
    """

    def __init__(
        self,
        volume_zarr_path: str,
        output_voxel_size_nm,
        jitter_voxels,
        dense_to_sparse_ratio: Optional[float] = None,
        good_regions: Optional[list] = None,
        rehearsal_fraction: Optional[float] = None,
    ):
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
        self._index_rehearsal()

    # ------------------------------------------------------------------
    # The draw
    # ------------------------------------------------------------------

    def draw(self, rng: np.random.Generator) -> Tuple[np.ndarray, bool]:
        """A patch centre in annotation voxels, and whether it is a rehearsal patch.

        The centre is not whole voxels yet (see PatchReader.snap). What is
        drawn from ``rng``, in order: rehearsal or not (only when there is a
        usable good region); then which region, or else which pool (only when
        both can be chosen), which voxel of it, and the jitter.
        """
        if (
            self.effective_rehearsal_fraction > 0.0
            and rng.random() < self.effective_rehearsal_fraction
        ):
            # One region is one patch, centred exactly: no jitter, or the
            # loss would spill outside the area that was actually judged.
            centres = self.rehearsal_centres
            return centres[rng.integers(0, centres.shape[0])].copy(), True

        use_dense = self.effective_dense_ratio >= 1.0 or (
            self.effective_dense_ratio > 0.0 and rng.random() < self.effective_dense_ratio
        )
        pool = self.dense if use_dense else self.sparse
        anchor = pool[rng.integers(0, pool.shape[0])].astype(np.float64)
        jitter = rng.integers(low=-self.jitter, high=self.jitter + 1, size=3).astype(np.float64)
        return anchor + jitter, False

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

        arr = zarr.open(s0_path, mode="r")
        self.shape_voxels = np.array(arr.shape, dtype=int)
        chunk_shape = np.array(arr.chunks, dtype=int)

        # In chunk-index order, so that a seed draws the same patches on any
        # filesystem and in a copied session.
        chunks = list_populated_chunks(s0_path)
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
        for index in chunks:
            chunk_origin = np.array(index, dtype=np.int64) * chunk_shape
            chunk_data = arr.blocks[index]
            if bbox_offsets.shape[0] == 0:
                # No imported crops: everything is sparse (painted).
                painted_local = np.argwhere(chunk_data >= 1).astype(np.int64)
                if not painted_local.size:
                    # The file exists (zarr writes fill chunks during slab
                    # writes) but holds no annotation; skip it.
                    continue
                n_fg_chunks += 1
                sparse_rows.append((painted_local + chunk_origin).astype(POOL_DTYPE))
                continue
            annotated_local = np.argwhere(chunk_data >= 1).astype(np.int64)
            if not annotated_local.size:
                continue
            is_fg = chunk_data[tuple(annotated_local.T)] >= 2
            annotated_global = annotated_local + chunk_origin
            in_dense = voxels_inside_any_bbox(annotated_global, bbox_offsets, bbox_ends)
            contributed = False
            if (in_dense & is_fg).any():
                dense_rows.append(annotated_global[in_dense & is_fg].astype(POOL_DTYPE))
                contributed = True
            if (~in_dense).any():
                sparse_rows.append(annotated_global[~in_dense].astype(POOL_DTYPE))
                contributed = True
            n_fg_chunks += int(contributed)

        self.dense = (
            np.concatenate(dense_rows, axis=0) if dense_rows else np.zeros((0, 3), dtype=POOL_DTYPE)
        )
        self.sparse = (
            np.concatenate(sparse_rows, axis=0) if sparse_rows else np.zeros((0, 3), dtype=POOL_DTYPE)
        )
        if self.dense.shape[0] == 0 and self.sparse.shape[0] == 0 and bbox_offsets.shape[0]:
            # Imported crops that are all background, and nothing painted:
            # centre on the crops' annotated voxels rather than refuse.
            self.dense = _annotated_voxels_in_crops(
                arr, chunks, chunk_shape, bbox_offsets, bbox_ends
            )
            n_fg_chunks = max(n_fg_chunks, 1 if self.dense.shape[0] else 0)
        self.annotated_chunks = n_fg_chunks

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
            try:
                offset_nm = np.array(region["offset_nm"], dtype=float)
                shape_nm = np.array(region["shape_nm"], dtype=float)
            except (KeyError, TypeError, ValueError):
                logger.warning(f"Skipping malformed good region: {region!r}")
                continue
            centre_nm = offset_nm + shape_nm / 2.0
            centre_voxels = (centre_nm - self.corner_nm) / self.output_voxel_size
            # A region marked against a different volume would sample pure
            # out-of-bounds zeros and quietly anchor the model to nothing.
            if np.any(centre_voxels < 0) or np.any(centre_voxels >= self.shape_voxels):
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
        """Say what is happening with the good regions, if there are any.

        Rehearsal switched off -- a deliberate choice -- and good regions that
        all fell outside the volume are different states, and are told apart.
        """
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
    """Every annotated voxel inside the imported crops, as (N, 3) global indices."""
    rows = []
    for index in chunks:
        chunk_origin = np.array(index, dtype=np.int64) * chunk_shape
        annotated = np.argwhere(arr.blocks[index] >= 1).astype(np.int64) + chunk_origin
        if annotated.size:
            inside = voxels_inside_any_bbox(annotated, bbox_offsets, bbox_ends)
            if inside.any():
                rows.append(annotated[inside].astype(POOL_DTYPE))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 3), dtype=POOL_DTYPE)
