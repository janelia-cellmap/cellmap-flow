"""
On-the-fly random-patch dataset for finetuning.

Architecture
------------
There is exactly one source of truth per session: an
``annotation_volume.zarr`` (sparse, full-dataset extent, OME-NGFF) that
holds **every** annotation — painted scribbles plus any imported YAML
crops, all merged at their physical offsets. This dataset reads patches
straight out of that single volume zarr; no per-tile materialization, no
parallel source list to keep in sync.

Sampling rule
-------------
Two-pool stratified sampling. FG voxels are partitioned by membership in
the volume's ``imported_crops`` bbox list (recorded in the volume zattrs
when YAML crops are imported):
  - **dense pool**: voxels inside any imported_crops bbox (abundant GT)
  - **sparse pool**: voxels outside all bboxes (painted scribbles, by
    construction always sparse and informative — the user paints there
    because the base model failed)

Each ``__getitem__`` picks a pool by ``dense_to_sparse_ratio`` (default
0.5/0.5 when both pools exist; auto-degrades to 1.0 when only one
exists), samples a random FG voxel from that pool, jitters the patch
center, and reads raw + annotation patches around it.

Without stratification, voxel-uniform sampling buries scribbles: a
typical session has ~40M dense voxels vs ~10K painted, so 999/1000
patches would be dense and the corrections you painted barely move the
gradient. Stratification guarantees scribbles get a defined share of
each epoch regardless of voxel count.

Index construction reads only **populated** chunks of the sparse zarr
(walks ``annotation/s0/`` for files matching ``z.y.x``). For an empty
volume that's an empty index; for a fully painted region it's the FG
voxels of those chunks.

Reviewer notes
--------------
- Workers each rebuild the FG index on spawn (cheap — only populated
  chunks are read). We don't pickle any open zarr/tensorstore handles.
- ``len(self)`` is ``patches_per_epoch``; it has no relationship to the
  number of populated chunks. The trainer treats this as the epoch length.
- The dataset returns ``(raw, annotation)`` tensors with shape
  ``(1, Z, Y, X)`` matching :class:`CorrectionDataset`'s contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_CHUNK_KEY_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _voxels_inside_any_bbox(
    voxels: np.ndarray, bbox_offsets: np.ndarray, bbox_ends: np.ndarray
) -> np.ndarray:
    """Return a boolean mask: ``True`` where ``voxels[i]`` lies inside any
    ``[bbox_offsets[j], bbox_ends[j])`` half-open box.

    voxels: (N, 3) int. bbox_offsets, bbox_ends: (M, 3) int. Vectorized
    over both: builds an (N, M) inside-test matrix and reduces along M.
    For typical M ~ 1-10 the temporary stays small.
    """
    if voxels.shape[0] == 0 or bbox_offsets.shape[0] == 0:
        return np.zeros(voxels.shape[0], dtype=bool)
    # (N, M, 3) broadcast: voxels[:, None, :] vs bbox_offsets[None, :, :]
    ge = np.all(voxels[:, None, :] >= bbox_offsets[None, :, :], axis=-1)
    lt = np.all(voxels[:, None, :] < bbox_ends[None, :, :], axis=-1)
    return np.any(ge & lt, axis=-1)


class VirtualPatchDataset(Dataset):
    """Yield random raw+annotation patches anchored on FG voxels in a volume zarr.

    Args:
        volume_zarr_path: path to the session's ``annotation_volume.zarr``.
        raw_dataset_path: path to the raw EM zarr the volume is aligned to.
        input_size_voxels: shape (Z, Y, X) of the raw patch returned per
            sample, in voxels at ``input_voxel_size_nm``.
        output_size_voxels: shape (Z, Y, X) of the annotation patch, in
            voxels at ``output_voxel_size_nm``.
        input_voxel_size_nm: voxel size for raw patches (the dataset's
            closest scale to the model's claimed input voxel size).
        output_voxel_size_nm: voxel size for annotation patches.
        patches_per_epoch: ``len(self)``; controls how many random patches
            comprise one epoch. ``None`` (the default) means "auto:
            substitute the total populated-chunk count" — every populated
            chunk gets ~one patch per epoch on average.
        jitter_voxels: half-range of the random offset applied to the patch
            center, in **annotation voxels**. Defaults to
            ``output_size_voxels // 4``.
        seed: RNG seed; per-worker offset added so multi-worker dataloaders
            sample distinct streams.
        dense_to_sparse_ratio: fraction in [0, 1] of patches drawn from
            the dense pool (FG voxels inside any imported_crops bbox).
            ``None`` (default) means auto: 0.5 if both pools have voxels,
            else 1.0 (use the non-empty pool exclusively).
    """

    def __init__(
        self,
        volume_zarr_path: str,
        raw_dataset_path: str,
        input_size_voxels: Tuple[int, int, int],
        output_size_voxels: Tuple[int, int, int],
        input_voxel_size_nm: Tuple[float, float, float],
        output_voxel_size_nm: Tuple[float, float, float],
        patches_per_epoch: Optional[int] = None,
        jitter_voxels: Optional[Tuple[int, int, int]] = None,
        seed: int = 0,
        input_norm_config: Optional[dict] = None,
        dense_to_sparse_ratio: Optional[float] = None,
    ):
        self.volume_zarr_path = volume_zarr_path
        self.raw_dataset_path = raw_dataset_path
        self.input_size = np.array(input_size_voxels, dtype=int)
        self.output_size = np.array(output_size_voxels, dtype=int)
        self.input_voxel_size = np.array(input_voxel_size_nm, dtype=float)
        self.output_voxel_size = np.array(output_voxel_size_nm, dtype=float)
        # Resolved to an int by _build_index() once the populated-chunk
        # count is known (when patches_per_epoch was passed as None).
        self.patches_per_epoch: Optional[int] = (
            int(patches_per_epoch) if patches_per_epoch is not None else None
        )
        self.jitter = (
            np.array(jitter_voxels, dtype=int)
            if jitter_voxels is not None
            else (self.output_size // 4)
        )
        self.seed = int(seed)
        self.dense_to_sparse_ratio = (
            float(dense_to_sparse_ratio)
            if dense_to_sparse_ratio is not None
            else None
        )
        self._effective_dense_ratio: float = 0.0  # set in _build_index

        # Input normalization to apply to every raw patch the dataset emits.
        # The dashboard's inference path normalizes raw via ``g.input_norms``
        # before feeding the model; the trainer (a separate LSF process)
        # has an empty ``g.input_norms``, so without this the trainer would
        # train on raw uint8 while inference sees normalized [-1, 1].
        # ``input_norm_config`` is the JSON-serializable dict from the YAML
        # (e.g. {"MinMaxNormalizer": {...}, "LambdaNormalizer": {...}}).
        self.input_norm_config: dict = dict(input_norm_config or {})
        self._input_normalizers = self._build_input_normalizers(self.input_norm_config)
        if not self._input_normalizers and self.input_norm_config:
            logger.warning(
                "input_norm_config provided but produced no normalizers; "
                "raw patches will be returned unnormalized."
            )
        if self._input_normalizers:
            logger.info(
                f"VirtualPatchDataset: applying {len(self._input_normalizers)} "
                f"input normalizer(s) per patch: "
                f"{[type(n).__name__ for n in self._input_normalizers]}"
            )
        else:
            logger.warning(
                "VirtualPatchDataset: no input normalizers configured. "
                "Raw patches will be returned in their native dtype/range. "
                "If inference normalizes to [-1, 1] (typical), the trained "
                "model will see different inputs at train vs inference time."
            )

        self.dataset_offset_nm: np.ndarray = np.zeros(3)
        self.volume_shape_voxels: np.ndarray = np.zeros(3, dtype=int)
        # Two-pool stratified sampling: dense FG voxels live inside any
        # imported_crops bbox; sparse FG voxels are everywhere else
        # (painted scribbles, by construction). Either may be empty.
        self._fg_index_dense: Optional[np.ndarray] = None
        self._fg_index_sparse: Optional[np.ndarray] = None
        self._volume_arr = None  # opened lazily after worker fork
        self._raw_idi = None     # opened lazily after worker fork
        # Cached per-worker RNG. None until first __getitem__ (after fork/spawn).
        # Without this cache, every __getitem__ would reseed and re-pick the
        # very first integer of the same stream — producing the same patch
        # forever and silently breaking training.
        self._cached_rng: Optional[np.random.Generator] = None

        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Walk the volume's populated chunks and build dense + sparse FG indices.

        We use the on-disk file layout (zarr v2 stores one file per chunk
        named ``z.y.x``) to enumerate just the chunks that have been
        written. Empty regions of the sparse volume produce no files and
        cost us nothing. FG voxels are then partitioned by membership in
        the volume's ``imported_crops`` bbox list.
        """
        s0_path = os.path.join(self.volume_zarr_path, "annotation", "s0")
        if not os.path.isdir(s0_path):
            raise ValueError(
                f"Volume zarr at {self.volume_zarr_path} has no annotation/s0/ "
                "directory; was it created?"
            )

        # Pull volume-level metadata once so we can map voxel coords to nm
        # and classify FG voxels as dense (inside an imported crop) or
        # sparse (outside).
        with open(os.path.join(self.volume_zarr_path, ".zattrs")) as f:
            root_attrs = json.load(f)
        self.dataset_offset_nm = np.array(
            root_attrs.get("dataset_offset_nm", [0, 0, 0]), dtype=float
        )
        imported = root_attrs.get("imported_crops", []) or []
        # Bbox list as two stacked (M, 3) arrays for vectorized membership
        # tests below. Empty when no YAML crops were imported.
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
        self.volume_shape_voxels = np.array(arr.shape, dtype=int)
        chunk_shape = np.array(arr.chunks, dtype=int)

        chunk_keys = [
            name for name in os.listdir(s0_path)
            if _CHUNK_KEY_RE.match(name)
        ]
        if not chunk_keys:
            raise ValueError(
                f"Volume zarr at {self.volume_zarr_path} has no populated chunks. "
                "Paint annotations or import crops first."
            )

        dense_rows: List[np.ndarray] = []
        sparse_rows: List[np.ndarray] = []
        n_fg_chunks = 0  # chunks that actually contributed FG voxels
        for key in chunk_keys:
            cz, cy, cx = (int(s) for s in key.split("."))
            chunk_origin = np.array([cz, cy, cx], dtype=np.int64) * chunk_shape
            chunk_data = arr.blocks[cz, cy, cx]
            fg_local = np.argwhere(chunk_data >= 2).astype(np.int64)
            if not fg_local.size:
                # On-disk file exists (zarr writes fill chunks during slab
                # writes) but contributes no FG; skip and don't count it.
                continue
            n_fg_chunks += 1
            fg_global = fg_local + chunk_origin
            if bbox_offsets.shape[0] == 0:
                # No imported crops → everything is sparse (painted).
                sparse_rows.append(fg_global)
                continue
            in_dense = _voxels_inside_any_bbox(fg_global, bbox_offsets, bbox_ends)
            if in_dense.any():
                dense_rows.append(fg_global[in_dense])
            if (~in_dense).any():
                sparse_rows.append(fg_global[~in_dense])

        self._fg_index_dense = (
            np.concatenate(dense_rows, axis=0) if dense_rows else np.zeros((0, 3), dtype=np.int64)
        )
        self._fg_index_sparse = (
            np.concatenate(sparse_rows, axis=0) if sparse_rows else np.zeros((0, 3), dtype=np.int64)
        )
        n_dense = int(self._fg_index_dense.shape[0])
        n_sparse = int(self._fg_index_sparse.shape[0])

        if n_dense == 0 and n_sparse == 0:
            raise ValueError(
                f"Volume zarr at {self.volume_zarr_path} has populated chunks "
                "but no foreground voxels (>=2). Did you only paint background?"
            )

        # Resolve dense ratio: explicit value wins, else auto-balance to
        # 0.5 when both pools have voxels, else fall back to whichever
        # pool is non-empty so we still draw patches.
        if self.dense_to_sparse_ratio is None:
            if n_dense > 0 and n_sparse > 0:
                self._effective_dense_ratio = 0.5
            elif n_dense > 0:
                self._effective_dense_ratio = 1.0
            else:
                self._effective_dense_ratio = 0.0
        else:
            ratio = max(0.0, min(1.0, self.dense_to_sparse_ratio))
            # Clamp away from a pool that's empty so __getitem__ never
            # tries to sample from an empty index.
            if n_dense == 0:
                self._effective_dense_ratio = 0.0
            elif n_sparse == 0:
                self._effective_dense_ratio = 1.0
            else:
                self._effective_dense_ratio = ratio

        # Default patches_per_epoch = number of FG-bearing chunks: each
        # such chunk gets ~1 patch per epoch on average. Cheap "auto cover
        # everything" mode the user can override via YAML or UI. We count
        # FG-bearing chunks (not all chunk files) because zarr writes
        # empty fill chunks during slab writes -- those don't represent
        # annotation work and shouldn't inflate epoch length.
        if self.patches_per_epoch is None:
            self.patches_per_epoch = max(1, n_fg_chunks)

        logger.info(
            f"VirtualPatchDataset: built FG index with {n_dense + n_sparse} voxels "
            f"(dense={n_dense}, sparse={n_sparse}) from {n_fg_chunks} FG-bearing "
            f"chunk(s) ({len(chunk_keys)} chunk files on disk) of "
            f"{self.volume_zarr_path}; "
            f"patches_per_epoch={self.patches_per_epoch}, "
            f"dense_ratio={self._effective_dense_ratio:.3f} "
            f"({'auto' if self.dense_to_sparse_ratio is None else 'explicit'}), "
            f"jitter={self.jitter.tolist()}"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Resolved by _build_index() in __init__.
        return int(self.patches_per_epoch or 0)

    def __getitem__(self, _idx: int):
        rng = self._worker_rng()
        # Pick a pool by the resolved dense ratio. Both indices may exist;
        # _build_index guarantees we never end up with the chosen pool empty.
        use_dense = (
            self._effective_dense_ratio >= 1.0
            or (self._effective_dense_ratio > 0.0 and rng.random() < self._effective_dense_ratio)
        )
        pool = self._fg_index_dense if use_dense else self._fg_index_sparse
        anchor_zyx = pool[rng.integers(0, pool.shape[0])].astype(np.float64)

        jitter_offset = rng.integers(
            low=-self.jitter, high=self.jitter + 1, size=3
        ).astype(np.float64)
        ann_center_voxels = anchor_zyx + jitter_offset

        # Convert annotation-space voxel center to physical (nm) for the raw read.
        ann_center_nm = (
            self.dataset_offset_nm + ann_center_voxels * self.output_voxel_size
        )

        ann_patch = self._read_annotation_patch(ann_center_voxels)
        raw_patch = self._read_raw_patch(ann_center_nm)

        raw_t = torch.from_numpy(raw_patch.astype(np.float32)[np.newaxis, ...])
        ann_t = torch.from_numpy(ann_patch.astype(np.float32)[np.newaxis, ...])
        return raw_t, ann_t

    # ------------------------------------------------------------------
    # Patch reads
    # ------------------------------------------------------------------

    def _open_volume(self):
        if self._volume_arr is None:
            self._volume_arr = zarr.open(
                os.path.join(self.volume_zarr_path, "annotation", "s0"), mode="r"
            )
        return self._volume_arr

    def _read_annotation_patch(self, center_voxels: np.ndarray) -> np.ndarray:
        """Crop a patch from the volume's annotation array.

        Out-of-bounds voxels are filled with 0 (= unannotated → masked
        out by the trainer's loss when ``mask_unannotated=True``).
        """
        out_size = self.output_size
        lo = (center_voxels - out_size / 2).astype(int)
        hi = lo + out_size

        clip_lo = np.maximum(lo, 0)
        clip_hi = np.minimum(hi, self.volume_shape_voxels)
        valid = np.all(clip_hi > clip_lo)

        patch = np.zeros(out_size, dtype=np.uint8)
        if valid:
            arr = self._open_volume()
            src_slices = tuple(slice(int(c), int(d)) for c, d in zip(clip_lo, clip_hi))
            dst_slices = tuple(
                slice(int(c - l), int(d - l))
                for c, d, l in zip(clip_lo, clip_hi, lo)
            )
            patch[dst_slices] = arr[src_slices]
        return patch

    def _read_raw_patch(self, center_nm: np.ndarray) -> np.ndarray:
        """Read an ``input_size`` patch from the raw dataset, centered at ``center_nm``.

        The raw read uses ``normalize=False`` because the trainer process's
        global ``g.input_norms`` is empty -- the dashboard's normalization
        config doesn't propagate across the LSF process boundary. We apply
        the dashboard's normalizers explicitly here from
        ``self._input_normalizers``, which is built from the manifest at
        construction time.
        """
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Coordinate, Roi

        if self._raw_idi is None:
            self._raw_idi = ImageDataInterface(
                self.raw_dataset_path,
                voxel_size=self.input_voxel_size,
                normalize=False,
            )
        idi = self._raw_idi
        read_shape_nm = self.input_size * self.input_voxel_size
        roi = Roi(
            offset=Coordinate(center_nm - read_shape_nm / 2),
            shape=Coordinate(read_shape_nm),
        )
        patch = idi.to_ndarray_ts(roi)

        # Apply the dashboard's normalizers locally (no global state).
        # Each normalizer is callable and returns an ndarray; the chain
        # mirrors what apply_norms() does inside the dashboard process.
        for norm in self._input_normalizers:
            patch = norm(patch)
        return patch

    @staticmethod
    def _build_input_normalizers(input_norm_config: dict) -> list:
        """Materialize the dict-form ``input_norm`` config into normalizer objects."""
        if not input_norm_config:
            return []
        try:
            from cellmap_flow.norm.input_normalize import get_normalizations

            return get_normalizations(input_norm_config)
        except Exception as e:
            logger.error(
                f"Failed to build input normalizers from config "
                f"{input_norm_config!r}: {e}. Patches will be unnormalized."
            )
            return []

    # ------------------------------------------------------------------
    # RNG plumbing
    # ------------------------------------------------------------------

    def _worker_rng(self) -> np.random.Generator:
        # Cache the Generator on self so consecutive __getitem__ calls draw
        # from the *advancing* state of the same RNG. Reseeding every call
        # made every patch identical (the first integer pulled from a freshly
        # seeded generator is deterministic).
        if self._cached_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            self._cached_rng = np.random.default_rng(
                self.seed + worker_id * 1_000_003
            )
        return self._cached_rng


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

VIRTUAL_MANIFEST_FILENAME = "_virtual_sources.json"


def write_manifest(corrections_dir: str, manifest: dict) -> str:
    """Persist a manifest sentinel that ``create_dataloader`` looks for."""
    os.makedirs(corrections_dir, exist_ok=True)
    path = os.path.join(corrections_dir, VIRTUAL_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(corrections_dir: str) -> Optional[dict]:
    """Return the manifest if present, else ``None``."""
    path = os.path.join(corrections_dir, VIRTUAL_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def dataset_from_manifest(manifest: dict) -> VirtualPatchDataset:
    """Instantiate a :class:`VirtualPatchDataset` from a manifest dict.

    Recognized manifest kinds:
      - ``volume_zarr_v1`` (current): trainer reads the session's
        annotation_volume.zarr directly. Field ``volume_zarr_path``.
    """
    kind = manifest.get("kind")
    if kind != "volume_zarr_v1":
        raise ValueError(
            f"Unsupported manifest kind: {kind!r}. Expected 'volume_zarr_v1'."
        )
    return VirtualPatchDataset(
        volume_zarr_path=manifest["volume_zarr_path"],
        raw_dataset_path=manifest["raw_dataset_path"],
        input_size_voxels=tuple(manifest["input_size_voxels"]),
        output_size_voxels=tuple(manifest["output_size_voxels"]),
        input_voxel_size_nm=tuple(manifest["input_voxel_size_nm"]),
        output_voxel_size_nm=tuple(manifest["output_voxel_size_nm"]),
        # None defaults to "cover all populated chunks" inside the dataset.
        patches_per_epoch=manifest.get("patches_per_epoch"),
        jitter_voxels=tuple(manifest["jitter_voxels"]) if manifest.get("jitter_voxels") else None,
        seed=manifest.get("seed", 0),
        input_norm_config=manifest.get("input_norm") or None,
        dense_to_sparse_ratio=manifest.get("dense_to_sparse_ratio"),
    )
