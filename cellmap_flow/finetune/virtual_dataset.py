"""
On-the-fly random-patch dataset for finetuning.

The dashboard's painted-volume / YAML-crop pipelines historically materialize
fixed-size annotation tiles to disk as ``_chunk_*.zarr`` entries. That works
but explodes disk usage on large dense crops (e.g. 600**3 → thousands of
chunks even after BG-only tile sub-sampling) and re-tiles whenever the user
wants to change patch shape, sampling ratio, etc.

This module is the lazy alternative. It holds **references** to source
annotation zarrs, builds a flat index of every foreground voxel once at
construction time, and at training time samples one annotated voxel per
``__getitem__`` call, reads a raw + annotation patch around it, and returns
them. No tiles on disk, every epoch sees fresh patch positions, and adding /
removing source crops is a manifest edit.

Sampling rule
-------------
"One sampler, one rule": every patch is centered on a uniformly-chosen
foreground voxel with a small random jitter. With ``mask_unannotated=True``
in the trainer, the patch's unannotated voxels (sparse mode) or out-of-ROI
voxels (near a dense crop's edge) are masked out and contribute zero gradient,
so we don't need to guarantee 100% coverage of the patch with annotation.

This deliberately omits more elaborate strategies (per-component
stratification, FG/BG mixing weights, per-source weights). Add them only if
the simple rule proves insufficient.

Design notes for reviewers
--------------------------
- Each source's full annotation array is read **once** at construction time
  to build the FG-voxel index. Memory cost is roughly the size of the
  annotation arrays, freed after indexing. Raw is **never** read at init —
  only the small patch we sample at ``__getitem__`` time.
- The class is process-fork-safe in the sense that workers re-open zarr
  handles lazily (we don't hold open tensorstore handles across pickling).
  See ``_open_raw`` and ``_get_annotation_array``.
- ``len(dataset)`` is set to ``patches_per_epoch``; it has no relationship
  to the number of source crops or to disk content. The trainer treats this
  as the epoch length.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from cellmap_flow.finetune.crop_loader import (
    CropEntry,
    _open_array,
    _read_voxel_size_and_offset,
    remap_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceSpec:
    """Per-source configuration: how to interpret the annotation values."""

    path: str
    fg_ids: Optional[List[int]] = None
    bg_ids: List[int] = field(default_factory=list)
    mode: str = "dense"
    connected_components: bool = False
    name: Optional[str] = None

    @classmethod
    def from_crop_entry(cls, entry: CropEntry) -> "SourceSpec":
        return cls(
            path=entry.path,
            fg_ids=entry.fg_ids,
            bg_ids=list(entry.bg_ids),
            mode=entry.mode,
            connected_components=entry.connected_components,
            name=entry.name,
        )


class VirtualPatchDataset(Dataset):
    """Yield random raw+annotation patches anchored on annotated voxels.

    Args:
        sources: list of :class:`SourceSpec` describing each annotation zarr.
        raw_dataset_path: path to the raw EM zarr the sources are aligned to.
        input_size_voxels: shape (Z, Y, X) of the raw patch returned per
            sample, in voxels at ``input_voxel_size``.
        output_size_voxels: shape (Z, Y, X) of the annotation patch returned
            per sample, in voxels at ``output_voxel_size``.
        input_voxel_size_nm: voxel size for raw patches; should match the
            scale snapped from the raw dataset.
        output_voxel_size_nm: voxel size for annotation patches; should match
            the scale snapped from the raw dataset.
        patches_per_epoch: ``len(self)``. Controls how many random patches
            comprise one epoch. Defaults to 500.
        jitter_voxels: half-range of the random offset applied to the patch
            center, in **annotation voxels**. ``None`` -> ``output_size//4``,
            which keeps the anchor FG voxel comfortably inside the patch.
        seed: RNG seed for reproducibility (per-worker offset added in
            ``__init__`` of dataloader workers).
    """

    def __init__(
        self,
        sources: List[SourceSpec],
        raw_dataset_path: str,
        input_size_voxels: Tuple[int, int, int],
        output_size_voxels: Tuple[int, int, int],
        input_voxel_size_nm: Tuple[float, float, float],
        output_voxel_size_nm: Tuple[float, float, float],
        patches_per_epoch: int = 500,
        jitter_voxels: Optional[Tuple[int, int, int]] = None,
        seed: int = 0,
    ):
        if not sources:
            raise ValueError("VirtualPatchDataset requires at least one source")

        self.sources = sources
        self.raw_dataset_path = raw_dataset_path
        self.input_size = np.array(input_size_voxels, dtype=int)
        self.output_size = np.array(output_size_voxels, dtype=int)
        self.input_voxel_size = np.array(input_voxel_size_nm, dtype=float)
        self.output_voxel_size = np.array(output_voxel_size_nm, dtype=float)
        self.patches_per_epoch = int(patches_per_epoch)
        self.jitter = (
            np.array(jitter_voxels, dtype=int)
            if jitter_voxels is not None
            else (self.output_size // 4)
        )
        self.seed = int(seed)

        # Per-source bookkeeping. ``_per_source`` parallels ``self.sources``.
        # Each entry holds the source's voxel-size, nm offset, remapped
        # annotation array (so we don't re-read it every __getitem__), and
        # the list of FG voxel coordinates (in annotation voxels).
        self._per_source: List[dict] = []
        self._fg_index: Optional[np.ndarray] = None  # (N, 4): src_idx, z, y, x

        self._build_index()

        # Worker-local cached IDIs for raw reads. Keyed only after the worker
        # forks/spawns; never serialized.
        self._raw_idi = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        rows: List[np.ndarray] = []
        for src_idx, spec in enumerate(self.sources):
            sub, voxel_size_nm, offset_nm = _read_voxel_size_and_offset(spec.path)
            arr = _open_array(spec.path, sub)
            data = arr[:]
            if data.ndim != 3:
                raise ValueError(
                    f"Source {spec.path}: expected 3D (z, y, x), got shape {data.shape}"
                )

            remapped = remap_labels(
                data,
                fg_ids=spec.fg_ids,
                bg_ids=spec.bg_ids,
                mode=spec.mode,
                connected_components=spec.connected_components,
            )

            fg_coords = np.argwhere(remapped >= 2).astype(np.int64)
            if fg_coords.size == 0:
                logger.warning(
                    f"Source {spec.path}: no foreground voxels after remap; "
                    "this source will never be sampled."
                )

            self._per_source.append(
                {
                    "voxel_size_nm": voxel_size_nm,
                    "offset_nm": offset_nm,
                    "remapped": remapped,
                    "shape": np.array(remapped.shape, dtype=int),
                    "n_fg": int(fg_coords.shape[0]),
                }
            )

            if fg_coords.shape[0] > 0:
                src_col = np.full((fg_coords.shape[0], 1), src_idx, dtype=np.int64)
                rows.append(np.concatenate([src_col, fg_coords], axis=1))

        if not rows:
            raise ValueError(
                "No foreground voxels found across any source. "
                "Check fg_ids and that the source zarrs contain nonzero labels."
            )
        self._fg_index = np.concatenate(rows, axis=0)
        logger.info(
            f"VirtualPatchDataset: built FG index with {self._fg_index.shape[0]} "
            f"voxels across {len(self.sources)} source(s); "
            f"patches_per_epoch={self.patches_per_epoch}, jitter={self.jitter.tolist()}"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, _idx: int):
        # Each worker gets a deterministic but distinct stream so that two
        # workers don't pull the same anchors. ``_idx`` itself is not used
        # because the conceptual mapping idx -> patch is meaningless here.
        rng = self._worker_rng()
        anchor = self._fg_index[rng.integers(0, self._fg_index.shape[0])]
        src_idx = int(anchor[0])
        anchor_zyx = anchor[1:4].astype(np.float64)

        # Jitter the patch center. Annotation patch is centered at the
        # jittered anchor; raw patch is centered on the same physical point
        # but covers a larger physical region (input_size > output_size for
        # this U-Net family because of context).
        jitter_offset = rng.integers(
            low=-self.jitter, high=self.jitter + 1, size=3
        ).astype(np.float64)
        ann_center_voxels = anchor_zyx + jitter_offset

        # Convert to physical (nm) coordinates for the raw read.
        per = self._per_source[src_idx]
        voxel_size_nm = per["voxel_size_nm"]
        ann_offset_nm = per["offset_nm"]
        ann_center_nm = ann_offset_nm + ann_center_voxels * voxel_size_nm

        ann_patch = self._read_annotation_patch(src_idx, ann_center_voxels)
        raw_patch = self._read_raw_patch(ann_center_nm)

        # Same channel/dtype convention as CorrectionDataset.
        raw_t = torch.from_numpy(raw_patch.astype(np.float32)[np.newaxis, ...])
        ann_t = torch.from_numpy(ann_patch.astype(np.float32)[np.newaxis, ...])
        return raw_t, ann_t

    # ------------------------------------------------------------------
    # Patch reads
    # ------------------------------------------------------------------

    def _read_annotation_patch(
        self, src_idx: int, center_voxels: np.ndarray
    ) -> np.ndarray:
        """Crop a patch from the (already-remapped) annotation array.

        Out-of-bounds voxels are filled with 0 (= unannotated -> masked out
        by the trainer's loss when ``mask_unannotated=True``).
        """
        per = self._per_source[src_idx]
        full = per["remapped"]
        shape = per["shape"]
        out_size = self.output_size

        # Patch bounds in voxel coordinates of the source array.
        lo = (center_voxels - out_size / 2).astype(int)
        hi = lo + out_size

        # Clamp to the valid region of the annotation array; pad the rest.
        clip_lo = np.maximum(lo, 0)
        clip_hi = np.minimum(hi, shape)
        valid = np.all(clip_hi > clip_lo)

        patch = np.zeros(out_size, dtype=full.dtype)
        if valid:
            src_slices = tuple(slice(int(c), int(d)) for c, d in zip(clip_lo, clip_hi))
            dst_slices = tuple(
                slice(int(c - l), int(d - l))
                for c, d, l in zip(clip_lo, clip_hi, lo)
            )
            patch[dst_slices] = full[src_slices]
        return patch

    def _read_raw_patch(self, center_nm: np.ndarray) -> np.ndarray:
        """Read an ``input_size`` patch from the raw dataset, centered at ``center_nm``."""
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Coordinate, Roi

        if self._raw_idi is None:
            self._raw_idi = ImageDataInterface(
                self.raw_dataset_path, voxel_size=self.input_voxel_size
            )
        idi = self._raw_idi
        read_shape_nm = self.input_size * self.input_voxel_size
        roi = Roi(
            offset=Coordinate(center_nm - read_shape_nm / 2),
            shape=Coordinate(read_shape_nm),
        )
        return idi.to_ndarray_ts(roi)

    # ------------------------------------------------------------------
    # RNG plumbing
    # ------------------------------------------------------------------

    def _worker_rng(self) -> np.random.Generator:
        """Per-worker RNG so multi-worker dataloaders don't sample identically."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        return np.random.default_rng(self.seed + worker_id * 1_000_003)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

VIRTUAL_MANIFEST_FILENAME = "_virtual_sources.json"


def write_manifest(corrections_dir: str, manifest: dict) -> str:
    """Persist a manifest that ``create_dataloader`` will pick up.

    Storing it as a sentinel file inside ``corrections_dir`` keeps the change
    surface small: the existing trainer entry point still receives
    ``--corrections <dir>`` and switches dataset class based on whether this
    file exists. Returns the manifest path.
    """
    import json

    os.makedirs(corrections_dir, exist_ok=True)
    path = os.path.join(corrections_dir, VIRTUAL_MANIFEST_FILENAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(corrections_dir: str) -> Optional[dict]:
    """Return the manifest if present, else ``None``."""
    import json

    path = os.path.join(corrections_dir, VIRTUAL_MANIFEST_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def dataset_from_manifest(manifest: dict) -> VirtualPatchDataset:
    """Instantiate a :class:`VirtualPatchDataset` from a manifest dict."""
    sources = [SourceSpec(**src) for src in manifest["sources"]]
    return VirtualPatchDataset(
        sources=sources,
        raw_dataset_path=manifest["raw_dataset_path"],
        input_size_voxels=tuple(manifest["input_size_voxels"]),
        output_size_voxels=tuple(manifest["output_size_voxels"]),
        input_voxel_size_nm=tuple(manifest["input_voxel_size_nm"]),
        output_voxel_size_nm=tuple(manifest["output_voxel_size_nm"]),
        patches_per_epoch=manifest.get("patches_per_epoch", 500),
        jitter_voxels=tuple(manifest["jitter_voxels"]) if manifest.get("jitter_voxels") else None,
        seed=manifest.get("seed", 0),
    )
