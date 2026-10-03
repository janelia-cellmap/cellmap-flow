"""The finetune training dataset: random patch pairs out of a session's annotation volume.

There is exactly one source of truth per session: an
``annotation_volume.zarr`` (sparse, full-dataset extent, OME-NGFF) that
holds **every** annotation -- painted scribbles plus any imported YAML
crops, all merged at their physical offsets. Patches are read straight out
of it: no per-tile materialization, no parallel source list to keep in sync.

A patch is a centre drawn from the pools (sampler.py: dense crops, sparse
scribbles, rehearsal regions), the annotation and raw read around it
(reader.py), and, when asked for, augmentation (augment.py).

Contract with the trainer:

- ``len(self)`` is ``patches_per_epoch``, the epoch length; it has no fixed
  relation to the number of populated chunks.
- An item is ``(raw, annotation)``, float32 tensors of shape
  ``(1, Z, Y, X)``, and a third, the anchor mask, once the session has a
  usable good region (``emits_anchor``).
- Loader workers are spawned, and each gets the dataset as a pickle, by its
  class path. The pools go with it; the arrays are opened in the worker.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cellmap_flow.finetune.data.augment import Augmentation
from cellmap_flow.finetune.data.reader import PatchReader
from cellmap_flow.finetune.data.sampler import PatchSampler

logger = logging.getLogger(__name__)


class VirtualPatchDataset(Dataset):
    """Yield random raw+annotation patches anchored on annotated voxels in a volume zarr.

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
            substitute the number of annotated chunks" -- every annotated
            chunk gets ~one patch per epoch on average.
        patches_per_chunk: how many patches the auto epoch length counts
            per annotated chunk (default 1). A patch much smaller than the
            chunks (a model trained on tiles, see
            ``loader.dataset_from_manifest``'s ``patch_voxels``) takes more
            to cover them.
        jitter_voxels: half-range of the random offset applied to the patch
            center, in **annotation voxels**. Defaults to
            ``output_size_voxels // 4``.
        seed: RNG seed; per-worker offset added so multi-worker dataloaders
            sample distinct streams.
        input_norm_config: the session's input normalization, applied to
            every raw patch (see reader.input_normalizers).
        dense_to_sparse_ratio: fraction in [0, 1] of patches drawn from
            the dense pool (FG voxels inside any imported_crops bbox).
            ``None`` (default) means auto: 0.5 if both pools have voxels,
            else 1.0 (use the non-empty pool exclusively).
        good_regions: the session's good regions, ``{"offset_nm",
            "shape_nm"}`` boxes to hold the model to its teacher in.
        rehearsal_fraction: the share of patches centred on a good region.
            ``None`` (default) means a quarter, when any region is usable.
        anchor_fraction: the share of the other patches centred at a random
            point of the volume and held to the teacher there. ``None``
            (default) means a quarter for a painted session, none otherwise.
        augment: flips, XY rotations, brightness and noise (augment.py).
        resample: read the raw resampled to ``input_voxel_size_nm`` when it
            has no level at it (the volume's ``resample``).
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
        patches_per_chunk: int = 1,
        jitter_voxels: Optional[Tuple[int, int, int]] = None,
        seed: int = 0,
        input_norm_config: Optional[dict] = None,
        dense_to_sparse_ratio: Optional[float] = None,
        good_regions: Optional[list] = None,
        rehearsal_fraction: Optional[float] = None,
        anchor_fraction: Optional[float] = None,
        augment: bool = False,
        resample: bool = False,
    ):
        self.volume_zarr_path = volume_zarr_path
        self.raw_dataset_path = raw_dataset_path
        self.augment = bool(augment)
        self.seed = int(seed)
        output_size = np.array(output_size_voxels, dtype=int)
        jitter = np.array(jitter_voxels, dtype=int) if jitter_voxels is not None else output_size // 4

        self.sampler = PatchSampler(
            volume_zarr_path,
            output_voxel_size_nm,
            jitter,
            dense_to_sparse_ratio=dense_to_sparse_ratio,
            good_regions=good_regions,
            rehearsal_fraction=rehearsal_fraction,
            anchor_fraction=anchor_fraction,
        )
        self.reader = PatchReader(
            volume_zarr_path,
            raw_dataset_path,
            input_size_voxels,
            output_size_voxels,
            input_voxel_size_nm,
            output_voxel_size_nm,
            corner_nm=self.sampler.corner_nm,
            shape_voxels=self.sampler.shape_voxels,
            input_norm_config=input_norm_config,
            resample=resample,
        )
        self.augmentation = Augmentation(self.reader.normalizers)

        # Default: one patch per annotated chunk, so each gets ~1 patch per
        # epoch on average -- a cheap "cover everything" the user can
        # override from the YAML or the dashboard. Chunks, not chunk files:
        # zarr writes empty fill chunks during slab writes, and those are no
        # annotation work.
        self.patches_per_epoch: int = (
            int(patches_per_epoch)
            if patches_per_epoch is not None
            else max(1, self.sampler.annotated_chunks * max(1, int(patches_per_chunk)))
        )
        # Cached per-worker RNG, None until the first __getitem__ (in the
        # worker, after spawn). Without the cache every __getitem__ would
        # reseed and re-pick the very first integer of the same stream --
        # the same patch forever, silently breaking training.
        self._cached_rng: Optional[np.random.Generator] = None

        sampler = self.sampler
        n_dense, n_sparse = int(sampler.dense.shape[0]), int(sampler.sparse.shape[0])
        n_sparse_fg = int(sampler.sparse_foreground_rows.size)
        logger.info(
            f"VirtualPatchDataset: patch centres from {sampler.annotated_chunks} annotated "
            f"chunk(s) ({sampler.chunk_files} chunk files on disk) of {self.volume_zarr_path}: "
            f"{n_sparse} painted voxels ({n_sparse_fg} foreground, {n_sparse - n_sparse_fg} "
            f"background), {n_dense} foreground voxels in imported crops; "
            f"patches_per_epoch={self.patches_per_epoch}, "
            f"dense_ratio={sampler.effective_dense_ratio:.3f} "
            f"({'auto' if sampler.dense_to_sparse_ratio is None else 'explicit'}), "
            f"jitter={sampler.jitter.tolist()}"
        )
        sampler.log_rehearsal_status()

    def __len__(self) -> int:
        return self.patches_per_epoch

    @property
    def emits_anchor(self) -> bool:
        """Whether __getitem__ yields the 3-tuple (raw, annotation, anchor).

        Only once there is something to anchor on: good regions, or random
        anchors (a painted session's default). Otherwise the dataset keeps
        its original 2-tuple contract.
        """
        return self.sampler.effective_rehearsal_fraction > 0.0 or self.sampler.effective_anchor_fraction > 0.0

    def __getitem__(self, _idx: int):
        rng = self._worker_rng()
        centre, is_rehearsal = self.sampler.draw(rng)
        centre = self.reader.snap(centre)

        ann_patch = self.reader.annotation(centre)
        raw_patch = self.reader.raw(centre)
        if self.augment:
            raw_patch = self.augmentation.intensity(raw_patch, rng)
        raw_patch = self.reader.normalize(raw_patch)
        if self.augment:
            # Before the tensors are built, and before `anchor` is derived
            # from ann_patch below, so the anchor mask inherits the same
            # transform rather than needing its own.
            raw_patch, ann_patch = self.augmentation.spatial(raw_patch, ann_patch, rng)
            self.augmentation.report()

        raw_t = torch.from_numpy(raw_patch.astype(np.float32)[np.newaxis, ...])
        ann_t = torch.from_numpy(ann_patch.astype(np.float32)[np.newaxis, ...])

        if not self.emits_anchor:
            return raw_t, ann_t

        # Per-voxel anchor mask, the third thing the loss needs to know.
        #
        #   annotated voxel          -> supervised loss, anchor 0
        #   unannotated in a good    -> anchor 1: hold the student to the
        #     region or a random        teacher here
        #     anchor patch
        #   unannotated anywhere     -> anchor 0, no loss at all: you did not
        #     else                      say it was right, only that you had
        #                               not got to it
        #
        # A scribble inside a good region therefore wins over the anchor,
        # which is what keeps a marked region correctable: notice a mistake
        # inside one, paint over it, and the paint takes precedence.
        if is_rehearsal:
            anchor = (ann_patch == 0).astype(np.float32)
        else:
            anchor = np.zeros_like(ann_patch, dtype=np.float32)
        anchor_t = torch.from_numpy(anchor[np.newaxis, ...])
        return raw_t, ann_t, anchor_t

    def _worker_rng(self) -> np.random.Generator:
        """This worker's generator: one stream per worker, advancing across __getitem__ calls."""
        if self._cached_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            self._cached_rng = np.random.default_rng(
                self.seed + worker_id * 1_000_003
            )
        return self._cached_rng
