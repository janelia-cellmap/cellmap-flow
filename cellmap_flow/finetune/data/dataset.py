"""The finetune training dataset: random patch pairs out of a session's annotation volume.

There is exactly one source of truth per session: an
``annotation_volume.zarr`` (sparse, full-dataset extent, OME-NGFF) that
holds **every** annotation -- painted scribbles plus any imported YAML
crops, all merged at their physical offsets. Patches are read straight out
of it: no per-tile materialization, no parallel source list to keep in sync.

A patch is a centre drawn from the pools (sampler.py: dense crops, sparse
scribbles, rehearsal regions), the annotation and raw read around it
(reader.py), and, when asked for, augmentation (augment.py).

A one-slice patch, a 2D network's, is cut in the plane its paint was drawn
in, x–y, z–y or z–x, where the voxels are cubes (``slice_planes``): there a
side slice looks as an x–y one does. Elsewhere it is always an x–y slice,
and paint drawn in a side plane, of which it sees a line, is warned about.

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
from cellmap_flow.finetune.data.sampler import PLANE_NAMES, PatchSampler

logger = logging.getLogger(__name__)


def _cubes(voxel_size) -> bool:
    """Whether voxels of ``voxel_size`` (nm) are as long along every axis."""
    voxel_size = np.asarray(voxel_size, dtype=float)
    return bool(np.allclose(voxel_size, voxel_size[0], rtol=1e-3))


def stored_voxel_size(raw_dataset_path: str, voxel_size, resample: bool) -> Optional[tuple]:
    """The voxel size (nm) of the raw level patches are read from at
    ``voxel_size``, as stored, before any resampling to it; None when its
    metadata cannot be read here.

    The level is the one PatchReader's ImageDataInterface opens. Resampled
    to cubes, voxels stored three times as deep are still blurred across z,
    so a side slice through them is not an x–y slice.
    """
    from cellmap_flow.io import multiscale, paths
    from cellmap_flow.io.metadata import read_array_meta

    try:
        path = paths.normalize_path(raw_dataset_path)
        level_path, _ = multiscale.select_dataset(path, voxel_size, mode="resample" if resample else "floor")
        return tuple(float(v) for v in read_array_meta(level_path).spatial().voxel_size)
    except Exception as e:
        logger.debug(f"Could not read the voxel size of {raw_dataset_path}: {e}")
        return None


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
            center, in **annotation voxels**, in the patch's order (for a
            slice cut across y or x, its first entry is across the slice).
            Defaults to ``output_size_voxels // 4``.
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

        # A 2D network's patch, one slice: cut in any plane where every voxel
        # it is read from is a cube, else always in x–y (sampler.py).
        one_slice = int(input_size_voxels[0]) == 1 and int(output_size_voxels[0]) == 1
        self.not_cubes = {}
        self.slice_planes = None
        if one_slice:
            voxel_sizes = {
                "the raw as stored": stored_voxel_size(raw_dataset_path, input_voxel_size_nm, resample),
                "the model's input": tuple(input_voxel_size_nm),
                "the model's output": tuple(output_voxel_size_nm),
            }
            self.not_cubes = {name: v for name, v in voxel_sizes.items() if v is not None and not _cubes(v)}
            self.slice_planes = "xy" if self.not_cubes else "any"
        self.sampler = PatchSampler(
            volume_zarr_path,
            output_voxel_size_nm,
            jitter,
            dense_to_sparse_ratio=dense_to_sparse_ratio,
            good_regions=good_regions,
            rehearsal_fraction=rehearsal_fraction,
            anchor_fraction=anchor_fraction,
            slice_planes=self.slice_planes,
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
        self._log_planes()
        sampler.log_rehearsal_status()

    def _log_planes(self) -> None:
        """Say which planes a one-slice patch is cut in, or warn that paint in
        a side plane is seen as a line."""
        counts = self.sampler.paint_planes
        if counts is None or not counts.sum():
            return
        shares = ", ".join(f"{100 * c / counts.sum():.0f}% {name}" for name, c in zip(PLANE_NAMES, counts))
        if self.slice_planes == "any":
            logger.info(
                f"VirtualPatchDataset: one-slice patches are cut in the plane their paint lies in "
                f"({shares} of the painted voxels), and random anchors in the same proportions; "
                "imported crops and good regions in x–y"
            )
            return
        side = int(counts[1:].sum())
        if side:
            name, voxel_size = next(iter(self.not_cubes.items()))
            logger.warning(
                f"VirtualPatchDataset: {side} of {int(counts.sum())} painted voxels ({shares}) lie in "
                "a z–x or z–y slice, one voxel thick across y or x. The network trains on x–y "
                "slices, which see that paint as a line a voxel wide and learn next to nothing "
                "from it (Cellpose learns flows along one axis, and segments strips). Side "
                "slices are trained on only where the voxels are cubes, and "
                f"{name} has voxels of {' x '.join(f'{v:g}' for v in voxel_size)} nm. "
                "Paint in x–y slices."
            )

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
        centre, is_rehearsal, depth_axis = self.sampler.draw(rng)
        centre = self.reader.snap(centre, depth_axis)

        # Turned so that the slice is first, whichever axis it is across.
        ann_patch = self.reader.annotation(centre, depth_axis)
        raw_patch = self.reader.raw(centre, depth_axis)
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
