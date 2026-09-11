"""SKOOTS-target training dataset for LoRA finetuning.

Precomputes SKOOTS targets (semantic/skeleton/vector, or -- with
`target_mode="skeleton_distance"` -- semantic/skeleton/distance-from-skeleton,
or -- with `target_mode="skeleton_semantic"` -- just semantic/skeleton, no
instance-splitting channel at all)
once per annotated crop at construction time -- not per patch, not per
epoch -- because `instance_skeletons`/`bake_vector_targets` need each
instance's *full* extent to skeletonize and rank nearest-skeleton-point
correctly. Running them on a small randomly-sampled patch would truncate
instances at the patch boundary and produce wrong skeletons/vectors near
every crop edge. Precomputation is expensive (~4 min for a 600x600x900/
101-instance crop), so results are cached to `cache_dir` and reused across
dataset restarts and training runs; the yaml/crop-loader path stays
identical to the rest of the finetuning pipeline (`crop_loader.remap_labels`
+ `_open_array`).

`__getitem__` returns `(raw, packed)` -- a 2-tuple, matching the strict
`for raw, target in dataloader` unpack in `LoRAFinetuner._train_epoch`.
`packed` concatenates the (C, Z, Y, X) target and (C, Z, Y, X) mask along
the channel axis into one (2C, Z, Y, X) tensor (C=5 for `target_mode="skoots"`,
C=3 for `"skeleton_distance"`); `SkootsTargetTransform`/
`SkeletonDistanceTargetTransform` (see target_transforms.py) are the inverse
of that packing on the training side. This avoids changing the shared
training loop's batch contract for every other output type.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cellmap_flow.finetune.crop_loader import (
    CropEntry,
    _open_array,
    _read_voxel_size_and_offset,
    parse_crops_yaml,
    remap_labels,
)
from cellmap_flow.finetune.skoots_targets import (
    build_skeleton_distance_targets,
    build_skeleton_semantic_targets,
    build_skoots_targets,
    stack_skeleton_distance_target_and_mask,
    stack_skeleton_semantic_target_and_mask,
    stack_targets_and_mask,
)

logger = logging.getLogger(__name__)


@dataclass
class _CropCache:
    name: str
    labels: np.ndarray  # skoots convention: 0=bg, 1..=instance id, shape (Z,Y,X)
    target: np.ndarray  # (C, Z, Y, X) float32: see SkootsDataset.target_mode
    # (C=5: semantic, skeleton, vec_z, vec_y, vec_x -- target_mode="skoots";
    #  C=3: semantic, skeleton, distance -- target_mode="skeleton_distance")
    # No stored `mask` array: every crop here is dense (mode="dense" in the
    # yaml), so `stack_targets_and_mask`'s mask is provably all-ones over the
    # whole crop -- persisting and loading a full-size float32 copy of that
    # (literally as large as `target` itself) would double cache disk usage
    # and load time for zero information. `__getitem__` synthesizes a
    # patch-sized all-ones mask instead.
    voxel_size_nm: np.ndarray  # (3,) float
    offset_nm: np.ndarray  # (3,) float
    fg_coords: np.ndarray  # (N, 3) int64, this crop's own voxel frame
    shape: np.ndarray  # (3,) int64


def _decimation_factor(voxel_size_nm: np.ndarray, target_voxel_size_nm: np.ndarray) -> np.ndarray:
    factor = np.round(target_voxel_size_nm / voxel_size_nm).astype(np.int64)
    if np.any(factor < 1) or np.any(np.abs(factor * voxel_size_nm - target_voxel_size_nm) > 1e-6):
        raise ValueError(
            f"target_voxel_size_nm {tuple(target_voxel_size_nm)} is not an integer multiple "
            f"of the crop's native voxel size {tuple(voxel_size_nm)}"
        )
    return factor


def _decimate_labels(
    labels: np.ndarray, voxel_size_nm: np.ndarray, target_voxel_size_nm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor-decimate an instance label volume to a coarser voxel size.

    Strided subsampling (not averaging/block-majority): instance ids are
    categorical, so blending them across a block is meaningless, and at a
    small integer factor (typically 2x) picking every Nth voxel is exactly
    what "the same volume, sampled on a coarser grid" means -- it doesn't
    shift the origin, unlike block-reduction schemes that report the block
    center.
    """
    factor = _decimation_factor(voxel_size_nm, target_voxel_size_nm)
    decimated = labels[:: factor[0], :: factor[1], :: factor[2]]
    return decimated, voxel_size_nm * factor


def _decimate_target_array(
    target: np.ndarray, factor: np.ndarray, continuous_channels: Tuple[int, ...]
) -> np.ndarray:
    """Strided-decimate a native-resolution (C, Z, Y, X) target to match
    `_decimate_labels`'s coarser grid.

    Binary/probability channels (semantic, skeleton) need only resampling.
    Physical-displacement channels (vectors, or scalar distance-from-skeleton)
    also need rescaling: a target built by skeletonizing at native resolution
    reports distances in *native* voxel units, but once the grid is
    downsampled by `factor` each output voxel is `factor` times larger, so a
    displacement of N native voxels must become N/factor coarse voxels to
    stay correctly calibrated for a model trained/served at the coarser
    resolution. Isotropic-factor assumption throughout (matches
    `_decimate_labels` and every crop this dataset has been used with).
    """
    decimated = target[:, :: factor[0], :: factor[1], :: factor[2]].copy()
    if continuous_channels:
        scale = float(factor[0])
        for c in continuous_channels:
            decimated[c] /= scale
    return decimated


def _crop_cache_paths(cache_dir: str, name: str) -> Dict[str, str]:
    d = os.path.join(cache_dir, name)
    return {
        "dir": d,
        "labels": os.path.join(d, "labels.npy"),
        "target": os.path.join(d, "target.npy"),
        "fg_coords": os.path.join(d, "fg_coords.npy"),
        "meta": os.path.join(d, "meta.npz"),
    }


class SkootsDataset(Dataset):
    """Randomly sampled SKOOTS training patches from a set of dense,
    instance-labeled crops (e.g. a `crop_loader.CropsConfig` YAML like
    jrc_axolotl-heart-1/mito_group.yaml).

    All crops are loaded and their SKOOTS targets built (or loaded from
    `cache_dir`) once, in full, at construction time -- see module
    docstring for why. Everything after that is in-memory slicing plus a
    per-patch raw EM read.
    """

    def __init__(
        self,
        crops_yaml_path: str,
        raw_dataset_path: str,
        output_size_voxels: Tuple[int, int, int],
        raw_voxel_size_nm: Optional[Tuple[float, float, float]] = None,
        target_voxel_size_nm: Optional[Tuple[float, float, float]] = None,
        context_voxels: Tuple[int, int, int] = (0, 0, 0),
        cache_dir: Optional[str] = None,
        patches_per_epoch: int = 1000,
        seed: int = 0,
        skeleton_radius: int = 2,
        min_branch_length: float = 3.0,
        target_mode: Literal["skoots", "skeleton_distance", "skeleton_semantic"] = "skoots",
    ) -> None:
        self.raw_dataset_path = raw_dataset_path
        self.output_size = np.array(output_size_voxels, dtype=np.int64)
        # None means "trust the raw dataset's own multiscale/attrs metadata"
        # (passed straight through to ImageDataInterface, which only
        # overrides its metadata-derived voxel_size when this is not None).
        self.raw_voxel_size_nm = (
            np.array(raw_voxel_size_nm, dtype=np.float64) if raw_voxel_size_nm is not None else None
        )
        # Some pretrained checkpoints (e.g. a valid-padding UNet exported at a
        # fixed tile size) were trained at a coarser voxel size than these
        # label crops' own native resolution -- feeding them native-resolution
        # patches would be shape-*and* physically-scale wrong (a conv net's
        # receptive field is calibrated in nm, not voxels). When set, crop
        # labels are nearest-neighbor-decimated to this resolution before
        # SKOOTS targets are ever built, so skeletons/vectors are computed
        # natively at the resolution the model will actually see, not
        # generated at native resolution and then degraded after the fact
        # (which would require averaging vector directions across voxels --
        # not meaningful).
        self.target_voxel_size_nm = (
            np.array(target_voxel_size_nm, dtype=np.float64) if target_voxel_size_nm is not None else None
        )
        # Extra halo added per side when reading the raw patch, beyond the
        # target/label/mask window -- for valid-padding architectures whose
        # output tile is smaller than their input tile (e.g. 56-voxel output
        # needs a 178-voxel input, a 61-voxel halo on each side). The
        # target/label window itself is never widened by this; only the raw
        # EM read is, since context comes from surrounding (possibly
        # unannotated) raw imagery.
        self.context_voxels = np.array(context_voxels, dtype=np.int64)
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(crops_yaml_path)), "skoots_cache")
        self.cache_dir = cache_dir
        self.patches_per_epoch = int(patches_per_epoch)
        self.seed = int(seed)
        self.skeleton_radius = int(skeleton_radius)
        self.min_branch_length = float(min_branch_length)
        if target_mode not in ("skoots", "skeleton_distance", "skeleton_semantic"):
            raise ValueError(f"Unknown target_mode: {target_mode!r}")
        self.target_mode = target_mode

        os.makedirs(self.cache_dir, exist_ok=True)
        self.crops: List[_CropCache] = self._load_or_build_crops(crops_yaml_path)
        if not self.crops:
            raise ValueError(f"No crops loaded from {crops_yaml_path}")

        # Uniform-per-crop weighting: every annotated region gets equal
        # representation in an epoch regardless of voxel count. The
        # alternative (weight by voxel count) is "spend samples
        # proportional to physical volume" -- more natural if crops were a
        # systematic dense survey of the dataset, but these are a handful
        # of hand-picked annotation crops, and voxel-count weighting would
        # let one large crop dominate the gradient simply for being
        # bigger, not more informative. Revisit if crop count/size spread
        # grows a lot.
        self.crop_weights = np.full(len(self.crops), 1.0 / len(self.crops))

        # Keyed by resolved voxel size so crops at different raw
        # resolutions (or raw_voxel_size_nm=None mixed with metadata that
        # differs per crop) don't silently share a mis-scaled handle.
        self._raw_idis: Dict[Tuple[float, ...], object] = {}
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Construction: load crops, build/cache targets once
    # ------------------------------------------------------------------

    def _load_or_build_crops(self, crops_yaml_path: str) -> List[_CropCache]:
        config = parse_crops_yaml(crops_yaml_path)
        crops = []
        for i, entry in enumerate(config.crops):
            name = entry.name or f"crop{i:03d}"
            crops.append(self._load_or_build_one(name, entry))
        return crops

    def _load_or_build_one(self, name: str, entry: CropEntry) -> _CropCache:
        # Resolution, skeleton radius, and branch-pruning length are all part
        # of the cache key: a cache built at a different target_voxel_size_nm
        # has the wrong shape entirely, and one built at a different
        # skeleton_radius or min_branch_length has the right shape but
        # silently-wrong skeleton/vector targets (see
        # skoots_targets.py::skeleton_mask_target,
        # ::_prune_skeleton_points) -- either way, reusing it would either
        # crash confusingly or, worse, load stale data that happens to look
        # valid.
        if self.target_voxel_size_nm is not None:
            cache_name = f"{name}_vs{'x'.join(str(int(v)) for v in self.target_voxel_size_nm)}nm"
        else:
            cache_name = name
        cache_name = f"{cache_name}_skelr{self.skeleton_radius}_prune{self.min_branch_length:g}"
        if self.target_mode != "skoots":
            cache_name = f"{cache_name}_{self.target_mode}"
        paths = _crop_cache_paths(self.cache_dir, cache_name)
        if os.path.exists(paths["meta"]):
            logger.info(f"SkootsDataset: loading cached targets for crop {name!r}")
            meta = np.load(paths["meta"])
            return _CropCache(
                name=name,
                labels=np.load(paths["labels"]),
                target=np.load(paths["target"]),
                voxel_size_nm=meta["voxel_size_nm"],
                offset_nm=meta["offset_nm"],
                fg_coords=np.load(paths["fg_coords"]),
                shape=meta["shape"],
            )

        logger.info(
            f"SkootsDataset: no cache for crop {name!r} at {paths['dir']}; "
            "building SKOOTS targets now (can take minutes for large/dense crops)"
        )
        sub, voxel_size_nm, offset_nm = _read_voxel_size_and_offset(entry.path)
        arr = _open_array(entry.path, sub)
        source = np.asarray(arr[:])
        remapped = remap_labels(
            source, entry.fg_ids, entry.bg_ids, entry.mode, entry.connected_components
        )
        # crop_loader convention (0=unannotated, 1=bg, 2+=instance) collapses
        # onto skoots_targets' convention (0=bg, 1+=instance) by treating
        # "unannotated" as background: every crop here is expected to be
        # mode="dense" (fully annotated), so there's no held-out
        # unannotated region to preserve separately. A sparse crop would
        # need an explicit "annotated" mask threaded through instead.
        native_labels = np.where(remapped >= 2, remapped.astype(np.int32) - 1, 0)

        # Skeletons/distances are computed at the crop's *native* resolution,
        # then the resulting target array (not the labels-before-
        # skeletonizing) is decimated to target_voxel_size_nm -- skeletonizing
        # already-decimated labels produces jaggier, more artifact-prone
        # skeletons than skeletonizing at full resolution and downsampling
        # the result (see corrections/skoots_mito ng_viz comparison).
        # skeleton_radius/min_branch_length are specified in
        # target_voxel_size_nm units (their public, documented meaning), so
        # they're scaled up by the same decimation factor here to keep the
        # same physical ball/pruning size when applied at native resolution.
        if self.target_voxel_size_nm is not None:
            factor = _decimation_factor(voxel_size_nm, self.target_voxel_size_nm)
            native_scale = int(factor[0])
        else:
            factor = None
            native_scale = 1

        if self.target_mode == "skeleton_distance":
            targets = build_skeleton_distance_targets(
                native_labels,
                skeleton_radius=self.skeleton_radius * native_scale,
                min_branch_length=self.min_branch_length * native_scale,
            )
            target, _mask = stack_skeleton_distance_target_and_mask(native_labels, targets)
            continuous_channels: Tuple[int, ...] = (2,)  # distance
        elif self.target_mode == "skeleton_semantic":
            targets = build_skeleton_semantic_targets(
                native_labels,
                skeleton_radius=self.skeleton_radius * native_scale,
                min_branch_length=self.min_branch_length * native_scale,
            )
            target, _mask = stack_skeleton_semantic_target_and_mask(native_labels, targets)
            continuous_channels = ()  # semantic/skeleton are both binary masks
        else:
            targets = build_skoots_targets(
                native_labels,
                skeleton_radius=self.skeleton_radius * native_scale,
                min_branch_length=self.min_branch_length * native_scale,
            )
            # stack_targets_and_mask's mask is always all-ones for a dense crop
            # (see its docstring); discarded here rather than persisted, see
            # `_CropCache`'s comment.
            target, _mask = stack_targets_and_mask(native_labels, targets)
            continuous_channels = (2, 3, 4)  # vec_z, vec_y, vec_x

        if factor is not None:
            labels, voxel_size_nm = _decimate_labels(native_labels, voxel_size_nm, self.target_voxel_size_nm)
            target = _decimate_target_array(target, factor, continuous_channels)
        else:
            labels = native_labels

        fg_coords = np.argwhere(labels > 0).astype(np.int64)
        if fg_coords.shape[0] == 0:
            raise ValueError(f"Crop {name!r} has no foreground voxels after remap")

        os.makedirs(paths["dir"], exist_ok=True)
        np.save(paths["labels"], labels)
        np.save(paths["target"], target)
        np.save(paths["fg_coords"], fg_coords)
        np.savez(
            paths["meta"],
            voxel_size_nm=voxel_size_nm,
            offset_nm=offset_nm,
            shape=np.array(labels.shape, dtype=np.int64),
        )

        return _CropCache(
            name=name,
            labels=labels,
            target=target,
            voxel_size_nm=voxel_size_nm,
            offset_nm=offset_nm,
            fg_coords=fg_coords,
            shape=np.array(labels.shape, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = self._worker_rng()
        crop = self.crops[rng.choice(len(self.crops), p=self.crop_weights)]

        # Uniform over the whole (dense) crop -- no foreground-centering bias.
        # An earlier version force-centered a hand-picked fraction of patches
        # (`fg_bias`) on a foreground voxel to guarantee the model saw enough
        # foreground; in practice that skewed the training distribution so
        # far from the crop's true foreground/background ratio that the
        # semantic head collapsed to predicting foreground almost everywhere.
        # Since every crop here is dense
        # (mode="dense"), a plain uniform draw already reproduces the crop's
        # real foreground fraction -- no constant to hand-tune, and no risk
        # of re-introducing the same collapse via a different arbitrary value.
        center = rng.integers(0, crop.shape)

        lo, hi = self._clamped_window(center, crop.shape)

        # Label/target/mask all slice with the exact same (lo, hi) computed
        # once above -- this is the entire point of precomputing targets
        # over the *full* crop rather than per patch: as long as the three
        # arrays share one crop-shaped coordinate frame, any window we cut
        # is automatically self-consistent, with no risk of the skeleton
        # target having been built from a different (truncated) instance
        # extent than the label patch it's paired with.
        target_patch = crop.target[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        # See `_CropCache`: dense crops have an all-ones mask everywhere, so
        # it's synthesized at patch size here rather than stored/sliced from
        # a full-volume array.
        mask_patch = np.ones_like(target_patch, dtype=np.float32)
        assert target_patch.shape[1:] == tuple(self.output_size)

        raw_patch = self._read_raw_patch(crop, lo, hi)

        raw_t = torch.from_numpy(raw_patch[np.newaxis].astype(np.float32))
        # Packed as (2C, Z, Y, X): [0:C]=target, [C:2C]=mask -- see module
        # docstring for why (keeps the shared 2-tuple batch contract).
        packed_t = torch.from_numpy(
            np.concatenate([target_patch, mask_patch], axis=0).astype(np.float32)
        )
        return raw_t, packed_t

    def _clamped_window(self, center: np.ndarray, shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Shift an `output_size`-shaped window to stay in-bounds, no padding.

        Unlike `VirtualPatchDataset` (which zero-pads out-of-bounds reads
        from one giant merged volume where samples can legitimately land
        near the dataset edge), these crops are small, fully in-memory,
        densely-annotated volumes: every crop is expected to be at least
        `output_size_voxels` in each dimension, so shifting the window
        keeps 100% of every patch as real annotated data with valid SKOOTS
        targets, rather than manufacturing an artificial zero-background
        border for the model to learn around.
        """
        size = self.output_size
        if np.any(shape < size):
            raise ValueError(
                f"Crop shape {tuple(shape)} is smaller than output_size_voxels "
                f"{tuple(size)}; cannot sample a full patch without padding."
            )
        lo = center - size // 2
        lo = np.clip(lo, 0, shape - size)
        hi = lo + size
        return lo, hi

    def _read_raw_patch(self, crop: _CropCache, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Read the raw EM patch spatially matching label window `[lo, hi)`.

        Converted to a physical-nm Roi, not a voxel slice, because the raw
        dataset can live at a coarser resolution/different multiscale level
        than the label crop -- there is no voxel-index correspondence
        between the two arrays, only a physical-coordinate one, so we must
        go through nm regardless of whether resolutions happen to match.
        """
        from funlib.geometry import Coordinate, Roi
        from cellmap_flow.image_data_interface import ImageDataInterface

        # If raw_voxel_size_nm is unset, read raw at the label crop's own
        # voxel size by default (rather than leaving it to
        # ImageDataInterface's own multiscale auto-selection), so every
        # crop's raw patch is guaranteed to come back at output_size_voxels
        # regardless of what level the raw dataset's metadata happens to
        # default to.
        voxel_size = self.raw_voxel_size_nm if self.raw_voxel_size_nm is not None else crop.voxel_size_nm
        idi_key = tuple(voxel_size)
        idi = self._raw_idis.get(idi_key)
        if idi is None:
            idi = ImageDataInterface(
                self.raw_dataset_path,
                voxel_size=Coordinate(voxel_size),
                normalize=False,
            )
            self._raw_idis[idi_key] = idi

        # Context/halo extends the raw read beyond the target/label window on
        # both sides -- it does not shrink or shift what the label/target
        # patch itself covers. The surrounding raw imagery this reaches into
        # may be outside the annotated crop entirely; that's fine, it's
        # meant to supply real (if unlabeled) neighborhood context to a
        # valid-padding architecture, not to be independently annotated.
        nm_lo = crop.offset_nm + (lo - self.context_voxels) * crop.voxel_size_nm
        nm_hi = crop.offset_nm + (hi + self.context_voxels) * crop.voxel_size_nm
        roi = Roi(offset=Coordinate(nm_lo), shape=Coordinate(nm_hi - nm_lo))
        raw = idi.to_ndarray_ts(roi)
        return raw.astype(np.float32) / 255.0

    def _worker_rng(self) -> np.random.Generator:
        # Cached on self, not reseeded per call: a freshly-seeded Generator
        # yields the same first draw every time, which would make every
        # __getitem__ in a worker return an identical patch.
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            self._rng = np.random.default_rng(self.seed + worker_id * 1_000_003)
        return self._rng
