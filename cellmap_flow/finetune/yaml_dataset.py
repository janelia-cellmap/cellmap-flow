"""
PyTorch Dataset for loading training data from a YAML config file.

Supports large ground truth crops with random patch sampling. Each crop
can be much larger than the model input/output size — patches are randomly
sampled from within the crop at each access.

YAML format:
    datasets:
      dataset_name:
        raw: /path/to/raw/zarr/sN
        crops:
          crop_name: /path/to/label/zarr/sN
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _read_ome_ngff_transforms(zarr_array_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read OME-NGFF coordinate transformations for a zarr array.

    Given a path like .../crop522/mito/s2, reads the parent (.../crop522/mito)
    multiscales metadata and finds the matching dataset entry.

    Returns:
        (scale, translation) as numpy arrays in nm, shape (3,) each [z, y, x]
    """
    array_path = Path(zarr_array_path)
    array_name = array_path.name  # e.g. "s2"
    parent_path = str(array_path.parent)  # e.g. ".../crop522/mito"

    parent = zarr.open(parent_path, mode="r")
    attrs = dict(parent.attrs)

    if "multiscales" not in attrs:
        raise ValueError(f"No multiscales metadata found at {parent_path}")

    multiscales = attrs["multiscales"]
    if not multiscales:
        raise ValueError(f"Empty multiscales at {parent_path}")

    datasets = multiscales[0].get("datasets", [])

    for ds in datasets:
        if ds.get("path") == array_name:
            transforms = ds.get("coordinateTransformations", [])
            scale = np.array([1.0, 1.0, 1.0])
            translation = np.array([0.0, 0.0, 0.0])
            for t in transforms:
                if t["type"] == "scale":
                    scale = np.array(t["scale"])
                elif t["type"] == "translation":
                    translation = np.array(t["translation"])
            return scale, translation

    raise ValueError(
        f"Could not find dataset entry for '{array_name}' in multiscales at {parent_path}. "
        f"Available: {[d['path'] for d in datasets]}"
    )


def augment_3d(raw: np.ndarray, mask: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply 3D augmentation to raw and mask volumes.

    Augmentations: random flips (Z/Y/X), random 90-degree XY rotations,
    random intensity scaling and Gaussian noise for raw.
    """
    # Random flips
    for axis in range(3):
        if np.random.rand() > 0.5:
            raw = np.flip(raw, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

    # Random 90-degree rotation in XY plane
    k = np.random.randint(0, 4)
    if k > 0:
        raw = np.rot90(raw, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

    # Intensity augmentation for raw only
    if normalize:
        scale = np.random.uniform(0.8, 1.2)
        raw = np.clip(raw * scale, -1, 1)
        noise = np.random.normal(0, 0.02, raw.shape).astype(np.float32)
        raw = np.clip(raw + noise, -1, 1)

    return raw, mask


class YamlCropDataset(Dataset):
    """
    Dataset that loads raw + ground truth from a YAML config with random
    patch sampling from large crops.

    Args:
        yaml_path: Path to YAML config file
        input_shape: Model input size in voxels [z, y, x] (for raw patches)
        output_shape: Model output size in voxels [z, y, x] (for target patches)
        samples_per_crop: Fixed number of random samples per crop per epoch.
            If None, uses proportional sampling.
        samples_per_epoch: Total samples per epoch, distributed proportionally.
            If None and samples_per_crop is None, auto-computes from crop volumes.
        augment: Whether to apply 3D augmentation
        normalize: Whether to normalize raw data
    """

    def __init__(
        self,
        yaml_path: str,
        input_shape: Tuple[int, int, int],
        output_shape: Tuple[int, int, int],
        samples_per_crop: Optional[int] = None,
        samples_per_epoch: Optional[int] = None,
        augment: bool = True,
        normalize: bool = True,
    ):
        self.input_shape = np.array(input_shape)
        self.output_shape = np.array(output_shape)
        self.augment = augment
        self.normalize = normalize

        # Load YAML config
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # Parse all crops
        self.crops: List[dict] = []
        self._parse_config(config)

        if len(self.crops) == 0:
            raise ValueError(f"No valid crops found in {yaml_path}")

        logger.info(f"Loaded {len(self.crops)} crops from {yaml_path}")

        # Build sample index
        self._build_sample_index(samples_per_crop, samples_per_epoch)

    def _parse_config(self, config: dict):
        """Parse YAML config and open zarr arrays."""
        datasets = config.get("datasets", {})

        for dataset_name, dataset_info in datasets.items():
            raw_path = dataset_info.get("raw")
            crops = dataset_info.get("crops", {})

            if not raw_path:
                logger.warning(f"Dataset {dataset_name}: no raw path, skipping")
                continue

            # Open raw zarr and read its transforms
            try:
                raw_array = zarr.open(raw_path, mode="r")
                raw_scale, raw_translation = _read_ome_ngff_transforms(raw_path)
                logger.info(
                    f"  Raw: {raw_path} shape={raw_array.shape} "
                    f"voxel_size={raw_scale} offset={raw_translation}"
                )
            except Exception as e:
                logger.error(f"Dataset {dataset_name}: cannot open raw {raw_path}: {e}")
                continue

            for crop_name, crop_path in crops.items():
                try:
                    crop_array = zarr.open(crop_path, mode="r")
                    crop_scale, crop_translation = _read_ome_ngff_transforms(crop_path)

                    crop_shape = np.array(crop_array.shape)

                    # Check that the crop is large enough for at least one output patch
                    if np.any(crop_shape < self.output_shape):
                        logger.warning(
                            f"  Crop {crop_name}: shape {crop_shape} < output_shape "
                            f"{self.output_shape}, skipping"
                        )
                        continue

                    self.crops.append({
                        "dataset_name": dataset_name,
                        "crop_name": crop_name,
                        "crop_array": crop_array,
                        "crop_shape": crop_shape,
                        "crop_scale": crop_scale,
                        "crop_translation": crop_translation,
                        "raw_array": raw_array,
                        "raw_shape": np.array(raw_array.shape),
                        "raw_scale": raw_scale,
                        "raw_translation": raw_translation,
                    })

                    logger.info(
                        f"  Crop {crop_name}: shape={crop_shape} "
                        f"voxel_size={crop_scale} offset={crop_translation}"
                    )

                except Exception as e:
                    logger.error(f"  Crop {crop_name}: error loading {crop_path}: {e}")

    def _build_sample_index(
        self,
        samples_per_crop: Optional[int],
        samples_per_epoch: Optional[int],
    ):
        """Build index mapping sample idx -> crop idx with sample counts."""
        self.sample_index: List[int] = []  # crop_idx for each sample

        if samples_per_crop is not None:
            # Fixed number per crop
            for i in range(len(self.crops)):
                self.sample_index.extend([i] * samples_per_crop)
        elif samples_per_epoch is not None:
            # Distribute proportional to crop volume
            volumes = []
            for crop_info in self.crops:
                vol = np.prod(crop_info["crop_shape"].astype(np.float64))
                volumes.append(vol)
            total_vol = sum(volumes)
            for i, vol in enumerate(volumes):
                n = max(1, round(samples_per_epoch * vol / total_vol))
                self.sample_index.extend([i] * n)
        else:
            # Auto: ~1 sample per non-overlapping output patch
            for i, crop_info in enumerate(self.crops):
                shape = crop_info["crop_shape"]
                n_patches = max(1, int(np.prod(shape / self.output_shape)))
                self.sample_index.extend([i] * n_patches)

        logger.info(f"Total samples per epoch: {len(self.sample_index)}")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        crop_idx = self.sample_index[idx]
        crop_info = self.crops[crop_idx]

        crop_array = crop_info["crop_array"]
        raw_array = crop_info["raw_array"]
        crop_shape = crop_info["crop_shape"]
        crop_scale = crop_info["crop_scale"]
        crop_trans = crop_info["crop_translation"]
        raw_scale = crop_info["raw_scale"]
        raw_trans = crop_info["raw_translation"]
        raw_shape = crop_info["raw_shape"]

        oz, oy, ox = self.output_shape
        iz, iy, ix = self.input_shape

        # 1. Random position within crop for the output (target) patch
        max_start = crop_shape - self.output_shape
        tz = np.random.randint(0, max(1, max_start[0] + 1))
        ty = np.random.randint(0, max(1, max_start[1] + 1))
        tx = np.random.randint(0, max(1, max_start[2] + 1))

        # 2. Read target patch from crop
        target = crop_array[tz:tz + oz, ty:ty + oy, tx:tx + ox]
        target = np.array(target, dtype=np.float32)

        # Convert instance segmentation to binary (0=background, >0=foreground)
        target = (target > 0).astype(np.float32)

        # 3. Compute physical center of the target patch
        target_center_voxel = np.array([
            tz + oz / 2.0,
            ty + oy / 2.0,
            tx + ox / 2.0,
        ])
        target_center_nm = target_center_voxel * crop_scale + crop_trans

        # 4. Convert to raw voxel coordinates
        raw_center_voxel = (target_center_nm - raw_trans) / raw_scale

        # 5. Compute raw patch start (centered on the physical location)
        raw_start = np.round(raw_center_voxel - self.input_shape / 2.0).astype(int)
        raw_end = raw_start + self.input_shape

        # 6. Handle boundaries: clamp and pad if needed
        pad_before = np.maximum(0, -raw_start)
        pad_after = np.maximum(0, raw_end - raw_shape)
        clamped_start = np.maximum(0, raw_start)
        clamped_end = np.minimum(raw_shape, raw_end)

        raw = raw_array[
            clamped_start[0]:clamped_end[0],
            clamped_start[1]:clamped_end[1],
            clamped_start[2]:clamped_end[2],
        ]
        raw = np.array(raw, dtype=np.float32)

        # Pad if we went out of bounds
        if np.any(pad_before > 0) or np.any(pad_after > 0):
            raw = np.pad(
                raw,
                list(zip(pad_before, pad_after)),
                mode="reflect",
            )

        # 7. Normalize raw
        if self.normalize:
            if raw.max() > 1.0:
                raw = (raw / 127.5) - 1.0

        # 8. Augmentation (only when raw and target have same shape)
        if self.augment and raw.shape == target.shape:
            raw, target = augment_3d(raw, target, normalize=self.normalize)

        # 9. Add channel dimension and convert to torch
        raw = torch.from_numpy(raw[np.newaxis, ...])      # (1, Z, Y, X)
        target = torch.from_numpy(target[np.newaxis, ...])  # (1, Z, Y, X)

        return raw, target


def create_yaml_dataloader(
    yaml_path: str,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    batch_size: int = 2,
    samples_per_crop: Optional[int] = None,
    samples_per_epoch: Optional[int] = None,
    augment: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
    normalize: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from a YAML crop config.

    Args:
        yaml_path: Path to YAML config file
        input_shape: Model input size in voxels [z, y, x]
        output_shape: Model output size in voxels [z, y, x]
        batch_size: Batch size
        samples_per_crop: Fixed samples per crop (None for proportional)
        samples_per_epoch: Total samples per epoch (None for auto)
        augment: Whether to augment
        num_workers: DataLoader workers
        shuffle: Whether to shuffle
        normalize: Whether to normalize raw

    Returns:
        DataLoader instance
    """
    dataset = YamlCropDataset(
        yaml_path=yaml_path,
        input_shape=input_shape,
        output_shape=output_shape,
        samples_per_crop=samples_per_crop,
        samples_per_epoch=samples_per_epoch,
        augment=augment,
        normalize=normalize,
    )

    actual_batch_size = min(batch_size, len(dataset)) if len(dataset) > 0 else batch_size

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    logger.info(
        f"Created YAML DataLoader with {len(dataset)} samples, "
        f"batch_size={actual_batch_size}, num_workers={num_workers}"
    )

    return dataloader
