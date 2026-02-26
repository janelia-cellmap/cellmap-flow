"""
PyTorch Dataset for loading user corrections.

This module provides a Dataset class that loads 3D EM data and correction
masks from Zarr files for training LoRA adapters.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CorrectionDataset(Dataset):
    """
    PyTorch Dataset for user corrections stored in Zarr format.

    Loads raw EM data and corrected masks from corrections.zarr/, with
    optional 3D augmentation.

    Args:
        corrections_zarr_path: Path to corrections.zarr directory
        patch_shape: Shape of patches to extract (Z, Y, X)
                    If None, uses full correction size
        augment: Whether to apply 3D augmentation
        model_name: If specified, only load corrections for this model

    Examples:
        >>> dataset = CorrectionDataset(
        ...     "test_corrections.zarr",
        ...     patch_shape=(64, 64, 64),
        ...     augment=True
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> raw, target = dataset[0]
        >>> print(f"Raw shape: {raw.shape}, Target shape: {target.shape}")
    """

    def __init__(
        self,
        corrections_zarr_path: str,
        patch_shape: Optional[Tuple[int, int, int]] = None,
        augment: bool = True,
        model_name: Optional[str] = None,
    ):
        self.corrections_path = Path(corrections_zarr_path)
        self.patch_shape = patch_shape
        self.augment = augment
        self.model_name = model_name

        # Load corrections
        self.corrections = self._load_corrections()

        if len(self.corrections) == 0:
            raise ValueError(
                f"No corrections found in {corrections_zarr_path}. "
                f"Generate corrections first."
            )

        logger.info(
            f"Loaded {len(self.corrections)} corrections from {corrections_zarr_path}"
        )

    def _load_corrections(self) -> List[dict]:
        """Load correction metadata from Zarr."""
        corrections = []

        logger.info(f"Loading corrections from: {self.corrections_path}")

        if not self.corrections_path.exists():
            logger.error(f"Corrections path does not exist: {self.corrections_path}")
            return corrections

        path_str = str(self.corrections_path)
        z = zarr.open_group(path_str, mode="r")

        for correction_id in z.keys():
            corr_group = z[correction_id]

            # Check if correction has required data
            # Support both 'mask' (from test scripts) and 'annotation' (from dashboard)
            has_raw = "raw" in corr_group
            has_mask = "mask" in corr_group
            has_annotation = "annotation" in corr_group

            has_raw_s0 = has_raw and "s0" in corr_group["raw"]
            has_mask_s0 = has_mask and "s0" in corr_group["mask"]
            has_annotation_s0 = has_annotation and "s0" in corr_group["annotation"]

            if not has_raw_s0 or not (has_mask_s0 or has_annotation_s0):
                logger.warning(
                    f"Skipping {correction_id}: missing raw/s0 or mask|annotation/s0"
                )
                continue

            # Use 'mask' if available, otherwise use 'annotation'
            mask_key = "mask" if has_mask_s0 else "annotation"

            # Get metadata
            attrs = dict(corr_group.attrs)

            # Filter by model name if specified
            if self.model_name and attrs.get("model_name") != self.model_name:
                continue

            raw_path = self.corrections_path / correction_id / "raw" / "s0"
            mask_path = self.corrections_path / correction_id / mask_key / "s0"

            if not raw_path.exists() or not mask_path.exists():
                logger.warning(
                    f"Skipping {correction_id}: missing paths "
                    f"raw_path={raw_path} (exists={raw_path.exists()}), "
                    f"mask_path={mask_path} (exists={mask_path.exists()})"
                )
                continue

            corrections.append(
                {
                    "id": correction_id,
                    "raw_path": str(raw_path),
                    "mask_path": str(mask_path),
                    "metadata": attrs,
                }
            )

        return corrections

    def __len__(self) -> int:
        """Return number of corrections."""
        return len(self.corrections)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a correction pair (raw, target).

        Args:
            idx: Index of correction

        Returns:
            Tuple of (raw, target) tensors:
            - raw: (1, Z, Y, X) float32 tensor
            - target: (1, Z, Y, X) float32 tensor, values in [0, 1]
        """
        correction = self.corrections[idx]

        # Load data using ImageDataInterface for consistent data loading
        from cellmap_flow.image_data_interface import ImageDataInterface

        try:
            raw = ImageDataInterface(
                correction["raw_path"], normalize=False
            ).to_ndarray_ts()
            mask = ImageDataInterface(
                correction["mask_path"], normalize=False
            ).to_ndarray_ts()
        except Exception as e:
            raise FileNotFoundError(
                f"Failed loading correction '{correction.get('id', idx)}' "
                f"raw_path='{correction.get('raw_path')}' "
                f"mask_path='{correction.get('mask_path')}': {e}"
            ) from e

        # Convert to float
        raw = raw.astype(np.float32)
        mask = mask.astype(np.float32)

        # Normalize mask to [0, 1]
        # Only normalize pixel-intensity masks (0-255 range), not class labels (0, 1, 2)
        # Class labels are small integers used by mask_unannotated logic in trainer
        if mask.max() > 2.0:
            mask = mask / 255.0

        # For models with different input/output sizes, we keep raw at full size
        # Patching is disabled for this case - use full corrections
        # Apply augmentation (only if raw and mask have same shape)
        if self.augment and raw.shape == mask.shape:
            raw, mask = self._augment_3d(raw, mask)
        elif self.augment and raw.shape != mask.shape:
            logger.debug(
                f"Skipping augmentation: raw {raw.shape} != mask {mask.shape}. "
                "Augmentation requires matching sizes."
            )

        # Add channel dimension and convert to torch
        raw = torch.from_numpy(raw[np.newaxis, ...])  # (1, Z, Y, X)
        mask = torch.from_numpy(mask[np.newaxis, ...])  # (1, Z, Y, X)

        return raw, mask

    def _random_crop(
        self, raw: np.ndarray, mask: np.ndarray, patch_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a random patch from the volumes.

        Args:
            raw: Raw data (Z, Y, X)
            mask: Mask data (Z, Y, X)
            patch_shape: Desired patch shape (Z, Y, X)

        Returns:
            Cropped (raw, mask) pair
        """
        z, y, x = raw.shape
        pz, py, px = patch_shape

        # If volume is smaller than patch, pad it
        if z < pz or y < py or x < px:
            pad_z = max(0, pz - z)
            pad_y = max(0, py - y)
            pad_x = max(0, px - x)

            raw = np.pad(raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="reflect")
            mask = np.pad(mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="reflect")
            z, y, x = raw.shape

        # Random offset
        z_offset = np.random.randint(0, max(1, z - pz + 1))
        y_offset = np.random.randint(0, max(1, y - py + 1))
        x_offset = np.random.randint(0, max(1, x - px + 1))

        # Crop
        raw_crop = raw[
            z_offset : z_offset + pz, y_offset : y_offset + py, x_offset : x_offset + px
        ]
        mask_crop = mask[
            z_offset : z_offset + pz, y_offset : y_offset + py, x_offset : x_offset + px
        ]

        return raw_crop, mask_crop

    def _augment_3d(
        self, raw: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 3D augmentation to raw and mask.

        Augmentations:
        - Random flips on Z/Y/X axes (50% each)
        - Random 90° rotations in XY plane (0°, 90°, 180°, 270°)
        - Random intensity scaling for raw (×0.8 to ×1.2)
        - Random Gaussian noise for raw (σ=0.01)

        Args:
            raw: Raw data (Z, Y, X)
            mask: Mask data (Z, Y, X)

        Returns:
            Augmented (raw, mask) pair
        """
        # Random flips
        if np.random.rand() > 0.5:
            raw = np.flip(raw, axis=0).copy()  # Flip Z
            mask = np.flip(mask, axis=0).copy()

        if np.random.rand() > 0.5:
            raw = np.flip(raw, axis=1).copy()  # Flip Y
            mask = np.flip(mask, axis=1).copy()

        if np.random.rand() > 0.5:
            raw = np.flip(raw, axis=2).copy()  # Flip X
            mask = np.flip(mask, axis=2).copy()

        # Random 90° rotation in XY plane
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3 (0°, 90°, 180°, 270°)
        if k > 0:
            raw = np.rot90(raw, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(1, 2)).copy()

        # Intensity augmentation for raw only
        # Random scaling (×0.8 to ×1.2)
        scale = np.random.uniform(0.8, 1.2)
        raw = np.clip(raw * scale, 0, 1)

        # Random Gaussian noise (σ=0.01)
        noise = np.random.normal(0, 0.01, raw.shape).astype(np.float32)
        raw = np.clip(raw + noise, 0, 1)

        return raw, mask


def create_dataloader(
    corrections_zarr_path: str,
    batch_size: int = 2,
    patch_shape: Optional[Tuple[int, int, int]] = None,
    augment: bool = True,
    num_workers: int = 4,
    shuffle: bool = True,
    model_name: Optional[str] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for corrections.

    Args:
        corrections_zarr_path: Path to corrections.zarr directory
        batch_size: Batch size (2-4 recommended for 3D data)
        patch_shape: Shape of patches to extract (Z, Y, X)
        augment: Whether to apply augmentation
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        model_name: If specified, only load corrections for this model

    Returns:
        DataLoader instance

    Examples:
        >>> dataloader = create_dataloader(
        ...     "test_corrections.zarr",
        ...     batch_size=2,
        ...     patch_shape=(64, 64, 64)
        ... )
        >>> for raw, target in dataloader:
        ...     print(f"Batch: raw={raw.shape}, target={target.shape}")
        ...     break
        Batch: raw=torch.Size([2, 1, 64, 64, 64]), target=torch.Size([2, 1, 64, 64, 64])
    """
    dataset = CorrectionDataset(
        corrections_zarr_path,
        patch_shape=patch_shape,
        augment=augment,
        model_name=model_name,
    )

    # Clamp batch size to number of samples so DataLoader doesn't error
    actual_batch_size = (
        min(batch_size, len(dataset)) if len(dataset) > 0 else batch_size
    )
    if actual_batch_size != batch_size:
        logger.info(
            f"Clamped batch_size from {batch_size} to {actual_batch_size} "
            f"(only {len(dataset)} samples available)"
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    logger.info(
        f"Created DataLoader with {len(dataset)} samples, "
        f"batch_size={actual_batch_size}, num_workers={num_workers}"
    )

    return dataloader
