#!/usr/bin/env python
"""
Test inference with finetuned LoRA adapter on corrections.

This script:
1. Loads the finetuned adapter
2. Runs inference on all corrections
3. Computes metrics (Dice score, IoU) on annotated regions only
4. Saves comparison visualizations
"""

import torch
import numpy as np
import zarr
from pathlib import Path

from cellmap_flow.models.models_config import FlyModelConfig
from cellmap_flow.finetune.lora_wrapper import load_lora_adapter


def compute_dice_score(pred, target, mask=None):
    """
    Compute Dice score between prediction and target.

    Args:
        pred: Prediction (0-1 probability)
        target: Ground truth (0 or 1)
        mask: Optional mask for annotated regions

    Returns:
        Dice score (0-1, higher is better)
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask

    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target)

    if union == 0:
        return 1.0  # Both empty

    return (2.0 * intersection) / union


def compute_iou(pred, target, mask=None, threshold=0.5):
    """
    Compute IoU (Intersection over Union).

    Args:
        pred: Prediction (0-1 probability)
        target: Ground truth (0 or 1)
        mask: Optional mask for annotated regions
        threshold: Threshold for binarizing predictions

    Returns:
        IoU score (0-1, higher is better)
    """
    pred_binary = (pred > threshold).astype(np.float32)

    if mask is not None:
        pred_binary = pred_binary * mask
        target = target * mask

    intersection = np.sum(pred_binary * target)
    union = np.sum(np.maximum(pred_binary, target))

    if union == 0:
        return 1.0

    return intersection / union


def main():
    print("="*60)
    print("Testing Finetuned Model Inference")
    print("="*60)
    print()

    # Paths
    model_checkpoint = "/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000"

    # Look for adapter in common locations
    adapter_paths = [
        "output/sparse_annotation_finetuning/lora_adapter",
        "scripts/output/sparse_annotation_finetuning/lora_adapter",
    ]

    adapter_path = None
    for path in adapter_paths:
        if Path(path).exists():
            adapter_path = path
            break

    if adapter_path is None:
        print("Error: Could not find LoRA adapter!")
        print("Searched:")
        for path in adapter_paths:
            print(f"  - {path}")
        return

    # Find corrections zarr
    corrections_dir = Path("/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections")
    sparse_files = sorted(corrections_dir.glob("sparse_corrections_*.zarr"))

    if not sparse_files:
        print("Error: No sparse corrections found!")
        return

    corrections_path = str(sparse_files[-1])
    print(f"Corrections: {corrections_path}")
    print(f"Adapter: {adapter_path}")
    print()

    # 1. Load base model
    print("1. Loading base model...")
    model_config = FlyModelConfig(
        checkpoint_path=model_checkpoint,
        channels=["mito"],  # This checkpoint only has 1 channel (mito)
        input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16),
    )
    base_model = model_config.config.model
    print(f"   ✓ Loaded: {type(base_model).__name__}")

    # 2. Load finetuned adapter
    print("\n2. Loading finetuned LoRA adapter...")
    finetuned_model = load_lora_adapter(
        base_model,
        adapter_path,
        is_trainable=False,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()
    print(f"   ✓ Loaded and moved to {device}")

    # 3. Load corrections (read-write mode to save predictions)
    print("\n3. Loading corrections...")
    corrections = zarr.open(corrections_path, mode='a')  # 'a' for append/write
    correction_ids = [k for k in corrections.keys() if not k.startswith('.')]
    print(f"   ✓ Found {len(correction_ids)} corrections")

    # Get voxel size from first correction
    first_corr = corrections[correction_ids[0]]
    voxel_size = np.array([16, 16, 16])  # Default

    # 4. Run inference on each correction and save results
    print("\n4. Running inference and saving finetuned predictions...")
    print("-"*60)

    all_dice_scores = []
    all_iou_scores = []

    for idx, corr_id in enumerate(correction_ids):
        corr = corrections[corr_id]

        # Load data
        raw = np.array(corr['raw/s0'])
        mask_gt = np.array(corr['mask/s0'])

        # Normalize raw to [-1, 1]
        raw_normalized = (raw.astype(np.float32) / 127.5) - 1.0

        # Add batch and channel dims
        raw_tensor = torch.from_numpy(raw_normalized).unsqueeze(0).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            pred_tensor = finetuned_model(raw_tensor)

        # Get prediction (model has 1 channel - mito)
        # Model already has Sigmoid at the end, so pred is already in [0, 1]
        pred = pred_tensor[0, 0].cpu().numpy()  # (56, 56, 56)

        # Create annotation mask and adjusted ground truth
        annotation_mask = (mask_gt > 0).astype(np.float32)
        target = np.clip(mask_gt.astype(np.float32) - 1, 0, None)  # Shift: 0->0, 1->0, 2->1

        # Compute metrics only on annotated regions
        dice = compute_dice_score(pred, target, mask=annotation_mask)
        iou = compute_iou(pred, target, mask=annotation_mask, threshold=0.5)

        all_dice_scores.append(dice)
        all_iou_scores.append(iou)

        # Count annotated voxels
        fg_voxels = np.sum(mask_gt == 2)  # Foreground annotations
        bg_voxels = np.sum(mask_gt == 1)  # Background annotations
        total_annotated = np.sum(annotation_mask)

        # Save finetuned predictions to corrections zarr
        # Check if prediction_finetuned group exists, create or overwrite
        if 'prediction_finetuned' in corr.keys():
            del corr['prediction_finetuned']

        pred_ft_group = corr.create_group('prediction_finetuned')
        pred_ft_group.create_dataset(
            's0',
            data=pred,
            dtype='float32',
            compression='gzip',
            compression_opts=6,
            chunks=(56, 56, 56)
        )

        # Add OME-NGFF metadata (copy from mask group which has the right offset)
        if 'multiscales' in corr['mask'].attrs:
            pred_ft_group.attrs['multiscales'] = corr['mask'].attrs['multiscales']

        print(f"[{idx+1}/{len(correction_ids)}] {corr_id[:8]}...")
        print(f"  Annotated: {total_annotated:,} voxels (FG: {fg_voxels:,}, BG: {bg_voxels:,})")
        print(f"  Dice: {dice:.4f}")
        print(f"  IoU:  {iou:.4f}")
        print(f"  Pred range: [{pred.min():.3f}, {pred.max():.3f}]")
        print(f"  ✓ Saved finetuned predictions")

    # 5. Summary statistics
    print("-"*60)
    print("\n5. Summary Statistics")
    print("="*60)
    print(f"Average Dice Score: {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"Average IoU:        {np.mean(all_iou_scores):.4f} ± {np.std(all_iou_scores):.4f}")
    print()
    print("Interpretation:")
    print("  - Dice/IoU > 0.90: Excellent")
    print("  - Dice/IoU > 0.80: Good")
    print("  - Dice/IoU > 0.70: Fair")
    print("  - Dice/IoU < 0.70: Needs improvement")
    print()
    print("Finetuned predictions saved to:")
    print(f"  {corrections_path}")
    print(f"  Each correction now has: raw, mask, prediction, prediction_finetuned")
    print("="*60)


if __name__ == "__main__":
    main()
