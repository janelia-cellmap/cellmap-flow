#!/usr/bin/env python3
"""
Analyze corrections to understand training data quality.

Checks how different the masks are from predictions, and whether
there's actually signal for the model to learn from.
"""

import argparse
import numpy as np
import zarr
from pathlib import Path


def analyze_correction(corr_group):
    """Analyze a single correction."""
    # Load data
    if 'prediction/s0' not in corr_group or 'mask/s0' not in corr_group:
        return None

    prediction = np.array(corr_group['prediction/s0'])
    mask = np.array(corr_group['mask/s0'])

    # Convert mask to float if needed
    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32)

    # Compute differences
    diff = np.abs(prediction - mask)

    # Compute stats
    stats = {
        'pred_mean': prediction.mean(),
        'pred_std': prediction.std(),
        'pred_min': prediction.min(),
        'pred_max': prediction.max(),
        'mask_mean': mask.mean(),
        'mask_std': mask.std(),
        'mask_sum': mask.sum(),
        'mask_fraction': (mask > 0.5).sum() / mask.size,
        'diff_mean': diff.mean(),
        'diff_std': diff.std(),
        'diff_max': diff.max(),
        'diff_median': np.median(diff),
        'diff_95th': np.percentile(diff, 95),
        'large_diffs': (diff > 0.1).sum() / diff.size,  # Fraction with >0.1 difference
        'huge_diffs': (diff > 0.3).sum() / diff.size,   # Fraction with >0.3 difference
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correction quality for training"
    )
    parser.add_argument(
        "--corrections",
        type=str,
        required=True,
        help="Path to corrections zarr"
    )

    args = parser.parse_args()

    print("="*70)
    print("Correction Analysis")
    print("="*70)
    print(f"Corrections: {args.corrections}")
    print()

    # Open corrections
    corrections_root = zarr.open(args.corrections, mode='r')
    correction_ids = [key for key in corrections_root.group_keys()]

    print(f"Found {len(correction_ids)} corrections")
    print()

    # Analyze each correction
    all_stats = []
    for i, corr_id in enumerate(correction_ids, 1):
        print(f"[{i}/{len(correction_ids)}] {corr_id}")
        corr_group = corrections_root[corr_id]

        stats = analyze_correction(corr_group)
        if stats is None:
            print("  ⚠ Missing prediction or mask")
            continue

        all_stats.append(stats)

        print(f"  Prediction: mean={stats['pred_mean']:.4f}, std={stats['pred_std']:.4f}, range=[{stats['pred_min']:.4f}, {stats['pred_max']:.4f}]")
        print(f"  Mask:       mean={stats['mask_mean']:.4f}, std={stats['mask_std']:.4f}, fraction={stats['mask_fraction']:.2%}")
        print(f"  Difference: mean={stats['diff_mean']:.4f}, median={stats['diff_median']:.4f}, max={stats['diff_max']:.4f}, 95th={stats['diff_95th']:.4f}")
        print(f"  Large diffs (>0.1): {stats['large_diffs']:.2%}, Huge diffs (>0.3): {stats['huge_diffs']:.2%}")
        print()

    if not all_stats:
        print("No valid corrections found!")
        return

    # Aggregate statistics
    print("="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)

    avg_diff_mean = np.mean([s['diff_mean'] for s in all_stats])
    avg_diff_max = np.mean([s['diff_max'] for s in all_stats])
    avg_large_diffs = np.mean([s['large_diffs'] for s in all_stats])
    avg_mask_fraction = np.mean([s['mask_fraction'] for s in all_stats])

    print(f"\nAverage difference: {avg_diff_mean:.4f}")
    print(f"Average max diff:   {avg_diff_max:.4f}")
    print(f"Average large diffs (>0.1): {avg_large_diffs:.2%}")
    print(f"Average mask fraction: {avg_mask_fraction:.2%}")

    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)

    if avg_diff_mean < 0.05:
        print("⚠ WARNING: Very small differences between predictions and masks!")
        print("  → The model predictions are already very close to the ground truth.")
        print("  → There may not be much signal for the model to learn from.")
        print("  → Consider:")
        print("     - Using more diverse corrections")
        print("     - Reducing erosion iterations (currently 5)")
        print("     - Finding regions where the model performs poorly")
    elif avg_diff_mean < 0.1:
        print("⚠ MODERATE: Small differences between predictions and masks.")
        print("  → Some learning signal present but may need more data or epochs.")
    else:
        print("✓ GOOD: Significant differences between predictions and masks.")
        print("  → Strong learning signal present.")

    if avg_large_diffs < 0.1:
        print("\n⚠ WARNING: Very few large differences!")
        print("  → Less than 10% of voxels have differences > 0.1")
        print("  → The model may struggle to learn meaningful patterns.")

    if avg_mask_fraction < 0.01:
        print("\n⚠ WARNING: Very sparse masks!")
        print("  → Masks are extremely sparse (< 1% positive)")
        print("  → Consider reducing erosion to preserve more structure.")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if avg_diff_mean < 0.1:
        print("\n1. Generate more challenging corrections:")
        print("   - Reduce EROSION_ITERATIONS in generate_mito_corrections.py")
        print("   - Find regions where model predictions are poor")
        print("   - Use manual corrections instead of synthetic erosion")

        print("\n2. Increase training intensity:")
        print("   - Use higher learning rate: --learning-rate 5e-4")
        print("   - Use larger LoRA rank: --lora-r 16")
        print("   - Train for more epochs: --num-epochs 50")

        print("\n3. Generate more corrections:")
        print("   - Increase NUM_CROPS in generate_mito_corrections.py to 50+")
    else:
        print("\nData looks reasonable. If training didn't improve:")
        print("   - Check that training loss actually decreased")
        print("   - Try higher learning rate: --learning-rate 5e-4")
        print("   - Try larger LoRA rank: --lora-r 16")
        print("   - Ensure you're comparing the right models")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
