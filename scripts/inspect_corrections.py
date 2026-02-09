#!/usr/bin/env python
"""
Inspect test corrections.

Usage:
    python scripts/inspect_corrections.py --corrections test_corrections.zarr
    python scripts/inspect_corrections.py --corrections test_corrections.zarr --save-slices --output-dir correction_slices
"""

import argparse
from pathlib import Path
import zarr
import numpy as np
from typing import Dict, List


def load_correction(zarr_path: str, correction_id: str) -> Dict:
    """Load a single correction from Zarr."""
    z = zarr.open(zarr_path, 'r')
    corr_group = z[correction_id]

    correction = {
        'id': correction_id,
        'raw': corr_group['raw/s0/data'][:],
        'prediction': corr_group['prediction/s0/data'][:],
        'mask': corr_group['mask/s0/data'][:],
        'metadata': dict(corr_group.attrs)
    }

    return correction


def print_correction_summary(correction: Dict):
    """Print summary of a correction."""
    print(f"\nCorrection: {correction['id'][:8]}...")
    print(f"  Model: {correction['metadata']['model_name']}")
    print(f"  Dataset: {correction['metadata']['dataset_path']}")
    print(f"  ROI: offset={correction['metadata']['roi_offset']}, shape={correction['metadata']['roi_shape']}")
    print(f"  Voxel size: {correction['metadata']['voxel_size']}")

    print(f"\n  Raw data:")
    print(f"    Shape: {correction['raw'].shape}")
    print(f"    Dtype: {correction['raw'].dtype}")
    print(f"    Range: [{correction['raw'].min()}, {correction['raw'].max()}]")

    print(f"\n  Prediction:")
    print(f"    Shape: {correction['prediction'].shape}")
    print(f"    Dtype: {correction['prediction'].dtype}")
    print(f"    Range: [{correction['prediction'].min()}, {correction['prediction'].max()}]")
    print(f"    Mean: {correction['prediction'].mean():.2f}")

    print(f"\n  Corrected mask:")
    print(f"    Shape: {correction['mask'].shape}")
    print(f"    Dtype: {correction['mask'].dtype}")
    print(f"    Range: [{correction['mask'].min()}, {correction['mask'].max()}]")
    print(f"    Coverage: {(correction['mask'] > 127).mean() * 100:.2f}%")

    # Compute difference
    diff = np.abs(correction['mask'].astype(np.int16) - correction['prediction'].astype(np.int16))
    print(f"\n  Difference (mask - prediction):")
    print(f"    Mean abs difference: {diff.mean():.2f}")
    print(f"    Max abs difference: {diff.max()}")
    print(f"    Changed pixels: {(diff > 0).sum() / diff.size * 100:.2f}%")


def save_correction_slices(correction: Dict, output_dir: Path):
    """Save middle slices of raw, prediction, and mask for visualization."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available, skipping slice saving")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get middle slice
    z_mid = correction['raw'].shape[0] // 2

    # Save raw
    raw_slice = correction['raw'][z_mid]
    Image.fromarray(raw_slice).save(
        output_dir / f"{correction['id'][:8]}_raw.png"
    )

    # Save prediction
    pred_slice = correction['prediction'][z_mid]
    Image.fromarray(pred_slice).save(
        output_dir / f"{correction['id'][:8]}_prediction.png"
    )

    # Save mask
    mask_slice = correction['mask'][z_mid]
    Image.fromarray(mask_slice).save(
        output_dir / f"{correction['id'][:8]}_mask.png"
    )

    # Save difference
    diff_slice = np.abs(
        mask_slice.astype(np.int16) - pred_slice.astype(np.int16)
    ).astype(np.uint8)
    Image.fromarray(diff_slice * 10).save(  # Multiply to make difference more visible
        output_dir / f"{correction['id'][:8]}_diff.png"
    )

    print(f"  Saved slices to: {output_dir}/{correction['id'][:8]}_*.png")


def list_corrections(zarr_path: str) -> List[str]:
    """List all correction IDs in the Zarr."""
    z = zarr.open(zarr_path, 'r')
    return list(z.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Inspect test corrections"
    )
    parser.add_argument(
        "--corrections",
        type=str,
        default="test_corrections.zarr",
        help="Path to corrections Zarr"
    )
    parser.add_argument(
        "--save-slices",
        action="store_true",
        help="Save middle slices as PNG images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="correction_slices",
        help="Output directory for slice images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of corrections to inspect"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Inspecting corrections: {args.corrections}")
    print(f"{'='*60}")

    # List corrections
    correction_ids = list_corrections(args.corrections)
    print(f"\nFound {len(correction_ids)} corrections")

    if args.limit:
        correction_ids = correction_ids[:args.limit]
        print(f"Limiting to first {args.limit} corrections")

    output_dir = Path(args.output_dir) if args.save_slices else None

    # Inspect each correction
    for i, correction_id in enumerate(correction_ids):
        correction = load_correction(args.corrections, correction_id)
        print_correction_summary(correction)

        if args.save_slices:
            save_correction_slices(correction, output_dir)

        print()

    print(f"{'='*60}")
    print(f"✓ Inspected {len(correction_ids)} corrections")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
