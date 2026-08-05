#!/usr/bin/env python
"""
Fix zarr structure of test_corrections.zarr for Neuroglancer compatibility.

Converts from:
    raw/s0/data/.zarray
to:
    raw/s0/.zarray

And adds OME-NGFF v0.4 metadata.
"""

import zarr
import shutil
from pathlib import Path
import numpy as np


def fix_correction_structure(corrections_path: str):
    """
    Fix the zarr structure to be Neuroglancer/OME-NGFF compatible.

    Args:
        corrections_path: Path to corrections.zarr
    """
    corrections_path = Path(corrections_path)
    if not corrections_path.exists():
        print(f"Error: {corrections_path} does not exist")
        return

    print(f"Fixing zarr structure in: {corrections_path}")
    print("=" * 60)

    # Open root group
    root = zarr.open_group(str(corrections_path), mode='a')

    # Process each correction
    correction_ids = [key for key in root.group_keys()]
    print(f"Found {len(correction_ids)} corrections\n")

    for i, corr_id in enumerate(correction_ids, 1):
        print(f"[{i}/{len(correction_ids)}] Processing {corr_id}...")
        corr_group = root[corr_id]

        # Get metadata
        voxel_size = corr_group.attrs.get('voxel_size', [16, 16, 16])

        # Fix each array (raw, prediction, mask)
        for array_name in ['raw', 'prediction', 'mask']:
            if array_name not in corr_group:
                print(f"  Warning: {array_name} not found, skipping")
                continue

            # Check if old structure exists (s0/data)
            old_path = f"{array_name}/s0/data"
            new_path = f"{array_name}/s0"

            if old_path in corr_group:
                # Load data from old location
                old_data = corr_group[old_path][:]
                print(f"  ✓ {array_name}: {old_data.shape} {old_data.dtype}")

                # Create new array at correct location
                corr_group.array(
                    new_path,
                    old_data,
                    chunks=(64, 64, 64),
                    dtype=old_data.dtype,
                    overwrite=True
                )

                # Delete old s0/data structure (if it's a group with 'data' inside)
                try:
                    s0_item = corr_group[array_name]['s0']
                    if isinstance(s0_item, zarr.hierarchy.Group):
                        # s0 is a group, check if it has 'data' array
                        if 'data' in dict(s0_item.arrays()):
                            del s0_item['data']
                except Exception as e:
                    # s0 is already an array, nothing to clean up
                    pass

            # Add OME-NGFF metadata
            array_group = corr_group[array_name]
            array_group.attrs['multiscales'] = [{
                'version': '0.4',
                'name': array_name,
                'axes': [
                    {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
                    {'name': 'x', 'type': 'space', 'unit': 'nanometer'}
                ],
                'datasets': [{
                    'path': 's0',
                    'coordinateTransformations': [{
                        'type': 'scale',
                        'scale': voxel_size
                    }]
                }]
            }]

        print(f"  ✓ Fixed structure and added OME-NGFF metadata")

    print("\n" + "=" * 60)
    print(f"✅ Fixed {len(correction_ids)} corrections")
    print("\nNew structure:")
    print("  corrections.zarr/<uuid>/raw/s0/.zarray")
    print("  corrections.zarr/<uuid>/prediction/s0/.zarray")
    print("  corrections.zarr/<uuid>/mask/s0/.zarray")
    print("\nOME-NGFF v0.4 metadata added to all arrays")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix zarr structure for Neuroglancer compatibility"
    )
    parser.add_argument(
        "--corrections",
        type=str,
        default="test_corrections.zarr",
        help="Path to corrections zarr"
    )

    args = parser.parse_args()
    fix_correction_structure(args.corrections)
