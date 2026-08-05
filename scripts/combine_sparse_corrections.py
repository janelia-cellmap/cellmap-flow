#!/usr/bin/env python3
"""
Combine individual sparse correction zarr files into a single zarr for training.

This script:
1. Scans a directory for sparse correction zarr files
2. Combines them into a single zarr with multiple correction groups
3. Preserves all metadata including timestamps
"""

import zarr
import numpy as np
from pathlib import Path
from datetime import datetime


def main():
    # Configuration
    SPARSE_DIR = Path("/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections/sparse_points")
    OUTPUT_PATH = "/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections/sparse_combined.zarr"

    print("="*60)
    print("Sparse Correction Combiner")
    print("="*60)
    print(f"Input directory: {SPARSE_DIR}")
    print(f"Output zarr: {OUTPUT_PATH}")
    print()

    if not SPARSE_DIR.exists():
        print(f"Error: Directory not found: {SPARSE_DIR}")
        return

    # Find all sparse correction zarr files
    zarr_files = sorted(SPARSE_DIR.glob("sparse_points_*.zarr"))

    if len(zarr_files) == 0:
        print(f"No sparse correction zarr files found in {SPARSE_DIR}")
        return

    print(f"Found {len(zarr_files)} sparse correction files")
    print()

    # Create output zarr
    output_root = zarr.open(OUTPUT_PATH, mode='w')

    # Process each file
    total_corrections = 0
    for idx, zarr_file in enumerate(zarr_files):
        print(f"[{idx+1}/{len(zarr_files)}] Processing {zarr_file.name}...")

        # Open source zarr
        source_root = zarr.open(str(zarr_file), mode='r')

        # Get all correction IDs in this zarr
        correction_ids = [k for k in source_root.keys() if not k.startswith('.')]

        for corr_id in correction_ids:
            source_group = source_root[corr_id]

            # Create group in output
            output_group = output_root.create_group(corr_id, overwrite=True)

            # Copy raw
            raw_group = output_group.create_group('raw')
            raw_data = np.array(source_group['raw/s0'])
            raw_group.create_dataset(
                's0',
                data=raw_data,
                dtype=raw_data.dtype,
                compression='gzip',
                compression_opts=6,
                chunks=(64, 64, 64)
            )
            if 'multiscales' in source_group['raw'].attrs:
                raw_group.attrs['multiscales'] = source_group['raw'].attrs['multiscales']

            # Copy mask
            mask_group = output_group.create_group('mask')
            mask_data = np.array(source_group['mask/s0'])
            mask_group.create_dataset(
                's0',
                data=mask_data,
                dtype=mask_data.dtype,
                compression='gzip',
                compression_opts=6,
                chunks=(56, 56, 56)
            )
            if 'multiscales' in source_group['mask'].attrs:
                mask_group.attrs['multiscales'] = source_group['mask'].attrs['multiscales']

            # Copy prediction
            pred_group = output_group.create_group('prediction')
            pred_data = np.array(source_group['prediction/s0'])
            pred_group.create_dataset(
                's0',
                data=pred_data,
                dtype=pred_data.dtype,
                compression='gzip',
                compression_opts=6,
                chunks=(56, 56, 56)
            )
            if 'multiscales' in source_group['prediction'].attrs:
                pred_group.attrs['multiscales'] = source_group['prediction'].attrs['multiscales']

            # Copy metadata
            for key, value in source_group.attrs.items():
                output_group.attrs[key] = value

            # Add source file info
            output_group.attrs['source_file'] = zarr_file.name

            total_corrections += 1

        print(f"  ✓ Copied {len(correction_ids)} corrections from {zarr_file.name}")

    # Add global metadata
    output_root.attrs.update({
        'combined_date': datetime.now().isoformat(),
        'num_corrections': total_corrections,
        'num_source_files': len(zarr_files),
        'description': 'Combined sparse point corrections for partial annotation training'
    })

    print()
    print("="*60)
    print(f"✓ Combined {total_corrections} corrections into {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
