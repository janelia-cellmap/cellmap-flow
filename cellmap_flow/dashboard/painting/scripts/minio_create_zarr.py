#!/usr/bin/env python
"""
Create a new empty zarr array and upload it to MinIO.

Usage:
    python minio_create_zarr.py /path/on/disk/new_annotation.zarr --shape 512,512,512
"""

import argparse
import subprocess
import json
from pathlib import Path
import zarr
import numpy as np


def create_empty_zarr(path, shape, chunks=None, dtype='uint8', compressor='blosc'):
    """Create an empty zarr array on disk."""
    path = Path(path)

    if chunks is None:
        # Default: 64x64x64 chunks
        chunks = tuple(min(64, s) for s in shape)

    # Set up compressor
    if compressor and compressor.lower() != 'none':
        if compressor.lower() == 'gzip':
            comp = zarr.Zlib(level=3)
        elif compressor.lower() == 'blosc':
            comp = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        else:
            comp = zarr.Zlib(level=3)  # default to gzip
    else:
        comp = None

    # Create zarr array
    z = zarr.open(
        str(path),
        mode='w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=comp
    )

    # Fill with zeros
    print(f"Filling array with zeros...")
    z[:] = 0

    print(f"✓ Created zarr array at {path}")
    print(f"  Shape: {shape}")
    print(f"  Chunks: {chunks}")
    print(f"  Dtype: {dtype}")
    print(f"  Compressor: {compressor}")
    print(f"  Filled with zeros: Yes")

    return path


def upload_to_minio(local_path, bucket, object_prefix=None):
    """Upload zarr to MinIO bucket."""
    local_path = Path(local_path)

    if object_prefix is None:
        object_prefix = local_path.name

    # Upload using mc mirror
    target = f"local/{bucket}/{object_prefix}"

    print(f"\nUploading to MinIO...")
    print(f"  Source: {local_path}")
    print(f"  Target: {target}")

    cmd = ["mc", "mirror", "--overwrite", str(local_path), target]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Upload failed: {result.stderr}")
        return False

    print(f"✓ Uploaded to MinIO")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create empty zarr and upload to MinIO"
    )
    parser.add_argument(
        "path",
        help="Path where zarr will be created on disk"
    )
    parser.add_argument(
        "--shape",
        required=True,
        help="Shape of the array (comma-separated, e.g. 512,512,512)"
    )
    parser.add_argument(
        "--chunks",
        help="Chunk size (comma-separated, default: 64,64,64)"
    )
    parser.add_argument(
        "--dtype",
        default="uint8",
        help="Data type (default: uint8)"
    )
    parser.add_argument(
        "--compressor",
        default="blosc",
        help="Compressor: gzip, blosc, or none (default: blosc)"
    )
    parser.add_argument(
        "--bucket",
        default="tmp",
        help="MinIO bucket name (default: tmp)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Create zarr but don't upload to MinIO"
    )

    args = parser.parse_args()

    # Parse shape
    shape = tuple(int(x) for x in args.shape.split(','))

    # Parse chunks if provided
    chunks = None
    if args.chunks:
        chunks = tuple(int(x) for x in args.chunks.split(','))

    # Handle compressor
    compressor = None if args.compressor.lower() == 'none' else args.compressor

    # Create zarr
    zarr_path = create_empty_zarr(
        args.path,
        shape=shape,
        chunks=chunks,
        dtype=args.dtype,
        compressor=compressor
    )

    # Upload to MinIO
    if not args.no_upload:
        success = upload_to_minio(zarr_path, args.bucket)
        if success:
            print(f"\n✓ Done! Access via MinIO at:")
            print(f"  http://<minio-ip>:<port>/{args.bucket}/{zarr_path.name}")
    else:
        print(f"\n✓ Zarr created at {zarr_path}")


if __name__ == "__main__":
    main()
