#!/usr/bin/env python
"""
Sync zarr files between disk and MinIO.

Usage:
    # Upload changes from disk to MinIO
    python minio_sync.py /path/to/zarr --upload --bucket tmp

    # Download changes from MinIO to disk
    python minio_sync.py /path/to/zarr --download --bucket tmp

    # Bidirectional sync (automatic)
    python minio_sync.py /path/to/zarr --bucket tmp
"""

import argparse
import subprocess
from pathlib import Path
import time


def sync_to_minio(local_path, bucket, object_prefix=None):
    """Upload local changes to MinIO."""
    local_path = Path(local_path)

    if object_prefix is None:
        object_prefix = local_path.name

    target = f"local/{bucket}/{object_prefix}"

    print(f"Uploading {local_path} -> {target}...")
    cmd = ["mc", "mirror", "--overwrite", str(local_path), target]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Upload failed: {result.stderr}")
        return False

    # Parse output for statistics
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Total' in line or 'Transferred' in line:
            print(f"  {line}")

    print(f"✓ Upload complete")
    return True


def sync_from_minio(local_path, bucket, object_prefix=None):
    """Download changes from MinIO to local."""
    local_path = Path(local_path)

    if object_prefix is None:
        object_prefix = local_path.name

    source = f"local/{bucket}/{object_prefix}"

    print(f"Downloading {source} -> {local_path}...")
    cmd = ["mc", "mirror", "--overwrite", source, str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Download failed: {result.stderr}")
        return False

    # Parse output for statistics
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Total' in line or 'Transferred' in line:
            print(f"  {line}")

    print(f"✓ Download complete")
    return True


def watch_and_sync(local_path, bucket, object_prefix=None, interval=5):
    """Watch for changes and sync periodically."""
    print(f"Watching {local_path} for changes...")
    print(f"Syncing to MinIO bucket '{bucket}' every {interval} seconds")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            sync_to_minio(local_path, bucket, object_prefix)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nStopping watcher...")


def main():
    parser = argparse.ArgumentParser(
        description="Sync zarr files between disk and MinIO"
    )
    parser.add_argument(
        "path",
        help="Path to zarr file/directory on disk"
    )
    parser.add_argument(
        "--bucket",
        default="tmp",
        help="MinIO bucket name (default: tmp)"
    )
    parser.add_argument(
        "--prefix",
        help="Object prefix in MinIO (default: same as directory name)"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload from disk to MinIO"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download from MinIO to disk"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for changes and sync continuously"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Sync interval in seconds when watching (default: 5)"
    )

    args = parser.parse_args()

    # Validate path
    local_path = Path(args.path).expanduser().resolve()

    if args.watch:
        # Continuous sync
        watch_and_sync(local_path, args.bucket, args.prefix, args.interval)
    elif args.upload:
        # One-time upload
        sync_to_minio(local_path, args.bucket, args.prefix)
    elif args.download:
        # One-time download
        sync_from_minio(local_path, args.bucket, args.prefix)
    else:
        # Default: upload
        print("No direction specified, defaulting to upload")
        sync_to_minio(local_path, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
