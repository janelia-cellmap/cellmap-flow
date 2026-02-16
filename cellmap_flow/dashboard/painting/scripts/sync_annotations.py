#!/usr/bin/env python3
"""
Sync annotations from MinIO storage to local zarr format.
"""
import zarr
import s3fs
import sys
from pathlib import Path

def sync_annotation(crop_id, minio_endpoint, output_base):
    """
    Copy annotation data from MinIO to local zarr.

    Args:
        crop_id: Crop ID (e.g., "5d291ea8-20260212-132326")
        minio_endpoint: MinIO endpoint (e.g., "10.36.107.11:9000")
        output_base: Base output directory
    """
    # Setup S3 filesystem
    s3 = s3fs.S3FileSystem(
        anon=False,
        key='minio',
        secret='minio123',
        client_kwargs={
            'endpoint_url': f'http://{minio_endpoint}',
            'region_name': 'us-east-1'
        }
    )

    # Source and destination paths
    zarr_name = f"{crop_id}.zarr"
    src_path = f"annotations/{zarr_name}/annotation"
    dst_path = Path(output_base) / zarr_name / "annotation"

    print(f"Syncing from MinIO: s3://{src_path}")
    print(f"         to local: {dst_path}")

    # Open source zarr from MinIO
    src_store = s3fs.S3Map(root=src_path, s3=s3)
    src_group = zarr.open_group(store=src_store, mode='r')

    # Create destination zarr on local filesystem
    dst_store = zarr.DirectoryStore(str(dst_path))
    dst_group = zarr.open_group(store=dst_store, mode='a')

    # Copy all arrays
    for key in src_group.array_keys():
        print(f"  Copying array: {key}")
        src_array = src_group[key]

        # Create or overwrite destination array
        dst_array = dst_group.create_dataset(
            key,
            shape=src_array.shape,
            chunks=src_array.chunks,
            dtype=src_array.dtype,
            overwrite=True
        )

        # Copy data
        dst_array[:] = src_array[:]

        # Copy attributes
        dst_array.attrs.update(src_array.attrs)

    # Copy group attributes
    dst_group.attrs.update(src_group.attrs)

    print(f"✓ Successfully synced annotation for {crop_id}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: sync_annotations.py <crop_id> <minio_endpoint> <output_base>")
        print("Example: sync_annotations.py 5d291ea8-20260212-132326 10.101.10.86:9000 /path/to/corrections/painting_bw.zarr")
        sys.exit(1)

    sync_annotation(sys.argv[1], sys.argv[2], sys.argv[3])
