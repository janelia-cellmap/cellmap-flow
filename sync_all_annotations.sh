#!/bin/bash
# Sync all annotations from MinIO to local disk

MINIO_ENDPOINT="10.36.107.11:9000"
OUTPUT_BASE="corrections/painting_bw.zarr"

echo "Syncing all annotations from MinIO to $OUTPUT_BASE"

# List all crops in MinIO
mc ls localminio/annotations/ | awk '{print $5}' | while read crop_dir; do
    if [ ! -z "$crop_dir" ]; then
        crop_id="${crop_dir%.zarr/}"
        echo "Syncing $crop_id..."
        python sync_annotations.py "$crop_id" "$MINIO_ENDPOINT" "$OUTPUT_BASE"
    fi
done

echo "✓ All annotations synced!"
