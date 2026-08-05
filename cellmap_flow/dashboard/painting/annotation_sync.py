"""
Annotation syncing from MinIO to local filesystem.

Handles periodic and on-demand syncing of painted annotations from
MinIO (where Neuroglancer writes) back to local disk for training.
"""

import json
import logging
import os
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import s3fs
import zarr

from .minio_manager import minio_state, annotation_volumes
from .zarr_utils import create_correction_zarr

logger = logging.getLogger(__name__)


def sync_all_annotations_from_minio(force: bool = True):
    """Sync all annotations from MinIO to local disk.

    Returns:
        Number of annotations synced, or -1 if MinIO is not initialized.
    """
    if not minio_state.get("ip") or not minio_state.get("port"):
        logger.info("MinIO not initialized, skipping annotation sync")
        return -1

    logger.info(f"Syncing annotations from MinIO before training/restart (force={force})...")
    s3 = s3fs.S3FileSystem(
        anon=False,
        key='minio',
        secret='minio123',
        client_kwargs={
            'endpoint_url': f"http://{minio_state['ip']}:{minio_state['port']}",
            'region_name': 'us-east-1'
        }
    )
    zarrs = s3.ls(minio_state['bucket'])
    zarr_ids = [Path(c).name.replace('.zarr', '') for c in zarrs if c.endswith('.zarr')]
    synced = 0
    for zid in zarr_ids:
        try:
            zarr_name = f"{zid}.zarr"
            attrs_path = f"{minio_state['bucket']}/{zarr_name}/.zattrs"
            if s3.exists(attrs_path):
                root_attrs = json.loads(s3.cat(attrs_path))
                if root_attrs.get("type") == "annotation_volume":
                    if sync_annotation_volume_from_minio(zid, force=force):
                        synced += 1
                    continue
        except Exception:
            pass
        if sync_annotation_from_minio(zid, force=force):
            synced += 1
    logger.info(f"Synced {synced}/{len(zarr_ids)} annotations")
    return synced


def sync_annotation_from_minio(crop_id, force=False):
    """
    Sync a single annotation crop from MinIO to local filesystem.

    Args:
        crop_id: Crop ID to sync (e.g., "5d291ea8-20260212-132326")
        force: Force sync even if not modified

    Returns:
        bool: True if synced successfully
    """
    if not minio_state["ip"] or not minio_state["port"] or not minio_state["output_base"]:
        logger.warning("MinIO not initialized or no output base set, skipping sync")
        return False

    try:
        # Setup S3 filesystem
        s3 = s3fs.S3FileSystem(
            anon=False,
            key='minio',
            secret='minio123',
            client_kwargs={
                'endpoint_url': f"http://{minio_state['ip']}:{minio_state['port']}",
                'region_name': 'us-east-1'
            }
        )

        # Check if annotation has been modified
        zarr_name = f"{crop_id}.zarr"
        src_path = f"{minio_state['bucket']}/{zarr_name}/annotation"
        dst_path = Path(minio_state['output_base']) / zarr_name / "annotation"

        # Check if source exists
        if not s3.exists(src_path):
            logger.debug(f"Source annotation not found in MinIO: {src_path}")
            return False

        # Check modification time
        try:
            s3_info = s3.info(f"{src_path}/s0/0.0.0")
            s3_mtime = s3_info.get('LastModified', None)

            # Check if we've synced this before
            last_sync = minio_state["last_sync"].get(crop_id, None)

            if not force and last_sync and s3_mtime and s3_mtime <= last_sync:
                # Not modified since last sync
                return False
        except Exception as e:
            logger.debug(f"Could not check modification time: {e}")
            # Continue with sync if we can't check mtime

        # Perform sync using zarr
        logger.info(f"Syncing annotation for {crop_id} from MinIO to local")

        src_store = s3fs.S3Map(root=src_path, s3=s3)
        src_group = zarr.open_group(store=src_store, mode='r')

        dst_store = zarr.DirectoryStore(str(dst_path))
        dst_group = zarr.open_group(store=dst_store, mode='a')

        # Copy all arrays
        for key in src_group.array_keys():
            src_array = src_group[key]
            dst_array = dst_group.create_dataset(
                key,
                shape=src_array.shape,
                chunks=src_array.chunks,
                dtype=src_array.dtype,
                overwrite=True
            )
            dst_array[:] = src_array[:]
            dst_array.attrs.update(src_array.attrs)

        # Copy group attributes
        dst_group.attrs.update(src_group.attrs)

        # Update last sync time
        minio_state["last_sync"][crop_id] = datetime.now()

        logger.info(f"Successfully synced annotation for {crop_id}")
        return True

    except Exception as e:
        logger.error(f"Error syncing annotation for {crop_id}: {e}")
        logger.error(traceback.format_exc())
        return False


def _get_volume_metadata(volume_id, zarr_path=None):
    """
    Get volume metadata from in-memory cache or reconstruct from zarr attrs.

    Used for server restart recovery -- if annotation_volumes dict was lost,
    reconstruct metadata from the zarr's stored attributes.
    """
    if volume_id in annotation_volumes:
        return annotation_volumes[volume_id]

    # Reconstruct from zarr
    if zarr_path is None:
        return None

    try:
        root = zarr.open(zarr_path, mode="r")
        attrs = dict(root.attrs)
        if attrs.get("type") != "annotation_volume":
            return None

        metadata = {
            "zarr_path": zarr_path,
            "model_name": attrs.get("model_name", ""),
            "output_size": attrs.get("chunk_size", [56, 56, 56]),
            "input_size": attrs.get("input_size", [178, 178, 178]),
            "input_voxel_size": attrs.get("input_voxel_size", [16, 16, 16]),
            "output_voxel_size": attrs.get("output_voxel_size", [16, 16, 16]),
            "dataset_path": attrs.get("dataset_path", ""),
            "dataset_offset_nm": attrs.get("dataset_offset_nm", [0, 0, 0]),
            "corrections_dir": str(Path(zarr_path).parent),
            "extracted_chunks": set(),
        }
        # Cache it
        annotation_volumes[volume_id] = metadata
        return metadata
    except Exception as e:
        logger.error(f"Error reconstructing volume metadata for {volume_id}: {e}")
        return None


def extract_correction_from_chunk(volume_id, chunk_indices, volume_metadata):
    """
    Extract a correction entry from a single annotated chunk in a sparse volume.

    Reads the annotation chunk, extracts raw data with context padding, and
    creates a standard correction zarr entry compatible with CorrectionDataset.

    Args:
        volume_id: Volume identifier
        chunk_indices: Tuple (cz, cy, cx) of chunk indices
        volume_metadata: Volume metadata dict

    Returns:
        bool: True if correction was created (chunk had annotations)
    """
    from cellmap_flow.image_data_interface import ImageDataInterface
    from funlib.geometry import Roi, Coordinate

    cz, cy, cx = chunk_indices
    chunk_size = np.array(volume_metadata["output_size"])
    output_voxel_size = np.array(volume_metadata["output_voxel_size"])
    input_size = np.array(volume_metadata["input_size"])
    input_voxel_size = np.array(volume_metadata["input_voxel_size"])
    dataset_offset_nm = np.array(volume_metadata["dataset_offset_nm"])
    corrections_dir = volume_metadata["corrections_dir"]

    # Read annotation data from the local synced volume
    vol_zarr_path = volume_metadata["zarr_path"]
    vol = zarr.open(vol_zarr_path, mode="r")

    z_start = cz * chunk_size[0]
    y_start = cy * chunk_size[1]
    x_start = cx * chunk_size[2]

    annotation_data = vol["annotation/s0"][
        z_start : z_start + chunk_size[0],
        y_start : y_start + chunk_size[1],
        x_start : x_start + chunk_size[2],
    ]

    # Skip if all zeros (unannotated or erased)
    if not np.any(annotation_data):
        return False

    # Compute physical position of this chunk's center
    chunk_offset_nm = dataset_offset_nm + np.array(
        [z_start, y_start, x_start]
    ) * output_voxel_size
    chunk_center_nm = chunk_offset_nm + (chunk_size * output_voxel_size) / 2

    # Extract raw data with full context padding
    read_shape_nm = input_size * input_voxel_size
    raw_roi = Roi(
        offset=Coordinate(chunk_center_nm - read_shape_nm / 2),
        shape=Coordinate(read_shape_nm),
    )

    logger.info(
        f"Extracting raw for chunk ({cz},{cy},{cx}): "
        f"ROI offset={raw_roi.offset}, shape={raw_roi.shape}"
    )

    idi = ImageDataInterface(
        volume_metadata["dataset_path"], voxel_size=input_voxel_size
    )
    raw_data = idi.to_ndarray_ts(raw_roi)

    # Create correction entry using existing function
    correction_id = f"{volume_id}_chunk_{cz}_{cy}_{cx}"
    correction_zarr_path = os.path.join(corrections_dir, f"{correction_id}.zarr")

    raw_offset_voxels = (
        (chunk_center_nm - read_shape_nm / 2) / input_voxel_size
    ).astype(int)
    annotation_offset_voxels = (chunk_offset_nm / output_voxel_size).astype(int)

    success, zarr_info = create_correction_zarr(
        zarr_path=correction_zarr_path,
        raw_crop_shape=input_size,
        raw_voxel_size=input_voxel_size,
        raw_offset=raw_offset_voxels,
        annotation_crop_shape=chunk_size,
        annotation_voxel_size=output_voxel_size,
        annotation_offset=annotation_offset_voxels,
        dataset_path=volume_metadata["dataset_path"],
        model_name=volume_metadata["model_name"],
        output_channels=1,
        raw_dtype=str(raw_data.dtype),
        create_mask=False,
    )

    if not success:
        logger.error(f"Failed to create correction zarr for chunk ({cz},{cy},{cx})")
        return False

    # Write data
    corr_zarr = zarr.open(correction_zarr_path, mode="r+")
    corr_zarr["raw/s0"][:] = raw_data
    corr_zarr["annotation/s0"][:] = annotation_data

    # Mark as sparse volume source
    corr_zarr.attrs["source"] = "sparse_volume"
    corr_zarr.attrs["volume_id"] = volume_id
    corr_zarr.attrs["chunk_indices"] = [cz, cy, cx]

    logger.info(f"Created correction {correction_id} from chunk ({cz},{cy},{cx})")
    return True


def sync_annotation_volume_from_minio(volume_id, force=False):
    """
    Sync an annotation volume from MinIO, detect annotated chunks, extract corrections.

    Steps:
    1. Sync the full annotation zarr from MinIO to local disk
    2. List chunk files in MinIO to find annotated chunks
    3. For each new annotated chunk, extract raw data and create correction entry

    Args:
        volume_id: Volume identifier
        force: Force re-extraction of all chunks

    Returns:
        bool: True if any corrections were created
    """
    if not minio_state["ip"] or not minio_state["port"] or not minio_state["output_base"]:
        logger.warning("MinIO not initialized, skipping volume sync")
        return False

    try:
        # Get volume metadata (from cache or reconstruct from zarr)
        zarr_name = f"{volume_id}.zarr"
        local_zarr_path = os.path.join(minio_state["output_base"], zarr_name)
        volume_meta = _get_volume_metadata(volume_id, local_zarr_path)

        if volume_meta is None:
            logger.warning(f"No metadata for volume {volume_id}, skipping")
            return False

        # Setup S3 filesystem
        s3 = s3fs.S3FileSystem(
            anon=False,
            key="minio",
            secret="minio123",
            client_kwargs={
                "endpoint_url": f"http://{minio_state['ip']}:{minio_state['port']}",
                "region_name": "us-east-1",
            },
        )

        bucket = minio_state["bucket"]
        src_annotation_path = f"{bucket}/{zarr_name}/annotation"

        # Check if annotation group exists in MinIO
        if not s3.exists(src_annotation_path):
            logger.debug(f"No annotation group in MinIO for {volume_id}")
            return False

        # Sync the full annotation volume from MinIO to local
        dst_annotation_path = Path(local_zarr_path) / "annotation"
        dst_annotation_path.mkdir(parents=True, exist_ok=True)

        src_store = s3fs.S3Map(root=src_annotation_path, s3=s3)
        src_group = zarr.open_group(store=src_store, mode="r")

        dst_store = zarr.DirectoryStore(str(dst_annotation_path))
        dst_group = zarr.open_group(store=dst_store, mode="a")

        # Copy array metadata and attributes
        for key in src_group.array_keys():
            src_array = src_group[key]
            # Only create array structure if it doesn't exist
            if key not in dst_group:
                dst_group.create_dataset(
                    key,
                    shape=src_array.shape,
                    chunks=src_array.chunks,
                    dtype=src_array.dtype,
                    fill_value=0,
                    overwrite=True,
                )
            dst_group[key].attrs.update(src_array.attrs)
        dst_group.attrs.update(src_group.attrs)

        # List chunk files in MinIO to find which chunks have been written
        s0_path = f"{bucket}/{zarr_name}/annotation/s0"
        try:
            chunk_files = s3.ls(s0_path)
        except FileNotFoundError:
            logger.debug(f"No chunks yet for volume {volume_id}")
            minio_state["last_sync"][volume_id] = datetime.now()
            return False

        # Parse chunk indices from filenames (format: z.y.x)
        annotated_chunks = []
        for f in chunk_files:
            basename = Path(f).name
            if re.match(r"^\d+\.\d+\.\d+$", basename):
                cz, cy, cx = map(int, basename.split("."))
                annotated_chunks.append((cz, cy, cx))

        if not annotated_chunks:
            logger.debug(f"No annotated chunks found for volume {volume_id}")
            minio_state["last_sync"][volume_id] = datetime.now()
            return False

        # Sync individual chunk data from MinIO to local
        for cz, cy, cx in annotated_chunks:
            chunk_key = f"{cz}.{cy}.{cx}"
            src_chunk_path = f"{s0_path}/{chunk_key}"
            dst_chunk_path = dst_annotation_path / "s0" / chunk_key
            dst_chunk_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.get(src_chunk_path, str(dst_chunk_path))
            except Exception as e:
                logger.debug(f"Error syncing chunk {chunk_key}: {e}")

        logger.info(
            f"Synced {len(annotated_chunks)} chunks for volume {volume_id}"
        )

        # Extract corrections for new/updated chunks
        extracted_chunks = volume_meta.get("extracted_chunks", set())
        created_any = False

        for chunk_idx in annotated_chunks:
            if not force and chunk_idx in extracted_chunks:
                continue

            try:
                created = extract_correction_from_chunk(
                    volume_id, chunk_idx, volume_meta
                )
                if created:
                    extracted_chunks.add(chunk_idx)
                    created_any = True
            except Exception as e:
                logger.error(
                    f"Error extracting correction for chunk {chunk_idx}: {e}"
                )
                logger.error(traceback.format_exc())

        # Update tracked state
        volume_meta["extracted_chunks"] = extracted_chunks
        minio_state["last_sync"][volume_id] = datetime.now()

        if created_any:
            logger.info(
                f"Created corrections for volume {volume_id}: "
                f"{len(extracted_chunks)} total chunks extracted"
            )

        return created_any

    except Exception as e:
        logger.error(f"Error syncing annotation volume {volume_id}: {e}")
        logger.error(traceback.format_exc())
        return False


def periodic_sync_annotations():
    """Background thread function to periodically sync annotations from MinIO."""
    logger.info("Starting periodic annotation sync thread")

    while True:
        try:
            time.sleep(30)  # Sync every 30 seconds

            if not minio_state["output_base"]:
                continue

            # Get list of all crops in MinIO
            if not minio_state["ip"] or not minio_state["port"]:
                continue

            try:
                s3 = s3fs.S3FileSystem(
                    anon=False,
                    key='minio',
                    secret='minio123',
                    client_kwargs={
                        'endpoint_url': f"http://{minio_state['ip']}:{minio_state['port']}",
                        'region_name': 'us-east-1'
                    }
                )

                # List all zarrs in bucket
                crops = s3.ls(minio_state['bucket'])
                zarr_ids = [Path(c).name.replace('.zarr', '') for c in crops if c.endswith('.zarr')]

                # Sync each zarr (route volumes vs crops)
                for zarr_id in zarr_ids:
                    try:
                        # Check if this is an annotation volume
                        zarr_name = f"{zarr_id}.zarr"
                        attrs_path = f"{minio_state['bucket']}/{zarr_name}/.zattrs"
                        if s3.exists(attrs_path):
                            root_attrs = json.loads(s3.cat(attrs_path))
                            if root_attrs.get("type") == "annotation_volume":
                                sync_annotation_volume_from_minio(zarr_id)
                                continue
                    except Exception:
                        pass
                    # Default: crop sync
                    sync_annotation_from_minio(zarr_id, force=False)

            except Exception as e:
                logger.debug(f"Error in periodic sync: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in sync thread: {e}")


def start_periodic_sync():
    """Start the periodic annotation sync thread if not already running."""
    if minio_state["sync_thread"] is None or not minio_state["sync_thread"].is_alive():
        thread = threading.Thread(target=periodic_sync_annotations, daemon=True)
        thread.start()
        minio_state["sync_thread"] = thread
        logger.info("Started periodic annotation sync thread")
