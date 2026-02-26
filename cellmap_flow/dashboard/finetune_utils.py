"""
Helper functions for finetuning annotation workflows.

Handles MinIO server management, annotation zarr creation, and
periodic synchronization of annotations between MinIO and local disk.
"""

import json
import os
import re
import socket
import subprocess
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import s3fs
import zarr

from cellmap_flow.globals import g

minio_state = g.minio_state
annotation_volumes = g.annotation_volumes
output_sessions = g.output_sessions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def get_or_create_session_path(base_output_path: str) -> str:
    """
    Get or create a timestamped session directory for the given base output path.

    If a session already exists for this base path, reuse it.
    Otherwise, create a new timestamped subdirectory.

    Args:
        base_output_path: Base output directory (e.g., "output/to/here")

    Returns:
        Timestamped session path (e.g., "output/to/here/20260213_123456")
    """
    base_output_path = os.path.expanduser(base_output_path)

    if base_output_path not in output_sessions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(base_output_path, timestamp)
        output_sessions[base_output_path] = session_path
        logger.info(f"Created new session path: {session_path}")

    return output_sessions[base_output_path]


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def get_local_ip():
    """Get the local IP address for MinIO server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_available_port(start_port=9000):
    """Find an available port pair for MinIO server (API on port, console on port+1)."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s1:
                s1.bind(("", port))
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.bind(("", port + 1))
                    return port
        except OSError:
            continue
    raise RuntimeError("Could not find available port for MinIO")


# ---------------------------------------------------------------------------
# Zarr creation
# ---------------------------------------------------------------------------

def create_correction_zarr(
    zarr_path,
    raw_crop_shape,
    raw_voxel_size,
    raw_offset,
    annotation_crop_shape,
    annotation_voxel_size,
    annotation_offset,
    dataset_path,
    model_name,
    output_channels,
    raw_dtype="uint8",
    create_mask=False,
):
    """
    Create a correction zarr with OME-NGFF v0.4 metadata.

    Structure:
        crop_id.zarr/
            raw/s0/          (uint8, shape=raw_crop_shape)
            annotation/s0/   (uint8, shape=annotation_crop_shape)
            mask/s0/         (optional, uint8, shape=annotation_crop_shape)
            .zattrs          (metadata)

    Returns:
        (success: bool, info: str)
    """
    try:
        def add_ome_ngff_metadata(group, name, voxel_size, translation_offset=None):
            """Add OME-NGFF v0.4 metadata."""
            if translation_offset is not None:
                physical_translation = [
                    float(o * v) for o, v in zip(translation_offset, voxel_size)
                ]
            else:
                physical_translation = [0.0, 0.0, 0.0]

            transforms = [{"type": "scale", "scale": [float(v) for v in voxel_size]}]

            if translation_offset is not None:
                transforms.append(
                    {"type": "translation", "translation": physical_translation}
                )

            group.attrs["multiscales"] = [
                {
                    "version": "0.4",
                    "name": name,
                    "axes": [
                        {"name": "z", "type": "space", "unit": "nanometer"},
                        {"name": "y", "type": "space", "unit": "nanometer"},
                        {"name": "x", "type": "space", "unit": "nanometer"},
                    ],
                    "datasets": [
                        {"path": "s0", "coordinateTransformations": transforms}
                    ],
                }
            ]

        root = zarr.open(zarr_path, mode="w")

        # Raw group
        raw_group = root.create_group("raw")
        raw_group.create_dataset(
            "s0",
            shape=tuple(raw_crop_shape),
            chunks=(64, 64, 64),
            dtype=raw_dtype,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )
        add_ome_ngff_metadata(raw_group, "raw", raw_voxel_size, raw_offset)

        # Annotation group
        annotation_group = root.create_group("annotation")
        annotation_group.create_dataset(
            "s0",
            shape=tuple(annotation_crop_shape),
            chunks=(64, 64, 64),
            dtype="uint8",
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )
        add_ome_ngff_metadata(
            annotation_group, "annotation", annotation_voxel_size, annotation_offset
        )

        # Optional mask group
        if create_mask:
            mask_group = root.create_group("mask")
            mask_group.create_dataset(
                "s0",
                shape=tuple(annotation_crop_shape),
                chunks=(64, 64, 64),
                dtype="uint8",
                compressor=zarr.Blosc(
                    cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE
                ),
                fill_value=0,
            )
            add_ome_ngff_metadata(
                mask_group, "mask", annotation_voxel_size, annotation_offset
            )

        # Root metadata
        root.attrs["roi"] = {
            "raw_offset": (
                raw_offset.tolist()
                if hasattr(raw_offset, "tolist")
                else list(raw_offset)
            ),
            "raw_shape": (
                raw_crop_shape.tolist()
                if hasattr(raw_crop_shape, "tolist")
                else list(raw_crop_shape)
            ),
            "annotation_offset": (
                annotation_offset.tolist()
                if hasattr(annotation_offset, "tolist")
                else list(annotation_offset)
            ),
            "annotation_shape": (
                annotation_crop_shape.tolist()
                if hasattr(annotation_crop_shape, "tolist")
                else list(annotation_crop_shape)
            ),
        }
        root.attrs["raw_voxel_size"] = (
            raw_voxel_size.tolist()
            if hasattr(raw_voxel_size, "tolist")
            else list(raw_voxel_size)
        )
        root.attrs["annotation_voxel_size"] = (
            annotation_voxel_size.tolist()
            if hasattr(annotation_voxel_size, "tolist")
            else list(annotation_voxel_size)
        )
        root.attrs["model_name"] = model_name
        root.attrs["dataset_path"] = dataset_path
        root.attrs["created_at"] = datetime.now().isoformat()

        logger.info(f"Created correction zarr at {zarr_path}")

        return True, zarr_path

    except Exception as e:
        logger.error(f"Error creating zarr: {e}")
        return False, str(e)


def create_annotation_volume_zarr(
    zarr_path,
    dataset_shape_voxels,
    output_voxel_size,
    dataset_offset_nm,
    chunk_size,
    dataset_path,
    model_name,
    input_size,
    input_voxel_size,
):
    """
    Create a sparse annotation volume zarr covering the full dataset extent.

    The volume has chunk_size = model output_size so each chunk maps to one
    training sample. Only metadata files are created (no chunk data), so the
    zarr is tiny regardless of dataset size.

    Label scheme: 0=unannotated (ignored), 1=background, 2=foreground.

    Returns:
        (success: bool, info: str)
    """
    try:
        root = zarr.open(zarr_path, mode="w")

        annotation_group = root.create_group("annotation")
        annotation_group.create_dataset(
            "s0",
            shape=tuple(dataset_shape_voxels),
            chunks=tuple(chunk_size),
            dtype="uint8",
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )

        # OME-NGFF v0.4 metadata with translation for dataset offset
        physical_translation = [float(o) for o in dataset_offset_nm]
        transforms = [
            {"type": "scale", "scale": [float(v) for v in output_voxel_size]},
            {"type": "translation", "translation": physical_translation},
        ]
        annotation_group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "annotation",
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [
                    {"path": "s0", "coordinateTransformations": transforms}
                ],
            }
        ]

        # Root metadata
        root.attrs["type"] = "annotation_volume"
        root.attrs["model_name"] = model_name
        root.attrs["dataset_path"] = dataset_path
        root.attrs["chunk_size"] = (
            chunk_size.tolist() if hasattr(chunk_size, "tolist") else list(chunk_size)
        )
        root.attrs["output_voxel_size"] = (
            output_voxel_size.tolist()
            if hasattr(output_voxel_size, "tolist")
            else list(output_voxel_size)
        )
        root.attrs["input_size"] = (
            input_size.tolist() if hasattr(input_size, "tolist") else list(input_size)
        )
        root.attrs["input_voxel_size"] = (
            input_voxel_size.tolist()
            if hasattr(input_voxel_size, "tolist")
            else list(input_voxel_size)
        )
        root.attrs["dataset_offset_nm"] = (
            dataset_offset_nm.tolist()
            if hasattr(dataset_offset_nm, "tolist")
            else list(dataset_offset_nm)
        )
        root.attrs["dataset_shape_voxels"] = (
            dataset_shape_voxels.tolist()
            if hasattr(dataset_shape_voxels, "tolist")
            else list(dataset_shape_voxels)
        )
        root.attrs["created_at"] = datetime.now().isoformat()

        logger.info(
            f"Created annotation volume zarr at {zarr_path} "
            f"(shape={dataset_shape_voxels}, chunks={chunk_size})"
        )

        return True, zarr_path

    except Exception as e:
        logger.error(f"Error creating annotation volume zarr: {e}")
        return False, str(e)


# ---------------------------------------------------------------------------
# MinIO management
# ---------------------------------------------------------------------------

def ensure_minio_serving(zarr_path, crop_id, output_base_dir=None):
    """
    Ensure MinIO is running and upload zarr file.

    Args:
        zarr_path: Path to zarr file to upload
        crop_id: Unique identifier for the crop
        output_base_dir: Base output directory (MinIO will use output_base_dir/.minio)

    Returns:
        MinIO URL for the zarr file
    """
    if minio_state["process"] is None or minio_state["process"].poll() is not None:
        # Determine MinIO storage location
        if output_base_dir:
            minio_root = Path(output_base_dir) / ".minio"
            minio_state["output_base"] = output_base_dir
        else:
            minio_root = Path("~/.minio-server").expanduser()
            minio_state["output_base"] = None

        minio_root.mkdir(parents=True, exist_ok=True)
        minio_state["minio_root"] = str(minio_root)

        ip = get_local_ip()
        port = find_available_port()

        env = os.environ.copy()
        env["MINIO_ROOT_USER"] = "minio"
        env["MINIO_ROOT_PASSWORD"] = "minio123"
        env["MINIO_API_CORS_ALLOW_ORIGIN"] = "*"

        minio_cmd = [
            "minio",
            "server",
            str(minio_root),
            "--address",
            f"{ip}:{port}",
            "--console-address",
            f"{ip}:{port+1}",
        ]

        logger.info(f"Starting MinIO server at {ip}:{port}")
        minio_proc = subprocess.Popen(
            minio_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(3)

        if minio_proc.poll() is not None:
            stderr = minio_proc.stderr.read().decode() if minio_proc.stderr else ""
            raise RuntimeError(f"MinIO failed to start: {stderr}")

        minio_state["process"] = minio_proc
        minio_state["port"] = port
        minio_state["ip"] = ip

        logger.info(f"MinIO started (PID: {minio_proc.pid})")

        # Configure mc client
        subprocess.run(
            [
                "mc",
                "alias",
                "set",
                "myserver",
                f"http://{ip}:{port}",
                "minio",
                "minio123",
            ],
            check=True,
            capture_output=True,
        )

        # Create bucket if needed
        result = subprocess.run(
            ["mc", "mb", f"myserver/{minio_state['bucket']}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "already" not in result.stderr.lower():
            logger.warning(f"Bucket creation returned: {result.stderr}")

        # Make bucket public
        subprocess.run(
            ["mc", "anonymous", "set", "public", f"myserver/{minio_state['bucket']}"],
            check=True,
            capture_output=True,
        )

        # Start periodic sync thread
        start_periodic_sync()

    # Upload zarr file
    zarr_name = Path(zarr_path).name
    target = f"myserver/{minio_state['bucket']}/{zarr_name}"

    logger.info(f"Uploading {zarr_name} to MinIO")
    result = subprocess.run(
        ["mc", "mirror", "--overwrite", zarr_path, target],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to upload to MinIO: {result.stderr}")

    logger.info(f"Uploaded {zarr_name} to MinIO")

    minio_url = (
        f"http://{minio_state['ip']}:{minio_state['port']}"
        f"/{minio_state['bucket']}/{zarr_name}"
    )
    return minio_url


# ---------------------------------------------------------------------------
# S3 / MinIO sync helpers
# ---------------------------------------------------------------------------

def _safe_epoch_timestamp(value) -> float:
    """Convert LastModified-like values to epoch seconds, best-effort."""
    if value is None:
        return 0.0
    if isinstance(value, datetime):
        return float(value.timestamp())
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value))
        return float(parsed.timestamp())
    except Exception:
        return 0.0


def _get_sync_worker_count() -> int:
    """
    Determine thread count for chunk sync.

    Prefer scheduler-provided CPU counts (e.g., LSF bsub -n), then fall back
    to process CPU affinity / system CPU count.
    """
    env_candidates = [
        "LSB_DJOB_NUMPROC",
        "LSB_MAX_NUM_PROCESSORS",
        "NSLOTS",
        "SLURM_CPUS_PER_TASK",
        "OMP_NUM_THREADS",
    ]
    for key in env_candidates:
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            continue

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _copy_chunks_parallel(s3, copy_pairs):
    """
    Copy chunk files from MinIO in parallel.

    Args:
        s3: s3fs filesystem instance
        copy_pairs: list of (src_chunk_path, dst_chunk_path_str)
    """
    if not copy_pairs:
        return

    available_workers = _get_sync_worker_count()
    workers = max(1, min(len(copy_pairs), available_workers))

    def _copy_one(src_dst):
        src_chunk_path, dst_chunk_path = src_dst
        s3.get(src_chunk_path, dst_chunk_path)
        return src_chunk_path

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_copy_one, pair) for pair in copy_pairs]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.debug(f"Error syncing chunk in parallel copy: {e}")


def _make_s3_filesystem():
    """Create an s3fs filesystem pointed at the local MinIO instance."""
    return s3fs.S3FileSystem(
        anon=False,
        key="minio",
        secret="minio123",
        client_kwargs={
            "endpoint_url": f"http://{minio_state['ip']}:{minio_state['port']}",
            "region_name": "us-east-1",
        },
    )


def _sync_zarr_group_metadata(s3, src_path, dst_path):
    """Sync zarr group structure and metadata from S3 to local disk.

    Ensures destination arrays exist with correct shape/dtype and copies attrs.
    """
    src_store = s3fs.S3Map(root=src_path, s3=s3)
    src_group = zarr.open_group(store=src_store, mode="r")

    dst_store = zarr.DirectoryStore(str(dst_path))
    dst_group = zarr.open_group(store=dst_store, mode="a")

    for key in src_group.array_keys():
        src_array = src_group[key]
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


def _diff_and_sync_chunks(s3, s0_path, dst_s0_path, known_chunk_state, force=False):
    """Diff remote vs known chunk state and sync changed chunks to local disk.

    Returns:
        (changed_keys, removed_keys, remote_chunk_state)
    """
    try:
        chunk_files = s3.ls(s0_path)
    except FileNotFoundError:
        chunk_files = []

    remote_chunk_state = {}
    for chunk_file in chunk_files:
        chunk_key = Path(chunk_file).name
        if not re.match(r"^\d+\.\d+\.\d+$", chunk_key):
            continue
        try:
            info = s3.info(chunk_file)
            remote_chunk_state[chunk_key] = _safe_epoch_timestamp(info.get("LastModified"))
        except Exception:
            remote_chunk_state[chunk_key] = 0.0

    if force:
        changed_keys = list(remote_chunk_state.keys())
    else:
        changed_keys = [k for k, v in remote_chunk_state.items() if known_chunk_state.get(k) != v]
    removed_keys = [k for k in known_chunk_state if k not in remote_chunk_state]

    if not changed_keys and not removed_keys:
        return [], [], remote_chunk_state

    # Copy changed chunks
    dst_s0_path = Path(dst_s0_path)
    dst_s0_path.mkdir(parents=True, exist_ok=True)
    copy_pairs = [(f"{s0_path}/{k}", str(dst_s0_path / k)) for k in changed_keys]
    _copy_chunks_parallel(s3, copy_pairs)

    # Remove stale local chunks
    for k in removed_keys:
        local_chunk = dst_s0_path / k
        try:
            if local_chunk.exists():
                local_chunk.unlink()
        except Exception as e:
            logger.debug(f"Error removing stale chunk {k}: {e}")

    return changed_keys, removed_keys, remote_chunk_state


# ---------------------------------------------------------------------------
# Annotation sync (crop-based)
# ---------------------------------------------------------------------------

def sync_annotation_from_minio(crop_id, force=False):
    """
    Sync a single annotation crop from MinIO to local filesystem.

    Args:
        crop_id: Crop ID to sync
        force: Force sync even if not modified

    Returns:
        bool: True if synced successfully
    """
    if not minio_state["ip"] or not minio_state["port"] or not minio_state["output_base"]:
        return False

    try:
        s3 = _make_s3_filesystem()

        zarr_name = f"{crop_id}.zarr"
        src_path = f"{minio_state['bucket']}/{zarr_name}/annotation"
        dst_path = Path(minio_state["output_base"]) / zarr_name / "annotation"

        if not s3.exists(src_path):
            return False

        known_chunk_state = minio_state["chunk_sync_state"].get(crop_id, {})
        s0_path = f"{src_path}/s0"
        changed, removed, remote_chunk_state = _diff_and_sync_chunks(
            s3, s0_path, dst_path / "s0", known_chunk_state, force=force
        )

        if not changed and not removed:
            return False

        logger.info(
            f"Syncing annotation for {crop_id} "
            f"(changed={len(changed)}, removed={len(removed)})"
        )

        _sync_zarr_group_metadata(s3, src_path, dst_path)

        minio_state["last_sync"][crop_id] = datetime.now()
        minio_state["chunk_sync_state"][crop_id] = remote_chunk_state

        logger.info(f"Successfully synced annotation for {crop_id}")
        return True

    except Exception as e:
        logger.error(f"Error syncing annotation for {crop_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Annotation sync (full-dataset sync)
# ---------------------------------------------------------------------------

def sync_all_annotations_from_minio(force: bool = True):
    """Sync all annotations from MinIO to local disk.

    Returns:
        Number of annotations synced, or -1 if MinIO is not initialized.
    """
    if not minio_state.get("ip") or not minio_state.get("port"):
        logger.info("MinIO not initialized, skipping annotation sync")
        return -1

    logger.info(f"Syncing all annotations from MinIO (force={force})...")
    s3 = _make_s3_filesystem()
    zarrs = s3.ls(minio_state["bucket"])
    zarr_ids = [Path(c).name.replace(".zarr", "") for c in zarrs if c.endswith(".zarr")]
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


# ---------------------------------------------------------------------------
# Volume metadata helpers
# ---------------------------------------------------------------------------

def _get_volume_metadata(volume_id, zarr_path=None):
    """
    Get volume metadata from in-memory cache or reconstruct from zarr attrs.

    Used for server restart recovery -- if annotation_volumes dict was lost,
    reconstruct metadata from the zarr's stored attributes.
    """
    if volume_id in annotation_volumes:
        return annotation_volumes[volume_id]

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
            "chunk_sync_state": {},
        }
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

    # Create correction entry
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

    corr_zarr.attrs["source"] = "sparse_volume"
    corr_zarr.attrs["volume_id"] = volume_id
    corr_zarr.attrs["chunk_indices"] = [cz, cy, cx]

    logger.info(f"Created correction {correction_id} from chunk ({cz},{cy},{cx})")
    return True


# ---------------------------------------------------------------------------
# Annotation volume sync
# ---------------------------------------------------------------------------

def sync_annotation_volume_from_minio(volume_id, force=False):
    """
    Sync an annotation volume from MinIO, detect annotated chunks, extract corrections.

    Steps:
    1. Sync the full annotation zarr from MinIO to local disk
    2. List chunk files in MinIO to find annotated chunks
    3. For each new annotated chunk, extract raw data and create correction entry

    Returns:
        bool: True if any corrections were created
    """
    if not minio_state["ip"] or not minio_state["port"] or not minio_state["output_base"]:
        logger.warning("MinIO not initialized, skipping volume sync")
        return False

    try:
        zarr_name = f"{volume_id}.zarr"
        local_zarr_path = os.path.join(minio_state["output_base"], zarr_name)
        volume_meta = _get_volume_metadata(volume_id, local_zarr_path)

        if volume_meta is None:
            logger.warning(f"No metadata for volume {volume_id}, skipping")
            return False

        s3 = _make_s3_filesystem()

        bucket = minio_state["bucket"]
        src_annotation_path = f"{bucket}/{zarr_name}/annotation"

        if not s3.exists(src_annotation_path):
            return False

        # Sync zarr group metadata
        dst_annotation_path = Path(local_zarr_path) / "annotation"
        dst_annotation_path.mkdir(parents=True, exist_ok=True)
        _sync_zarr_group_metadata(s3, src_annotation_path, dst_annotation_path)

        # Diff and sync chunks
        s0_path = f"{bucket}/{zarr_name}/annotation/s0"
        known_chunk_state = volume_meta.get("chunk_sync_state", {})
        changed_chunk_keys, removed_chunk_keys, remote_chunk_state = _diff_and_sync_chunks(
            s3, s0_path, dst_annotation_path / "s0", known_chunk_state, force=force
        )

        if not changed_chunk_keys and not removed_chunk_keys:
            minio_state["last_sync"][volume_id] = datetime.now()
            return False

        logger.info(
            f"Synced {len(changed_chunk_keys)} changed chunks for volume {volume_id}"
        )

        # Extract corrections for changed chunks
        extracted_chunks = volume_meta.get("extracted_chunks", set())
        changed_chunk_indices = [
            tuple(map(int, k.split(".")))
            for k in changed_chunk_keys
        ]
        created_any = False

        for chunk_idx in changed_chunk_indices:
            try:
                created = extract_correction_from_chunk(
                    volume_id, chunk_idx, volume_meta
                )
                if created:
                    extracted_chunks.add(chunk_idx)
                    created_any = True
                else:
                    extracted_chunks.discard(chunk_idx)
            except Exception as e:
                logger.error(f"Error extracting correction for chunk {chunk_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Update tracked state
        volume_meta["extracted_chunks"] = extracted_chunks
        volume_meta["chunk_sync_state"] = remote_chunk_state
        minio_state["last_sync"][volume_id] = datetime.now()

        if created_any or changed_chunk_keys or removed_chunk_keys:
            logger.info(
                f"Volume {volume_id}: {len(extracted_chunks)} total chunks extracted"
            )

        return bool(created_any or changed_chunk_keys or removed_chunk_keys)

    except Exception as e:
        logger.error(f"Error syncing annotation volume {volume_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Periodic sync
# ---------------------------------------------------------------------------

def periodic_sync_annotations():
    """Background thread function to periodically sync annotations from MinIO."""
    while True:
        try:
            time.sleep(30)
            if not minio_state["output_base"]:
                continue
            if not minio_state["ip"] or not minio_state["port"]:
                continue
            sync_all_annotations_from_minio(force=False)
        except Exception as e:
            logger.debug(f"Error in periodic sync: {e}")


def start_periodic_sync():
    """Start the periodic annotation sync thread if not already running."""
    if minio_state["sync_thread"] is None or not minio_state["sync_thread"].is_alive():
        thread = threading.Thread(target=periodic_sync_annotations, daemon=True)
        thread.start()
        minio_state["sync_thread"] = thread
        logger.info("Started periodic annotation sync thread")


