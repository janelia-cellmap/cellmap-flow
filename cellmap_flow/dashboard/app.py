import json
import os
import socket
import neuroglancer
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import logging
import subprocess
import yaml
import tempfile
import re
from collections import deque
import queue
from cellmap_flow.utils.web_utils import get_free_port
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.utils.load_py import load_safe_config
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.web_utils import (
    encode_to_str,
    decode_to_json,
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
)
from cellmap_flow.utils.serilization_utils import serialize_norms_posts_to_json
from cellmap_flow.models.run import update_run_models
from cellmap_flow.globals import g
import numpy as np
import time
import uuid
import zarr
from pathlib import Path
import threading
import s3fs
from cellmap_flow.finetune.job_manager import FinetuneJobManager

logger = logging.getLogger(__name__)

# Global log buffer for streaming to frontend
log_buffer = deque(maxlen=1000)  # Keep last 1000 lines
log_clients = []  # List of queues for connected clients


# Custom handler to capture logs
class LogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_buffer.append(log_entry)
        # Send to all connected clients
        for client_queue in log_clients:
            try:
                client_queue.put_nowait(log_entry)
            except queue.Full:
                pass


# Explicitly set template and static folder paths for package installation
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Add custom log handler to logger
log_handler = LogHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

NEUROGLANCER_URL = None
INFERENCE_SERVER = None


# Blockwise task directory will be set from globals or use default
def get_blockwise_tasks_dir():
    tasks_dir = getattr(g, "blockwise_tasks_dir", None) or os.path.expanduser(
        "~/.cellmap_flow/blockwise_tasks"
    )
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir


CUSTOM_CODE_FOLDER = os.path.expanduser(
    os.environ.get(
        "CUSTOM_CODE_FOLDER",
        "~/Desktop/cellmap/cellmap-flow/example/example_norm",
    )
)

# Global finetuning job manager
finetune_job_manager = FinetuneJobManager()

# MinIO state for finetune annotation crops
minio_state = {
    "process": None,  # subprocess.Popen object
    "port": None,  # int
    "ip": None,  # str
    "bucket": "annotations",
    "minio_root": None,  # Path to MinIO storage directory
    "output_base": None,  # Base output directory for syncing back
    "last_sync": {},  # Track last sync time per crop_id
    "sync_thread": None,  # Background sync thread
}

# Track annotation volumes for sparse annotation workflow
# Maps volume_id -> volume metadata dict
annotation_volumes = {}

# Session management for timestamped output directories
# Maps base_output_path -> timestamped_session_path
output_sessions = {}


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
        # Create new timestamped session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = os.path.join(base_output_path, timestamp)
        output_sessions[base_output_path] = session_path
        logger.info(f"Created new session path: {session_path}")

    return output_sessions[base_output_path]


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
    """Find an available port for MinIO server."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find available port for MinIO")


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
            annotation/s0/   (uint8, shape=annotation_crop_shape, empty for manual annotation)
            mask/s0/         (optional, uint8, shape=annotation_crop_shape)
            .zattrs          (metadata)

    Args:
        zarr_path: Path to create zarr
        raw_crop_shape: Shape in voxels for raw [z, y, x]
        raw_voxel_size: Voxel size in nm for raw [z, y, x]
        raw_offset: Offset in voxels for raw [z, y, x]
        annotation_crop_shape: Shape in voxels for annotation [z, y, x]
        annotation_voxel_size: Voxel size in nm for annotation [z, y, x]
        annotation_offset: Offset in voxels for annotation [z, y, x]
        dataset_path: Source dataset path
        model_name: Model name for metadata
        output_channels: Number of output channels
        create_mask: Whether to create a mask group (default: False)

    Returns:
        (success: bool, info: str)
    """
    try:
        # Helper to add OME-NGFF metadata
        def add_ome_ngff_metadata(group, name, voxel_size, translation_offset=None):
            """Add OME-NGFF v0.4 metadata."""
            # Calculate physical translation
            if translation_offset is not None:
                physical_translation = [
                    float(o * v) for o, v in zip(translation_offset, voxel_size)
                ]
            else:
                physical_translation = [0.0, 0.0, 0.0]

            # Coordinate transformations
            transforms = [{"type": "scale", "scale": [float(v) for v in voxel_size]}]

            if translation_offset is not None:
                transforms.append(
                    {"type": "translation", "translation": physical_translation}
                )

            # OME-NGFF v0.4 metadata
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

        # Open zarr root
        root = zarr.open(zarr_path, mode="w")

        # Create raw group (will be filled by user later)
        raw_group = root.create_group("raw")
        raw_s0 = raw_group.create_dataset(
            "s0",
            shape=tuple(raw_crop_shape),
            chunks=(64, 64, 64),
            dtype=raw_dtype,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )
        add_ome_ngff_metadata(raw_group, "raw", raw_voxel_size, raw_offset)

        # Create annotation group (empty, will be filled by user annotations)
        annotation_group = root.create_group("annotation")
        annotation_s0 = annotation_group.create_dataset(
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

        # Optionally create mask group (will be filled by user annotations)
        if create_mask:
            mask_group = root.create_group("mask")
            mask_s0 = mask_group.create_dataset(
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

        # Add root metadata
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
        logger.info(
            f"  Raw crop shape: {raw_crop_shape}, voxel size: {raw_voxel_size}, offset: {raw_offset}"
        )
        logger.info(
            f"  Annotation crop shape: {annotation_crop_shape}, voxel size: {annotation_voxel_size}, offset: {annotation_offset}"
        )
        logger.info(f"  Mask created: {create_mask}")

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

    Args:
        zarr_path: Path to create zarr
        dataset_shape_voxels: Full dataset shape in output voxels [z, y, x]
        output_voxel_size: nm per voxel for output [z, y, x]
        dataset_offset_nm: Dataset offset in nm [z, y, x]
        chunk_size: Chunk size in voxels = model output_size [z, y, x]
        dataset_path: Source dataset path
        model_name: Model name for metadata
        input_size: Model input size in voxels [z, y, x]
        input_voxel_size: nm per voxel for input [z, y, x]

    Returns:
        (success: bool, info: str)
    """
    try:
        root = zarr.open(zarr_path, mode="w")

        # Create annotation group with chunks = output_size
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
        physical_translation = [
            float(o) for o in dataset_offset_nm
        ]
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

        # Root metadata marking this as an annotation volume
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

        logger.info(f"Created annotation volume zarr at {zarr_path}")
        logger.info(
            f"  Shape: {dataset_shape_voxels}, chunks: {chunk_size}, "
            f"voxel size: {output_voxel_size}"
        )

        return True, zarr_path

    except Exception as e:
        logger.error(f"Error creating annotation volume zarr: {e}")
        return False, str(e)


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
    # Check if MinIO is already running
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
            raise RuntimeError("MinIO failed to start")

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
        logger.info("MC client configured")

        # Create bucket if needed
        result = subprocess.run(
            ["mc", "mb", f"myserver/{minio_state['bucket']}"],
            capture_output=True,
            text=True,
        )

        # Ignore error if bucket already exists
        if result.returncode != 0 and "already" not in result.stderr.lower():
            logger.warning(f"Bucket creation returned: {result.stderr}")

        # Make bucket public
        subprocess.run(
            ["mc", "anonymous", "set", "public", f"myserver/{minio_state['bucket']}"],
            check=True,
            capture_output=True,
        )
        logger.info(f"Bucket {minio_state['bucket']} is public")

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

    # Return MinIO URL
    minio_url = f"http://{minio_state['ip']}:{minio_state['port']}/{minio_state['bucket']}/{zarr_name}"
    return minio_url


def sync_all_annotations_from_minio():
    """Sync all annotations from MinIO to local disk.

    Returns:
        Number of annotations synced, or -1 if MinIO is not initialized.
    """
    if not minio_state.get("ip") or not minio_state.get("port"):
        logger.info("MinIO not initialized, skipping annotation sync")
        return -1

    logger.info("Syncing annotations from MinIO before training...")
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
                    if sync_annotation_volume_from_minio(zid, force=True):
                        synced += 1
                    continue
        except Exception:
            pass
        if sync_annotation_from_minio(zid, force=True):
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
        import traceback
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
                import traceback
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
        import traceback
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


@app.route("/api/logs/stream")
def stream_logs():
    """Stream logs via Server-Sent Events (SSE)"""

    def generate():
        # Send existing log buffer first
        for log_line in log_buffer:
            yield f"data: {log_line}\n\n"

        # Create a queue for this client
        client_queue = queue.Queue(maxsize=100)
        log_clients.append(client_queue)

        try:
            while True:
                try:
                    log_line = client_queue.get(timeout=30)
                    yield f"data: {log_line}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            # Clean up when client disconnects
            if client_queue in log_clients:
                log_clients.remove(client_queue)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/")
def index():
    # Render the main page with tabs
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()
    model_mergers = get_model_mergers_list()
    model_catalog = g.model_catalog

    # Build User catalog from jobs, using model configs to get paths
    user_models = {}
    for j in g.jobs:
        model_path = ""
        # Try to find the model config for this job
        if hasattr(g, 'models_config') and g.models_config:
            for model_config in g.models_config:
                if hasattr(model_config, 'name') and model_config.name == j.model_name:
                    # Try to get checkpoint_path or script_path
                    if hasattr(model_config, 'checkpoint_path') and model_config.checkpoint_path:
                        model_path = str(model_config.checkpoint_path)
                    elif hasattr(model_config, 'script_path') and model_config.script_path:
                        model_path = str(model_config.script_path)
                    break
        user_models[j.model_name] = model_path

    model_catalog["User"] = user_models
    default_post_process = {d.to_dict()["name"]: d.to_dict() for d in g.postprocess}
    default_input_norm = {d.to_dict()["name"]: d.to_dict() for d in g.input_norms}
    logger.warning(f"Model catalog: {model_catalog}")
    logger.warning(f"Default postprocess: {default_post_process}")
    logger.warning(f"Default input norm: {default_input_norm}")

    return render_template(
        "index.html",
        neuroglancer_url=NEUROGLANCER_URL,
        inference_servers=INFERENCE_SERVER,
        input_normalizers=input_norms,
        output_postprocessors=output_postprocessors,
        model_mergers=model_mergers,
        default_post_process=default_post_process,
        default_input_norm=default_input_norm,
        model_catalog=model_catalog,
        default_models=[j.model_name for j in g.jobs],
    )


@app.route("/pipeline-builder")
def pipeline_builder():
    """Render the drag-and-drop pipeline builder interface with current state from globals"""
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()

    # Get available models from models_config (not from catalog)
    available_models = {}
    # if hasattr(g, 'models_config') and g.models_config:
    #     for model_config in g.models_config:
    #         model_dict = model_config.to_dict()
    #         available_models[model_config.name] = model_dict

    logger.warning(f"\n{'='*80}")
    logger.warning(f"AVAILABLE MODELS DEBUG:")
    logger.warning(f"  Initial available_models keys: {list(available_models.keys())}")
    logger.warning(
        f"  g.models_config: {g.models_config if hasattr(g, 'models_config') else 'NOT SET'}"
    )
    logger.warning(f"  Sample model with config:")
    for model_name, model_data in list(available_models.items())[:1]:
        logger.warning(f"    {model_name}: {model_data}")
    models_with_config = {}
    for model_name in available_models.keys():
        # Find matching config (strip _server suffix for matching)
        model_name_stripped = model_name.replace("_server", "")
        for model_config in g.models_config:
            config_name = getattr(model_config, "name", "").replace("_server", "")
            if config_name == model_name_stripped:
                if hasattr(model_config, "to_dict"):
                    models_with_config[model_name] = {
                        "name": model_name,
                        "config": model_config.to_dict(),
                    }
                break
        # If no config found, just use the name
        if model_name not in models_with_config:
            models_with_config[model_name] = {"name": model_name}
    available_models = models_with_config

    # Check if we have stored pipeline state from previous apply
    if hasattr(g, "pipeline_normalizers") and len(g.pipeline_normalizers) > 0:
        # Use stored pipeline state (includes IDs, positions, params)
        current_normalizers = g.pipeline_normalizers
        current_postprocessors = g.pipeline_postprocessors
        current_models = g.pipeline_models
        # Enrich current_models with config from g.models_config if available
        if hasattr(g, "models_config") and g.models_config:
            for model_dict in current_models:
                if "config" not in model_dict:
                    # Strip _server suffix for matching
                    model_name = model_dict["name"].replace("_server", "")
                    for model_config in g.models_config:
                        config_name = getattr(model_config, "name", "").replace(
                            "_server", ""
                        )
                        if config_name == model_name:
                            if hasattr(model_config, "to_dict"):
                                model_dict["config"] = model_config.to_dict()
                            break
        current_inputs = g.pipeline_inputs
        current_outputs = g.pipeline_outputs
        current_edges = g.pipeline_edges
    else:
        # Fall back to converting from globals.input_norms and globals.postprocess
        current_normalizers = []
        for idx, norm in enumerate(g.input_norms):
            norm_dict = (
                norm.to_dict() if hasattr(norm, "to_dict") else {"name": str(norm)}
            )
            norm_name = norm_dict.get("name", str(norm))
            # Extract params: all dict items except 'name'
            params = {k: v for k, v in norm_dict.items() if k != "name"}
            current_normalizers.append(
                {
                    "id": f"norm-{idx}-{int(time.time()*1000)}",
                    "name": norm_name,
                    "params": params,
                }
            )

        # Current models (from jobs and models_config)
        current_models = []
        logger.warning(f"\n{'='*80}")
        logger.warning(f"Building current_models from g.jobs:")
        logger.warning(f"  g.jobs count: {len(g.jobs)}")
        logger.warning(f"  g.models_config exists: {hasattr(g, 'models_config')}")
        if hasattr(g, "models_config"):
            logger.warning(
                f"  g.models_config count: {len(g.models_config) if g.models_config else 0}"
            )
            logger.warning(f"  g.models_config type: {type(g.models_config)}")
            logger.warning(f"  g.models_config value: {g.models_config}")
            if g.models_config:
                logger.warning(
                    f"  g.models_config names: {[getattr(mc, 'name', 'NO_NAME') for mc in g.models_config]}"
                )
                for mc in g.models_config:
                    logger.warning(
                        f"    Config object: {mc}, has to_dict: {hasattr(mc, 'to_dict')}"
                    )

        # If models_config is empty but we have jobs, try to get configs from model_catalog
        if (not hasattr(g, "models_config") or not g.models_config) and hasattr(
            g, "model_catalog"
        ):
            logger.warning(
                f"  models_config is empty, checking model_catalog for configs..."
            )
            # Check if available_models dict has configs
            if available_models:
                logger.warning(
                    f"  available_models has {len(available_models)} entries with potential configs"
                )

        for idx, job in enumerate(g.jobs):
            if hasattr(job, "model_name"):
                logger.warning(f"\n  Processing job {idx}: model_name={job.model_name}")
                model_dict = {
                    "id": f"model-{idx}-{int(time.time()*1000)}",
                    "name": job.model_name,
                    "params": {},
                }
                # Try to find the corresponding ModelConfig to get full configuration
                config_found = False

                # First try g.models_config
                if hasattr(g, "models_config") and g.models_config:
                    # Strip _server suffix for matching
                    job_model_name = job.model_name.replace("_server", "")
                    for model_config in g.models_config:
                        model_config_name = getattr(model_config, "name", None)
                        config_name_stripped = (
                            model_config_name.replace("_server", "")
                            if model_config_name
                            else None
                        )
                        logger.warning(
                            f"    Checking model_config: {model_config_name} (stripped: {config_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})"
                        )
                        if (
                            config_name_stripped
                            and config_name_stripped == job_model_name
                        ):
                            # Export the full model config using to_dict()
                            if hasattr(model_config, "to_dict"):
                                model_dict["config"] = model_config.to_dict()
                                logger.warning(
                                    f"    ✓ Config attached from models_config: {model_dict['config']}"
                                )
                                config_found = True
                            break

                # Fallback: check available_models dict (which was enriched earlier)
                if not config_found and available_models:
                    job_model_name = job.model_name.replace("_server", "")
                    for model_name, model_data in available_models.items():
                        model_name_stripped = model_name.replace("_server", "")
                        logger.warning(
                            f"    Checking available_models: {model_name} (stripped: {model_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})"
                        )
                        if (
                            model_name_stripped == job_model_name
                            and isinstance(model_data, dict)
                            and "config" in model_data
                        ):
                            model_dict["config"] = model_data["config"]
                            logger.warning(
                                f"    ✓ Config attached from available_models: {model_dict['config']}"
                            )
                            config_found = True
                            break

                # Second fallback: check previously saved pipeline_model_configs
                if not config_found and hasattr(g, "pipeline_model_configs"):
                    job_model_name = job.model_name.replace("_server", "")
                    for saved_name, saved_config in g.pipeline_model_configs.items():
                        saved_name_stripped = saved_name.replace("_server", "")
                        logger.warning(
                            f"    Checking pipeline_model_configs: {saved_name} (stripped: {saved_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})"
                        )
                        if saved_name_stripped == job_model_name:
                            model_dict["config"] = saved_config
                            logger.warning(
                                f"    ✓ Config attached from pipeline_model_configs: {model_dict['config']}"
                            )
                            config_found = True
                            break

                if not config_found:
                    logger.warning(
                        f"    ✗ No matching config found for {job.model_name}"
                    )
                    logger.warning(
                        f"       TIP: Import a YAML with full model configs to populate g.pipeline_model_configs"
                    )
                current_models.append(model_dict)
        logger.warning(f"{'='*80}\n")

        current_postprocessors = []
        for idx, post in enumerate(g.postprocess):
            post_dict = (
                post.to_dict() if hasattr(post, "to_dict") else {"name": str(post)}
            )
            post_name = post_dict.get("name", str(post))
            # Extract params: all dict items except 'name'
            params = {k: v for k, v in post_dict.items() if k != "name"}
            current_postprocessors.append(
                {
                    "id": f"post-{idx}-{int(time.time()*1000)}",
                    "name": post_name,
                    "params": params,
                }
            )

        current_inputs = []
        current_outputs = []
        current_edges = []

    # Get current dataset_path from globals
    dataset_path = getattr(g, "dataset_path", None) or ""

    # Get available model mergers
    model_mergers = get_model_mergers_list()

    return render_template(
        "pipeline_builder_v2.html",
        input_normalizers=input_norms or {},
        available_models=available_models or {},
        output_postprocessors=output_postprocessors or {},
        model_mergers=model_mergers or {},
        current_normalizers=current_normalizers,
        current_models=current_models,
        current_postprocessors=current_postprocessors,
        current_inputs=current_inputs,
        current_outputs=current_outputs,
        current_edges=current_edges,
        dataset_path=dataset_path,
    )


def is_output_segmentation():
    if len(g.postprocess) == 0:
        return False

    for postprocess in g.postprocess[::-1]:
        if postprocess.is_segmentation is not None:
            return postprocess.is_segmentation


@app.route("/update/equivalences", methods=["POST"])
def update_equivalences():
    equivalences_info = request.get_json()
    dataset = equivalences_info["dataset"]
    equivalences_str = equivalences_info["equivalences"]
    equivalences = [
        [np.uint64(item) for item in sublist] for sublist in equivalences_str
    ]

    with g.viewer.txn() as s:
        for layer in s.layers:
            if layer.source[0].url.endswith(dataset):
                layer.equivalences = equivalences
                break
    return jsonify({"message": "Equivalences updated successfully"})


@app.route("/api/models", methods=["POST"])
def submit_models():
    data = request.get_json()
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    selected_models = data.get("selected_models", [])
    update_run_models(selected_models)
    logger.warning(f"Selected models: {selected_models}")
    return jsonify(
        {
            "message": "Data received successfully",
            "models": selected_models,
        }
    )


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()

    # add dashboard url to data so we can update the state from the server
    data["dashboard_url"] = request.host_url

    # we wanmt to set the time such that each request is unique
    data["time"] = time.time()

    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    custom_code = data.get("custom_code", None)
    if "custom_code" in data:
        del data["custom_code"]
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    g.input_norms = get_normalizations(data["input_norm"])
    g.postprocess = get_postprocessors(data["postprocess"])

    with g.viewer.txn() as s:
        # g.raw.invalidate()
        g.raw = get_raw_layer(g.dataset_path)
        s.layers["data"] = g.raw
        for job in g.jobs:
            model = job.model_name
            host = job.host
            # response = requests.post(f"{host}/input_normalize", json=data)
            # print(f"Response from {host}: {response.json()}")
            st_data = encode_to_str(data)

            if is_output_segmentation():
                s.layers[model] = neuroglancer.SegmentationLayer(
                    source=f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )
            else:
                s.layers[model] = neuroglancer.ImageLayer(
                    source=f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )

    logger.warning(f"Input normalizers: {g.input_norms}")

    if custom_code:

        try:
            # Save custom code to a file with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_code_{timestamp}.py"
            filepath = os.path.join(CUSTOM_CODE_FOLDER, filename)

            with open(filepath, "w") as file:
                file.write(custom_code)

            config = load_safe_config(filepath)
            logger.warning(f"Custom code loaded successfully: {config}")

            logger.warning(get_input_normalizers())

        except Exception as e:
            logger.warning(f"Error executing custom code: {e}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
            "found_custom_normalizer": get_input_normalizers(),
        }
    )


@app.route("/api/pipeline/validate", methods=["POST"])
def validate_pipeline():
    """Validate a pipeline configuration"""
    try:
        data = request.get_json()

        # Validate normalizers
        normalizer_names = [n.get("name") for n in data.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        # Extract just the normalizer names from the list of dicts
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return (
                    jsonify(
                        {"valid": False, "error": f"Unknown normalizer: {norm_name}"}
                    ),
                    400,
                )

        # Validate postprocessors
        processor_names = [p.get("name") for p in data.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        # Extract just the postprocessor names from the list of dicts
        available_proc_names = [proc["name"] for proc in available_procs]
        for proc_name in processor_names:
            if proc_name not in available_proc_names:
                return (
                    jsonify(
                        {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}
                    ),
                    400,
                )

        return jsonify({"valid": True, "message": "Pipeline is valid"})

    except Exception as e:
        logger.error(f"Error validating pipeline: {e}")
        return jsonify({"valid": False, "error": str(e)}), 500


@app.route("/api/dataset-path", methods=["GET", "POST"])
def dataset_path_api():
    """Get or set the dataset path in globals"""
    if request.method == "GET":
        dataset_path = getattr(g, "dataset_path", None) or ""
        return jsonify({"dataset_path": dataset_path})
    elif request.method == "POST":
        data = request.get_json()
        dataset_path = data.get("dataset_path", "")
        g.dataset_path = dataset_path
        logger.warning(f"Dataset path updated to: {dataset_path}")
        return jsonify({"success": True, "dataset_path": g.dataset_path})


@app.route("/api/blockwise-config", methods=["GET", "POST"])
def blockwise_config_api():
    """Get or set blockwise configuration in globals"""
    if request.method == "GET":
        return jsonify(
            {
                "queue": g.queue,
                "charge_group": g.charge_group,
                "nb_cores_master": g.nb_cores_master,
                "nb_cores_worker": g.nb_cores_worker,
                "nb_workers": g.nb_workers,
                "tmp_dir": g.tmp_dir,
                "blockwise_tasks_dir": g.blockwise_tasks_dir,
            }
        )
    elif request.method == "POST":
        data = request.get_json()
        g.queue = data.get("queue")
        g.charge_group = data.get("charge_group")
        g.nb_cores_master = int(data.get("nb_cores_master"))
        g.nb_cores_worker = int(data.get("nb_cores_worker"))
        g.nb_workers = int(data.get("nb_workers"))
        g.tmp_dir = data.get("tmp_dir")
        g.blockwise_tasks_dir = data.get("blockwise_tasks_dir")
        logger.warning(
            f"Blockwise config updated: queue={g.queue}, charge_group={g.charge_group}, cores_master={g.nb_cores_master}, cores_worker={g.nb_cores_worker}, workers={g.nb_workers}, tmp_dir={g.tmp_dir}, blockwise_tasks_dir={g.blockwise_tasks_dir}"
        )
        return jsonify(
            {
                "success": True,
                "config": {
                    "queue": g.queue,
                    "charge_group": g.charge_group,
                    "nb_cores_master": g.nb_cores_master,
                    "nb_cores_worker": g.nb_cores_worker,
                    "nb_workers": g.nb_workers,
                    "tmp_dir": g.tmp_dir,
                    "blockwise_tasks_dir": g.blockwise_tasks_dir,
                },
            }
        )


@app.route("/api/pipeline/apply", methods=["POST"])
def apply_pipeline():
    """Apply a pipeline configuration to the current inference"""
    try:
        data = request.get_json()
        logger.warning(f"\n{'='*80}")
        logger.warning(f"APPLY PIPELINE - Received data:")
        logger.warning(f"  Input normalizers: {data.get('input_normalizers', [])}")
        logger.warning(f"  Postprocessors: {data.get('postprocessors', [])}")

        # Validate first
        validation = validate_pipeline_config(data)
        if not validation["valid"]:
            return jsonify(validation), 400

        # Apply normalizers
        input_norms_config = {
            n["name"]: n.get("params", {}) for n in data.get("input_normalizers", [])
        }
        logger.warning(f"\nNormalizers config dict: {input_norms_config}")
        g.input_norms = get_normalizations(input_norms_config)

        # Apply postprocessors
        postprocs_config = {
            p["name"]: p.get("params", {}) for p in data.get("postprocessors", [])
        }
        logger.warning(f"Postprocessors config dict: {postprocs_config}")
        g.postprocess = get_postprocessors(postprocs_config)

        # Save complete pipeline visual state to globals
        g.pipeline_inputs = data.get("inputs", [])
        g.pipeline_outputs = data.get("outputs", [])
        g.pipeline_edges = data.get("edges", [])
        g.pipeline_normalizers = data.get("input_normalizers", [])
        g.pipeline_models = data.get("models", [])
        g.pipeline_postprocessors = data.get("postprocessors", [])

        # Also save model configs separately for easier access
        if not hasattr(g, "pipeline_model_configs"):
            g.pipeline_model_configs = {}
        for model in data.get("models", []):
            if "config" in model and model["config"]:
                g.pipeline_model_configs[model["name"]] = model["config"]

        # Log the updated globals state
        logger.warning(f"\n{'='*80}")
        logger.warning(f"UPDATED GLOBALS (g) STATE:")
        logger.warning(f"{'='*80}")
        logger.warning(f"\ng.input_norms ({len(g.input_norms)} items):")
        for idx, norm in enumerate(g.input_norms):
            logger.warning(f"  [{idx}] {norm}")

        logger.warning(f"\ng.postprocess ({len(g.postprocess)} items):")
        for idx, post in enumerate(g.postprocess):
            logger.warning(f"  [{idx}] {post}")

        logger.warning(f"\ng.jobs ({len(g.jobs)} items):")
        for idx, job in enumerate(g.jobs):
            logger.warning(
                f"  [{idx}] model_name={getattr(job, 'model_name', 'N/A')}, host={getattr(job, 'host', 'N/A')}"
            )

        logger.warning(
            f"\ng.pipeline_inputs ({len(g.pipeline_inputs)} items): {g.pipeline_inputs}"
        )
        logger.warning(
            f"\ng.pipeline_outputs ({len(g.pipeline_outputs)} items): {g.pipeline_outputs}"
        )
        logger.warning(
            f"\ng.pipeline_edges ({len(g.pipeline_edges)} items): {g.pipeline_edges}"
        )
        logger.warning(
            f"\ng.pipeline_normalizers ({len(g.pipeline_normalizers)} items): {g.pipeline_normalizers}"
        )
        logger.warning(
            f"\ng.pipeline_models ({len(g.pipeline_models)} items): {g.pipeline_models}"
        )
        logger.warning(
            f"\ng.pipeline_postprocessors ({len(g.pipeline_postprocessors)} items): {g.pipeline_postprocessors}"
        )

        logger.warning(f"{'='*80}\n")

        return jsonify(
            {
                "message": "Pipeline applied successfully",
                "normalizers_applied": len(g.input_norms),
                "postprocessors_applied": len(g.postprocess),
            }
        )

    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return jsonify({"error": str(e)}), 500


def validate_pipeline_config(config):
    """Helper function to validate pipeline configuration"""
    try:
        normalizer_names = [n.get("name") for n in config.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        # Extract just the normalizer names from the list of dicts
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return {"valid": False, "error": f"Unknown normalizer: {norm_name}"}

        processor_names = [p.get("name") for p in config.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        # Extract just the postprocessor names from the list of dicts
        available_proc_names = [proc["name"] for proc in available_procs]
        for proc_name in processor_names:
            if proc_name not in available_proc_names:
                return {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}

        return {"valid": True}

    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.route("/api/blockwise/validate", methods=["POST"])
def validate_blockwise():
    """Validate if pipeline is ready for blockwise processing"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})

        # Check required components
        if not pipeline.get("inputs") or len(pipeline["inputs"]) == 0:
            return {"valid": False, "error": "No input nodes defined"}

        if not pipeline.get("outputs") or len(pipeline["outputs"]) == 0:
            return {"valid": False, "error": "No output nodes defined"}

        if not pipeline.get("models") or len(pipeline["models"]) == 0:
            return {"valid": False, "error": "No models defined"}

        # Check blockwise config
        if (
            not pipeline.get("blockwise_config")
            or len(pipeline["blockwise_config"]) == 0
        ):
            return {"valid": False, "error": "No blockwise configuration defined"}

        # Check input has dataset_path
        input_node = pipeline["inputs"][0]
        if not input_node.get("params", {}).get("dataset_path"):
            return {"valid": False, "error": "Input node missing dataset_path"}

        # Check output has dataset_path
        output_node = pipeline["outputs"][0]
        if not output_node.get("params", {}).get("dataset_path"):
            return {"valid": False, "error": "Output node missing dataset_path"}

        logger.info("Pipeline validation passed")
        return {"valid": True, "message": "Pipeline is ready for blockwise processing"}

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"valid": False, "error": str(e)}


@app.route("/api/blockwise/generate", methods=["POST"])
def generate_blockwise_task():
    """Generate blockwise task YAML files"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})

        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}

        # Get blockwise config
        blockwise_config = pipeline["blockwise_config"][0]
        input_node = pipeline["inputs"][0]
        output_node = pipeline["outputs"][0]

        # Get output path and ensure it ends with .zarr
        output_path = output_node["params"]["dataset_path"]
        if output_path:
            # Remove trailing slashes
            output_path = output_path.rstrip("/\\")
            # Add .zarr if not already present
            if not output_path.endswith(".zarr"):
                output_path = output_path + ".zarr"

        # Create task YAML content
        task_name = f"cellmap_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_yaml = {
            "data_path": input_node["params"]["dataset_path"],
            "output_path": output_path,
            "task_name": task_name,
            "charge_group": blockwise_config["params"]["charge_group"],
            "queue": blockwise_config["params"]["queue"],
            "workers": blockwise_config["params"]["nb_workers"],
            "cpu_workers": blockwise_config["params"]["nb_cores_worker"],
            "tmp_dir": blockwise_config["params"]["tmp_dir"],
            "models": [],
        }

        # Add bounding_boxes from INPUT node if they exist
        bounding_boxes = input_node.get("params", {}).get("bounding_boxes", [])
        if (
            bounding_boxes
            and isinstance(bounding_boxes, list)
            and len(bounding_boxes) > 0
        ):
            task_yaml["bounding_boxes"] = bounding_boxes
            logger.info(f"Adding bounding_boxes to YAML: {len(bounding_boxes)} box(es)")

        # Add separate_bounding_boxes_zarrs flag from INPUT node if set
        separate_zarrs = input_node.get("params", {}).get(
            "separate_bounding_boxes_zarrs", False
        )
        if separate_zarrs:
            task_yaml["separate_bounding_boxes_zarrs"] = True
            logger.info("Adding separate_bounding_boxes_zarrs: True")

        # Add model_mode if multiple models are present and a merge mode is selected
        model_count = len(pipeline.get("models", []))
        model_mode = pipeline.get("model_mode", "")
        if model_count > 1 and model_mode:
            task_yaml["model_mode"] = model_mode
            logger.info(f"Adding model_mode: {model_mode} for {model_count} models")

        # Add models with full config
        for model in pipeline.get("models", []):
            model_entry = {
                "name": model.get("name"),
                **model.get("params", model.get("config", {})),
            }
            # Parse string representations of lists/tuples back to actual lists for specific fields
            import ast
            import re

            for field in [
                "channels",
                "input_size",
                "output_size",
                "input_voxel_size",
                "output_voxel_size",
            ]:
                if field in model_entry:
                    value = model_entry[field]
                    # If it's already a list, keep it
                    if isinstance(value, (list, tuple)):
                        model_entry[field] = list(value)
                        logger.info(
                            f"Field {field} is already a list: {model_entry[field]}"
                        )
                    # If it's a string that looks like a list/tuple, parse it
                    elif isinstance(value, str):
                        value_stripped = value.strip().strip(
                            "'\""
                        )  # Remove outer quotes
                        if (
                            value_stripped.startswith("[")
                            or value_stripped.startswith("(")
                        ) and (
                            value_stripped.endswith("]") or value_stripped.endswith(")")
                        ):
                            try:
                                # Fix unquoted identifiers: convert [mito] to ['mito']
                                # Replace word characters not inside quotes with quoted versions
                                fixed_value = re.sub(
                                    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b",
                                    r"'\1'",
                                    value_stripped,
                                )
                                # Remove duplicate quotes: ''mito'' -> 'mito'
                                fixed_value = re.sub(r"''+", "'", fixed_value)
                                logger.info(
                                    f"Fixing {field}: {value_stripped!r} -> {fixed_value!r}"
                                )

                                parsed = ast.literal_eval(fixed_value)
                                if isinstance(parsed, (list, tuple)):
                                    model_entry[field] = list(parsed)
                                    logger.info(
                                        f"Parsed {field} from string {value!r} to list {model_entry[field]}"
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to parse {field}: {value}, error: {e}"
                                )

            task_yaml["models"].append(model_entry)

        # Serialize normalizers and postprocessors to json_data format
        # READ FROM TOP-LEVEL PIPELINE (THEY ARE STORED AT pipeline["normalizers"] and pipeline["postprocessors"])
        # Normalizers and postprocessors are drawn in the pipeline and stored at top level, not in INPUT node
        normalizers_list = pipeline.get("normalizers", [])
        postprocessors_list = pipeline.get("postprocessors", [])

        # Create json_data for blockwise processor - maintain order by using list iteration order
        if normalizers_list or postprocessors_list:
            try:
                # Build normalizers dict - preserve insertion order from normalizers_list
                norm_fns = {}
                for norm in normalizers_list:
                    # norm can be dict with "name" and "params" keys
                    if isinstance(norm, dict):
                        norm_name = norm.get("name")
                        norm_params = norm.get("params", {})
                    else:
                        continue
                    if norm_name:
                        norm_fns[norm_name] = norm_params

                # Build postprocessors dict - preserve insertion order from postprocessors_list
                post_fns = {}
                for post in postprocessors_list:
                    # post can be dict with "name" and "params" keys
                    if isinstance(post, dict):
                        post_name = post.get("name")
                        post_params = post.get("params", {})
                    else:
                        continue
                    if post_name:
                        post_fns[post_name] = post_params

                # Create json_data as dict (not JSON string) using the correct key constants
                json_data_dict = {
                    INPUT_NORM_DICT_KEY: norm_fns,
                    POSTPROCESS_DICT_KEY: post_fns,
                }
                # Store as dict (YAML will handle it properly)
                task_yaml["json_data"] = json_data_dict
                logger.info(
                    f"Added json_data as dict with {len(normalizers_list)} normalizers and {len(postprocessors_list)} postprocessors"
                )
            except Exception as e:
                logger.warning(f"Failed to create json_data: {e}")

        # Add output_channels from OUTPUT node if configured
        output_channels = output_node.get("params", {}).get("output_channels", [])
        if (
            output_channels
            and isinstance(output_channels, list)
            and len(output_channels) > 0
        ):
            task_yaml["output_channels"] = output_channels
            logger.info(f"Adding output_channels to YAML: {output_channels}")

        # Convert to YAML format with proper list handling
        # sort_keys=False preserves dict insertion order (Python 3.7+)
        yaml_content = yaml.dump(
            task_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        # Save to file
        yaml_filename = f"{task_name}.yaml"
        tasks_dir = get_blockwise_tasks_dir()
        yaml_path = os.path.join(tasks_dir, yaml_filename)

        # Check if we need to generate multiple YAMLs (one per bbox with separate output paths)
        # Use the output_path (which already has .zarr appended if needed)
        output_base_path = output_path
        yaml_paths = []

        if separate_zarrs and bounding_boxes and len(bounding_boxes) > 0:
            # Generate separate YAML for each bounding box
            logger.info(
                f"Generating separate YAMLs for {len(bounding_boxes)} bounding box(es)"
            )
            for bbox_idx, bbox in enumerate(bounding_boxes):
                # Create a copy of task_yaml for this bbox
                bbox_task_yaml = task_yaml.copy()

                # Keep only this bbox in bounding_boxes
                bbox_task_yaml["bounding_boxes"] = [bbox]

                # Set output path to box_X subdirectory
                bbox_output_path = os.path.join(output_base_path, f"box_{bbox_idx + 1}")
                bbox_task_yaml["output_path"] = bbox_output_path

                # Update task name to include bbox index
                bbox_task_name = f"{task_name}_box{bbox_idx + 1}"
                bbox_task_yaml["task_name"] = bbox_task_name

                # Convert to YAML
                bbox_yaml_content = yaml.dump(
                    bbox_task_yaml,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

                # Save bbox YAML
                bbox_yaml_filename = f"{bbox_task_name}.yaml"
                bbox_yaml_path = os.path.join(tasks_dir, bbox_yaml_filename)
                with open(bbox_yaml_path, "w") as f:
                    f.write(bbox_yaml_content)

                yaml_paths.append(bbox_yaml_path)
                logger.info(f"Generated bbox {bbox_idx + 1} YAML at: {bbox_yaml_path}")
        else:
            # Single YAML for all bboxes
            with open(yaml_path, "w") as f:
                f.write(yaml_content)
            yaml_paths = [yaml_path]
            logger.info(f"Generated blockwise task YAML at: {yaml_path}")

        logger.info(f"Task YAML content:\n{yaml_content}")

        return {
            "success": True,
            "task_yaml": yaml_content,
            "task_config": task_yaml,
            "task_paths": yaml_paths,  # All paths for multiple YAMLs
            "task_name": task_name,
            "message": "Blockwise task generated successfully",
        }

    except Exception as e:
        logger.error(f"Task generation error: {str(e)}")
        return {"success": False, "error": str(e)}


@app.route("/api/blockwise/precheck", methods=["POST"])
def precheck_blockwise_task():
    """Precheck blockwise task configuration using already-generated YAML"""
    try:
        from cellmap_flow.blockwise.blockwise_processor import (
            CellMapFlowBlockwiseProcessor,
        )

        data = request.get_json()
        yaml_paths = data.get("yaml_paths", [])

        if not yaml_paths:
            return {
                "success": False,
                "error": "No YAML paths provided. Please generate task first.",
            }

        # Try to instantiate the processor to validate configuration with the first YAML
        try:
            _ = CellMapFlowBlockwiseProcessor(yaml_paths[0], create=True)
            logger.info(f"Blockwise precheck passed for: {yaml_paths[0]}")
            return {"success": True, "message": "success"}
        except Exception as e:
            logger.error(f"Blockwise precheck failed: {str(e)}")
            return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Precheck error: {str(e)}")
        return {"success": False, "error": str(e)}


@app.route("/api/blockwise/submit", methods=["POST"])
def submit_blockwise_task():
    """Submit blockwise task to LSF"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})
        job_name = data.get("job_name", f"cellmap_flow_{int(time.time())}")

        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}

        # Generate task YAML
        gen_result = generate_blockwise_task()
        if not gen_result.get("success"):
            return {"success": False, "error": gen_result.get("error")}

        yaml_paths = gen_result.get("task_paths", [gen_result.get("task_path")])
        blockwise_config = pipeline["blockwise_config"][0]

        # Build bsub command - use multiple_cli to handle multiple YAML files
        cores_master = blockwise_config["params"]["nb_cores_master"]
        charge_group = blockwise_config["params"]["charge_group"]
        queue = blockwise_config["params"]["queue"]

        bsub_cmd = [
            "bsub",
            "-J",
            job_name,
            "-n",
            str(cores_master),
            "-P",
            charge_group,
            # "-q", queue,
            "python",
            "-m",
            "cellmap_flow.blockwise.multiple_cli",
        ] + yaml_paths  # Add all YAML paths

        logger.info(f"Submitting LSF job: {' '.join(bsub_cmd)}")

        # Submit job - use same environment as parent process
        result = subprocess.run(
            bsub_cmd, capture_output=True, text=True, env=os.environ
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            logger.info(f"Job submitted successfully: {output}")

            # Extract job ID from bsub output (format: "Job <12345> is submitted")
            match = re.search(r"<(\d+)>", output)
            job_id = match.group(1) if match else "unknown"

            return {
                "success": True,
                "job_id": job_id,
                "task_paths": yaml_paths,
                "command": " ".join(bsub_cmd),
                "message": f"Task submitted as job {job_id}",
            }
        else:
            error_msg = result.stderr or result.stdout
            logger.error(f"LSF submission failed: {error_msg}")
            return {"success": False, "error": f"LSF error: {error_msg}"}

    except Exception as e:
        logger.error(f"Submission error: {str(e)}")
        return {"success": False, "error": str(e)}


# Global state for BBX generator
bbx_generator_state = {
    "dataset_path": None,
    "num_boxes": 0,
    "bounding_boxes": [],
    "viewer": None,
    "viewer_process": None,
    "viewer_url": None,
    "viewer_state": None,
}


@app.route("/api/bbx-generator", methods=["POST"])
def start_bbx_generator():
    """Start the Neuroglancer viewer for creating bounding boxes"""
    try:
        # Set Neuroglancer server to bind to 0.0.0.0 for external access
        neuroglancer.set_server_bind_address("0.0.0.0")

        data = request.json
        dataset_path = data.get("dataset_path", "")
        num_boxes = data.get("num_boxes", 1)

        if not dataset_path:
            return jsonify({"error": "Dataset path is required"}), 400

        # Create Neuroglancer viewer
        viewer = neuroglancer.Viewer()

        with viewer.txn() as s:
            # Set coordinate space
            s.dimensions = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[8, 8, 8],
            )

            # Add image layer
            s.layers["fibsem"] = get_raw_layer(dataset_path)

            # Add LOCAL annotation layer for bounding boxes
            s.layers["annotations"] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=[1, 1, 1],
                ),
            )

        # Store state
        bbx_generator_state["dataset_path"] = dataset_path
        bbx_generator_state["num_boxes"] = num_boxes
        bbx_generator_state["bounding_boxes"] = []
        bbx_generator_state["viewer"] = viewer

        # Get the viewer URL and fix localhost reference
        viewer_url = str(viewer)

        # Replace localhost with the actual request host for external access
        # Parse the URL and replace localhost with the client's host
        if "localhost" in viewer_url:
            # Get the client's host from the request
            client_host = request.host.split(":")[
                0
            ]  # Get just the host part without port
            viewer_url = viewer_url.replace("localhost", client_host)
            logger.info(f"Replaced localhost with {client_host} in viewer URL")

        bbx_generator_state["viewer_url"] = viewer_url
        bbx_generator_state["viewer_state"] = viewer.state

        logger.info(f"Starting BBX generator with viewer URL: {viewer_url}")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Target boxes: {num_boxes}")

        # For iframe access, we need to return the raw viewer URL
        # Neuroglancer server should be accessible at the returned URL
        return jsonify(
            {
                "success": True,
                "viewer_url": viewer_url,
                "dataset_path": dataset_path,
                "num_boxes": num_boxes,
            }
        )

    except Exception as e:
        logger.error(f"Error starting BBX generator: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bbx-generator/status", methods=["GET"])
def get_bbx_generator_status():
    """Get current status of bounding box generation"""
    try:
        # Extract bounding boxes from viewer if it exists
        bboxes = []
        if bbx_generator_state.get("viewer"):
            viewer = bbx_generator_state["viewer"]
            try:
                with viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if hasattr(annotations_layer, "annotations"):
                            for ann in annotations_layer.annotations:
                                # Check if this is a bounding box annotation
                                if (
                                    type(ann).__name__
                                    == "AxisAlignedBoundingBoxAnnotation"
                                ):
                                    point_a = ann.point_a
                                    point_b = ann.point_b

                                    # Ensure point_a is the min and point_b is the max
                                    offset = [
                                        min(point_a[j], point_b[j]) for j in range(3)
                                    ]
                                    max_point = [
                                        max(point_a[j], point_b[j]) for j in range(3)
                                    ]
                                    shape = [
                                        int(max_point[j] - offset[j]) for j in range(3)
                                    ]
                                    offset = [int(x) for x in offset]

                                    bboxes.append(
                                        {
                                            "offset": offset,
                                            "shape": shape,
                                        }
                                    )
                    except KeyError:
                        logger.warning("Annotations layer not found in viewer")
            except Exception as e:
                logger.warning(f"Error extracting bboxes from viewer: {str(e)}")

        bbx_generator_state["bounding_boxes"] = bboxes

        return jsonify(
            {
                "dataset_path": bbx_generator_state.get("dataset_path"),
                "num_boxes": bbx_generator_state.get("num_boxes"),
                "bounding_boxes": bboxes,
                "count": len(bboxes),
            }
        )

    except Exception as e:
        logger.error(f"Error getting BBX status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bbx-generator/finalize", methods=["POST"])
def finalize_bbx_generation():
    """Finalize bounding box generation and return results"""
    try:
        # Extract final bounding boxes from viewer
        bboxes = []
        if bbx_generator_state.get("viewer"):
            viewer = bbx_generator_state["viewer"]
            try:
                with viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if hasattr(annotations_layer, "annotations"):
                            for ann in annotations_layer.annotations:
                                # Check if this is a bounding box annotation
                                if (
                                    type(ann).__name__
                                    == "AxisAlignedBoundingBoxAnnotation"
                                ):
                                    point_a = ann.point_a
                                    point_b = ann.point_b

                                    # Ensure point_a is the min and point_b is the max
                                    offset = [
                                        min(point_a[j], point_b[j]) for j in range(3)
                                    ]
                                    max_point = [
                                        max(point_a[j], point_b[j]) for j in range(3)
                                    ]
                                    shape = [
                                        int(max_point[j] - offset[j]) for j in range(3)
                                    ]
                                    offset = [int(x) for x in offset]

                                    bboxes.append(
                                        {
                                            "offset": offset,
                                            "shape": shape,
                                        }
                                    )
                    except KeyError:
                        logger.warning("Annotations layer not found in viewer")
            except Exception as e:
                logger.warning(f"Error extracting final bboxes: {str(e)}")

        # Reset state
        bbx_generator_state["dataset_path"] = None
        bbx_generator_state["num_boxes"] = 0
        bbx_generator_state["bounding_boxes"] = []
        bbx_generator_state["viewer_url"] = None
        bbx_generator_state["viewer"] = None

        return jsonify(
            {"success": True, "bounding_boxes": bboxes, "count": len(bboxes)}
        )

    except Exception as e:
        logger.error(f"Error finalizing BBX generation: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/finetune/models", methods=["GET"])
def get_finetune_models():
    """Get available models for finetuning with their configurations."""
    try:
        models = []

        # Extract from g.models_config
        if hasattr(g, "models_config") and g.models_config:
            for model_config in g.models_config:
                try:
                    config = model_config.config
                    models.append(
                        {
                            "name": model_config.name,
                            "write_shape": list(config.write_shape),
                            "output_voxel_size": list(config.output_voxel_size),
                            "output_channels": config.output_channels,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract config for {model_config.name}: {e}"
                    )

        # If no models found in g.models_config, try to get from running jobs
        # This handles the case where models were submitted via GUI after app started
        if len(models) == 0 and hasattr(g, "jobs") and g.jobs:
            logger.warning("No models in g.models_config, checking running jobs")
            # Try to get model configs from jobs
            for job in g.jobs:
                if hasattr(job, "model_name"):
                    job_model_name = job.model_name
                    # Look for config in pipeline_model_configs (if available)
                    if hasattr(g, "pipeline_model_configs") and job_model_name in g.pipeline_model_configs:
                        config_dict = g.pipeline_model_configs[job_model_name]
                        try:
                            models.append(
                                {
                                    "name": job_model_name,
                                    "write_shape": config_dict.get("write_shape", []),
                                    "output_voxel_size": config_dict.get("output_voxel_size", []),
                                    "output_channels": config_dict.get("output_channels", 1),
                                }
                            )
                            logger.info(f"Found config for {job_model_name} in pipeline_model_configs")
                        except Exception as e:
                            logger.warning(f"Could not extract config for {job_model_name}: {e}")
                    else:
                        logger.warning(f"No configuration found for running job: {job_model_name}")
                        logger.warning(f"  → Model needs write_shape, output_voxel_size, and output_channels for finetuning")
                        logger.warning(f"  → Consider restarting with a proper YAML configuration file")

        # Determine selected model
        selected = models[0]["name"] if len(models) == 1 else None

        return jsonify({"models": models, "selected_model": selected})

    except Exception as e:
        logger.error(f"Error getting finetune models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/finetune/view-center", methods=["GET"])
def get_view_center():
    """Get current view center position from Neuroglancer viewer."""
    try:
        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Access viewer state using transaction
        with g.viewer.txn() as s:
            # Get the current view position (center of view)
            position = s.position

            # Get the viewer dimensions to extract scales
            dimensions = s.dimensions
            scales_nm = None

            if dimensions and hasattr(dimensions, "scales"):
                # CoordinateSpace has scales attribute directly
                scales_nm = list(dimensions.scales)
                logger.info(f"Viewer scales (raw): {scales_nm}")

                # Check units and convert if needed
                if hasattr(dimensions, "units"):
                    units = dimensions.units
                    # units can be a string (same for all axes) or list
                    if isinstance(units, str):
                        units = [units] * len(scales_nm)

                    # Convert to nm if needed
                    converted_scales = []
                    for scale, unit in zip(scales_nm, units):
                        if unit == "m":
                            converted_scales.append(scale * 1e9)  # meters to nanometers
                        elif unit == "nm":
                            converted_scales.append(scale)
                        else:
                            logger.warning(f"Unknown unit: {unit}, assuming nm")
                            converted_scales.append(scale)
                    scales_nm = converted_scales

                logger.info(f"Viewer scales (nm): {scales_nm}")
            else:
                logger.warning("Could not extract scales from viewer dimensions")

            # Convert to list if it's a numpy array or coordinate object
            if hasattr(position, "tolist"):
                position = position.tolist()
            elif hasattr(position, "__iter__"):
                position = list(position)

            logger.info(f"Got view center position: {position}")

            return jsonify(
                {"success": True, "position": position, "scales_nm": scales_nm}
            )

    except Exception as e:
        logger.error(f"Error getting view center position: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/create-crop", methods=["POST"])
def create_annotation_crop():
    """Create an annotation crop centered at view center position."""
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Roi, Coordinate

        data = request.get_json()
        model_name = data.get("model_name")
        output_path = data.get("output_path")  # User-specified output directory

        if not hasattr(g, "models_config") or not g.models_config:
            return jsonify({"success": False, "error": "No models loaded"}), 400

        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Get view center and scales automatically from viewer
        with g.viewer.txn() as s:
            # Get the current view position (center of view)
            position = s.position

            # Get the viewer dimensions to extract scales
            dimensions = s.dimensions
            viewer_scales_nm = None

            if dimensions and hasattr(dimensions, "scales"):
                # CoordinateSpace has scales attribute directly
                scales_nm = list(dimensions.scales)

                # Check units and convert if needed
                if hasattr(dimensions, "units"):
                    units = dimensions.units
                    # units can be a string (same for all axes) or list
                    if isinstance(units, str):
                        units = [units] * len(scales_nm)

                    # Convert to nm if needed
                    converted_scales = []
                    for scale, unit in zip(scales_nm, units):
                        if unit == "m":
                            converted_scales.append(scale * 1e9)  # meters to nanometers
                        elif unit == "nm":
                            converted_scales.append(scale)
                        else:
                            logger.warning(f"Unknown unit: {unit}, assuming nm")
                            converted_scales.append(scale)
                    viewer_scales_nm = converted_scales
                else:
                    viewer_scales_nm = scales_nm

            # Convert to list if it's a numpy array or coordinate object
            if hasattr(position, "tolist"):
                view_center = position.tolist()
            elif hasattr(position, "__iter__"):
                view_center = list(position)
            else:
                view_center = position

            view_center = np.array(view_center)

        logger.info(f"Auto-detected view center: {view_center}")
        logger.info(f"Auto-detected viewer scales: {viewer_scales_nm} nm")

        # Find model config
        model_config = None
        for mc in g.models_config:
            if mc.name == model_name:
                model_config = mc
                break

        if not model_config:
            return (
                jsonify({"success": False, "error": f"Model {model_name} not found"}),
                404,
            )

        # Get model parameters
        config = model_config.config
        read_shape = np.array(config.read_shape)  # Physical size in nm for raw data
        write_shape = np.array(config.write_shape)  # Physical size in nm for prediction
        input_voxel_size = np.array(config.input_voxel_size)  # nm per voxel for input
        output_voxel_size = np.array(
            config.output_voxel_size
        )  # nm per voxel for output
        output_channels = config.output_channels

        # Convert view center to nm using viewer scales
        if viewer_scales_nm is not None:
            viewer_scales_nm = np.array(viewer_scales_nm)
            view_center_nm = view_center * viewer_scales_nm
            logger.info(
                f"Converted view center from {view_center} (viewer coords) to {view_center_nm} nm"
            )
            logger.info(f"  Using viewer scales: {viewer_scales_nm} nm")
        else:
            # Fallback: assume it's already in nm
            view_center_nm = view_center
            logger.warning(
                "No viewer scales provided, assuming view center is already in nm"
            )

        # Calculate raw crop size in voxels (use read_shape and input_voxel_size)
        raw_crop_shape_voxels = (read_shape / input_voxel_size).astype(int)

        # Calculate annotation crop size in voxels (use write_shape and output_voxel_size)
        annotation_crop_shape_voxels = (write_shape / output_voxel_size).astype(int)

        # Calculate crop offset for raw (center the crop at view center)
        half_read_shape = read_shape / 2
        raw_crop_offset_nm = view_center_nm - half_read_shape
        raw_crop_offset_voxels = (raw_crop_offset_nm / input_voxel_size).astype(int)

        # Calculate crop offset for annotation (center the crop at view center)
        half_write_shape = write_shape / 2
        annotation_crop_offset_nm = view_center_nm - half_write_shape
        annotation_crop_offset_voxels = (
            annotation_crop_offset_nm / output_voxel_size
        ).astype(int)

        # Generate unique crop ID
        crop_id = f"{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create zarr structure with timestamped session directory
        if output_path:
            # Use user-specified output path with timestamped session
            session_path = get_or_create_session_path(output_path)
            corrections_dir = os.path.join(session_path, "corrections")
            os.makedirs(corrections_dir, exist_ok=True)

            # Initialize as zarr group if not already
            import zarr
            zarr.open_group(corrections_dir, mode='a')

            zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")
            logger.info(f"Using session path: {session_path}")
            logger.info(f"Corrections directory: {corrections_dir}")
        else:
            # Fallback to default location
            corrections_dir = os.path.expanduser("~/.cellmap_flow/corrections")
            os.makedirs(corrections_dir, exist_ok=True)

            # Initialize as zarr group if not already
            import zarr
            zarr.open_group(corrections_dir, mode='a')

            zarr_path = os.path.join(corrections_dir, f"{crop_id}.zarr")

        # Get dataset path
        dataset_path = getattr(g, "dataset_path", "unknown")

        # Create ImageDataInterface first to get the data dtype
        logger.info(f"Creating ImageDataInterface for {dataset_path}")
        logger.info(f"Using input voxel size: {input_voxel_size} nm")
        try:
            idi = ImageDataInterface(dataset_path, voxel_size=input_voxel_size)
            # Get the dtype from the tensorstore
            raw_dtype = str(idi.ts.dtype)
            logger.info(f"Dataset dtype: {raw_dtype}")
        except Exception as e:
            logger.error(f"Error creating ImageDataInterface: {e}")
            return (
                jsonify(
                    {"success": False, "error": f"Failed to access dataset: {str(e)}"}
                ),
                500,
            )

        # Create zarr with OME-NGFF metadata (no mask needed)
        success, zarr_info = create_correction_zarr(
            zarr_path=zarr_path,
            raw_crop_shape=raw_crop_shape_voxels,
            raw_voxel_size=input_voxel_size,
            raw_offset=raw_crop_offset_voxels,
            annotation_crop_shape=annotation_crop_shape_voxels,
            annotation_voxel_size=output_voxel_size,
            annotation_offset=annotation_crop_offset_voxels,
            dataset_path=dataset_path,
            model_name=model_name,
            output_channels=output_channels,
            raw_dtype=raw_dtype,
            create_mask=False,
        )

        if not success:
            return jsonify({"success": False, "error": zarr_info}), 500

        # Read and fill raw data from the dataset
        logger.info(f"Reading raw data from {dataset_path}")
        try:

            # Define ROI for the crop in physical coordinates (nm)
            # Center the crop at view_center_nm
            roi = Roi(
                offset=Coordinate(view_center_nm - read_shape / 2),
                shape=Coordinate(read_shape),
            )
            logger.info(f"Reading ROI: offset={roi.offset}, shape={roi.shape}")

            # Read the data using tensorstore interface
            raw_data = idi.to_ndarray_ts(roi)
            logger.info(
                f"Read raw data with shape: {raw_data.shape}, dtype: {raw_data.dtype}"
            )

            # Write to zarr
            raw_zarr = zarr.open(zarr_path, mode="r+")
            raw_zarr["raw/s0"][:] = raw_data
            logger.info(f"Wrote raw data to {zarr_path}/raw/s0")

        except Exception as e:
            logger.error(f"Error reading/writing raw data: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return (
                jsonify(
                    {"success": False, "error": f"Failed to read raw data: {str(e)}"}
                ),
                500,
            )

        # Start/ensure MinIO is running and upload
        minio_url = ensure_minio_serving(zarr_path, crop_id, output_base_dir=corrections_dir)

        return jsonify(
            {
                "success": True,
                "crop_id": crop_id,
                "zarr_path": zarr_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "metadata": {
                    "center_position_nm": view_center_nm.tolist(),
                    "raw_crop_offset": raw_crop_offset_voxels.tolist(),
                    "raw_crop_shape": raw_crop_shape_voxels.tolist(),
                    "raw_voxel_size": input_voxel_size.tolist(),
                    "annotation_crop_offset": annotation_crop_offset_voxels.tolist(),
                    "annotation_crop_shape": annotation_crop_shape_voxels.tolist(),
                    "annotation_voxel_size": output_voxel_size.tolist(),
                },
            }
        )

    except Exception as e:
        logger.error(f"Error creating annotation crop: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/create-volume", methods=["POST"])
def create_annotation_volume():
    """Create a sparse annotation volume covering the full dataset extent."""
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface
        from funlib.geometry import Coordinate

        data = request.get_json()
        model_name = data.get("model_name")
        output_path = data.get("output_path")

        if not hasattr(g, "models_config") or not g.models_config:
            return jsonify({"success": False, "error": "No models loaded"}), 400

        # Find model config
        model_config = None
        for mc in g.models_config:
            if mc.name == model_name:
                model_config = mc
                break

        if not model_config:
            return (
                jsonify({"success": False, "error": f"Model {model_name} not found"}),
                404,
            )

        # Get model parameters
        config = model_config.config
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        input_voxel_size = np.array(config.input_voxel_size)
        output_voxel_size = np.array(config.output_voxel_size)

        # Compute output_size and input_size in voxels
        output_size = (write_shape / output_voxel_size).astype(int)
        input_size = (read_shape / input_voxel_size).astype(int)

        # Get dataset path
        dataset_path = getattr(g, "dataset_path", None)
        if not dataset_path:
            return (
                jsonify({"success": False, "error": "No dataset path configured"}),
                400,
            )

        # Get full dataset extent
        logger.info(f"Getting dataset extent from {dataset_path}")
        try:
            idi = ImageDataInterface(dataset_path, voxel_size=output_voxel_size)
            dataset_roi = idi.roi
            dataset_offset_nm = np.array(dataset_roi.offset)
            dataset_shape_nm = np.array(dataset_roi.shape)

            # Convert to voxels at output resolution
            dataset_shape_voxels = (dataset_shape_nm / output_voxel_size).astype(int)

            # Snap up to chunk_size (output_size) multiples
            dataset_shape_voxels = (
                np.ceil(dataset_shape_voxels / output_size).astype(int) * output_size
            )

            logger.info(
                f"Dataset extent: offset={dataset_offset_nm} nm, "
                f"shape={dataset_shape_voxels} voxels (at {output_voxel_size} nm/voxel)"
            )
        except Exception as e:
            logger.error(f"Error getting dataset extent: {e}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"Failed to access dataset: {str(e)}",
                    }
                ),
                500,
            )

        # Generate volume ID
        volume_id = (
            f"vol-{uuid.uuid4().hex[:8]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Set up output directory
        if output_path:
            session_path = get_or_create_session_path(output_path)
            corrections_dir = os.path.join(session_path, "corrections")
            os.makedirs(corrections_dir, exist_ok=True)
            zarr.open_group(corrections_dir, mode="a")
            zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")
            logger.info(f"Using session path: {session_path}")
        else:
            corrections_dir = os.path.expanduser("~/.cellmap_flow/corrections")
            os.makedirs(corrections_dir, exist_ok=True)
            zarr.open_group(corrections_dir, mode="a")
            zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")

        # Create the annotation volume zarr
        success, zarr_info = create_annotation_volume_zarr(
            zarr_path=zarr_path,
            dataset_shape_voxels=dataset_shape_voxels,
            output_voxel_size=output_voxel_size,
            dataset_offset_nm=dataset_offset_nm,
            chunk_size=output_size,
            dataset_path=dataset_path,
            model_name=model_name,
            input_size=input_size,
            input_voxel_size=input_voxel_size,
        )

        if not success:
            return jsonify({"success": False, "error": zarr_info}), 500

        # Upload to MinIO
        minio_url = ensure_minio_serving(
            zarr_path, volume_id, output_base_dir=corrections_dir
        )

        # Store volume metadata for sync to use
        annotation_volumes[volume_id] = {
            "zarr_path": zarr_path,
            "model_name": model_name,
            "output_size": output_size.tolist(),
            "input_size": input_size.tolist(),
            "input_voxel_size": input_voxel_size.tolist(),
            "output_voxel_size": output_voxel_size.tolist(),
            "dataset_path": dataset_path,
            "dataset_offset_nm": dataset_offset_nm.tolist(),
            "corrections_dir": corrections_dir,
            "extracted_chunks": set(),
        }

        return jsonify(
            {
                "success": True,
                "volume_id": volume_id,
                "zarr_path": zarr_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "metadata": {
                    "dataset_shape_voxels": dataset_shape_voxels.tolist(),
                    "chunk_size": output_size.tolist(),
                    "output_voxel_size": output_voxel_size.tolist(),
                    "dataset_offset_nm": dataset_offset_nm.tolist(),
                },
            }
        )

    except Exception as e:
        logger.error(f"Error creating annotation volume: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/add-to-viewer", methods=["POST"])
def add_crop_to_viewer():
    """Add annotation crop or volume layer to Neuroglancer viewer."""
    try:
        data = request.get_json()
        crop_id = data.get("crop_id")
        minio_url = data.get("minio_url")

        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Add layer to viewer
        with g.viewer.txn() as s:
            layer_name = data.get("layer_name", f"annotation_{crop_id}")
            # Configure source with writing enabled
            source_config = {
                "url": f"s3+{minio_url}",
                "subsources": {"default": {"writingEnabled": True}, "bounds": {}},
            }
            layer = neuroglancer.SegmentationLayer(source=source_config)
            s.layers[layer_name] = layer

        logger.info(f"Added layer {layer_name} to viewer")

        return jsonify(
            {
                "success": True,
                "message": "Layer added to viewer",
                "layer_name": layer_name,
            }
        )

    except Exception as e:
        logger.error(f"Error adding layer to viewer: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/sync-annotations", methods=["POST"])
def sync_annotations_manually():
    """Manually trigger sync of annotations from MinIO to local disk."""
    try:
        data = request.get_json()
        crop_id = data.get("crop_id", None)  # If None, sync all
        force = data.get("force", True)  # Force sync by default for manual trigger

        if crop_id:
            # Sync single crop
            success = sync_annotation_from_minio(crop_id, force=force)
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Synced annotation for {crop_id}"
                })
            else:
                return jsonify({
                    "success": False,
                    "message": f"No updates to sync for {crop_id}"
                })
        else:
            # Sync all crops
            if not minio_state["ip"] or not minio_state["port"]:
                return jsonify({"success": False, "error": "MinIO not initialized"}), 400

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

                zarrs = s3.ls(minio_state['bucket'])
                zarr_ids = [Path(c).name.replace('.zarr', '') for c in zarrs if c.endswith('.zarr')]

                synced_count = 0
                for zid in zarr_ids:
                    # Route volumes vs crops
                    try:
                        zarr_name = f"{zid}.zarr"
                        attrs_path = f"{minio_state['bucket']}/{zarr_name}/.zattrs"
                        if s3.exists(attrs_path):
                            root_attrs = json.loads(s3.cat(attrs_path))
                            if root_attrs.get("type") == "annotation_volume":
                                if sync_annotation_volume_from_minio(zid, force=force):
                                    synced_count += 1
                                continue
                    except Exception:
                        pass
                    if sync_annotation_from_minio(zid, force=force):
                        synced_count += 1

                return jsonify({
                    "success": True,
                    "message": f"Synced {synced_count} annotations",
                    "synced_count": synced_count,
                    "total_crops": len(zarr_ids)
                })

            except Exception as e:
                logger.error(f"Error syncing all annotations: {e}")
                return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        logger.error(f"Error in sync endpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/submit", methods=["POST"])
def submit_finetuning():
    """
    Submit a finetuning job to the LSF cluster.

    Request body:
    {
        "model_name": "model_name",
        "lora_r": 8,
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 0.0001
    }
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        corrections_path_str = data.get("corrections_path")
        lora_r = data.get("lora_r", 8)
        num_epochs = data.get("num_epochs", 10)
        batch_size = data.get("batch_size", 2)
        learning_rate = data.get("learning_rate", 1e-4)
        checkpoint_path_override = data.get("checkpoint_path")  # Optional override
        auto_serve = data.get("auto_serve", True)  # Auto-serve by default
        loss_type = data.get("loss_type", "mse")
        label_smoothing = data.get("label_smoothing", 0.1)
        distillation_lambda = data.get("distillation_lambda", 0.0)
        distillation_scope = data.get("distillation_scope", "unlabeled")
        margin = data.get("margin", 0.3)
        balance_classes = data.get("balance_classes", False)

        if not model_name:
            return jsonify({"success": False, "error": "model_name is required"}), 400

        if not corrections_path_str:
            return jsonify({"success": False, "error": "corrections_path is required. Please specify the output path where annotation crops are saved."}), 400

        # Find model config
        model_config = None
        for config in g.models_config:
            if config.name == model_name:
                model_config = config
                break

        if not model_config:
            return jsonify({"success": False, "error": f"Model {model_name} not found"}), 404

        # Get the corrections path from the user's input
        # This will be the base path they entered (e.g., "output/to/here")
        # We need to find the actual corrections directory with the session timestamp
        base_corrections_path = Path(corrections_path_str)

        # Check if this looks like a session path with corrections subdirectory
        actual_corrections_path = None
        if base_corrections_path.name == "corrections" and base_corrections_path.exists():
            # User provided the full path including "/corrections"
            actual_corrections_path = base_corrections_path
            session_path = base_corrections_path.parent
        else:
            # User provided the base path - find the session with corrections
            session_path = get_or_create_session_path(str(base_corrections_path))
            actual_corrections_path = Path(session_path) / "corrections"

        if not actual_corrections_path.exists():
            return jsonify({
                "success": False,
                "error": f"Corrections path does not exist: {actual_corrections_path}. Please create annotation crops first."
            }), 400

        # Derive output base from session path for finetuning outputs
        output_base = Path(session_path)
        logger.info(f"Using session path for finetuning: {session_path}")
        logger.info(f"Corrections path: {actual_corrections_path}")
        logger.info(f"Output base: {output_base}")

        # Auto-sync annotations from MinIO before training
        try:
            sync_all_annotations_from_minio()
        except Exception as e:
            logger.warning(f"Error syncing annotations before training: {e}")

        # Detect sparse annotations: check if any correction has source=sparse_volume
        has_sparse = False
        try:
            for p in actual_corrections_path.iterdir():
                if p.suffix == ".zarr" and (p / ".zattrs").exists():
                    attrs = json.loads((p / ".zattrs").read_text())
                    if attrs.get("source") == "sparse_volume":
                        has_sparse = True
                        break
        except Exception as e:
            logger.warning(f"Error checking for sparse annotations: {e}")

        sparse_auto_switched = False
        if has_sparse:
            logger.info("Detected sparse annotations, will use mask_unannotated=True")
            # Auto-switch to better defaults for sparse scribbles
            if loss_type == "mse":  # only override if user hasn't explicitly chosen
                loss_type = "margin"
                distillation_lambda = 0.5
                sparse_auto_switched = True
                logger.info("Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations")

        # Submit job
        finetune_job = finetune_job_manager.submit_finetuning_job(
            model_config=model_config,
            corrections_path=actual_corrections_path,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_base=output_base,
            checkpoint_path_override=Path(checkpoint_path_override) if checkpoint_path_override else None,
            auto_serve=auto_serve,
            mask_unannotated=has_sparse,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            balance_classes=balance_classes,
        )

        logger.info(f"Submitted finetuning job: {finetune_job.job_id}")

        # Get LSF job ID or local PID
        lsf_job_id = None
        if finetune_job.lsf_job:
            if hasattr(finetune_job.lsf_job, 'job_id'):
                lsf_job_id = finetune_job.lsf_job.job_id
            elif hasattr(finetune_job.lsf_job, 'process'):
                lsf_job_id = f"PID:{finetune_job.lsf_job.process.pid}"

        response = {
            "success": True,
            "job_id": finetune_job.job_id,
            "lsf_job_id": lsf_job_id,
            "output_dir": str(finetune_job.output_dir),
            "message": "Finetuning job submitted successfully"
        }
        if sparse_auto_switched:
            response["note"] = "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"

        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting finetuning job: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/jobs", methods=["GET"])
def get_finetuning_jobs():
    """Get list of all finetuning jobs."""
    try:
        jobs = finetune_job_manager.list_jobs()
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/job/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    """Get detailed status of a specific job."""
    try:
        status = finetune_job_manager.get_job_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, **status})
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/job/<job_id>/logs", methods=["GET"])
def get_job_logs(job_id):
    """Get training logs for a specific job."""
    try:
        logs = finetune_job_manager.get_job_logs(job_id)
        if logs is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        logger.error(f"Error getting job logs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/job/<job_id>/logs/stream", methods=["GET"])
def stream_job_logs(job_id):
    """Server-Sent Events stream for live training logs."""

    import re as _re

    # Patterns to filter out of the log stream
    _log_filters = [
        _re.compile(r"^DEBUG", _re.IGNORECASE),
        _re.compile(r"^\s+base_model\.\S+\.lora_"),  # gradient norm lines
        _re.compile(r"^INFO:werkzeug:"),
        _re.compile(r"^Array metadata \(scale="),  # server chunk metadata
        _re.compile(r"^Host name:"),
        _re.compile(r"^DEBUG trainer:"),
    ]

    def _should_show(line):
        for pat in _log_filters:
            if pat.search(line):
                return False
        return True

    def generate():
        # Check if job exists
        if job_id not in finetune_job_manager.jobs:
            yield f"data: Job {job_id} not found\n\n"
            return

        finetune_job = finetune_job_manager.jobs[job_id]

        # Send existing log content first
        if finetune_job.log_file.exists():
            try:
                with open(finetune_job.log_file, "r") as f:
                    existing_content = f.read()
                    for line in existing_content.split("\n"):
                        if line and _should_show(line):
                            yield f"data: {line}\n\n"
            except Exception as e:
                logger.error(f"Error reading log file: {e}")

        # Then tail for new content
        last_position = finetune_job.log_file.stat().st_size if finetune_job.log_file.exists() else 0

        while finetune_job.status.value in ["PENDING", "RUNNING"]:
            try:
                if finetune_job.log_file.exists():
                    with open(finetune_job.log_file, "r") as f:
                        f.seek(last_position)
                        new_content = f.read()
                        last_position = f.tell()

                        if new_content:
                            for line in new_content.split("\n"):
                                if line and _should_show(line):
                                    yield f"data: {line}\n\n"

                time.sleep(1)  # Poll every second

            except Exception as e:
                logger.error(f"Error streaming logs: {e}")
                break

        yield f"data: === Training {finetune_job.status.value} ===\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/finetune/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running finetuning job."""
    try:
        success = finetune_job_manager.cancel_job(job_id)

        if success:
            return jsonify({"success": True, "message": f"Job {job_id} cancelled"})
        else:
            return jsonify({"success": False, "error": "Failed to cancel job"}), 400

    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/job/<job_id>/inference-server", methods=["GET"])
def get_inference_server_status(job_id):
    """Get inference server status for a finetuning job."""
    try:
        job = finetune_job_manager.get_job(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({
            "success": True,
            "ready": job.inference_server_ready,
            "url": job.inference_server_url,
            "model_name": job.finetuned_model_name,
            "model_script_path": str(job.model_script_path) if job.model_script_path else None
        })

    except Exception as e:
        logger.error(f"Error getting inference server status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/viewer/add-finetuned-layer", methods=["POST"])
def add_finetuned_layer_to_viewer():
    """Add finetuned model layer to Neuroglancer viewer and register model in system."""
    try:
        data = request.get_json()
        server_url = data.get("server_url")
        model_name = data.get("model_name")
        model_script_path = data.get("model_script_path")

        if not server_url or not model_name:
            return jsonify({"success": False, "error": "Missing server_url or model_name"}), 400

        logger.info(f"Registering finetuned model: {model_name} at {server_url}")

        # 1. Load model config from script if provided
        if model_script_path and Path(model_script_path).exists():
            try:
                model_config = load_safe_config(model_script_path)

                # Add to models_config if not already there
                if not hasattr(g, 'models_config'):
                    g.models_config = []

                # Remove old finetuned model configs with same base name
                base_model_name = model_name.rsplit("_finetuned_", 1)[0] if "_finetuned_" in model_name else model_name
                g.models_config = [
                    mc for mc in g.models_config
                    if not (hasattr(mc, 'name') and mc.name.startswith(f"{base_model_name}_finetuned"))
                ]

                # Add new config
                g.models_config.append(model_config)
                logger.info(f"✓ Loaded model config from {model_script_path}")
            except Exception as e:
                logger.warning(f"Could not load model config: {e}")

        # 2. Add to model_catalog under "Finetuned" group
        if not hasattr(g, 'model_catalog'):
            g.model_catalog = {}

        if "Finetuned" not in g.model_catalog:
            g.model_catalog["Finetuned"] = {}

        # Remove old finetuned models with same base name
        base_model_name = model_name.rsplit("_finetuned_", 1)[0] if "_finetuned_" in model_name else model_name
        g.model_catalog["Finetuned"] = {
            name: path for name, path in g.model_catalog["Finetuned"].items()
            if not name.startswith(f"{base_model_name}_finetuned")
        }

        # Add new finetuned model
        g.model_catalog["Finetuned"][model_name] = model_script_path if model_script_path else ""
        logger.info(f"✓ Added to model catalog: Finetuned/{model_name}")

        # 3. Create a Job object for the running inference server
        from cellmap_flow.utils.bsub_utils import LSFJob

        # Extract job_id from the finetune job (the training job is running the server)
        # Find the corresponding finetune job
        finetune_job = None
        for job_id, ft_job in finetune_job_manager.jobs.items():
            if ft_job.finetuned_model_name == model_name:
                finetune_job = ft_job
                break

        if finetune_job and finetune_job.job_id:
            # Create Job object pointing to the running server
            inference_job = LSFJob(job_id=finetune_job.job_id, model_name=model_name)
            inference_job.host = server_url
            inference_job.status = finetune_job.status

            # Remove old jobs for this base model
            g.jobs = [
                j for j in g.jobs
                if not (hasattr(j, 'model_name') and j.model_name and j.model_name.startswith(f"{base_model_name}_finetuned"))
            ]

            # Add to jobs
            g.jobs.append(inference_job)
            logger.info(f"✓ Created Job object for {model_name} with job_id {finetune_job.job_id}")
        else:
            logger.warning(f"Could not find finetune job for {model_name}, Job object not created")

        # 4. Add neuroglancer layer
        layer_name = model_name  # Use model name directly (not prefixed with "finetuned_")

        with g.viewer.txn() as s:
            # Remove old finetuned layer if it exists
            if layer_name in s.layers:
                logger.info(f"Removing old finetuned layer: {layer_name}")
                del s.layers[layer_name]

            # Add new layer pointing to inference server
            from cellmap_flow.utils.neuroglancer_utils import get_norms_post_args
            from cellmap_flow.utils.web_utils import ARGS_KEY

            st_data = get_norms_post_args(g.input_norms, g.postprocess)

            # Create image layer for finetuned model (same style as normal models)
            import neuroglancer
            s.layers[layer_name] = neuroglancer.ImageLayer(
                source=f"zarr://{server_url}/{model_name}{ARGS_KEY}{st_data}{ARGS_KEY}",
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){{emitRGB(color * normalized());}}""",
            )

        logger.info(f"✓ Added neuroglancer layer: {layer_name} -> {server_url}")

        return jsonify({
            "success": True,
            "layer_name": layer_name,
            "model_name": model_name,
            "reload_page": True  # Signal frontend to reload to see new model in catalog
        })

    except Exception as e:
        logger.error(f"Error adding finetuned layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/finetune/job/<job_id>/restart", methods=["POST"])
def restart_finetuning_job(job_id):
    """Restart training on the same GPU via signal file.

    The CLI watches for a restart_signal.json file and restarts training
    without needing a new job/GPU allocation.
    """
    try:
        data = request.get_json() or {}

        # Sync annotations from MinIO before restarting training
        try:
            sync_all_annotations_from_minio()
        except Exception as e:
            logger.warning(f"Error syncing annotations before restart: {e}")

        # Extract updated parameters
        updated_params = {}
        for key in ["num_epochs", "batch_size", "learning_rate", "loss_type", "distillation_lambda", "distillation_scope", "margin"]:
            if key in data and data[key] is not None:
                updated_params[key] = data[key]

        # Send restart signal (same job, same GPU)
        job = finetune_job_manager.restart_finetuning_job(
            job_id=job_id,
            updated_params=updated_params
        )

        return jsonify({
            "success": True,
            "job_id": job.job_id,
            "message": "Restart signal sent. Training will restart on the same GPU.",
        })

    except Exception as e:
        logger.error(f"Error restarting job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def create_and_run_app(neuroglancer_url=None, inference_servers=None):
    global NEUROGLANCER_URL, INFERENCE_SERVER
    NEUROGLANCER_URL = neuroglancer_url
    INFERENCE_SERVER = inference_servers
    hostname = socket.gethostname()
    port = 0
    logger.warning(f"Host name: {hostname}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    # app.run(debug=True)
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
