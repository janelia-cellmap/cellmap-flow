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
            compressor=None,  # neuroglancer browser reader cannot decode blosc
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
            compressor=None,  # neuroglancer browser reader cannot decode blosc
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
                compressor=None,  # neuroglancer browser reader cannot decode blosc
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
    claimed_output_voxel_size=None,
    claimed_input_voxel_size=None,
    input_norm_config=None,
):
    """
    Create a sparse annotation volume zarr covering the full dataset extent.

    The volume has chunk_size = model output_size so each chunk maps to one
    training sample. Only metadata files are created (no chunk data), so the
    zarr is tiny regardless of dataset size.

    Label scheme: 0=unannotated (ignored), 1=background, 2=foreground.

    Args:
        output_voxel_size, input_voxel_size: the EFFECTIVE voxel sizes used
            for the actual grid alignment (typically the dataset's closest
            available scale to the model's claimed voxel size).
        claimed_output_voxel_size, claimed_input_voxel_size: optional —
            the model's originally-declared voxel sizes, recorded for
            provenance.

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
            compressor=None,  # neuroglancer browser reader cannot decode blosc
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
        # Record the model's originally-declared voxel sizes for provenance.
        # These may differ from the active output_voxel_size/input_voxel_size
        # above when we've snapped to the dataset's closest available scale.
        if claimed_output_voxel_size is not None:
            root.attrs["claimed_output_voxel_size"] = (
                claimed_output_voxel_size.tolist()
                if hasattr(claimed_output_voxel_size, "tolist")
                else list(claimed_output_voxel_size)
            )
        if claimed_input_voxel_size is not None:
            root.attrs["claimed_input_voxel_size"] = (
                claimed_input_voxel_size.tolist()
                if hasattr(claimed_input_voxel_size, "tolist")
                else list(claimed_input_voxel_size)
            )
        # Snapshot of the dashboard's input_norm at volume-creation time.
        # Used as the baseline for Resume Existing (the new session inherits
        # this normalization). Stored as the raw YAML-style dict so it round-
        # trips via json.load / yaml.safe_load without any extra parsing.
        if input_norm_config is not None:
            root.attrs["input_norm"] = input_norm_config
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

def ensure_minio_serving(zarr_path, crop_id, output_base_dir=None, mc_target_name=None):
    """
    Ensure MinIO is running and upload zarr file.

    Args:
        zarr_path: Path to zarr file to upload
        crop_id: Unique identifier for the crop
        output_base_dir: Base output directory (MinIO will use output_base_dir/.minio)
        mc_target_name: Optional override for the MinIO bucket object name.
            Defaults to `basename(zarr_path)` (the historical behavior).
            When provided, `mc mirror` is invoked with this as the target
            directory name instead, producing a MinIO URL like
            `http://.../bucket/<mc_target_name>/...` regardless of the
            source filename on disk. Used by the multi-ROI workflow
            (Patch 44) to keep a stable bucket name across sessions even
            when the source is a dated snapshot like
            `roi3_20260414_144600.zarr`.

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

        # Configure mc client (retry — MinIO may not accept connections
        # immediately after the port is bound)
        mc_cmd = ["mc", "alias", "set", "myserver", f"http://{ip}:{port}", "minio", "minio123"]
        for attempt in range(5):
            result = subprocess.run(mc_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            time.sleep(1)
        else:
            raise subprocess.CalledProcessError(result.returncode, mc_cmd, result.stdout, result.stderr)

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

    # Upload zarr file. When mc_target_name is provided, it overrides the
    # source basename as the MinIO bucket object key — used by the multi-ROI
    # workflow to keep a stable "roi3_annotation.zarr" bucket name even when
    # the on-disk source is a dated snapshot like "roi3_20260414_144600.zarr".
    zarr_name = mc_target_name or Path(zarr_path).name
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

    # Use localhost for SSH tunnel compatibility (browser can't reach compute node IPs)
    minio_url = (
        f"http://localhost:{minio_state['port']}"
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

    Returns:
        set of src_chunk_path values that successfully copied. The caller
        must use this to filter the saved chunk_sync_state — chunks that
        FAILED to copy must NOT be marked as known, or they will be
        permanently locked out of subsequent non-force diff syncs.
    """
    if not copy_pairs:
        return set()

    available_workers = _get_sync_worker_count()
    workers = max(1, min(len(copy_pairs), available_workers))

    def _copy_one(src_dst):
        src_chunk_path, dst_chunk_path = src_dst
        s3.get(src_chunk_path, dst_chunk_path)
        return src_chunk_path

    successful = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pair = {executor.submit(_copy_one, pair): pair for pair in copy_pairs}
        for fut in as_completed(future_to_pair):
            pair = future_to_pair[fut]
            try:
                successful.add(fut.result())
            except Exception as e:
                logger.warning(
                    f"Error syncing chunk in parallel copy ({pair[0]}): {e}"
                )
    return successful


def _make_s3_filesystem():
    """Create an s3fs filesystem pointed at the local MinIO instance.

    Uses skip_instance_cache=True to defeat fsspec's default
    instance-caching behavior (`s3fs.S3FileSystem(...) is s3fs.S3FileSystem(...)`
    returns True for identical kwargs). Without this, every caller of
    this factory shares a single S3FileSystem object with a single
    dircache — a partial s3.ls result poisons subsequent s3.exists()
    calls and causes the partial-ls-unlink bug (validated 2026-05-10:
    chunks vanished post-Save in both spine and direct-ssh modes; a
    "fresh" verify_s3 was actually the same poisoned instance, and
    Patch D's smoking-gun warning never fired despite the bug
    triggering).

    Cost of skip_instance_cache=True: a few microseconds of S3FileSystem
    object construction per call. Cheap relative to the remote calls
    each instance subsequently makes.
    """
    return s3fs.S3FileSystem(
        anon=False,
        key="minio",
        secret="minio123",
        client_kwargs={
            "endpoint_url": f"http://{minio_state['ip']}:{minio_state['port']}",
            "region_name": "us-east-1",
        },
        skip_instance_cache=True,
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
        if key in dst_group:
            dst_array = dst_group[key]
            shape_mismatch = (
                tuple(dst_array.shape) != tuple(src_array.shape)
                or tuple(dst_array.chunks) != tuple(src_array.chunks)
                or dst_array.dtype != src_array.dtype
            )
        else:
            shape_mismatch = True
        if shape_mismatch:
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
    """Diff remote vs known chunk state and pull changed chunks to local disk.

    Local disk is the source of truth — YAML imports are written locally
    first and only later mirrored to MinIO; painted scribbles flow MinIO
    → local through this function. We never delete on-disk chunks based
    on remote state: an "absent" chunk on MinIO is almost always a
    transient (paginated listing truncated, in-flight `mc mirror`,
    server restart, network blip), not a real user erase. Painting BG
    over a chunk in neuroglancer rewrites the chunk file, it does not
    remove it. Treating remote-missing as "user erased it" once cost a
    full session of training (3456 chunks wiped from disk after one bad
    listing, FG index emptied, loss silently went to 0).

    Returns:
        (changed_keys, removed_keys=[], remote_chunk_state)
        ``removed_keys`` is always empty; the slot is preserved so
        callers' tuple-unpacking keeps working.
    """
    try:
        chunk_files = s3.ls(s0_path)
    except FileNotFoundError:
        # Remote bucket has no annotation/s0 yet (just created) — keep
        # whatever we have locally and try again next cycle.
        return [], [], dict(known_chunk_state)
    except Exception as e:
        logger.warning(f"_diff_and_sync_chunks: s3.ls({s0_path}) failed: {e}; "
                       "treating as transient, skipping sync this cycle.")
        return [], [], dict(known_chunk_state)

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
    successful_src = _copy_chunks_parallel(s3, copy_pairs)

    # Drop FAILED chunks from remote_chunk_state so they are not recorded
    # in known_chunk_state — otherwise the next non-force diff would see
    # `known_chunk_state.get(k) == v` and exclude them, locking them out
    # of every subsequent periodic sync. The only recovery in that lockout
    # state is force=True, which is operationally untenable for the
    # background periodic-sync thread.
    src_to_key = {f"{s0_path}/{k}": k for k in changed_keys}
    successful_keys = {src_to_key[s] for s in successful_src if s in src_to_key}
    failed_keys = [k for k in changed_keys if k not in successful_keys]
    if failed_keys:
        logger.warning(
            f"_diff_and_sync_chunks: {len(failed_keys)} of "
            f"{len(changed_keys)} chunk copies failed; will retry on next "
            f"sync. failed={failed_keys[:5]}{' ...' if len(failed_keys) > 5 else ''}"
        )
    for k in failed_keys:
        remote_chunk_state.pop(k, None)

    # Remove stale local chunks — but VERIFY each removal against MinIO
    # using a FRESH s3fs instance, to guard against partial-listing bugs
    # in s3.ls. Without verification, a partial s3.ls would unlink chunks
    # from disk that are really still on MinIO, and subsequent syncs would
    # see them as "removed already" and never re-copy — user paint
    # vanishes from disk despite a green "Save" success.
    #
    # CRITICAL: the verification MUST use a fresh s3fs instance, not the
    # `s3` parameter that just did the partial s3.ls. s3fs maintains an
    # internal listing cache PER INSTANCE that records both positive
    # presences and negative absences; if s3.ls returned partial, the
    # same instance's `s3.exists()` reports False for the missing keys
    # (cached from the partial list). A fresh instance has its own
    # (empty) cache and performs a real HEAD against MinIO.
    #
    # Bounded cost: O(removed_keys) HEADs + 1 fresh s3fs construction.
    # removed_keys is typically empty, so usually no-op.
    confirmed_removed_keys = []
    spurious_removed_keys = []
    if removed_keys:
        verify_s3 = _make_s3_filesystem()
        for k in removed_keys:
            src_chunk_path = f"{s0_path}/{k}"
            try:
                still_on_minio = verify_s3.exists(src_chunk_path)
            except Exception as e:
                logger.warning(
                    f"_diff_and_sync_chunks: s3.exists check failed for {k}; "
                    f"skipping unlink to be safe: {e}"
                )
                spurious_removed_keys.append(k)
                continue
            if still_on_minio:
                spurious_removed_keys.append(k)
                continue
            confirmed_removed_keys.append(k)
            local_chunk = dst_s0_path / k
            try:
                if local_chunk.exists():
                    local_chunk.unlink()
            except Exception as e:
                logger.debug(f"Error removing stale chunk {k}: {e}")

    if spurious_removed_keys:
        logger.warning(
            f"_diff_and_sync_chunks: skipped unlink for "
            f"{len(spurious_removed_keys)}/{len(removed_keys)} keys whose "
            f"s3.exists confirmed they are still on MinIO (partial-ls bug "
            f"suppressed). spurious={spurious_removed_keys[:5]}"
            f"{' ...' if len(spurious_removed_keys) > 5 else ''}"
        )
        # Re-add spurious keys back to remote_chunk_state with their
        # known LM so the caller's `volume_meta["chunk_sync_state"] =
        # remote_chunk_state` doesn't drop them and trigger them as
        # "new" (and re-copy) on next sync. They were never really removed.
        for k in spurious_removed_keys:
            remote_chunk_state[k] = known_chunk_state[k]

    # Mutate removed_keys in-place so the return value reflects what was
    # ACTUALLY unlinked (callers use this for downstream bookkeeping).
    removed_keys[:] = confirmed_removed_keys

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

    logger.debug(f"Syncing all annotations from MinIO (force={force})...")
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
    logger.debug(f"Synced {synced}/{len(zarr_ids)} annotations")
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

    # Idempotency short-circuit: if an existing extract zarr's annotation/s0
    # byte-matches the source labels for this chunk, the raw EM source
    # hasn't changed (raw zarr is treated as immutable), so re-running this
    # function would produce a byte-equivalent output — pure wasted CPU +
    # GPFS I/O. The check costs ~175 KiB read per chunk vs. the ~13 MB
    # raw-EM read + scipy resample + zarr write that follows, so on miss
    # the cost is negligible relative to the work it gates.
    #
    # Most common trigger of needless re-extract: dashboard process
    # restart wipes the in-memory `chunk_sync_state` for the volume, so
    # the next _diff_and_sync_chunks marks every existing chunk as
    # "changed" relative to the empty known-state, even though all
    # existing extracts on disk are already up to date.
    correction_id = f"{volume_id}_chunk_{cz}_{cy}_{cx}"
    correction_zarr_path = os.path.join(corrections_dir, f"{correction_id}.zarr")
    if os.path.isdir(correction_zarr_path):
        try:
            existing = zarr.open(correction_zarr_path, mode="r")
            existing_ann = existing["annotation/s0"][:]
            if (existing_ann.shape == annotation_data.shape
                    and np.array_equal(existing_ann, annotation_data)):
                logger.debug(
                    f"Idempotent skip for chunk ({cz},{cy},{cx}): "
                    f"existing extract already up to date"
                )
                return True
        except Exception as e:
            logger.warning(
                f"Idempotency check failed for chunk ({cz},{cy},{cx}); "
                f"will re-extract: {e}"
            )

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

    # If a stale zarr exists (e.g. copied in during Resume Existing Volume),
    # wipe it before recreating. zarr's mode="w" only overwrites top-level
    # metadata and can leave stale subarrays behind, causing
    # KeyError: 'annotation/s0' when we later index into the group.
    if os.path.isdir(correction_zarr_path):
        import shutil
        shutil.rmtree(correction_zarr_path, ignore_errors=True)

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
            logger.debug(f"No metadata for volume {volume_id}, skipping")
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

        # Extract corrections for changed chunks. Skip entirely when a
        # virtual-sources manifest is present: the trainer reads the volume
        # zarr directly via VirtualPatchDataset and never touches per-chunk
        # extracts, so this loop just slowly fills disk with thousands of
        # 178**3 raw cubes that nothing reads. (See
        # cellmap_flow/finetune/virtual_dataset.py for the manifest format.)
        from cellmap_flow.finetune.virtual_dataset import read_manifest

        corrections_dir = volume_meta.get("corrections_dir") or os.path.dirname(
            local_zarr_path
        )
        manifest = read_manifest(corrections_dir) if corrections_dir else None

        extracted_chunks = volume_meta.get("extracted_chunks", set())
        changed_chunk_indices = [
            tuple(map(int, k.split(".")))
            for k in changed_chunk_keys
        ]
        created_any = False

        if manifest is not None:
            logger.debug(
                f"Volume {volume_id}: skipping per-chunk extract (manifest present); "
                f"{len(changed_chunk_indices)} changed chunks ignored."
            )
        else:
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
            synced = sync_all_annotations_from_minio(force=False)
            # After each successful sync, refresh the bounding-box overlay so
            # the user sees where they've painted without clicking a button.
            if synced and synced > 0:
                try:
                    from cellmap_flow.dashboard.routes.finetune import (
                        refresh_annotated_regions_layer,
                    )
                    refresh_annotated_regions_layer()
                except Exception as e:
                    logger.debug(f"Periodic sync: refresh_annotated_regions_layer failed: {e}")
        except Exception as e:
            logger.debug(f"Error in periodic sync: {e}")


def start_periodic_sync():
    """Start the periodic annotation sync thread if not already running."""
    if minio_state["sync_thread"] is None or not minio_state["sync_thread"].is_alive():
        thread = threading.Thread(target=periodic_sync_annotations, daemon=True)
        thread.start()
        minio_state["sync_thread"] = thread
        logger.info("Started periodic annotation sync thread")



# ---------------------------------------------------------------------------
# Instance-correction helpers (Phase 2 port from vacc-compat finetune_utils.py)
# ---------------------------------------------------------------------------

def create_instance_annotation_volume_from_seg(
    output_zarr_path,
    instance_zarr_path,
    dataset_path,
    model_name,
    input_size,
    input_voxel_size,
    dilation_radius_voxels=5,
    chunk_size=None,
    annotation_dtype="uint16",
):
    """Seed a paintable annotation volume from an existing instance zarr.

    Produces a writable annotation zarr (uint16/uint32) in cellmap-flow's
    `target_transforms.AffinityTargetTransform` label scheme:
      0 = unannotated (ignored in loss)
      1 = background (confident — the dilation shell around each instance)
      2+ = instance IDs (one distinct label per mitochondrion)

    The annotation volume's shape/offset/resolution match the input instance
    zarr, so NG renders it at the same physical location as the source.

    Args:
        output_zarr_path: Where to write the new annotation zarr.
        instance_zarr_path: Path to the uint32 instance zarr produced by
            `run_postprocess_on_subvolume.py` (must have per-scale `.zattrs`
            with `resolution` and `offset`, and an `s0` scale).
        dataset_path: Raw EM zarr path (for `extract_correction_from_chunk`
            to pull raw context later during training-data extraction).
        model_name: Model identifier (e.g. "mito_aff_trichocyst").
        input_size: Model's read_shape as 3-vector (e.g. [178, 178, 178]).
        input_voxel_size: Model's input voxel size in nm (e.g. [16, 16, 16]).
        dilation_radius_voxels: Number of voxels to dilate each instance by
            to form the background shell. 5 @ 16nm output = 80 nm shell.
        chunk_size: Annotation chunks z,y,x. Defaults to the model write_shape
            (56 for mito_aff), which matches the training-data convention.
        annotation_dtype: "uint16" (up to 65534 instances, enough for our
            ROIs which have 1192–2610) or "uint32" for larger workloads.

    Returns:
        (success: bool, zarr_path_or_error: str)
    """
    from scipy.ndimage import binary_dilation

    if chunk_size is None:
        chunk_size = [56, 56, 56]

    # Read source metadata from the instance zarr's s0 .zattrs (same path the
    # dashboard's extra_layers loader uses via get_raw_layer).
    try:
        src_s0 = zarr.open(os.path.join(instance_zarr_path, "s0"), mode="r")
    except Exception as e:
        return False, f"Failed to open instance zarr s0: {e}"
    src_attrs = dict(src_s0.attrs)
    try:
        source_offset_nm = [float(v) for v in src_attrs["offset"]]
        source_voxel_size = [float(v) for v in src_attrs["resolution"]]
    except KeyError as e:
        return False, f"instance zarr s0 missing attr: {e}"
    source_shape = tuple(src_s0.shape)
    logger.info(
        f"Seeding annotation volume from {instance_zarr_path}: "
        f"shape={source_shape}, offset_nm={source_offset_nm}, "
        f"voxel_nm={source_voxel_size}, dilation_r={dilation_radius_voxels} vox"
    )

    # Load the instance array fully (ROI-sized, fits in memory — ~300 MB uint32).
    instances = src_s0[:].astype(np.uint32)
    n_source_instances = int(instances.max())
    if n_source_instances + 1 > np.iinfo(np.dtype(annotation_dtype)).max:
        return False, (
            f"instance count {n_source_instances} + shell label 1 exceeds "
            f"{annotation_dtype} max {np.iinfo(np.dtype(annotation_dtype)).max}; "
            "use annotation_dtype='uint32'"
        )

    fg_mask = instances > 0
    # Dilation shell: grow each instance by R voxels and subtract the original.
    # iterate a face-connected 3D structuring element R times for a ball-ish shell.
    logger.info(
        f"Computing dilation shell (radius={dilation_radius_voxels} voxels)..."
    )
    dilated = binary_dilation(
        fg_mask, iterations=int(dilation_radius_voxels)
    )
    shell_mask = dilated & (~fg_mask)

    # Build annotation: shell=1, instance voxels=(id+1) to reserve label 1 for
    # background. Unannotated stays 0. Everything done in the target dtype.
    annotation = np.zeros(source_shape, dtype=annotation_dtype)
    annotation[shell_mask] = 1
    annotation[fg_mask] = (instances[fg_mask] + 1).astype(annotation_dtype)

    n_shell = int(shell_mask.sum())
    n_fg = int(fg_mask.sum())
    logger.info(
        f"annotation labels: {n_fg} fg voxels ({n_source_instances} instances), "
        f"{n_shell} shell (bg) voxels, "
        f"{int((annotation == 0).sum())} unannotated"
    )

    # Create the zarr skeleton via the existing helper.
    success, info = create_annotation_volume_zarr(
        zarr_path=output_zarr_path,
        dataset_shape_voxels=list(source_shape),
        output_voxel_size=list(source_voxel_size),
        dataset_offset_nm=list(source_offset_nm),
        chunk_size=list(chunk_size),
        dataset_path=dataset_path,
        model_name=model_name,
        input_size=list(input_size),
        input_voxel_size=list(input_voxel_size),
        annotation_dtype=annotation_dtype,
        annotation_type="instance_annotation_volume",
    )
    if not success:
        return False, info

    # Write the seeded annotation into annotation/s0.
    try:
        root = zarr.open(output_zarr_path, mode="r+")
        root["annotation/s0"][:] = annotation
        # Record the seed source + parameters so we can re-seed later
        # without losing track of what this zarr was made from.
        root.attrs["seed_source_instance_zarr"] = str(instance_zarr_path)
        root.attrs["seed_dilation_radius_voxels"] = int(dilation_radius_voxels)
        root.attrs["seed_n_instances"] = n_source_instances
    except Exception as e:
        return False, f"Failed to write seeded annotation: {e}"

    # Explicitly drop the large intermediate arrays before returning to the
    # Flask request handler — Python's refcount GC should free them at the
    # function's frame pop anyway, but under a tight SLURM cgroup (e.g.
    # --mem=128G shared with the dashboard + base inference + LoRA serves)
    # we want to minimize overlap with any subsequent in-request allocations.
    import gc
    del instances, fg_mask, dilated, shell_mask, annotation
    gc.collect()

    return True, output_zarr_path



def minio_backing_store_populated(output_dir, zarr_name):
    """Return True if MinIO's on-disk backing store already has a non-empty
    s0 directory for this instance-correction zarr.

    This is the clobber-guard check used by `create_instance_correction` to
    refuse re-seeding a zarr whose MinIO state may contain unpulled user
    edits. We check the filesystem rather than query MinIO because this
    decision is made *before* `ensure_minio_serving` runs, so the MinIO
    server may not yet be up.

    The layout is determined by `ensure_minio_serving`'s `MINIO_ROOT =
    <output_dir>/.minio`, MinIO's bucket name (`annotations`), and the
    zarr's object path (`<zarr_name>/annotation/s0`). If that directory
    exists and contains any entries, the backing store is considered
    populated and re-seeding should be refused.
    """
    s0_backing = (
        Path(output_dir) / ".minio" / "annotations" / zarr_name / "annotation" / "s0"
    )
    if not s0_backing.exists():
        return False
    try:
        return any(s0_backing.iterdir())
    except Exception:
        return False



def sync_instance_correction_from_minio(zarr_path, dst_path=None):
    """Force-snapshot the MinIO state of a paintable instance-correction zarr
    to a local destination path.

    Unlike `sync_annotation_volume_from_minio`, this does NOT trigger any
    `extract_correction_from_chunk` side effects — it is a plain write-through
    snapshot. Its purpose is to produce a durable on-disk copy of the
    MinIO-backed paintable layer so that:
      - Run 12 training-data extraction can read `annotation/s0` from a
        user-visible path without going through MinIO;
      - a dated snapshot can be produced at any time for rollback / audit;
      - the dashboard can safely be restarted (re-POSTing
        `create-instance-correction` would otherwise trigger
        `ensure_minio_serving`'s initial `mc mirror local->MinIO` and
        clobber the user's brush edits with the stale seed).

    Args:
        zarr_path: Absolute path to the user-visible instance-correction
            zarr (e.g. `.../instance_corrections/roi3_annotation.zarr`).
            Used to derive the MinIO bucket key from its basename
            (`<bucket>/<basename(zarr_path)>/annotation`). The file itself
            is not opened — only the name is read.
        dst_path: Absolute path to write the snapshot to. Defaults to
            `zarr_path` (in-place pull-back into the user-visible zarr).
            Prefer a fresh dated path (e.g.
            `.../roi3_annotation_FINAL_session14_<ts>.zarr`) to avoid any
            hardlink / aliasing hazards with provenance snapshots — a
            plain in-place sync overwrites chunk files in the existing
            inodes, and if those inodes are hardlinked to another path
            (e.g. the bootstrap-via-`cp -rl` seed path), the "other" path
            is corrupted as a side effect. Using a distinct `dst_path`
            sidesteps this entirely.

    Returns:
        (success: bool, info_or_error: dict or str). On success, info is
        {"zarr_path", "dst_path", "chunks_synced", "chunks_removed"}.
    """
    if not minio_state["ip"] or not minio_state["port"]:
        return False, "MinIO not running"

    if dst_path is None:
        dst_path = zarr_path

    try:
        zarr_name = os.path.basename(os.path.normpath(zarr_path))
        s3 = _make_s3_filesystem()
        bucket = minio_state["bucket"]
        src_root = f"{bucket}/{zarr_name}"

        if not s3.exists(f"{src_root}/annotation/s0"):
            return False, (
                f"no MinIO bucket entry at {src_root}/annotation/s0 "
                "(was create-instance-correction ever POSTed for this zarr?)"
            )

        # Use `zarr.copy_store` for a complete byte-for-byte copy from
        # MinIO to the local destination. Copies every key under the
        # bucket root verbatim: top-level `.zattrs` + `.zgroup`,
        # `annotation/.zattrs` + `.zgroup`, `annotation/s0/.zarray`
        # (which preserves the Patch 39 `compressor=None` setting —
        # critical, because the raw chunks in MinIO are uncompressed
        # bytes and opening them with the wrong compressor in .zarray
        # produces a blosc decompression error), `annotation/s0/.zattrs`,
        # and every chunk file.
        #
        # Why not `_sync_zarr_group_metadata` + `_diff_and_sync_chunks`
        # (the pre-Patch-46 approach)? Because that pair was written for
        # the legacy crop-based annotation_volume workflow, where the
        # destination was always pre-created by a session setup step and
        # its `.zarray` already had the correct compressor. For fresh
        # destinations (e.g. save_roi.sh writing a timestamped snapshot),
        # `_sync_zarr_group_metadata` would call `create_dataset` without
        # specifying a compressor, so zarr's default (Blosc) would be
        # written into the new `.zarray` — and subsequent reads would
        # blosc-decode the raw bytes and fail. Also, that pair only
        # syncs the `annotation/` subgroup, not the top-level zarr group
        # metadata, so the new snapshot lacks a `.zgroup` marker at the
        # root and `zarr.open(path)` doesn't recognize it as a group.
        #
        # `zarr.copy_store` copies every key under the source store and
        # is semantically a deep byte-for-byte clone. Slower than the
        # parallel chunk-diff path, but correct.
        src_store = s3fs.S3Map(root=src_root, s3=s3, check=False)
        os.makedirs(dst_path, exist_ok=True)
        dst_store = zarr.DirectoryStore(str(dst_path))
        n_copied, n_skipped, n_bytes = zarr.copy_store(
            src_store, dst_store, if_exists="replace"
        )

        logger.info(
            f"Synced instance correction {zarr_name} from MinIO -> "
            f"{dst_path}: {n_copied} keys copied "
            f"({n_skipped} skipped, {n_bytes} bytes)"
        )

        return True, {
            "zarr_path": str(zarr_path),
            "dst_path": str(dst_path),
            "keys_copied": int(n_copied),
            "keys_skipped": int(n_skipped),
            "bytes_copied": int(n_bytes),
        }

    except Exception as e:
        logger.error(f"Error syncing instance correction {zarr_path} -> {dst_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)



def cc3d_relabel_instance_correction(zarr_path, target_label, snapshot_dir=None):
    """Split a single label in a paintable instance-correction zarr by
    running 26-connectivity cc3d on its voxel mask and reassigning all
    components except the largest to fresh unused instance IDs.

    Typical workflow: the user erases a thin bridge between two fused
    mitochondria in NG's brush tool (still sharing `target_label`), then
    POSTs this route with that label. cc3d finds the now-separated
    components; we keep the largest as `target_label` and reassign the
    smaller components to `max(existing) + 1 ...`. A hard reload of the
    NG tab shows the split colors. Validated end-to-end on ROI3 in
    Session 13 — productionizes `scripts/oneshot_cc3d_split_roi3.py`.

    Reads and writes the MinIO-backed zarr in place (MinIO is the source
    of truth for in-progress brush edits; the user-visible `zarr_path` is
    only used to derive the bucket object name and to place the rollback
    snapshot next to). Before writing, snapshots the current MinIO state
    to a local DirectoryStore for rollback safety.

    Args:
        zarr_path: Absolute path to the user-visible instance-correction
            zarr. Its basename is used as the MinIO object key.
        target_label: The instance ID to split (must be >= 2, since 0 is
            unannotated and 1 is background in the AffinityTargetTransform
            scheme).
        snapshot_dir: Where to drop the pre-split rollback snapshot.
            Defaults to `<parent_of_zarr_path>/snapshots/`.

    Returns:
        (success: bool, info_or_error: dict or str). On success, info is
        {"zarr_path", "target_label", "n_components", "kept_voxels",
         "splits": [{"new_label", "voxels"}, ...], "snapshot_path"}.
    """
    if not minio_state["ip"] or not minio_state["port"]:
        return False, "MinIO not running"

    try:
        import cc3d  # connected-components-3d
    except ImportError as e:
        return False, f"cc3d not installed in the dashboard env: {e}"

    if int(target_label) < 2:
        return False, (
            f"target_label must be >= 2 (got {target_label}); "
            "label 0 is unannotated and label 1 is background"
        )

    try:
        zarr_name = os.path.basename(os.path.normpath(zarr_path))
        s3 = _make_s3_filesystem()
        bucket = minio_state["bucket"]
        src_root = f"{bucket}/{zarr_name}"
        if not s3.exists(f"{src_root}/annotation/s0"):
            return False, (
                f"no MinIO bucket entry at {src_root}/annotation/s0 "
                "(was create-instance-correction ever POSTed for this zarr?)"
            )

        # Open the MinIO-backed zarr r+. S3Map with check=False skips the
        # initial bucket-probe round trip.
        store = s3fs.S3Map(root=src_root, s3=s3, check=False)
        root = zarr.open(store, mode="r+")
        ann = root["annotation/s0"]
        logger.info(
            f"[cc3d-relabel] opened {src_root}: shape={ann.shape} "
            f"dtype={ann.dtype} chunks={ann.chunks}"
        )

        # Snapshot to a local DirectoryStore before any writes so the user
        # has a guaranteed rollback point even if cc3d or the writeback
        # misbehaves. `zarr.copy_store` streams chunks one at a time without
        # materializing the full volume twice in memory.
        if snapshot_dir is None:
            snapshot_dir = os.path.join(
                os.path.dirname(os.path.normpath(zarr_path)), "snapshots"
            )
        os.makedirs(snapshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        snap_basename = zarr_name.replace(".zarr", "") + f"_snapshot_{ts}.zarr"
        snapshot_path = os.path.join(snapshot_dir, snap_basename)
        logger.info(f"[cc3d-relabel] snapshotting to {snapshot_path}")
        local_store = zarr.DirectoryStore(snapshot_path)
        zarr.copy_store(store, local_store, if_exists="replace")

        # Load the full annotation into memory — fits comfortably at ROI
        # scale (~600 MB uint16) and avoids per-chunk GET churn during the
        # cc3d + reassignment passes.
        logger.info("[cc3d-relabel] loading full annotation/s0 into memory")
        arr = ann[:]

        mask = arr == int(target_label)
        n_target_voxels = int(mask.sum())
        if n_target_voxels == 0:
            return False, f"no voxels match label {target_label}"
        logger.info(
            f"[cc3d-relabel] target mask: {n_target_voxels} voxels labeled "
            f"{target_label}"
        )

        labeled, n_comp = cc3d.connected_components(
            mask, connectivity=26, return_N=True
        )
        logger.info(f"[cc3d-relabel] cc3d found {n_comp} connected components")

        if n_comp == 1:
            logger.info(
                "[cc3d-relabel] only one connected component — nothing to "
                "split. (Erase may not have fully cut the bridge, or the "
                "label was already single-component.)"
            )
            # No writeback: the snapshot is still useful as a backup, but
            # nothing on MinIO changed.
            return True, {
                "zarr_path": str(zarr_path),
                "target_label": int(target_label),
                "n_components": int(n_comp),
                "kept_voxels": n_target_voxels,
                "splits": [],
                "snapshot_path": snapshot_path,
                "note": "single component, no split performed",
            }

        # Pick the largest component as the one to keep under target_label,
        # reassign every other component to fresh unused IDs. cc3d labels
        # background as 0 and foreground components as 1..n_comp.
        stats = cc3d.statistics(labeled)
        sizes = stats["voxel_counts"]
        fg_sizes = sorted(
            ((i + 1, int(sizes[i + 1])) for i in range(n_comp)),
            key=lambda x: -x[1],
        )
        biggest_comp, kept_voxels = fg_sizes[0]
        logger.info(
            f"[cc3d-relabel] keeping component {biggest_comp} "
            f"({kept_voxels} voxels) as label {target_label}"
        )

        existing_max = int(arr.max())
        next_id = existing_max + 1
        splits = []
        for comp_i, sz in fg_sizes[1:]:
            new_id = next_id
            next_id += 1
            arr[labeled == comp_i] = new_id
            splits.append({"new_label": int(new_id), "voxels": int(sz)})
            logger.info(
                f"[cc3d-relabel]   component {comp_i}: {sz} voxels "
                f"-> new label {new_id}"
            )

        # Write the full array back; zarr's S3 backend breaks this into
        # per-chunk PUTs. Every chunk is PUT (not just changed ones) —
        # acceptable at ROI scale, the prototype measured ~15 s for 900 MB.
        logger.info("[cc3d-relabel] writing back to MinIO")
        ann[:] = arr

        return True, {
            "zarr_path": str(zarr_path),
            "target_label": int(target_label),
            "n_components": int(n_comp),
            "kept_voxels": int(kept_voxels),
            "splits": splits,
            "snapshot_path": snapshot_path,
        }

    except Exception as e:
        logger.error(f"Error in cc3d_relabel_instance_correction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e)


