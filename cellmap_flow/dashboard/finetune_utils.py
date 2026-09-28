"""
Helper functions for finetuning annotation workflows.

Handles MinIO server management, annotation zarr creation, and
periodic synchronization of annotations between MinIO and local disk.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import time
import logging
import threading
import uuid
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


_SESSION_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def latest_session_on_disk(base_output_path: str):
    """The newest session under ``base_output_path`` that has something to train on.

    Sessions are ``<base>/<YYYYmmdd_HHMMSS>/`` directories; this takes the
    latest one whose corrections/ holds a virtual-sources manifest. The
    in-memory base -> session map dies with the dashboard, so after a
    restart get_or_create_session_path hands out a new, empty session, and
    submitting training for the base path failed with "Corrections path
    does not exist" although the session painted before the restart was
    right there. Returns None when there is none.
    """
    from cellmap_flow.finetune.virtual_dataset import VIRTUAL_MANIFEST_FILENAME

    base = os.path.expanduser(str(base_output_path))
    try:
        entries = sorted(os.listdir(base), reverse=True)
    except OSError:
        return None
    for entry in entries:
        session = os.path.join(base, entry)
        if _SESSION_DIR_RE.match(entry) and os.path.isfile(
            os.path.join(session, "corrections", VIRTUAL_MANIFEST_FILENAME)
        ):
            return session
    return None


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
    claimed_output_voxel_size=None,
    claimed_input_voxel_size=None,
    input_norm_config=None,
    postprocess_config=None,
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
        # Same rationale as input_norm above: without this, a served
        # finetuned model generated from this correction data has no way to
        # know it needs e.g. a SigmoidPostprocessor on its output.
        if postprocess_config is not None:
            root.attrs["postprocess"] = postprocess_config
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

def _require_minio_binaries():
    """Fail early, with a fix, if the MinIO binaries are missing.

    Otherwise the missing binary surfaces as a bare
    ``FileNotFoundError: [Errno 2] ... 'minio'`` from subprocess, after the user
    has already picked an output path and created a session directory.

    MinIO no longer publishes prebuilt community server binaries (dl.min.io is
    410 Gone and the GitHub releases carry no assets), so conda-forge -- which
    still builds from source -- is the only practical way to install them.
    """
    missing = [name for name in ("minio", "mc") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"Required MinIO binaries not found on PATH: {', '.join(missing)}. "
            "Annotation volumes are served to Neuroglancer through a local MinIO "
            "server, so painting cannot start without them.\n\n"
            "Install with:\n"
            "    mamba install minio-server minio-client -c conda-forge"
        )


# The mc alias for this dashboard's MinIO. It is defined per call through
# MC_HOST_<alias> (see _mc_env), not with `mc alias set`, which writes
# ~/.mc/config.json: that file is shared by every dashboard the user runs,
# so two of them repointed each other's alias and one's uploads went to the
# other's server.
MC_ALIAS = "myserver"
MINIO_READY_TIMEOUT = 30.0

# Serializes starting MinIO: two requests arriving together could each see
# no server and start one.
_minio_lock = threading.Lock()


def _mc_env(ip, port):
    env = os.environ.copy()
    env[f"MC_HOST_{MC_ALIAS}"] = f"http://minio:minio123@{ip}:{port}"
    return env


def _wait_for_minio_ready(ip, port, process, timeout=MINIO_READY_TIMEOUT) -> bool:
    """Poll MinIO's readiness endpoint until it answers 200, the process exits, or time runs out."""
    import urllib.request

    url = f"http://{ip}:{port}/minio/health/ready"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def _start_minio(output_base_dir):
    """Start MinIO and set it up; record it in minio_state only once all of that worked.

    The state used to be recorded right after the process started, before
    the alias, bucket and policy were set up, so a failure in any of those
    left a live process that later calls took for a working server -- no
    bucket and no sync thread until the dashboard restarted. It waited a
    fixed 3 s instead of asking MinIO whether it was ready, and piped its
    output into a pipe nobody read, which blocks MinIO once it has logged
    about 64 KB. Its log now goes to a file beside its data directory.
    """
    if output_base_dir:
        minio_root = Path(output_base_dir) / ".minio"
    else:
        minio_root = Path("~/.minio-server").expanduser()
    minio_root.mkdir(parents=True, exist_ok=True)
    log_path = minio_root.parent / f"{minio_root.name}.log"

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

    logger.info(f"Starting MinIO server at {ip}:{port} (log: {log_path})")
    with open(log_path, "ab") as log:
        minio_proc = subprocess.Popen(minio_cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

    try:
        if not _wait_for_minio_ready(ip, port, minio_proc):
            raise RuntimeError(
                f"MinIO did not become ready at {ip}:{port} within "
                f"{MINIO_READY_TIMEOUT:.0f}s; see {log_path}"
            )
        mc_env = _mc_env(ip, port)
        bucket = f"{MC_ALIAS}/{minio_state['bucket']}"

        # Create bucket if needed
        result = subprocess.run(["mc", "mb", bucket], capture_output=True, text=True, env=mc_env)
        if result.returncode != 0 and "already" not in result.stderr.lower():
            raise RuntimeError(f"Could not create the MinIO bucket {bucket}: {result.stderr}")

        # Make bucket public
        subprocess.run(
            ["mc", "anonymous", "set", "public", bucket],
            check=True,
            capture_output=True,
            env=mc_env,
        )
    except Exception:
        minio_proc.terminate()
        try:
            minio_proc.wait(timeout=10)
        except Exception:
            minio_proc.kill()
        raise

    minio_state["output_base"] = output_base_dir if output_base_dir else None
    minio_state["minio_root"] = str(minio_root)
    minio_state["log_path"] = str(log_path)
    minio_state["port"] = port
    minio_state["ip"] = ip
    minio_state["process"] = minio_proc
    logger.info(f"MinIO started (PID: {minio_proc.pid})")

    # Start periodic sync thread
    start_periodic_sync()


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
    _require_minio_binaries()

    with _minio_lock:
        if minio_state["process"] is None or minio_state["process"].poll() is not None:
            _start_minio(output_base_dir)

    # Upload zarr file
    zarr_name = Path(zarr_path).name
    target = f"{MC_ALIAS}/{minio_state['bucket']}/{zarr_name}"

    logger.info(f"Uploading {zarr_name} to MinIO")
    result = subprocess.run(
        ["mc", "mirror", "--overwrite", zarr_path, target],
        capture_output=True,
        text=True,
        env=_mc_env(minio_state["ip"], minio_state["port"]),
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


def _chunk_version(entry) -> str:
    """What identifies one version of a remote chunk: its ETag, from a listing.

    Change detection used LastModified from a HEAD per chunk, which has
    one-second resolution: two brush strokes to the same chunk within a
    second, with a sync between them, left the second stroke unsynced for
    good. The ETag changes with the content. LastModified and size are the
    fallback for a store that lists no ETag.
    """
    if not isinstance(entry, dict):
        return ""
    etag = entry.get("ETag") or entry.get("etag")
    if etag:
        return str(etag).strip('"')
    return f"{_safe_epoch_timestamp(entry.get('LastModified'))}:{entry.get('size')}"


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

    Each chunk is downloaded to a temporary file beside its destination and
    moved into place with os.replace, so a reader -- the trainer on its LSF
    node reads this same volume -- sees the old chunk or the new one, never
    a half-written file (which blosc rejects).

    Args:
        s3: s3fs filesystem instance
        copy_pairs: list of (src_chunk_path, dst_chunk_path_str)

    Returns:
        The source paths that could not be copied. Their chunks keep what
        was on disk, and the caller must not record them as synced.
    """
    if not copy_pairs:
        return set()

    available_workers = _get_sync_worker_count()
    workers = max(1, min(len(copy_pairs), available_workers))

    def _copy_one(src_dst):
        src_chunk_path, dst_chunk_path = src_dst
        dst = Path(dst_chunk_path)
        # Dot-prefixed, so nothing that lists chunk keys (z.y.x) sees it.
        tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.part")
        try:
            s3.get(src_chunk_path, str(tmp))
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink()
        return src_chunk_path

    failed = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_copy_one, pair): pair[0] for pair in copy_pairs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                failed.add(futures[fut])
                logger.warning(f"Could not sync chunk {futures[fut]}: {e}; will retry next sync.")
    return failed


def _make_s3_filesystem():
    """Create an s3fs filesystem pointed at the local MinIO instance.

    Both cache opt-outs are load-bearing, not tuning knobs.

    fsspec caches filesystem *instances* keyed on their constructor
    arguments, so every call here would otherwise hand back the same object
    -- and with it the same ``dircache``. s3fs fills ``dircache`` on ``ls()``
    and never expires it by default. The periodic sync thread starts when the
    annotation volume is created, so its first listing of ``annotation/s0``
    runs before the user has painted anything and caches a chunk-less
    listing. Every later sync then reuses that stale listing,
    ``_diff_and_sync_chunks`` sees no chunk keys, and painted scribbles never
    reach disk -- while ``_sync_zarr_group_metadata`` keeps working, because
    ``cat()``/``exists()`` address objects directly and bypass the cache.
    The symptom is a permanent "Synced 0/N annotations" and a training run
    that dies with "No corrections found".

    Listings here are small and served by a local MinIO, so not caching them
    costs nothing.
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
        use_listings_cache=False,
    )


def _sync_zarr_group_metadata(s3, src_path, dst_path):
    """Sync zarr group structure and metadata from S3 to local disk.

    Creates destination arrays that do not exist yet, and copies attrs.
    An array that exists locally with a different shape, chunking or dtype
    is left alone and reported: re-creating it with overwrite=True -- as
    this used to -- deletes every local chunk of that array, which the
    "never delete on-disk chunks" rule in _diff_and_sync_chunks exists to
    prevent.

    Returns:
        The keys of arrays whose local layout does not match MinIO's. The
        caller must not copy chunks into those, since they would not fit.
    """
    src_store = s3fs.S3Map(root=src_path, s3=s3)
    src_group = zarr.open_group(store=src_store, mode="r")

    dst_store = zarr.DirectoryStore(str(dst_path))
    dst_group = zarr.open_group(store=dst_store, mode="a")

    mismatched = set()
    for key in src_group.array_keys():
        src_array = src_group[key]
        if key in dst_group:
            dst_array = dst_group[key]
            if (
                tuple(dst_array.shape) != tuple(src_array.shape)
                or tuple(dst_array.chunks) != tuple(src_array.chunks)
                or dst_array.dtype != src_array.dtype
            ):
                logger.error(
                    f"Local array {dst_path}/{key} ({dst_array.shape}, chunks "
                    f"{dst_array.chunks}, {dst_array.dtype}) does not match MinIO's "
                    f"({src_array.shape}, chunks {src_array.chunks}, {src_array.dtype}); "
                    "leaving the local array and its chunks as they are, and not syncing it."
                )
                mismatched.add(key)
                continue
        else:
            dst_group.create_dataset(
                key,
                shape=src_array.shape,
                chunks=src_array.chunks,
                dtype=src_array.dtype,
                fill_value=0,
            )
        dst_group[key].attrs.update(src_array.attrs)

    dst_group.attrs.update(src_group.attrs)
    return mismatched


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
        # One listing, with each object's ETag: no per-chunk HEAD request.
        chunk_files = s3.ls(s0_path, detail=True)
    except FileNotFoundError:
        # Remote bucket has no annotation/s0 yet (just created) — keep
        # whatever we have locally and try again next cycle.
        return [], [], dict(known_chunk_state)
    except Exception as e:
        logger.warning(f"_diff_and_sync_chunks: s3.ls({s0_path}) failed: {e}; "
                       "treating as transient, skipping sync this cycle.")
        return [], [], dict(known_chunk_state)

    remote_chunk_state = {}
    for entry in chunk_files:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("Key") or ""
        else:
            name = str(entry)
        chunk_key = Path(name).name
        if not re.match(r"^\d+\.\d+\.\d+$", chunk_key):
            continue
        remote_chunk_state[chunk_key] = _chunk_version(entry)

    if force:
        changed_keys = list(remote_chunk_state.keys())
    else:
        changed_keys = [k for k, v in remote_chunk_state.items() if known_chunk_state.get(k) != v]

    if not changed_keys:
        return [], [], remote_chunk_state

    # Copy changed chunks. We never delete: known_chunk_state may shrink
    # if remote drops keys, but the on-disk file stays.
    dst_s0_path = Path(dst_s0_path)
    dst_s0_path.mkdir(parents=True, exist_ok=True)
    copy_pairs = [(f"{s0_path}/{k}", str(dst_s0_path / k)) for k in changed_keys]
    failed = _copy_chunks_parallel(s3, copy_pairs)

    # A chunk that failed to copy keeps its previous state (or none), so the
    # next sync tries it again. It used to be recorded as synced along with
    # the rest, and so was never retried: its strokes were silently missing
    # from training until someone forced a full resync.
    failed_keys = {Path(src).name for src in failed}
    for key in failed_keys:
        if key in known_chunk_state:
            remote_chunk_state[key] = known_chunk_state[key]
        else:
            remote_chunk_state.pop(key, None)
    changed_keys = [k for k in changed_keys if k not in failed_keys]

    return changed_keys, [], remote_chunk_state


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

# One sync at a time. The periodic thread, the Save button, submit and
# restart all call these, and two of them diffing the same chunk state and
# downloading the same chunks at once raced each other over both. Reentrant,
# since a full sync syncs each volume.
_sync_lock = threading.RLock()


def sync_all_annotations_from_minio(force: bool = True):
    """Sync all annotations from MinIO to local disk.

    Returns:
        Number of annotations synced, or -1 if MinIO is not initialized.
    """
    with _sync_lock:
        return _sync_all_annotations_from_minio(force)


def _sync_all_annotations_from_minio(force):
    if not minio_state.get("ip") or not minio_state.get("port"):
        logger.info("MinIO not initialized, skipping annotation sync")
        return -1

    logger.info(f"Syncing all annotations from MinIO (force={force})...")
    s3 = _make_s3_filesystem()
    zarrs = s3.ls(minio_state["bucket"])
    zarr_ids = [Path(c).name.replace(".zarr", "") for c in zarrs if c.endswith(".zarr")]
    synced = 0
    failed = 0
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
        except Exception as e:
            # Not necessarily a problem -- a crop zarr has no root .zattrs and
            # is handled below -- but silently swallowing this hid real
            # failures behind a count that looked like a quiet steady state.
            logger.debug(f"Could not read root attrs for {zid}: {e}")
            failed += 1
        if sync_annotation_from_minio(zid, force=force):
            synced += 1

    # "Synced 0/1" counted volumes that *changed*, so the healthy idle case
    # and a broken sync printed the same line -- which is what made a real
    # sync failure take a day to spot. Say which of the two this is.
    unchanged = len(zarr_ids) - synced
    if synced:
        summary = f"{synced} updated, {unchanged} unchanged"
    else:
        summary = f"no changes ({len(zarr_ids)} checked)"
    if failed:
        summary += f", {failed} could not be read"
    logger.info(f"Annotation sync: {summary}")
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
    creates a standard correction zarr entry.

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

def sync_annotation_volume_from_minio(volume_id, force=False, zarr_path=None):
    """
    Sync an annotation volume from MinIO, detect annotated chunks, extract corrections.

    Steps:
    1. Sync the full annotation zarr from MinIO to local disk
    2. List chunk files in MinIO to find annotated chunks
    3. For each new annotated chunk, extract raw data and create correction entry

    The chunks go to the volume's own zarr_path. They used to go to
    <output_base>/<volume>.zarr, where output_base is fixed by the first
    ensure_minio_serving call of the dashboard's life -- so the strokes of a
    volume created under another output path (or resumed into one) landed
    in the first session's directory, and the trainer, reading the volume's
    own manifest, never saw them. ``zarr_path`` overrides the destination
    (a resumed copy); output_base is only the fallback for a volume this
    dashboard has no record of.

    Returns:
        bool: True if any corrections were created
    """
    with _sync_lock:
        return _sync_annotation_volume_from_minio(volume_id, force, zarr_path)


def _sync_annotation_volume_from_minio(volume_id, force, zarr_path):
    if not minio_state["ip"] or not minio_state["port"]:
        logger.warning("MinIO not initialized, skipping volume sync")
        return False

    try:
        zarr_name = f"{volume_id}.zarr"
        local_zarr_path = zarr_path or (annotation_volumes.get(volume_id) or {}).get("zarr_path")
        if not local_zarr_path:
            if not minio_state.get("output_base"):
                logger.warning(f"No local path for volume {volume_id}, skipping")
                return False
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
        mismatched = _sync_zarr_group_metadata(s3, src_annotation_path, dst_annotation_path)
        if "s0" in mismatched:
            return False

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

# How often a periodic sync that keeps failing says so, in seconds.
SYNC_FAILURE_WARNING_INTERVAL = 300
_sync_failures = {"count": 0, "last_warned": None}


def _periodic_sync_once():
    """One round of the periodic sync; failures are warned about, not hidden."""
    try:
        if not minio_state["output_base"]:
            return
        if not minio_state["ip"] or not minio_state["port"]:
            return
        # Pull annotations to disk, and stop there. This thread must
        # never write to the viewer: python owns the whole state
        # document, so any write makes the browser run
        # `trackable.reset(); restoreState(...)` and rebuild every layer
        # -- taking the draw tool out of the user's hand and dropping
        # whatever strokes were still buffered behind the brush's commit
        # debounce. The annotated-regions boxes are refreshed on demand
        # instead, from the "Show Annotated Regions" button.
        sync_all_annotations_from_minio(force=False)
        if _sync_failures["count"]:
            logger.info(f"Periodic annotation sync recovered after {_sync_failures['count']} failure(s)")
        _sync_failures.update(count=0, last_warned=None)
    except Exception as e:
        # This was logged at DEBUG, which hid a sync outage -- the very
        # failure that once took a day to spot -- until training ran on
        # stale annotations. Warn, once per interval rather than every 30s.
        _sync_failures["count"] += 1
        now = time.monotonic()
        last = _sync_failures["last_warned"]
        if last is None or now - last >= SYNC_FAILURE_WARNING_INTERVAL:
            _sync_failures["last_warned"] = now
            logger.warning(
                f"Periodic annotation sync failed ({_sync_failures['count']} time(s) in a row): {e}. "
                "Painted annotations are not reaching disk until it recovers."
            )


def periodic_sync_annotations():
    """Background thread function to periodically sync annotations from MinIO."""
    while True:
        time.sleep(30)
        _periodic_sync_once()


def start_periodic_sync():
    """Start the periodic annotation sync thread if not already running."""
    if minio_state["sync_thread"] is None or not minio_state["sync_thread"].is_alive():
        thread = threading.Thread(target=periodic_sync_annotations, daemon=True)
        thread.start()
        minio_state["sync_thread"] = thread
        logger.info("Started periodic annotation sync thread")

