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
    annotation_dtype="uint8",
    annotation_type="annotation_volume",
):
    """
    Create a sparse annotation volume zarr covering the full dataset extent.

    The volume has chunk_size = model output_size so each chunk maps to one
    training sample. Only metadata files are created (no chunk data), so the
    zarr is tiny regardless of dataset size.

    Label scheme: 0=unannotated (ignored), 1=background, 2=foreground.

    Args:
        dataset_offset_nm: the world position of voxel 0's *centre*, which is
            also written as the OME translation (new_volume_geometry gives it).
        output_voxel_size, input_voxel_size: the EFFECTIVE voxel sizes used
            for the actual grid alignment (typically the dataset's closest
            available scale to the model's claimed voxel size).
        claimed_output_voxel_size, claimed_input_voxel_size: optional —
            the model's originally-declared voxel sizes, recorded for
            provenance.
        annotation_dtype: the label dtype; uint8 unless the labels are
            instance ids, which need uint16 or uint32.
        annotation_type: the root ``type`` attribute.

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
            dtype=annotation_dtype,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.SHUFFLE),
            fill_value=0,
        )

        # dataset_offset_nm is voxel 0's centre, so it is the OME translation
        # as it stands (see virtual_dataset.volume_corner_nm).
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
        root.attrs["type"] = annotation_type
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
    minio_state["log_path"] = str(log_path)
    minio_state["port"] = port
    minio_state["ip"] = ip
    minio_state["process"] = minio_proc
    logger.info(f"MinIO started (PID: {minio_proc.pid})")

    # Start periodic sync thread
    start_periodic_sync()


def _pull_painted_chunks(zarr_path, volume_id):
    """Sync the volume's chunks from MinIO into ``zarr_path``, if MinIO has any.

    A fresh volume has nothing in the bucket yet; that costs one request.
    """
    zarr_name = Path(zarr_path).name
    try:
        s3 = _make_s3_filesystem()
        if not s3.exists(f"{minio_state['bucket']}/{zarr_name}/annotation/s0"):
            return
    except Exception as e:
        logger.warning(f"Could not check MinIO for painted chunks of {zarr_name}: {e}")
        return
    sync_annotation_volume_from_minio(volume_id, zarr_path=str(zarr_path))


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
    _require_minio_binaries()

    with _minio_lock:
        if minio_state["process"] is None or minio_state["process"].poll() is not None:
            _start_minio(output_base_dir)

    # `mc mirror --overwrite` pushes every local chunk over MinIO's copy, so
    # anything painted since the last sync -- up to 30 s of strokes, or all
    # of them for a resumed session whose .minio holds strokes never synced
    # -- would be overwritten by a stale local chunk. Pull those first.
    _pull_painted_chunks(zarr_path, crop_id)

    # Upload zarr file. mc_target_name, when given, is the bucket key in
    # place of the zarr's own name.
    zarr_name = mc_target_name or Path(zarr_path).name
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
# Annotation sync (full-dataset sync)
# ---------------------------------------------------------------------------

# One sync at a time. The periodic thread, the Save button, submit and
# restart all call these, and two of them diffing the same chunk state and
# downloading the same chunks at once raced each other over both. Reentrant,
# since a full sync syncs each volume.
_sync_lock = threading.RLock()


def sync_all_annotations_from_minio(force: bool = True):
    """Sync every annotation volume in MinIO to local disk.

    Returns:
        Number of volumes that had changed chunks, or -1 if MinIO is not
        initialized.
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
    volumes = 0
    synced = 0
    failed = 0
    for zid in zarr_ids:
        # Only annotation volumes are synced: they are all the dashboard
        # serves now. Anything else in the bucket is a crop zarr from the
        # create-crop route, which is gone and whose crops nothing trained on.
        attrs_path = f"{minio_state['bucket']}/{zid}.zarr/.zattrs"
        try:
            if not s3.exists(attrs_path):
                continue
            if json.loads(s3.cat(attrs_path)).get("type") != "annotation_volume":
                continue
        except Exception as e:
            # Silently swallowing this hid real failures behind a count that
            # looked like a quiet steady state.
            logger.debug(f"Could not read root attrs for {zid}: {e}")
            failed += 1
            continue
        volumes += 1
        if sync_annotation_volume_from_minio(zid, force=force):
            synced += 1

    # "Synced 0/1" counted volumes that *changed*, so the healthy idle case
    # and a broken sync printed the same line -- which is what made a real
    # sync failure take a day to spot. Say which of the two this is.
    unchanged = volumes - synced
    if synced:
        summary = f"{synced} updated, {unchanged} unchanged"
    else:
        summary = f"no changes ({volumes} checked)"
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
            "chunk_sync_state": {},
        }
        annotation_volumes[volume_id] = metadata
        return metadata
    except Exception as e:
        logger.error(f"Error reconstructing volume metadata for {volume_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Annotation volume sync
# ---------------------------------------------------------------------------

def sync_annotation_volume_from_minio(volume_id, force=False, zarr_path=None):
    """
    Pull an annotation volume's changed chunks from MinIO to local disk.

    Syncs the annotation group's metadata, then diffs MinIO's chunk listing
    against what was last synced and copies the chunks that changed. The
    trainer reads the volume itself, through the session's manifest.

    The chunks go to the volume's own zarr_path. They used to go to
    <output_base>/<volume>.zarr, where output_base is fixed by the first
    ensure_minio_serving call of the dashboard's life -- so the strokes of a
    volume created under another output path (or resumed into one) landed
    in the first session's directory, and the trainer, reading the volume's
    own manifest, never saw them. ``zarr_path`` overrides the destination
    (a resumed copy); output_base is only the fallback for a volume this
    dashboard has no record of.

    Returns:
        bool: True if any chunk was pulled
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
            return False

        logger.info(
            f"Synced {len(changed_chunk_keys)} changed chunks for volume {volume_id}"
        )
        volume_meta["chunk_sync_state"] = remote_chunk_state
        return True

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



# ---------------------------------------------------------------------------
# Instance-correction helpers
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


