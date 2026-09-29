"""Bringing painted chunks from MinIO back to the volumes on disk.

The browser paints into MinIO; the trainer reads the volume on disk. A sync
lists a volume's chunks in the bucket with their ETags, copies the ones that
changed since the last sync into the volume's own zarr, and records what it
saw in the volume's registry record (``chunk_sync_state``). Local disk is the
source of truth: nothing on disk is ever deleted because MinIO lacks it.

``state`` is the dashboard's ``minio_state`` and ``volumes`` its volume
registry, passed to every call; only the lock and the failure counter of
the periodic sync are kept here.
"""

import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import s3fs
import zarr

from cellmap_flow.finetune.session import minio
from cellmap_flow.finetune.session.volume import NotAnAnnotationVolume, read_volume

logger = logging.getLogger(__name__)

# One sync at a time. The periodic thread, the Save button, submit and
# restart all call these, and two of them diffing the same chunk state and
# downloading the same chunks at once raced each other over both. Reentrant,
# since a full sync syncs each volume.
_sync_lock = threading.RLock()

SYNC_PERIOD_SECONDS = 30
# How often a periodic sync that keeps failing says so, in seconds.
SYNC_FAILURE_WARNING_INTERVAL = 300
_sync_failures = {"count": 0, "last_warned": None}

_CHUNK_KEY = re.compile(r"^\d+\.\d+\.\d+$")


def worker_count() -> int:
    """How many threads chunk copies and writes may use.

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


def chunk_version(entry) -> str:
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


def copy_chunks_parallel(s3, copy_pairs) -> set:
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

    workers = max(1, min(len(copy_pairs), worker_count()))

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


def sync_zarr_group_metadata(s3, src_path, dst_path) -> set:
    """Sync zarr group structure and metadata from S3 to local disk.

    Creates destination arrays that do not exist yet, and copies attrs.
    An array that exists locally with a different shape, chunking or dtype
    is left alone and reported: re-creating it with overwrite=True -- as
    this used to -- deletes every local chunk of that array, which the
    "never delete on-disk chunks" rule in diff_and_sync_chunks exists to
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


def diff_and_sync_chunks(s3, s0_path, dst_s0_path, known_state, force=False) -> tuple:
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
        return [], [], dict(known_state)
    except Exception as e:
        logger.warning(f"diff_and_sync_chunks: s3.ls({s0_path}) failed: {e}; "
                       "treating as transient, skipping sync this cycle.")
        return [], [], dict(known_state)

    remote_chunk_state = {}
    for entry in chunk_files:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("Key") or ""
        else:
            name = str(entry)
        chunk_key = Path(name).name
        if not _CHUNK_KEY.match(chunk_key):
            continue
        remote_chunk_state[chunk_key] = chunk_version(entry)

    if force:
        changed_keys = list(remote_chunk_state.keys())
    else:
        changed_keys = [k for k, v in remote_chunk_state.items() if known_state.get(k) != v]

    if not changed_keys:
        return [], [], remote_chunk_state

    # Copy changed chunks. We never delete: known_state may shrink if remote
    # drops keys, but the on-disk file stays.
    dst_s0_path = Path(dst_s0_path)
    dst_s0_path.mkdir(parents=True, exist_ok=True)
    copy_pairs = [(f"{s0_path}/{k}", str(dst_s0_path / k)) for k in changed_keys]
    failed = copy_chunks_parallel(s3, copy_pairs)

    # A chunk that failed to copy keeps its previous state (or none), so the
    # next sync tries it again. It used to be recorded as synced along with
    # the rest, and so was never retried: its strokes were silently missing
    # from training until someone forced a full resync.
    failed_keys = {Path(src).name for src in failed}
    for key in failed_keys:
        if key in known_state:
            remote_chunk_state[key] = known_state[key]
        else:
            remote_chunk_state.pop(key, None)
    changed_keys = [k for k in changed_keys if k not in failed_keys]

    return changed_keys, [], remote_chunk_state


def volume_record(volume_id, zarr_path=None, *, volumes):
    """The registry record of ``volume_id``, rebuilt from ``zarr_path``'s attrs
    (and registered) when there is none yet: after a dashboard restart, say.
    None if neither gives one."""
    if volume_id in volumes:
        return volumes[volume_id]
    if zarr_path is None:
        return None
    try:
        record = read_volume(zarr_path)
    except NotAnAnnotationVolume:
        return None
    except Exception as e:
        logger.error(f"Error reconstructing volume metadata for {volume_id}: {e}")
        return None
    volumes[volume_id] = record
    return record


def sync_volume(volume_id, force=False, zarr_path=None, *, state, volumes) -> bool:
    """
    Pull an annotation volume's changed chunks from MinIO to local disk.

    Syncs the annotation group's metadata, then diffs MinIO's chunk listing
    against what was last synced and copies the chunks that changed. The
    trainer reads the volume itself, through the session's manifest.

    The chunks go to the volume's own zarr_path. They used to go to
    <output_base>/<volume>.zarr, where output_base is fixed by the first
    ensure_serving call of the dashboard's life -- so the strokes of a
    volume created under another output path (or resumed into one) landed
    in the first session's directory, and the trainer, reading the volume's
    own manifest, never saw them. ``zarr_path`` overrides the destination
    (a resumed copy); output_base is only the fallback for a volume this
    dashboard has no record of.

    Returns:
        bool: True if any chunk was pulled
    """
    with _sync_lock:
        return _sync_volume(volume_id, force, zarr_path, state, volumes)


def _sync_volume(volume_id, force, zarr_path, state, volumes):
    if not state["ip"] or not state["port"]:
        logger.warning("MinIO not initialized, skipping volume sync")
        return False

    try:
        zarr_name = f"{volume_id}.zarr"
        local_zarr_path = zarr_path or (volumes.get(volume_id) or {}).get("zarr_path")
        if not local_zarr_path:
            if not state.get("output_base"):
                logger.warning(f"No local path for volume {volume_id}, skipping")
                return False
            local_zarr_path = os.path.join(state["output_base"], zarr_name)
        record = volume_record(volume_id, local_zarr_path, volumes=volumes)

        if record is None:
            logger.warning(f"No metadata for volume {volume_id}, skipping")
            return False

        s3 = minio.make_s3_filesystem(state)

        bucket = state["bucket"]
        src_annotation_path = f"{bucket}/{zarr_name}/annotation"

        if not s3.exists(src_annotation_path):
            return False

        dst_annotation_path = Path(local_zarr_path) / "annotation"
        dst_annotation_path.mkdir(parents=True, exist_ok=True)
        mismatched = sync_zarr_group_metadata(s3, src_annotation_path, dst_annotation_path)
        if "s0" in mismatched:
            return False

        s0_path = f"{bucket}/{zarr_name}/annotation/s0"
        known_state = record.get("chunk_sync_state", {})
        changed_chunk_keys, removed_chunk_keys, remote_chunk_state = diff_and_sync_chunks(
            s3, s0_path, dst_annotation_path / "s0", known_state, force=force
        )

        if not changed_chunk_keys and not removed_chunk_keys:
            return False

        logger.info(
            f"Synced {len(changed_chunk_keys)} changed chunks for volume {volume_id}"
        )
        record["chunk_sync_state"] = remote_chunk_state
        return True

    except Exception as e:
        logger.error(f"Error syncing annotation volume {volume_id}: {e}")
        logger.error(traceback.format_exc())
        return False


def sync_all(force: bool = True, *, state, volumes) -> int:
    """Sync every annotation volume in MinIO to local disk.

    Returns:
        Number of volumes that had changed chunks, or -1 if MinIO is not
        initialized.
    """
    with _sync_lock:
        return _sync_all(force, state, volumes)


def _sync_all(force, state, volumes):
    if not state.get("ip") or not state.get("port"):
        logger.info("MinIO not initialized, skipping annotation sync")
        return -1

    # DEBUG: the periodic sync runs this every 30 s.
    logger.debug(f"Syncing all annotations from MinIO (force={force})...")
    s3 = minio.make_s3_filesystem(state)
    zarrs = s3.ls(state["bucket"])
    zarr_ids = [Path(c).name.replace(".zarr", "") for c in zarrs if c.endswith(".zarr")]
    checked = 0
    synced = 0
    failed = 0
    for zid in zarr_ids:
        # Only annotation volumes are synced: they are all the dashboard
        # serves now. Anything else in the bucket is a crop zarr from the
        # create-crop route, which is gone and whose crops nothing trained on.
        attrs_path = f"{state['bucket']}/{zid}.zarr/.zattrs"
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
        checked += 1
        if sync_volume(zid, force=force, state=state, volumes=volumes):
            synced += 1

    # "Synced 0/1" counted volumes that *changed*, so the healthy idle case
    # and a broken sync printed the same line -- which is what made a real
    # sync failure take a day to spot. Say which of the two this is.
    unchanged = checked - synced
    if synced:
        summary = f"{synced} updated, {unchanged} unchanged"
    else:
        summary = f"no changes ({checked} checked)"
    if failed:
        summary += f", {failed} could not be read"
    logger.info(f"Annotation sync: {summary}")
    return synced


def periodic_sync_once(*, state, volumes) -> None:
    """One round of the periodic sync; failures are warned about, not hidden."""
    try:
        if not state["output_base"]:
            return
        if not state["ip"] or not state["port"]:
            return
        # Pull annotations to disk, and stop there. This thread must
        # never write to the viewer: python owns the whole state
        # document, so any write makes the browser run
        # `trackable.reset(); restoreState(...)` and rebuild every layer
        # -- taking the draw tool out of the user's hand and dropping
        # whatever strokes were still buffered behind the brush's commit
        # debounce. The annotated-regions boxes are refreshed on demand
        # instead, from the "Show Annotated Regions" button.
        sync_all(force=False, state=state, volumes=volumes)
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


def periodic_sync(*, state, volumes) -> None:
    """The periodic sync thread's loop: a round every SYNC_PERIOD_SECONDS, forever."""
    while True:
        time.sleep(SYNC_PERIOD_SECONDS)
        periodic_sync_once(state=state, volumes=volumes)


def start_periodic_sync(state, volumes) -> None:
    """Start the periodic annotation sync thread if not already running."""
    if state["sync_thread"] is None or not state["sync_thread"].is_alive():
        thread = threading.Thread(
            target=periodic_sync, kwargs={"state": state, "volumes": volumes}, daemon=True
        )
        thread.start()
        state["sync_thread"] = thread
        logger.info("Started periodic annotation sync thread")
