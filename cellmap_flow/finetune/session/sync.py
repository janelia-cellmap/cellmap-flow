"""Bringing painted chunks from MinIO back to the volumes on disk.

The browser paints into MinIO; the trainer reads the volume on disk. A sync
lists a volume's chunks in the bucket with their ETags, copies the changed
ones into the volume's own zarr, and records the ETags in the volume's
registry record (``chunk_sync_state``). ``state`` (MinIO's) and ``volumes``
(the registry) are passed to every call; only the lock and the periodic
sync's failure counter live here.
"""

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import s3fs
import zarr

from cellmap_flow.finetune.session import minio
from cellmap_flow.finetune.session.manifest import CHUNK_KEY_RE
from cellmap_flow.finetune.session.volume import NotAnAnnotationVolume, read_volume

logger = logging.getLogger(__name__)

# One sync at a time: the periodic thread, the Save button, submit and
# restart all sync, and two at once raced over the same chunk state and
# downloads. Reentrant, since a full sync syncs each volume.
_sync_lock = threading.RLock()

SYNC_PERIOD_SECONDS = 30
# How often a periodic sync that keeps failing says so, in seconds.
SYNC_FAILURE_WARNING_INTERVAL = 300
_sync_failures = {"count": 0, "last_warned": None}


def worker_count() -> int:
    """Threads for chunk copies and writes: the scheduler's CPU count (LSF,
    SGE, SLURM, OpenMP), else the CPUs this process may run on."""
    for key in ("LSB_DJOB_NUMPROC", "LSB_MAX_NUM_PROCESSORS", "NSLOTS", "SLURM_CPUS_PER_TASK",
                "OMP_NUM_THREADS"):
        try:
            value = int(os.environ.get(key, ""))
        except ValueError:
            continue
        if value > 0:
            return value
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def chunk_version(entry) -> str:
    """One version of a remote chunk: its ETag, from a listing, which changes
    with the content. LastModified has one-second resolution, so two strokes
    within a second lost the second; it and the size are only the fallback
    for a store that lists no ETag."""
    if not isinstance(entry, dict):
        return ""
    etag = entry.get("ETag") or entry.get("etag")
    if etag:
        return str(etag).strip('"')
    return f"{entry.get('LastModified')}:{entry.get('size')}"


def copy_chunks_parallel(s3, copy_pairs) -> set:
    """Copy ``(src, dst)`` chunk files from MinIO; returns the sources that failed.

    Each lands in a temporary file beside its destination and is renamed
    over it, so the trainer, which reads this same volume, sees the old chunk
    or the new one and never a half-written file. A failed chunk keeps what
    was on disk.
    """
    if not copy_pairs:
        return set()

    def copy_one(src, dst):
        dst = Path(dst)
        # Dot-prefixed, so nothing that lists chunk keys (z.y.x) sees it.
        tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.part")
        try:
            s3.get(src, str(tmp))
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink()

    failed = set()
    with ThreadPoolExecutor(max_workers=max(1, min(len(copy_pairs), worker_count()))) as executor:
        futures = {executor.submit(copy_one, src, dst): src for src, dst in copy_pairs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                failed.add(futures[fut])
                logger.warning(f"Could not sync chunk {futures[fut]}: {e}; will retry next sync.")
    return failed


def sync_zarr_group_metadata(s3, src_path, dst_path) -> set:
    """Create the arrays of MinIO's group ``src_path`` that ``dst_path`` lacks,
    and copy the attrs; returns the arrays whose local shape, chunks or dtype
    differ from MinIO's. Those are left alone: re-creating one would delete
    its chunks on disk, and chunks must not be copied into it."""
    src_group = zarr.open_group(store=s3fs.S3Map(root=src_path, s3=s3), mode="r")
    dst_group = zarr.open_group(store=zarr.DirectoryStore(str(dst_path)), mode="a")

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
                key, shape=src_array.shape, chunks=src_array.chunks, dtype=src_array.dtype, fill_value=0
            )
        dst_group[key].attrs.update(src_array.attrs)

    dst_group.attrs.update(src_group.attrs)
    return mismatched


def diff_and_sync_chunks(s3, s0_path, dst_s0_path, known_state, force=False) -> tuple:
    """Copy the chunks of MinIO's ``s0_path`` that changed since ``known_state``.

    Nothing on disk is ever deleted because MinIO lacks it: a chunk missing
    from a listing is almost always a transient (a truncated page, a mirror
    in flight, a restart), and painting background rewrites a chunk rather
    than removing it. Trusting one bad listing once wiped 3456 chunks of a
    session.

    Returns:
        ``(changed_keys, [], remote_state)``. A chunk that failed to copy is
        left out of both, keeping its previous state, so the next sync
        retries it. The empty list is where removed keys used to be.
    """
    try:
        # One listing, with each object's ETag: no per-chunk HEAD request.
        entries = s3.ls(s0_path, detail=True)
    except FileNotFoundError:
        # Nothing painted yet.
        return [], [], dict(known_state)
    except Exception as e:
        logger.warning(f"diff_and_sync_chunks: s3.ls({s0_path}) failed: {e}; "
                       "treating as transient, skipping sync this cycle.")
        return [], [], dict(known_state)

    remote_state = {}
    for entry in entries:
        name = (entry.get("name") or entry.get("Key") or "") if isinstance(entry, dict) else str(entry)
        key = Path(name).name
        if CHUNK_KEY_RE.match(key):
            remote_state[key] = chunk_version(entry)

    changed = [k for k, v in remote_state.items() if force or known_state.get(k) != v]
    if not changed:
        return [], [], remote_state

    dst_s0_path = Path(dst_s0_path)
    dst_s0_path.mkdir(parents=True, exist_ok=True)
    failed = {Path(src).name for src in copy_chunks_parallel(
        s3, [(f"{s0_path}/{k}", str(dst_s0_path / k)) for k in changed]
    )}
    for key in failed:
        if key in known_state:
            remote_state[key] = known_state[key]
        else:
            remote_state.pop(key, None)
    return [k for k in changed if k not in failed], [], remote_state


def volume_record(volume_id, zarr_path=None, *, volumes):
    """The registry record of ``volume_id``, rebuilt from ``zarr_path``'s attrs
    (and registered) when there is none yet: after a dashboard restart, say.
    None if neither gives one."""
    if volume_id in volumes:
        return volumes[volume_id]
    if zarr_path is None:
        return None
    try:
        # Syncing needs no geometry; a volume without it syncs, and the
        # manifest refuses it (build_manifest).
        record = read_volume(zarr_path, require_geometry=False)
    except NotAnAnnotationVolume:
        return None
    except Exception as e:
        logger.error(f"Error reconstructing volume metadata for {volume_id}: {e}")
        return None
    volumes[volume_id] = record
    return record


def sync_volume(volume_id, force=False, zarr_path=None, *, state, volumes) -> bool:
    """Pull an annotation volume's changed chunks from MinIO; True if any came.

    They go to the volume's own zarr: ``zarr_path`` (a resumed copy), else
    the registered one, else -- for a volume this process has no record of
    -- ``<output_base>/<volume_id>.zarr``. Sending every volume's strokes to
    the first session's output_base put them where its trainer never looked.
    """
    with _sync_lock:
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
            src_annotation = f"{state['bucket']}/{zarr_name}/annotation"
            if not s3.exists(src_annotation):
                return False
            dst_annotation = Path(local_zarr_path) / "annotation"
            dst_annotation.mkdir(parents=True, exist_ok=True)
            if "s0" in sync_zarr_group_metadata(s3, src_annotation, dst_annotation):
                return False

            changed, _, remote_state = diff_and_sync_chunks(
                s3, f"{src_annotation}/s0", dst_annotation / "s0",
                record.get("chunk_sync_state", {}), force=force,
            )
            if not changed:
                return False
            logger.info(f"Synced {len(changed)} changed chunks for volume {volume_id}")
            record["chunk_sync_state"] = remote_state
            return True
        except Exception:
            logger.exception(f"Error syncing annotation volume {volume_id}")
            return False


def sync_all(force: bool = True, *, state, volumes) -> int:
    """Sync every annotation volume in the bucket; how many changed, or -1
    if MinIO is not initialized."""
    with _sync_lock:
        if not state.get("ip") or not state.get("port"):
            logger.info("MinIO not initialized, skipping annotation sync")
            return -1

        logger.debug(f"Syncing all annotations from MinIO (force={force})...")
        s3 = minio.make_s3_filesystem(state)
        zarr_ids = [Path(c).name.replace(".zarr", "") for c in s3.ls(state["bucket"]) if c.endswith(".zarr")]
        checked = synced = failed = 0
        for zid in zarr_ids:
            # Only annotation volumes; anything else in the bucket is a crop
            # zarr of the removed create-crop route.
            attrs_path = f"{state['bucket']}/{zid}.zarr/.zattrs"
            try:
                if not s3.exists(attrs_path) or json.loads(s3.cat(attrs_path)).get("type") != "annotation_volume":
                    continue
            except Exception as e:
                logger.debug(f"Could not read root attrs for {zid}: {e}")
                failed += 1
                continue
            checked += 1
            if sync_volume(zid, force=force, state=state, volumes=volumes):
                synced += 1

        # Say whether nothing changed or something is broken: one line for
        # both made a real sync failure take a day to spot.
        if synced:
            summary = f"{synced} updated, {checked - synced} unchanged"
        else:
            summary = f"no changes ({checked} checked)"
        if failed:
            summary += f", {failed} could not be read"
        logger.info(f"Annotation sync: {summary}")
        return synced


def periodic_sync_once(*, state, volumes) -> None:
    """One round of the periodic sync; failures are warned about, not hidden."""
    try:
        if not state["output_base"] or not state["ip"] or not state["port"]:
            return
        # Pull annotations to disk and nothing else. Any write to the viewer
        # makes the browser rebuild every layer, taking the draw tool out of
        # the user's hand and dropping strokes still buffered behind the
        # brush; the annotated-regions boxes are refreshed from a button.
        sync_all(force=False, state=state, volumes=volumes)
        if _sync_failures["count"]:
            logger.info(f"Periodic annotation sync recovered after {_sync_failures['count']} failure(s)")
        _sync_failures.update(count=0, last_warned=None)
    except Exception as e:
        # A warning, once per interval: at DEBUG, an outage went unnoticed
        # until training ran on stale annotations.
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
