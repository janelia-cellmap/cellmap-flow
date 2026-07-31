import json
import logging
import os
import threading
import time
from datetime import datetime

from flask import jsonify, request

from cellmap_flow.dashboard.finetune_utils import ensure_minio_serving
from cellmap_flow.dashboard.routes.finetune.common import (
    ensure_corrections_storage,
    rewrite_minio_url_for_proxy,
)
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


# Module-level progress tracker for in-flight Resume operations, keyed by a
# load_id supplied by the client. Same pattern as
# yaml_crops._PROGRESS / _set_progress so the dashboard can poll for updates
# while the long copytree + mirror is in flight.
_RESUME_PROGRESS: dict = {}
_RESUME_PROGRESS_LOCK = threading.Lock()
_RESUME_PROGRESS_TTL_SECONDS = 300


def _set_resume_progress(load_id, **fields):
    if not load_id:
        return
    with _RESUME_PROGRESS_LOCK:
        entry = _RESUME_PROGRESS.setdefault(load_id, {"created_at": time.time()})
        entry.update(fields)
        entry["updated_at"] = time.time()
        now = time.time()
        stale = [
            k for k, v in _RESUME_PROGRESS.items()
            if now - v.get("updated_at", v.get("created_at", now)) > _RESUME_PROGRESS_TTL_SECONDS
        ]
        for k in stale:
            _RESUME_PROGRESS.pop(k, None)


def get_resume_progress_response(load_id):
    if not load_id:
        return jsonify({"success": False, "error": "Missing 'load_id' query param"}), 400
    with _RESUME_PROGRESS_LOCK:
        snapshot = _RESUME_PROGRESS.get(load_id)
        snapshot = dict(snapshot) if snapshot else None
    if snapshot is None:
        return jsonify({"success": False, "error": f"Unknown load_id {load_id}"}), 404
    return jsonify({"success": True, "progress": snapshot})


def _copytree_with_progress(src, dst, load_id, label, parent_done, parent_total):
    """``shutil.copytree`` replacement that copies files in parallel and emits
    per-file progress. NFS round-trip latency dominates per-file cost, so
    threading gives a big speedup on small-file workloads (sparse zarr chunks).
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed

    file_pairs: list[tuple[str, str]] = []
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for d in dirs:
            os.makedirs(os.path.join(target_root, d), exist_ok=True)
        for f in files:
            file_pairs.append(
                (os.path.join(root, f), os.path.join(target_root, f))
            )

    files_in_src = len(file_pairs)
    if files_in_src == 0:
        return 0

    # Use exactly what LSF allocated (LSB_DJOB_NUMPROC, falling back to CPU
    # affinity). No artificial ceiling — going above the slot count means
    # using cores LSF didn't give us; going below leaves throughput on the
    # table.
    from cellmap_flow.dashboard.finetune_utils import _get_sync_worker_count

    workers = max(1, min(_get_sync_worker_count(), files_in_src))

    def _copy_one(pair):
        s, d = pair
        shutil.copy2(s, d)

    copied_so_far = 0
    progress_step = max(1, files_in_src // 50)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_copy_one, p) for p in file_pairs]
        for fut in as_completed(futures):
            fut.result()  # surface any exception
            copied_so_far += 1
            if copied_so_far % progress_step == 0 or copied_so_far == files_in_src:
                _set_resume_progress(
                    load_id,
                    phase="copying",
                    current=label,
                    files_done=copied_so_far,
                    files_total=files_in_src,
                    parent_done=parent_done,
                    parent_total=parent_total,
                )
    return files_in_src


def _register_annotation_volume(volume_id, **volume_data):
    if not hasattr(g, "annotation_volumes"):
        g.annotation_volumes = {}
    g.annotation_volumes[volume_id] = {
        **volume_data,
        "extracted_chunks": set(),
        "chunk_sync_state": {},
    }


def list_existing_sessions_response(data):
    try:
        output_path = data.get("output_path", "")
        if not output_path:
            return jsonify({"success": False, "error": "output_path required"}), 400

        base = os.path.expanduser(output_path)
        if not os.path.isdir(base):
            return jsonify({"success": True, "sessions": []})

        sessions = []
        for entry in sorted(os.listdir(base), reverse=True):
            session_dir = os.path.join(base, entry)
            corrections_dir = os.path.join(session_dir, "corrections")
            if not os.path.isdir(corrections_dir):
                continue

            volumes = []
            chunks = []
            for item in os.listdir(corrections_dir):
                if not item.endswith(".zarr"):
                    continue
                full = os.path.join(corrections_dir, item)
                if "_chunk_" in item:
                    chunks.append(item)
                else:
                    volumes.append({"volume_id": item.replace(".zarr", ""), "path": full})

            if volumes or chunks:
                sessions.append(
                    {
                        "session_id": entry,
                        "session_path": session_dir,
                        "volumes": volumes,
                        "chunk_count": len(chunks),
                    }
                )

        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def load_existing_volume_response(data):
    try:
        import shutil

        from cellmap_flow.dashboard.finetune_utils import minio_state

        source_session_path = data.get("source_session_path")
        output_path = data.get("output_path")
        load_id = data.get("load_id")
        if load_id:
            _set_resume_progress(
                load_id,
                phase="starting",
                done=False,
                files_done=0,
                files_total=0,
                parent_done=0,
                parent_total=0,
            )
        if not source_session_path or not output_path:
            return jsonify(
                {"success": False, "error": "source_session_path and output_path required"}
            ), 400

        source_session_path = os.path.expanduser(source_session_path)
        source_corrections = os.path.join(source_session_path, "corrections")
        if not os.path.isdir(source_corrections):
            return jsonify({"success": False, "error": f"No corrections found in {source_session_path}"}), 404

        volume_entries = [
            entry
            for entry in os.listdir(source_corrections)
            if entry.endswith(".zarr") and "_chunk_" not in entry
        ]
        if not volume_entries:
            return jsonify(
                {"success": False, "error": f"No annotation volume found in {source_corrections}"}
            ), 404

        volume_dir = volume_entries[0]
        volume_id = volume_dir.replace(".zarr", "")
        new_session_path, new_corrections = ensure_corrections_storage(output_path)

        all_zarr_entries = [item for item in os.listdir(source_corrections) if item.endswith(".zarr")]
        has_volume_zarr = any("_chunk_" not in e for e in all_zarr_entries)
        if has_volume_zarr:
            # New unified flow: trainer reads the volume zarr directly via
            # VirtualPatchDataset; the per-chunk _chunk_*.zarr extracts from
            # the legacy materialize pipeline are dead weight (and on big
            # sessions can be thousands of files).
            zarr_entries = [e for e in all_zarr_entries if "_chunk_" not in e]
            skipped_chunk_extracts = len(all_zarr_entries) - len(zarr_entries)
            if skipped_chunk_extracts:
                logger.info(
                    f"Resume: skipping {skipped_chunk_extracts} legacy "
                    f"_chunk_*.zarr extracts; trainer will read the volume "
                    "zarr directly via the manifest."
                )
        else:
            # Legacy session with only per-chunk extracts (no volume zarr).
            # Copy them so the trainer's CorrectionDataset path still works.
            zarr_entries = all_zarr_entries
            skipped_chunk_extracts = 0
        copied = []
        for idx, item in enumerate(zarr_entries):
            src = os.path.join(source_corrections, item)
            dst = os.path.join(new_corrections, item)
            if os.path.exists(dst):
                logger.info(f"Skipping {item} (already exists in target)")
                continue
            if load_id:
                _set_resume_progress(
                    load_id,
                    phase="copying",
                    current=item,
                    files_done=0,
                    files_total=0,
                    parent_done=idx,
                    parent_total=len(zarr_entries),
                    done=False,
                )
            _copytree_with_progress(
                src, dst, load_id, label=item,
                parent_done=idx, parent_total=len(zarr_entries),
            )
            copied.append(item)

        source_minio = os.path.join(source_corrections, ".minio")
        new_minio = os.path.join(new_corrections, ".minio")
        copied_minio = False
        if os.path.isdir(source_minio):
            if minio_state.get("process") is not None and minio_state["process"].poll() is None:
                logger.warning(
                    "MinIO already running with a different output_base; cannot rebind. "
                    "Falling back to mc mirror upload - painted data may be incomplete "
                    "if the source had unsynced chunks."
                )
            elif not os.path.exists(new_minio):
                if load_id:
                    _set_resume_progress(
                        load_id,
                        phase="copying_minio",
                        current=".minio",
                        files_done=0, files_total=0,
                        parent_done=len(zarr_entries),
                        parent_total=len(zarr_entries) + 1,
                        done=False,
                    )
                _copytree_with_progress(
                    source_minio, new_minio, load_id, label=".minio",
                    parent_done=len(zarr_entries), parent_total=len(zarr_entries) + 1,
                )
                copied_minio = True

        if load_id:
            _set_resume_progress(
                load_id,
                phase="mirroring_minio",
                current=volume_dir,
                done=False,
            )

        lineage_file = os.path.join(new_session_path, "loaded_from.json")
        with open(lineage_file, "w") as f:
            json.dump(
                {
                    "source_session_path": source_session_path,
                    "loaded_at": datetime.now().isoformat(),
                    "copied_files": copied,
                },
                f,
                indent=2,
            )

        new_volume_path = os.path.join(new_corrections, volume_dir)
        zattrs_file = os.path.join(new_volume_path, ".zattrs")
        volume_meta = {}
        if os.path.exists(zattrs_file):
            with open(zattrs_file) as f:
                volume_meta = json.load(f)

        s0_dir = os.path.join(new_volume_path, "annotation", "s0")
        s0_count = 0
        if os.path.isdir(s0_dir):
            s0_count = sum(1 for entry in os.listdir(s0_dir) if not entry.startswith("."))

        minio_url = ensure_minio_serving(new_volume_path, volume_id, output_base_dir=new_corrections)
        minio_url = rewrite_minio_url_for_proxy(minio_url, request)
        _register_annotation_volume(
            volume_id,
            zarr_path=new_volume_path,
            model_name=volume_meta.get("model_name"),
            output_size=volume_meta.get("chunk_size"),
            input_size=volume_meta.get("input_size"),
            input_voxel_size=volume_meta.get("input_voxel_size"),
            output_voxel_size=volume_meta.get("output_voxel_size"),
            dataset_path=volume_meta.get("dataset_path"),
            dataset_offset_nm=volume_meta.get("dataset_offset_nm"),
            corrections_dir=new_corrections,
        )
        refresh_annotated_regions_layer()

        if load_id:
            _set_resume_progress(
                load_id,
                phase="done",
                done=True,
                volume_id=volume_id,
                copied_count=len(copied),
                painted_chunk_count=s0_count,
            )

        return jsonify(
            {
                "success": True,
                "volume_id": volume_id,
                "new_session_path": new_session_path,
                "zarr_path": new_volume_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "copied_count": len(copied),
                "copied_minio": copied_minio,
                "painted_chunk_count": s0_count,
                "skipped_chunk_extracts": skipped_chunk_extracts,
                "metadata": volume_meta,
            }
        )
    except Exception as e:
        if load_id:
            _set_resume_progress(load_id, phase="error", done=True, error=str(e))
        logger.error(f"Error loading existing volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
