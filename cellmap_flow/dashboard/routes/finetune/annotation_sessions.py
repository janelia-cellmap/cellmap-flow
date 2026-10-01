"""Earlier sessions: listing them, and resuming one's volume in a new session.

Routes: POST ``/api/finetune/list-existing-sessions``, POST
``/api/finetune/load-existing-volume`` (the resume) and GET
``/api/finetune/load-existing-volume-progress`` (how far a resume has got).
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime

from flask import jsonify, request

from cellmap_flow.dashboard.finetune_utils import ensure_minio_serving
from cellmap_flow.dashboard.progress import Progress
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    ensure_corrections_storage,
    rewrite_minio_url_for_proxy,
    session_store,
    write_volume_manifest,
)
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session import manifest as session_manifest
from cellmap_flow.finetune.session import sync as session_sync
from cellmap_flow.finetune.session.manifest import read_manifest
from cellmap_flow.finetune.session.volume import read_volume

logger = logging.getLogger(__name__)

# Each resume's progress, by the load_id the page sent with it: the phase
# (copying the zarrs, then MinIO's data, then mirroring), and the files and
# zarrs copied so far.
_RESUME_PROGRESS = Progress()


@finetune_bp.route("/api/finetune/load-existing-volume-progress", methods=["GET"])
def load_existing_volume_progress():
    """How far the resume with this load_id has got."""
    return _RESUME_PROGRESS.response(request.args.get("load_id"))


def _copytree_with_progress(src, dst, load_id, label, parent_done, parent_total):
    """``shutil.copytree`` replacement that copies files in parallel and emits
    per-file progress. NFS round-trip latency dominates per-file cost, so
    threading gives a big speedup on small-file workloads (sparse zarr chunks).
    """
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
    workers = max(1, min(session_sync.worker_count(), files_in_src))

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
                _RESUME_PROGRESS.update(
                    load_id,
                    phase="copying",
                    current=label,
                    files_done=copied_so_far,
                    files_total=files_in_src,
                    parent_done=parent_done,
                    parent_total=parent_total,
                )
    return files_in_src


def _annotation_volume_dirs(corrections_dir):
    """The annotation volumes in a corrections directory, the one to use first.

    Only zarrs whose attrs say ``type: annotation_volume`` count: imported
    crop zarrs and legacy _chunk_ extracts sit in the same directory and
    were counted as volumes too. Resume used to take whichever of them
    os.listdir() happened to return first. The first entry here is the one
    the session's manifest trains on, else the most recently written.

    The _chunk_ extracts are per-chunk copies that older dashboards wrote
    beside the volume, often thousands of them; they are skipped by name,
    without opening their attrs.
    """
    volumes = []
    for entry in os.listdir(corrections_dir):
        if not entry.endswith(".zarr") or "_chunk_" in entry:
            continue
        attrs_file = os.path.join(corrections_dir, entry, ".zattrs")
        try:
            with open(attrs_file) as f:
                if json.load(f).get("type") != "annotation_volume":
                    continue
        except (OSError, ValueError):
            continue
        volumes.append((os.path.getmtime(attrs_file), entry))
    volumes = [entry for _, entry in sorted(volumes, reverse=True)]
    try:
        trained = (read_manifest(corrections_dir) or {}).get("volume_zarr_path")
    except (OSError, ValueError):
        trained = None
    if trained and os.path.basename(str(trained).rstrip("/")) in volumes:
        name = os.path.basename(str(trained).rstrip("/"))
        volumes.remove(name)
        volumes.insert(0, name)
    return volumes


def _volume_dataset(volume_path):
    """The raw dataset an annotation volume was painted on (its attrs'
    ``dataset_path``), or None when it does not say."""
    try:
        with open(os.path.join(volume_path, ".zattrs")) as f:
            return json.load(f).get("dataset_path")
    except (OSError, ValueError):
        return None


def _same_dataset(a, b):
    """Whether two dataset paths name one dataset: a trailing slash and a
    trailing scale level (``/s2``) aside."""
    def bare(path):
        return re.sub(r"/s\d+$", "", str(path).rstrip("/"))
    return bare(a) == bare(b)


def _populated_chunk_count(volume_path):
    """How many chunks of a volume's annotation/s0 are on disk, painted or imported."""
    s0_dir = os.path.join(volume_path, "annotation", "s0")
    try:
        return sum(1 for entry in os.listdir(s0_dir) if not entry.startswith("."))
    except OSError:
        return 0


@finetune_bp.route("/api/finetune/list-existing-sessions", methods=["POST"])
def list_existing_sessions():
    data = request.get_json() or {}
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

            volumes = [
                {"volume_id": item.replace(".zarr", ""), "path": os.path.join(corrections_dir, item)}
                for item in _annotation_volume_dirs(corrections_dir)
            ]
            if volumes:
                sessions.append(
                    {
                        "session_id": entry,
                        "session_path": session_dir,
                        "volumes": volumes,
                        # The volumes' populated chunks. This used to count
                        # legacy per-chunk extracts, which no session gets
                        # any more, so it said 0 for most sessions.
                        "chunk_count": sum(_populated_chunk_count(v["path"]) for v in volumes),
                        "dataset_path": _volume_dataset(volumes[0]["path"]),
                    }
                )

        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/load-existing-volume", methods=["POST"])
def load_existing_volume():
    data = request.get_json() or {}
    try:
        session = get_session()
        minio_state = session.minio_state
        source_session_path = data.get("source_session_path")
        output_path = data.get("output_path")
        load_id = data.get("load_id")
        _RESUME_PROGRESS.update(
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

        volume_entries = _annotation_volume_dirs(source_corrections)
        if not volume_entries:
            return jsonify(
                {"success": False, "error": f"No annotation volume found in {source_corrections}"}
            ), 404

        volume_dir = volume_entries[0]
        volume_id = volume_dir.replace(".zarr", "")

        # A volume's voxels are positions in the dataset it was painted on.
        # Resumed in a dashboard showing another dataset, the strokes were
        # drawn over the wrong EM, training read the old dataset, and the
        # finetuned layer served the old dataset over the new one's view.
        painted_on = _volume_dataset(os.path.join(source_corrections, volume_dir))
        if painted_on and session.dataset_path and not _same_dataset(painted_on, session.dataset_path):
            return jsonify({"success": False, "error": (
                f"That session was painted on {painted_on}, and this dashboard has "
                f"{session.dataset_path} open. Start the dashboard on {painted_on} to resume it."
            )}), 409

        new_session_path, new_corrections = ensure_corrections_storage(output_path)

        all_zarr_entries = [item for item in os.listdir(source_corrections) if item.endswith(".zarr")]
        # The trainer reads the volume zarr through the manifest. The
        # per-chunk _chunk_*.zarr extracts that older dashboards wrote beside
        # it are dead weight, and a big session has thousands of them.
        zarr_entries = [e for e in all_zarr_entries if "_chunk_" not in e]
        skipped_chunk_extracts = len(all_zarr_entries) - len(zarr_entries)
        if skipped_chunk_extracts:
            logger.info(
                f"Resume: skipping {skipped_chunk_extracts} legacy "
                f"_chunk_*.zarr extracts; trainer will read the volume "
                "zarr directly via the manifest."
            )
        copied = []
        for idx, item in enumerate(zarr_entries):
            src = os.path.join(source_corrections, item)
            dst = os.path.join(new_corrections, item)
            if os.path.exists(dst):
                logger.info(f"Skipping {item} (already exists in target)")
                continue
            _RESUME_PROGRESS.update(
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
                _RESUME_PROGRESS.update(
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

        # The good regions sit beside corrections/, not in it, and the
        # trainer and the good-regions routes read them from the new
        # session. Left behind, a resumed session showed no regions and
        # trained with no rehearsal or anchored distillation.
        source_regions = session_manifest.good_regions_path(source_corrections)
        new_regions = session_manifest.good_regions_path(new_corrections)
        copied_good_regions = os.path.isfile(source_regions) and not os.path.exists(new_regions)
        if copied_good_regions:
            shutil.copy2(source_regions, new_regions)

        _RESUME_PROGRESS.update(
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
                    "copied_good_regions": copied_good_regions,
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

        s0_count = _populated_chunk_count(new_volume_path)

        minio_url = ensure_minio_serving(new_volume_path, volume_id, output_base_dir=new_corrections)
        minio_url = rewrite_minio_url_for_proxy(minio_url)
        # Whatever geometry the copied .zattrs has; what it lacks stays None.
        record = read_volume(new_volume_path, require_geometry=False)
        record.pop("chunk_sync_state")
        session_store().register_volume(volume_id, **record)
        # A resumed session is trained the same way a fresh one is. The
        # geometry comes from the copied .zattrs, so a volume written before
        # those keys existed gets no manifest and cannot be trained --
        # write_volume_manifest says so in the log.
        write_volume_manifest(session.annotation_volumes[volume_id])
        refresh_annotated_regions_layer()

        _RESUME_PROGRESS.update(
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
            _RESUME_PROGRESS.update(load_id, phase="error", done=True, error=str(e))
        logger.error(f"Error loading existing volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
