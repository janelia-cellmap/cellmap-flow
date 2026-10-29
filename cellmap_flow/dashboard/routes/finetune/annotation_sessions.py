"""Earlier sessions: listing them, and resuming one's volume in a new session.

Routes: POST ``/api/finetune/list-existing-sessions``, POST
``/api/finetune/load-existing-volume`` (the resume) and GET
``/api/finetune/load-existing-volume-progress`` (how far a resume has got).

A session is its current volume (``_current_volume_dir``: the one its
manifest trains on), as for every other route (SessionStore.session_volume):
the list counts that volume's chunks and names its model, and a resume
copies and registers that one alone. The other volumes in a session's
directory -- those Use this model retired lie in its ``superseded/``, out of
the way already -- are neither counted nor copied: they are the same paint
on other models' grids, and a resume of every volume would make the oldest
of them the resumed session's volume.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime

import worldarray as wa
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
from cellmap_flow.finetune.session.manifest import trained_volume_name
from cellmap_flow.finetune.session.volume import read_root_attrs, read_volume

logger = logging.getLogger(__name__)

# Each resume's progress, by the load_id the page sent with it: the phase
# (copying the zarrs, then MinIO's data, then mirroring), and the files and
# zarrs copied so far.
_RESUME_PROGRESS = Progress()


@finetune_bp.route("/api/finetune/load-existing-volume-progress", methods=["GET"])
def load_existing_volume_progress():
    """How far the resume with this load_id has got."""
    return _RESUME_PROGRESS.response(request.args.get("load_id"))


def _copytree_with_progress(src, dst, load_id, label, parent_done, parent_total, skip=None):
    """``shutil.copytree`` replacement that copies files in parallel and emits
    per-file progress. NFS round-trip latency dominates per-file cost, so
    threading gives a big speedup on small-file workloads (sparse zarr chunks).
    ``skip``, a function of a directory's path relative to ``src``, leaves
    out the directories it is true for, and everything under them.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    file_pairs: list[tuple[str, str]] = []
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_root, exist_ok=True)
        if skip is not None:
            dirs[:] = [d for d in dirs if not skip(os.path.normpath(os.path.join(rel, d)))]
        for d in dirs:
            os.makedirs(os.path.join(target_root, d), exist_ok=True)
        for f in files:
            file_pairs.append(
                (os.path.join(root, f), os.path.join(target_root, f))
            )

    files_in_src = len(file_pairs)
    if files_in_src == 0:
        return 0

    # Exactly the threads the job was given (worldarray's thread_budget:
    # OMP_NUM_THREADS, else the scheduler's slots, else the CPUs this process
    # may run on). No artificial ceiling — going above the slot count means
    # using cores LSF didn't give us; going below leaves throughput on the
    # table.
    workers = max(1, min(wa.thread_budget(), files_in_src))

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


def _current_volume_dir(corrections_dir):
    """The directory name (``<id>.zarr``) of the session's current volume in
    ``corrections_dir``, or None when it has no annotation volume: the one
    its manifest trains on (trained_volume_name), else the most recently
    written.

    Only zarrs whose attrs say ``type: annotation_volume`` count: imported
    crop zarrs and legacy _chunk_ extracts sit in the same directory and
    were counted as volumes too. Resume used to take whichever of them
    os.listdir() happened to return first. Only the directory's top level
    is looked in: a volume Use this model retired is in its ``superseded/``
    (store.SUPERSEDED_DIRNAME), and is never the session's.

    The _chunk_ extracts are per-chunk copies that older dashboards wrote
    beside the volume, often thousands of them; they are skipped by name,
    without opening their attrs.
    """
    volumes = []
    for entry in os.listdir(corrections_dir):
        if not entry.endswith(".zarr") or "_chunk_" in entry:
            continue
        path = os.path.join(corrections_dir, entry)
        try:
            if read_root_attrs(path).get("type") != "annotation_volume":
                continue
            volumes.append((os.path.getmtime(os.path.join(path, ".zattrs")), entry))
        except (OSError, ValueError):
            continue
    trained = trained_volume_name(corrections_dir)
    if trained in (entry for _, entry in volumes):
        return trained
    return max(volumes)[1] if volumes else None


def _volume_attrs(volume_path):
    """An annotation volume's root attrs (its ``dataset_path``, the raw it
    was painted on, and ``model_name``, the model it is for), or ``{}`` when
    they cannot be read."""
    try:
        return read_root_attrs(volume_path)
    except ValueError:
        return {}


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

            current = _current_volume_dir(corrections_dir)
            if current:
                path = os.path.join(corrections_dir, current)
                attrs = _volume_attrs(path)
                sessions.append(
                    {
                        "session_id": entry,
                        "session_path": session_dir,
                        # The session's current volume alone, what a resume
                        # copies.
                        "volumes": [{"volume_id": current.replace(".zarr", ""), "path": path}],
                        # The volume's own populated chunks, not the legacy
                        # per-chunk extracts', which no session gets any more.
                        "chunk_count": _populated_chunk_count(path),
                        # The model it is for; None for a volume painted
                        # before any model.
                        "model_name": attrs.get("model_name"),
                        "dataset_path": attrs.get("dataset_path"),
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

        volume_dir = _current_volume_dir(source_corrections)
        if not volume_dir:
            return jsonify(
                {"success": False, "error": f"No annotation volume found in {source_corrections}"}
            ), 404
        volume_id = volume_dir.replace(".zarr", "")

        # A volume's voxels are positions in the dataset it was painted on.
        # Resumed in a dashboard showing another dataset, the strokes were
        # drawn over the wrong EM, training read the old dataset, and the
        # finetuned layer served the old dataset over the new one's view.
        painted_on = _volume_attrs(os.path.join(source_corrections, volume_dir)).get("dataset_path")
        if painted_on and session.dataset_path and not _same_dataset(painted_on, session.dataset_path):
            return jsonify({"success": False, "error": (
                f"That session was painted on {painted_on}, and this dashboard has "
                f"{session.dataset_path} open. Start the dashboard on {painted_on} to resume it."
            )}), 409

        new_session_path, new_corrections = ensure_corrections_storage(output_path)

        # The session's current volume alone. Its other volumes, crop zarrs
        # and the per-chunk _chunk_*.zarr extracts older dashboards wrote
        # (thousands, in a big session) are not the session: the trainer
        # reads the one volume through the manifest.
        copied = []
        dst = os.path.join(new_corrections, volume_dir)
        if os.path.exists(dst):
            logger.info(f"Skipping {volume_dir} (already exists in target)")
        else:
            _RESUME_PROGRESS.update(
                load_id,
                phase="copying",
                current=volume_dir,
                files_done=0,
                files_total=0,
                parent_done=0,
                parent_total=1,
                done=False,
            )
            _copytree_with_progress(
                os.path.join(source_corrections, volume_dir), dst, load_id, label=volume_dir,
                parent_done=0, parent_total=1,
            )
            copied.append(volume_dir)

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
                    parent_done=1,
                    parent_total=2,
                    done=False,
                )
                # MinIO keeps each object under <bucket>/<key>/, so a
                # volume's strokes are <bucket>/<id>.zarr/: the other
                # volumes' stay behind with them.
                bucket = minio_state["bucket"]
                _copytree_with_progress(
                    source_minio, new_minio, load_id, label=".minio",
                    parent_done=1, parent_total=2,
                    skip=lambda rel: (os.path.dirname(rel) == bucket and rel.endswith(".zarr")
                                      and os.path.basename(rel) != volume_dir),
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
        volume_meta = read_root_attrs(new_volume_path)

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
                "metadata": volume_meta,
            }
        )
    except Exception as e:
        if load_id:
            _RESUME_PROGRESS.update(load_id, phase="error", done=True, error=str(e))
        logger.error(f"Error loading existing volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
