"""Training jobs: submitting one, restarting it, and following it.

Routes:
- POST ``/api/finetune/submit``: a job for a model over a session's corrections;
- POST ``/api/finetune/job/<job_id>/restart``: the next iteration, with new settings;
- GET ``/api/finetune/jobs``: every job, including those a dashboard before
  this one started (they are looked for first);
- GET ``/api/finetune/job/<job_id>/status`` and ``.../logs``;
- GET ``/api/finetune/job/<job_id>/logs/stream``: the log as server-sent events;
- POST ``/api/finetune/job/<job_id>/cancel`` and ``.../stop-early``.
"""

import os
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from flask import Response, jsonify, request

from cellmap_flow.dashboard.finetune_layers import follow_jobs
from cellmap_flow.dashboard.finetune_utils import sync_all_annotations_from_minio
from cellmap_flow.dashboard.requests import FinetuneRestart, FinetuneSubmit, parse
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    LOG_FILTER_PATTERNS,
    autodetect_output_type,
    build_restart_params,
    current_chain,
    detect_sparse_annotations,
    find_model_config,
    get_lsf_job_id,
    resolve_finetune_session,
    training_settings,
)
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.job_manager.state import can_restart
from cellmap_flow.finetune.session.store import SESSION_DIR_RE
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import exists_now

logger = logging.getLogger(__name__)


def _step_names(config):
    """Step names from an input_norm/postprocess chain, whichever shape it is.

    current_chain() gives a list of dicts carrying a "name" key, in the order
    the steps run (a dict's keys would lose it: jsonify sorts them). A
    manifest written before the chains were lists may still hold the legacy
    name-keyed dict. Only used for logging, so an unrecognised shape gives no
    names rather than raising.
    """
    if isinstance(config, dict):
        return list(config.keys())
    if isinstance(config, list):
        return [d.get("name") for d in config if isinstance(d, dict)]
    return []


def _backfill_manifest(corrections_dir):
    """Write a manifest for a session that predates the volume routes writing one.

    Returns the manifest if one could be written, else None: the session
    then has nothing the trainer can read, and submit refuses it.
    """
    from cellmap_flow.dashboard.routes.finetune.common import session_store, write_volume_manifest
    from cellmap_flow.finetune.session.manifest import read_manifest

    _, volume = session_store().session_volume(corrections_dir)
    if volume is not None:
        if write_volume_manifest(volume) is None:
            return None
        return read_manifest(str(corrections_dir))

    logger.info(
        f"No annotation volume registered for {corrections_dir}, so no "
        "manifest to backfill; the session cannot be trained without one."
    )
    return None


def _refresh_virtual_manifest_for_training(corrections_dir, manifest, overrides, context):
    """Write the dashboard's chains and the run's ``overrides`` (see
    requests._ManifestOverrides) into a session's manifest before ``context``,
    a submit or a restart."""
    from cellmap_flow.finetune.session.manifest import write_manifest

    current_norm, current_postprocess = current_chain()
    if current_norm and manifest.get("input_norm") != current_norm:
        logger.info(
            "Refreshing manifest input_norm before %s "
            "(was: %s, now: %s)",
            context,
            _step_names(manifest.get("input_norm")),
            _step_names(current_norm),
        )
    manifest["input_norm"] = current_norm

    if current_postprocess and manifest.get("postprocess") != current_postprocess:
        logger.info(
            "Refreshing manifest postprocess before %s "
            "(was: %s, now: %s)",
            context,
            _step_names(manifest.get("postprocess")),
            _step_names(current_postprocess),
        )
    manifest["postprocess"] = current_postprocess

    for key, value in overrides.items():
        logger.info(f"Applying the {key} override before {context}: {manifest.get(key)} -> {value}")
        manifest[key] = value

    write_manifest(str(corrections_dir), manifest)


def _pull_annotations(context):
    """Pull from MinIO what changed since the last sync; the number of volumes pulled.

    force=False diffs the chunk keys and downloads only what differs, so it
    is cheap when nothing changed. 0 when MinIO is not running, or the sync
    failed (logged, under ``context``): the volume on disk is then what
    there is.
    """
    try:
        t0 = time.perf_counter()
        pulled = sync_all_annotations_from_minio(force=False) or 0
        elapsed = time.perf_counter() - t0
    except Exception as e:
        logger.warning(f"{context}: error syncing annotations from MinIO: {e}")
        return 0
    if pulled < 0:
        logger.info(f"{context}: MinIO is not running, so there is nothing to pull.")
        return 0
    if pulled:
        logger.info(
            f"{context}: pulled new annotations for {pulled} volume(s) in "
            f"{elapsed:.2f}s; training uses them."
        )
    else:
        logger.info(
            f"{context}: nothing new to pull ({elapsed:.2f}s) -- annotations "
            f"on disk are already current."
        )
    return pulled


def _rehydrate_jobs():
    """Reattach to jobs a previous dashboard process left running.

    Looks in every session under the base paths this dashboard knows: the
    ones it made sessions for, and the output path saved in the user prefs,
    which is what a freshly restarted dashboard starts with.
    """
    from cellmap_flow.dashboard.routes.finetune.common import load_user_prefs

    manager = get_session().finetune_job_manager
    rehydrate = getattr(manager, "rehydrate_session", None)
    if rehydrate is None:
        return
    # Before the jobs are found: finding one starts its monitor, which tells
    # the viewer when its server is up.
    follow_jobs(manager)
    bases = {os.path.expanduser(b) for b in get_session().output_sessions}
    saved = load_user_prefs().get("outputPath")
    if saved:
        bases.add(os.path.expanduser(saved))
    for base in bases:
        try:
            sessions = [
                os.path.join(base, entry) for entry in os.listdir(base)
                if SESSION_DIR_RE.match(entry)
            ]
        except OSError:
            continue
        for session in sessions:
            try:
                rehydrate(session)
            except Exception as e:
                logger.warning(f"Could not look for running jobs in {session}: {e}")


@finetune_bp.route("/api/finetune/jobs", methods=["GET"])
def get_finetuning_jobs():
    try:
        _rehydrate_jobs()
        return jsonify({"success": True, "jobs": get_session().finetune_job_manager.list_jobs()})
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    try:
        status = get_session().finetune_job_manager.get_job_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify({"success": True, **status})
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/logs", methods=["GET"])
def get_job_logs(job_id):
    try:
        manager = get_session().finetune_job_manager
        job = (getattr(manager, "jobs", {}) or {}).get(job_id)
        if job is not None and Path(job.log_file).exists():
            # Whole lines only, with the byte offset they end at: the client
            # opens the live stream from there, rather than having the stream
            # send the whole log again on top of this.
            logs, offset = _read_complete_lines(job.log_file, 0)
            return jsonify({"success": True, "logs": logs, "offset": offset})
        logs = manager.get_job_logs(job_id)
        if logs is None:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify({"success": True, "logs": logs, "offset": 0})
    except Exception as e:
        logger.error(f"Error getting job logs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/submit", methods=["POST"])
def submit_finetuning():
    body, refused = parse(FinetuneSubmit, request.get_json() or {})
    if refused:
        return refused
    try:
        model_config = find_model_config(body.model_name)
        if not model_config:
            return jsonify({"success": False, "error": f"Model {body.model_name} not found"}), 404

        session_path, actual_corrections_path = resolve_finetune_session(body.corrections_path)
        if not actual_corrections_path.exists():
            return jsonify(
                {
                    "success": False,
                    "error": f"Corrections path does not exist: {actual_corrections_path}. Please create annotation crops first.",
                }
            ), 400

        from cellmap_flow.finetune.session.manifest import read_manifest

        existing_manifest = read_manifest(str(actual_corrections_path))
        if existing_manifest is None:
            # Sessions started before the volume routes wrote a manifest have
            # a perfectly trainable volume zarr and no sentinel pointing at
            # it. Backfill from the registered volume rather than making the
            # user start over.
            existing_manifest = _backfill_manifest(actual_corrections_path)

        if existing_manifest is not None:
            _refresh_virtual_manifest_for_training(
                actual_corrections_path, existing_manifest, body.overrides(), "submit"
            )

        output_type, offsets = autodetect_output_type(model_config, body.output_type, body.offsets)

        # Whether the session is sparse is read from the volume on disk, which
        # lags the browser's strokes (they go straight to MinIO) by up to the
        # periodic sync's 30 s, or for good while that sync fails. So pull
        # first, as restart does: strokes painted just before Submit made a
        # distance model train distance/bce on scribbles, without the mask.
        # The sync used to also write per-chunk extracts and could take
        # minutes; it is only the chunk diff now.
        _pull_annotations("Submit pre-sync")
        has_sparse = detect_sparse_annotations(actual_corrections_path)
        settings = training_settings(
            output_type=output_type,
            loss_type=body.loss_type,
            label_smoothing=body.label_smoothing,
            distillation_lambda=body.distillation_lambda,
            sparse=has_sparse,
        )

        session = get_session()
        # Before the job exists: submitting starts its monitor, which tells
        # the viewer when its server is up and when each iteration is done.
        follow_jobs(session.finetune_job_manager)
        finetune_job = session.finetune_job_manager.submit_finetuning_job(
            model_config=model_config,
            corrections_path=actual_corrections_path,
            lora_r=body.lora_r,
            num_epochs=body.num_epochs,
            batch_size=body.batch_size,
            learning_rate=body.learning_rate,
            output_base=Path(session_path),
            checkpoint_path_override=Path(body.checkpoint_path) if body.checkpoint_path else None,
            auto_serve=body.auto_serve,
            mask_unannotated=settings.mask_unannotated,
            loss_type=settings.loss_type,
            label_smoothing=settings.label_smoothing,
            distillation_lambda=settings.distillation_lambda,
            distillation_scope=body.distillation_scope,
            margin=body.margin,
            balance_classes=body.balance_classes,
            augment=body.augment,
            queue=body.queue,
            # The request's, else the dashboard's own, else the site's.
            charge_group=body.charge_group or session.charge_group or current_site().default_charge_group,
            walltime=session.walltime,
            output_type=settings.output_type,
            select_channel=body.select_channel,
            offsets=offsets,
        )

        response = {
            "success": True,
            "job_id": finetune_job.job_id,
            "lsf_job_id": get_lsf_job_id(finetune_job),
            "output_dir": str(finetune_job.output_dir),
            # Over the parent so every run in the session tree overlays.
            "tensorboard_command": f"tensorboard --logdir {os.path.dirname(str(finetune_job.output_dir))}",
            "output_type": settings.output_type,
            "message": "Finetuning job submitted successfully",
        }
        if settings.note:
            response["note"] = settings.note
        return jsonify(response)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting finetuning job: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def _read_complete_lines(path, offset):
    """(text of the whole lines after byte ``offset``, byte offset after them).

    Byte offsets, so they can be handed to the client as SSE event ids and
    back, and a partial last line is left for the next read -- a read can
    land mid-line ("Epoch 7/10 - Lo"), and emitting that split the record in
    two, neither half matching the client's "Epoch N/M - Loss:" pattern.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        chunk = f.read()
    cut = chunk.rfind(b"\n") + 1
    return chunk[:cut].decode("utf-8", errors="replace"), offset + cut


def _requested_offset(request):
    """Where the client wants the log from: the SSE Last-Event-ID, else ?offset=."""
    for raw in (request.headers.get("Last-Event-ID"), request.args.get("offset")):
        try:
            if raw not in (None, ""):
                return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return 0


# Statuses in which the job can still write to its log.
_LIVE_STATUSES = ("PENDING", "RUNNING", "WAITING_FOR_RESTART")

# The line bpeek puts before a job's output, which is not part of the log.
_BPEEK_HEADER = re.compile(rb"\A<< output from stdout >>\r?\n")


@finetune_bp.route("/api/finetune/job/<job_id>/logs/stream", methods=["GET"])
def stream_job_logs(job_id):
    """Server-sent log stream.

    Every block of lines carries ``id: <byte offset>``, the position in the
    log after it, and a (re)connection resumes from the client's
    Last-Event-ID -- which EventSource sends by itself when it reconnects --
    or from ``?offset=``. The stream ends with ``event: done`` once the job
    is finished. It used to send the whole log on every connection and just
    stop at the end, so the browser's automatic reconnect replayed the whole
    log, plus "=== Training COMPLETED ===", every few seconds for as long as
    the page stayed open.
    """
    log_filters = [re.compile(pattern) for pattern in LOG_FILTER_PATTERNS]
    start_offset = _requested_offset(request)

    def iter_visible_lines(text):
        for line in text.splitlines():
            if line and not any(pattern.search(line) for pattern in log_filters):
                yield line

    def sse_data_block(lines, event_id=None):
        head = f"id: {event_id}\n" if event_id is not None else ""
        if not lines:
            # Nothing to show, but still say how far the log has been read,
            # so a reconnect does not re-read it.
            return head + "\n" if head else None
        payload = "\n".join(lines)
        return head + "data: " + payload.replace("\n", "\ndata: ") + "\n\n"

    def sse_done(status):
        return f"event: done\ndata: {status}\n\n"

    def read_bpeek_output(lsf_job_id):
        """The job's output so far, as bpeek shows it: bytes, without bpeek's
        "<< output from stdout >>" header. None when bpeek cannot be run."""
        try:
            result = subprocess.run(["bpeek", str(lsf_job_id)], capture_output=True, timeout=2)
        except Exception as e:
            logger.debug(f"bpeek call failed for job {lsf_job_id}: {e}")
            return None
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if stderr and "Not yet started" not in stderr:
            logger.debug(f"bpeek stderr for job {lsf_job_id}: {stderr}")
        return _BPEEK_HEADER.sub(b"", result.stdout or b"", count=1)

    def generate():
        heartbeat_interval_s = 1.0
        last_heartbeat = time.perf_counter()

        fjm = get_session().finetune_job_manager
        if job_id not in fjm.jobs:
            yield f"data: Job {job_id} not found\n\n"
            yield sse_done("NOT_FOUND")
            return

        finetune_job = fjm.jobs[job_id]
        lsf_job_id = None
        if finetune_job.lsf_job and hasattr(finetune_job.lsf_job, "job_id"):
            lsf_job_id = finetune_job.lsf_job.job_id

        # How many bytes of the log the client has, whichever source they
        # came from. The job's output is the log: the trainer's output goes
        # through `tee` into both, byte for byte. So bpeek, read until the
        # log file can be seen from here (jobs.spec.exists_now), and the file
        # count the same bytes, and neither replays what the other sent: on
        # switching sources, or when the browser reconnects with the id of
        # a block bpeek sent. A block without an id, as bpeek's were, made a
        # reconnect start the file from 0, and the log showed twice.
        position = start_offset
        use_bpeek = lsf_job_id is not None
        last_bpeek_poll = 0.0
        bpeek_poll_interval_s = 1.0

        def read_file():
            """The new whole lines of the log as an SSE block, or None."""
            nonlocal position
            size = finetune_job.log_file.stat().st_size
            if size < position:
                position = 0  # the log was replaced
            text, new_position = _read_complete_lines(finetune_job.log_file, position)
            if new_position == position:
                return None
            position = new_position
            return sse_data_block(list(iter_visible_lines(text)), position)

        def read_bpeek():
            """The new whole lines of the job's output as an SSE block, or None."""
            nonlocal position, use_bpeek
            output = read_bpeek_output(lsf_job_id)
            if output is None:
                use_bpeek = False
                return None
            whole = output[: output.rfind(b"\n") + 1]
            if len(whole) <= position:
                return None
            text = whole[position:].decode("utf-8", errors="replace")
            position = len(whole)
            return sse_data_block(list(iter_visible_lines(text)), position)

        while True:
            try:
                now = time.perf_counter()
                block = None
                if exists_now(finetune_job.log_file):
                    block = read_file()
                elif use_bpeek and now - last_bpeek_poll >= bpeek_poll_interval_s:
                    last_bpeek_poll = now
                    block = read_bpeek()
                if block:
                    yield block

                if finetune_job.status.value not in _LIVE_STATUSES:
                    break
                if now - last_heartbeat >= heartbeat_interval_s:
                    yield ": ping\n\n"
                    last_heartbeat = now
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error streaming logs: {e}")
                break

        # The loop above exits as soon as the job is finished, which can be
        # before the last lines of the log have been read. Without this final
        # drain the closing epochs -- and the "Training Complete!" summary --
        # are never streamed. The log is final now, so a last line without
        # its newline goes out too.
        finished = finetune_job.status.value not in _LIVE_STATUSES
        try:
            if exists_now(finetune_job.log_file):
                block = read_file()
                if block:
                    yield block
                if finished:
                    with open(finetune_job.log_file, "rb") as f:
                        f.seek(position)
                        tail = f.read()
                    if tail:
                        position += len(tail)
                        block = sse_data_block(
                            list(iter_visible_lines(tail.decode("utf-8", errors="replace"))), position
                        )
                        if block:
                            yield block
        except Exception as e:
            logger.error(f"Error draining final log content: {e}")

        if finished:
            yield sse_done(finetune_job.status.value)
        # Otherwise the loop broke on an error while the job still runs: end
        # without "done", so the browser reconnects and resumes from its id.

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@finetune_bp.route("/api/finetune/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    try:
        success = get_session().finetune_job_manager.cancel_job(job_id)
        if success:
            return jsonify({"success": True, "message": f"Job {job_id} cancelled"})
        return jsonify({"success": False, "error": "Failed to cancel job"}), 400
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/stop-early", methods=["POST"])
def stop_training_early(job_id):
    try:
        jobs = getattr(get_session().finetune_job_manager, "jobs", {}) or {}
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"success": False, "error": f"Job {job_id} not found"}), 404

        output_dir = Path(job.output_dir)
        if not output_dir.exists():
            return jsonify({"success": False, "error": f"Job output dir missing: {output_dir}"}), 400

        signal_path = output_dir / "stop_signal.json"
        with open(signal_path, "w") as f:
            json.dump(
                {
                    "requested_at": datetime.now().isoformat(),
                    "reason": "user_requested_stop_early",
                },
                f,
                indent=2,
            )

        return jsonify(
            {
                "success": True,
                "message": (
                    "Stop requested. Training will exit after the current epoch; "
                    "the inference server will then start so you can restart with "
                    "updated parameters."
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error requesting stop-early: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def _restart_training_settings(job, updated_params, corrections_dir):
    """The target and loss settings a restart sends, adjusted as submit adjusts them.

    The form holds what the user picked, not what submit replaced it with
    (training_settings), and the trainer only checks what it is sent. So a
    restart from the unchanged form undid submit's choice: a distance model
    was sent the form's margin loss, and its next iteration failed at setup,
    every time, and mse went back to training on scribbles. The same rules
    are applied here, to the form's settings over the job's own, with the
    session's sparsity read again after the sync: strokes painted since
    submit can make it sparse.

    All five settings are sent, so the trainer's settings are the adjusted
    ones whichever of them the form left out. A job that does not know its
    session (one from before jobs recorded it) keeps its mask_unannotated.
    """
    current = {**job.params, **updated_params}
    settings = training_settings(
        output_type=current.get("output_type"),
        loss_type=current.get("loss_type"),
        label_smoothing=current.get("label_smoothing"),
        distillation_lambda=current.get("distillation_lambda"),
        sparse=bool(corrections_dir) and detect_sparse_annotations(corrections_dir),
    )
    sent = {key: value for key, value in settings._asdict().items() if key != "note" and value is not None}
    if not corrections_dir:
        sent.pop("mask_unannotated")
    return sent


@finetune_bp.route("/api/finetune/job/<job_id>/restart", methods=["POST"])
def restart_finetuning_job(job_id):
    data = request.get_json() or {}
    body, refused = parse(FinetuneRestart, data)
    if refused:
        return refused
    try:
        restart_t0 = time.perf_counter()

        from cellmap_flow.finetune.session.manifest import read_manifest

        # Refused before anything is written or pulled: a restart the job
        # cannot take used to rewrite the session's manifest with the form's
        # settings, which later submits then inherited, and sync, and only
        # then fail (with a 500).
        manager = get_session().finetune_job_manager
        job_record = (getattr(manager, "jobs", {}) or {}).get(job_id)
        if job_record is None:
            return jsonify({"success": False, "error": f"Job {job_id} not found"}), 404
        if not can_restart(job_record):
            return jsonify({"success": False, "error": (
                f"Job {job_id} is in state {job_record.status.value} - can only restart a "
                f"job that is waiting for a restart (its training iteration has "
                f"finished or diverged)"
            )}), 409
        corrections_dir = str(job_record.corrections_path or "")

        existing_manifest = (
            read_manifest(corrections_dir) if corrections_dir else None
        )
        if existing_manifest is None and corrections_dir:
            # Same backfill as submit: the trainer rebuilds its dataset from
            # the manifest on every restart.
            existing_manifest = _backfill_manifest(corrections_dir)

        if existing_manifest is not None:
            _refresh_virtual_manifest_for_training(
                corrections_dir, existing_manifest, body.overrides(), "restart"
            )

        # Every restart asks MinIO whether anything changed. That question is
        # cheap and is the only way to answer it -- the browser writes its
        # strokes straight to MinIO, so the dashboard has no way of knowing
        # locally whether you drew anything since the last run. Asking *is*
        # the check: force=False diffs chunk keys and downloads only what
        # differs, so a parameters-only restart pulls nothing and the log says
        # so. What it must not do is skip the question, because the trainer
        # rebuilds its dataloader from the volume zarr on disk each iteration
        # and the background sync only runs every 30s.
        #
        # This used to be skipped whenever a manifest was present, because the
        # sync also materialized per-chunk raw extracts the virtual dataset
        # never reads, which on a big session took minutes. The sync is just
        # the chunk diff now.
        pulled = _pull_annotations(f"Restart pre-sync for job {job_id}")

        updated_params = build_restart_params(data)
        updated_params.update(_restart_training_settings(job_record, updated_params, corrections_dir))
        job = manager.restart_finetuning_job(job_id=job_id, updated_params=updated_params)
        total_elapsed = time.perf_counter() - restart_t0
        logger.info(f"Restart request processed for job {job_id}: total={total_elapsed:.2f}s")
        if pulled:
            message = (
                f"Restart request sent. Picked up new annotations from "
                f"{pulled} volume(s); training will restart on the same GPU."
            )
        else:
            message = (
                "Restart request sent. No new annotations to pull; training "
                "will restart on the same GPU."
            )
        return jsonify(
            {
                "success": True,
                "job_id": job.job_id,
                "annotations_synced": pulled,
                "message": message,
            }
        )
    except Exception as e:
        logger.error(f"Error restarting job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
