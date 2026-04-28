import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from flask import Response, jsonify

from cellmap_flow.dashboard.finetune_utils import sync_all_annotations_from_minio
from cellmap_flow.dashboard.routes.finetune.common import (
    LOG_FILTER_PATTERNS,
    autodetect_output_type,
    build_restart_params,
    detect_sparse_annotations,
    find_model_config,
    get_lsf_job_id,
    resolve_finetune_session,
)
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


def list_finetuning_jobs_response():
    try:
        return jsonify({"success": True, "jobs": g.finetune_job_manager.list_jobs()})
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def get_job_status_response(job_id):
    try:
        status = g.finetune_job_manager.get_job_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify({"success": True, **status})
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def get_job_logs_response(job_id):
    try:
        logs = g.finetune_job_manager.get_job_logs(job_id)
        if logs is None:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        logger.error(f"Error getting job logs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def submit_finetuning_response(data):
    try:
        model_name = data.get("model_name")
        corrections_path_str = data.get("corrections_path")
        if not model_name:
            return jsonify({"success": False, "error": "model_name is required"}), 400
        if not corrections_path_str:
            return jsonify(
                {
                    "success": False,
                    "error": "corrections_path is required. Please specify the output path where annotation crops are saved.",
                }
            ), 400

        model_config = find_model_config(model_name)
        if not model_config:
            return jsonify({"success": False, "error": f"Model {model_name} not found"}), 404

        session_path, actual_corrections_path = resolve_finetune_session(corrections_path_str)
        if not actual_corrections_path.exists():
            return jsonify(
                {
                    "success": False,
                    "error": f"Corrections path does not exist: {actual_corrections_path}. Please create annotation crops first.",
                }
            ), 400

        # Pre-training sync: only needed by the legacy CorrectionDataset path,
        # which reads per-chunk _chunk_*.zarr extracts. The new VirtualPatchDataset
        # reads the annotation_volume.zarr directly, so when a manifest is present
        # the sync is wasted work and can hang submit for many minutes when the
        # volume contains imported YAML data.
        from cellmap_flow.finetune.virtual_dataset import read_manifest, write_manifest

        existing_manifest = read_manifest(str(actual_corrections_path))
        if existing_manifest is None:
            try:
                sync_all_annotations_from_minio(force=False)
            except Exception as e:
                logger.warning(f"Error syncing annotations before training: {e}")
        else:
            # Refresh the manifest's input_norm with the dashboard's current
            # value so the trainer applies the same normalization the user
            # currently sees at inference. Lets users tweak normalization in
            # the UI and re-Submit without rebuilding the session.
            current_norm = getattr(g, "input_norm_config", None) or {}
            if current_norm and existing_manifest.get("input_norm") != current_norm:
                logger.info(
                    "Refreshing manifest input_norm before submit "
                    "(was: %s, now: %s)",
                    list((existing_manifest.get("input_norm") or {}).keys()),
                    list(current_norm.keys()),
                )
            existing_manifest["input_norm"] = current_norm
            write_manifest(str(actual_corrections_path), existing_manifest)
            logger.info(
                "Virtual sources manifest present; skipping pre-training MinIO sync."
            )

        loss_type = data.get("loss_type", "mse")
        distillation_lambda = data.get("distillation_lambda", 0.0)
        has_sparse = detect_sparse_annotations(actual_corrections_path)
        sparse_auto_switched = False
        if has_sparse and loss_type == "mse":
            loss_type = "margin"
            distillation_lambda = 0.5
            sparse_auto_switched = True
            logger.info(
                "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"
            )

        output_type, offsets = autodetect_output_type(
            model_config,
            data.get("output_type", None),
            data.get("offsets", None),
        )

        finetune_job = g.finetune_job_manager.submit_finetuning_job(
            model_config=model_config,
            corrections_path=actual_corrections_path,
            lora_r=data.get("lora_r", 8),
            num_epochs=data.get("num_epochs", 10),
            batch_size=data.get("batch_size", 2),
            learning_rate=data.get("learning_rate", 1e-4),
            output_base=Path(session_path),
            checkpoint_path_override=(
                Path(data["checkpoint_path"]) if data.get("checkpoint_path") else None
            ),
            auto_serve=data.get("auto_serve", True),
            mask_unannotated=has_sparse,
            loss_type=loss_type,
            label_smoothing=data.get("label_smoothing", 0.1),
            distillation_lambda=distillation_lambda,
            distillation_scope=data.get("distillation_scope", "unlabeled"),
            margin=data.get("margin", 0.3),
            balance_classes=data.get("balance_classes", False),
            queue=data.get("queue", "gpu_h100"),
            output_type=output_type,
            select_channel=data.get("select_channel", None),
            offsets=offsets,
        )

        response = {
            "success": True,
            "job_id": finetune_job.job_id,
            "lsf_job_id": get_lsf_job_id(finetune_job),
            "output_dir": str(finetune_job.output_dir),
            "output_type": output_type,
            "message": "Finetuning job submitted successfully",
        }
        if sparse_auto_switched:
            response["note"] = (
                "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"
            )
        return jsonify(response)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting finetuning job: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def stream_job_logs_response(job_id):
    log_filters = [re.compile(pattern) for pattern in LOG_FILTER_PATTERNS]

    def iter_visible_lines(text):
        for line in text.splitlines():
            if line and not any(pattern.search(line) for pattern in log_filters):
                yield line

    def sse_data_block(lines):
        if not lines:
            return None
        payload = "\n".join(lines)
        return "data: " + payload.replace("\n", "\ndata: ") + "\n\n"

    def read_bpeek_content(lsf_job_id):
        try:
            result = subprocess.run(
                ["bpeek", str(lsf_job_id)],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"bpeek call failed for job {lsf_job_id}: {e}")
            return None

        output = result.stdout or ""
        stderr = (result.stderr or "").strip()
        if stderr and "Not yet started" not in stderr:
            logger.debug(f"bpeek stderr for job {lsf_job_id}: {stderr}")
        return output

    def generate():
        heartbeat_interval_s = 1.0
        last_heartbeat = time.perf_counter()

        fjm = g.finetune_job_manager
        if job_id not in fjm.jobs:
            yield f"data: Job {job_id} not found\n\n"
            return

        finetune_job = fjm.jobs[job_id]
        lsf_job_id = None
        if finetune_job.lsf_job and hasattr(finetune_job.lsf_job, "job_id"):
            lsf_job_id = finetune_job.lsf_job.job_id

        use_bpeek = lsf_job_id is not None
        last_bpeek_line_count = 0
        last_bpeek_poll = 0.0
        bpeek_poll_interval_s = 0.25

        if use_bpeek:
            initial = read_bpeek_content(lsf_job_id)
            if initial is None:
                use_bpeek = False
            else:
                last_bpeek_line_count = len(initial.splitlines())
                block = sse_data_block(list(iter_visible_lines(initial)))
                if block:
                    yield block

        if not use_bpeek and finetune_job.log_file.exists():
            try:
                with open(finetune_job.log_file, "r") as f:
                    block = sse_data_block(list(iter_visible_lines(f.read())))
                if block:
                    yield block
            except Exception as e:
                logger.error(f"Error reading log file: {e}")

        last_position = finetune_job.log_file.stat().st_size if finetune_job.log_file.exists() else 0

        while finetune_job.status.value in ["PENDING", "RUNNING"]:
            try:
                now = time.perf_counter()
                if use_bpeek and lsf_job_id and now - last_bpeek_poll >= bpeek_poll_interval_s:
                    last_bpeek_poll = now
                    content = read_bpeek_content(lsf_job_id)
                    if content is None:
                        use_bpeek = False
                    else:
                        current_lines = content.splitlines()
                        delta_lines = current_lines if len(current_lines) < last_bpeek_line_count else current_lines[last_bpeek_line_count:]
                        last_bpeek_line_count = len(current_lines)
                        if delta_lines:
                            block = sse_data_block(list(iter_visible_lines("\n".join(delta_lines))))
                            if block:
                                yield block

                if not use_bpeek and finetune_job.log_file.exists():
                    with open(finetune_job.log_file, "r") as f:
                        f.seek(last_position)
                        new_content = f.read()
                        last_position = f.tell()
                    if new_content:
                        block = sse_data_block(list(iter_visible_lines(new_content)))
                        if block:
                            yield block

                if now - last_heartbeat >= heartbeat_interval_s:
                    yield ": ping\n\n"
                    last_heartbeat = now
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error streaming logs: {e}")
                break

        yield f"data: === Training {finetune_job.status.value} ===\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def cancel_job_response(job_id):
    try:
        success = g.finetune_job_manager.cancel_job(job_id)
        if success:
            return jsonify({"success": True, "message": f"Job {job_id} cancelled"})
        return jsonify({"success": False, "error": "Failed to cancel job"}), 400
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def stop_training_early_response(job_id):
    try:
        jobs = getattr(g.finetune_job_manager, "jobs", {}) or {}
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


def get_inference_server_status_response(job_id):
    try:
        job = g.finetune_job_manager.get_job(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404
        return jsonify(
            {
                "success": True,
                "ready": job.inference_server_ready,
                "url": job.inference_server_url,
                "model_name": job.finetuned_model_name,
                "model_script_path": str(job.model_script_path) if job.model_script_path else None,
            }
        )
    except Exception as e:
        logger.error(f"Error getting inference server status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def restart_finetuning_job_response(job_id, data):
    try:
        restart_t0 = time.perf_counter()

        # Pre-sync is only needed by the legacy CorrectionDataset path. With
        # a virtual-sources manifest the trainer reads the volume zarr
        # directly, so the sync would just download chunks the trainer never
        # touches — and on big sessions can hang Restart for minutes.
        from cellmap_flow.finetune.virtual_dataset import read_manifest, write_manifest

        jobs = getattr(g.finetune_job_manager, "jobs", {}) or {}
        job_record = jobs.get(job_id)
        corrections_dir = (
            str(getattr(job_record, "corrections_path", "") or "")
            if job_record is not None
            else ""
        )

        existing_manifest = (
            read_manifest(corrections_dir) if corrections_dir else None
        )
        if existing_manifest is not None:
            # Refresh manifest input_norm so the next training cycle obeys
            # whatever the user currently has set in the dashboard. Same UX
            # as bumping LR / lora_r and clicking Restart.
            current_norm = getattr(g, "input_norm_config", None) or {}
            if current_norm and existing_manifest.get("input_norm") != current_norm:
                logger.info(
                    "Refreshing manifest input_norm before restart "
                    "(was: %s, now: %s)",
                    list((existing_manifest.get("input_norm") or {}).keys()),
                    list(current_norm.keys()),
                )
            existing_manifest["input_norm"] = current_norm
            write_manifest(corrections_dir, existing_manifest)
            logger.info(
                f"Virtual sources manifest present for job {job_id}; "
                "skipping pre-restart MinIO sync."
            )
        else:
            try:
                sync_t0 = time.perf_counter()
                synced = sync_all_annotations_from_minio(force=False)
                sync_elapsed = time.perf_counter() - sync_t0
                logger.info(
                    f"Restart pre-sync complete for job {job_id}: synced={synced}, "
                    f"elapsed={sync_elapsed:.2f}s"
                )
            except Exception as e:
                logger.warning(f"Error syncing annotations before restart: {e}")

        job = g.finetune_job_manager.restart_finetuning_job(
            job_id=job_id,
            updated_params=build_restart_params(data),
        )
        total_elapsed = time.perf_counter() - restart_t0
        logger.info(f"Restart request processed for job {job_id}: total={total_elapsed:.2f}s")
        return jsonify(
            {
                "success": True,
                "job_id": job.job_id,
                "message": "Restart request sent. Training will restart on the same GPU.",
            }
        )
    except Exception as e:
        logger.error(f"Error restarting job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
