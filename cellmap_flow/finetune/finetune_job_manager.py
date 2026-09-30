"""
Job manager for orchestrating finetuning jobs on LSF cluster.

This module provides:
- FinetuneJob: Track metadata and status of a single finetuning job
- FinetuneJobManager: Orchestrate job lifecycle from submission to completion
"""

import json
import logging
import os
import threading
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Any

from cellmap_flow.finetune import markers
from cellmap_flow.finetune.job_manager import state, submit
from cellmap_flow.finetune.job_manager.listener import Listeners
from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES, FinetuneJob, JobStatus
from cellmap_flow.finetune.job_manager.tailer import LogTailer, finished_iterations, trainer_outputs_from_log
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import JobStatus as LSFJobStatus
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.utils.restart_token import (
    TOKEN_HEADER,
    read_restart_token,
    write_restart_token,
)

logger = logging.getLogger(__name__)


# The trainer's markers; see finetune/markers.py.
_STATUS_MARKER_RE = markers.STATUS_MARKER_RE


def finetune_export_kwargs(output_dir, params=None) -> dict:
    """Which artifact a finished run produced, as FinetuneModelConfig kwargs.

    A LoRA run exports lora_adapter/; a full finetune (--lora-r 0) exports
    full_finetune/model_state_dict.pt. Decided by what is on disk first --
    the job's own record of lora_r is the fallback for a run that has not
    written its export yet -- so the dashboard never points a viewer at an
    adapter directory that a rank-0 run never made.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    weights = output_dir / "full_finetune" / "model_state_dict.pt"
    adapter = output_dir / "lora_adapter"
    if weights.exists():
        return {"weights_path": str(weights)}
    if adapter.exists():
        return {"lora_adapter_path": str(adapter)}
    if params and int(params.get("lora_r", 8) or 0) <= 0:
        return {"weights_path": str(weights)}
    return {"lora_adapter_path": str(adapter)}


class FinetuneJobManager:
    """
    Orchestrate finetuning jobs from submission to completion.

    Manages the full lifecycle:
    1. Validation and job submission to LSF
    2. Background monitoring of training progress
    3. Post-training model registration
    4. Job cancellation and cleanup
    """

    def __init__(self):
        """Initialize the job manager."""
        self.jobs: Dict[str, FinetuneJob] = {}
        self.logger = logging.getLogger(__name__)
        # Told when a job's server comes up and when an iteration finishes.
        # None by default: what a job's model looks like in the dashboard is
        # the dashboard's (dashboard.finetune_layers).
        self._listeners = Listeners()
        # The models whose made-up settings have been warned about (submit.model_settings).
        self._made_up_warned = set()

    def add_listener(self, listener) -> None:
        """Tell ``listener`` about job events; see listener.FinetuneJobListener.
        Adding one already added does nothing."""
        self._listeners.add(listener)

    def remove_listener(self, listener) -> None:
        """Stop telling ``listener``."""
        self._listeners.remove(listener)

    def _build_submission_metadata(
        self,
        *,
        model_config,
        model_type: str,
        checkpoint_path: Optional[Path],
        corrections_path: Path,
        num_corrections: int,
        output_dir: Path,
        lora_r: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        loss_type: str,
        label_smoothing: float,
        distillation_lambda: Optional[float],
        distillation_scope: str,
        margin: float,
        balance_classes: bool,
        augment: bool,
        channels: List[str],
        input_voxel_size: List[int],
        output_voxel_size: List[int],
        output_type: str,
        queue: str,
        charge_group: str,
        command: str,
    ) -> dict:
        """Build metadata persisted for a submitted finetuning job."""
        return {
            "job_id": str(uuid.uuid4()),
            "model_name": model_config.name,
            "model_type": model_type,
            "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "model_script": str(model_config.script_path) if hasattr(model_config, "script_path") else None,
            "repo": model_config.repo if model_type == "huggingface" else None,
            "revision": getattr(model_config, "revision", None) if model_type == "huggingface" else None,
            "model_entry": model_config.to_dict() if model_type in submit.MODEL_ENTRY_TYPES else None,
            "corrections_path": str(corrections_path),
            "num_corrections": num_corrections,
            "output_dir": str(output_dir),
            "params": {
                "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "lora_r": lora_r,
                "lora_alpha": lora_r * 2,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss_type": loss_type,
                "label_smoothing": label_smoothing,
                "distillation_lambda": distillation_lambda,
                "distillation_scope": distillation_scope,
                "margin": margin,
                "balance_classes": balance_classes,
                "augment": augment,
                "channels": channels,
                "input_voxel_size": input_voxel_size,
                "output_voxel_size": output_voxel_size,
                "output_type": output_type,
                "queue": queue,
                "charge_group": charge_group,
            },
            "queue": queue,
            "charge_group": charge_group,
            "created_at": datetime.now().isoformat(),
            "command": command,
        }

    def submit_finetuning_job(
        self,
        model_config,
        corrections_path: Path,
        lora_r: int = 8,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        output_base: Optional[Path] = None,
        queue: Optional[str] = None,
        charge_group: Optional[str] = None,
        checkpoint_path_override: Optional[Path] = None,
        auto_serve: bool = True,
        mask_unannotated: bool = False,
        loss_type: str = "combined",
        label_smoothing: float = 0.0,
        # None leaves it to the trainer: 1.0 with good regions, else 0.
        distillation_lambda: Optional[float] = None,
        distillation_scope: str = "unlabeled",
        margin: float = 0.3,
        balance_classes: bool = False,
        augment: bool = False,
        output_type: str = "binary",
        select_channel: Optional[int] = None,
        offsets: Optional[str] = None,
    ) -> FinetuneJob:
        """
        Submit finetuning job to LSF cluster.

        Args:
            model_config: Model configuration object (FlyModelConfig, etc.)
            corrections_path: Path to corrections.zarr directory
            lora_r: LoRA rank (default: 8)
            num_epochs: Number of training epochs (default: 10)
            batch_size: Training batch size (default: 8)
            learning_rate: Learning rate (default: 1e-4)
            output_base: Base directory for outputs (default: output/finetuning)
            queue: LSF queue name (default: the site's, gpu_h100 at Janelia)
            charge_group: LSF charge group (default: the site's, cellmap at Janelia)
            checkpoint_path_override: Optional path to override checkpoint detection (default: None)
            auto_serve: Automatically start inference server after training (default: True)

        Returns:
            FinetuneJob object tracking the submitted job

        Raises:
            ValueError: If validation fails
            RuntimeError: If job submission fails
        """
        # === Validation ===

        # 1. Check model config
        if not model_config:
            raise ValueError("Model config is required")
        site = current_site()
        queue = site.default_queue if queue is None else queue
        charge_group = site.default_charge_group if charge_group is None else charge_group

        # Get model type from the config class's cli_name (e.g., "fly",
        # "dacapo", "huggingface"); refuses types the trainer cannot train.
        model_type = submit.resolve_model_type(model_config)

        # 2. Get checkpoint path if available (optional)
        # For script models: we'll pass the script path instead
        # For fly/dacapo models: we need the checkpoint path
        checkpoint_path = submit.find_checkpoint(model_config, checkpoint_path_override)

        # 3. Check there is something to train on.
        num_corrections = submit.count_corrections(corrections_path)

        # === Setup output directory ===

        if output_base is None:
            output_base = Path("output/finetuning")
        else:
            output_base = Path(output_base)

        # Create timestamped run directory under output_base/runs/ (output_base
        # already encodes the "finetuning" segment — see default above).
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = model_config.name.replace("/", "_").replace(" ", "_")
        run_dir_name = f"{model_basename}_{timestamp}"
        output_dir = output_base / "runs" / run_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Before submitting, so the job's server finds it and requires it.
        write_restart_token(output_dir)

        log_file = output_dir / "training_log.txt"
        # The session's models/, next to runs/, where every iteration's
        # serving YAML goes.
        models_dir = output_base / "models"

        self.logger.info(f"Output directory: {output_dir}")

        # === Build training command ===

        channels, input_voxel_size, output_voxel_size = submit.model_settings(
            model_config, self._made_up_warned
        )

        # Extract data path for inference server if auto-serve is enabled
        serve_data_path = None
        if auto_serve:
            try:
                serve_data_path = submit.extract_data_path_from_corrections(corrections_path)
                self.logger.info(f"Extracted dataset path for inference: {serve_data_path}")
            except Exception as e:
                self.logger.warning(f"Could not extract dataset path from corrections: {e}")
                self.logger.warning("Auto-serve will be disabled")
                auto_serve = False

        cli_command = submit.build_command(
            model_config=model_config,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            corrections_path=corrections_path,
            output_dir=output_dir,
            log_file=log_file,
            channels=channels,
            input_voxel_size=input_voxel_size,
            output_voxel_size=output_voxel_size,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            auto_serve=auto_serve,
            serve_data_path=serve_data_path,
            mask_unannotated=mask_unannotated,
            balance_classes=balance_classes,
            augment=augment,
            output_type=output_type,
            select_channel=select_channel,
            offsets=offsets,
            models_dir=models_dir,
            queue=queue,
            charge_group=charge_group,
        )

        self.logger.info(f"Training command: {cli_command}")

        # === Save job metadata ===

        metadata = self._build_submission_metadata(
            model_config=model_config,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            corrections_path=corrections_path,
            num_corrections=num_corrections,
            output_dir=output_dir,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            balance_classes=balance_classes,
            augment=augment,
            channels=channels,
            input_voxel_size=input_voxel_size,
            output_voxel_size=output_voxel_size,
            output_type=output_type,
            queue=queue,
            charge_group=charge_group,
            command=cli_command,
        )

        metadata["models_dir"] = str(models_dir)

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_file}")

        # === Submit job (LSF or local) ===

        job_name = f"finetune_{model_basename}_{timestamp}"

        lsf_job = submit.launch(job_name, cli_command, queue=queue, charge_group=charge_group)

        # === Create FinetuneJob tracking object ===

        job_id = metadata["job_id"]

        finetune_job = FinetuneJob(
            job_id=job_id,
            lsf_job=lsf_job,
            model_name=model_config.name,
            output_dir=output_dir,
            params=metadata["params"],
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            log_file=log_file,
            total_epochs=num_epochs,
            corrections_path=corrections_path,
        )

        self.jobs[job_id] = finetune_job
        # What a dashboard started later needs to find this job again: the
        # scheduler's id for it, and where it stands (see rehydrate_session).
        self._update_metadata(
            finetune_job, lsf_job_id=finetune_job.to_dict()["lsf_job_id"],
            status=finetune_job.status.value,
        )

        self._start_monitor(finetune_job)

        return finetune_job

    def _start_monitor(self, finetune_job: FinetuneJob):
        monitor_thread = threading.Thread(
            target=self.monitor_job,
            args=(finetune_job,),
            daemon=True
        )
        monitor_thread.start()
        self.logger.info(f"Started monitoring thread for job {finetune_job.job_id}")

    def _update_metadata(self, finetune_job: FinetuneJob, **fields):
        """Merge ``fields`` into the job's metadata.json, replacing it atomically."""
        path = Path(finetune_job.output_dir) / "metadata.json"
        try:
            metadata = json.loads(path.read_text()) if path.exists() else {}
            metadata.update(fields)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(metadata, indent=2))
            os.replace(tmp, path)
        except Exception as e:
            self.logger.warning(f"Could not update {path}: {e}")

    def rehydrate_session(self, session_path) -> int:
        """Pick up the jobs of a session that are still alive on the cluster.

        Jobs live only in this process's memory, so after a dashboard restart
        a running job disappeared: it could not be seen or cancelled, and
        with auto-serve it waited for a restart until walltime. Each job's
        metadata.json now records its LSF job id and status; a run whose
        recorded status is not final and that bjobs still reports as pending
        or running is monitored again, which also tells the listeners again
        once the log shows its server. Local runs (a PID, not an LSF
        job) are not reattached. Returns how many jobs were picked up.

        All the session's candidates are asked about in one bjobs call, and
        whatever bjobs says has ended -- including a job it no longer knows
        at all -- is recorded as final, so it is not asked about again. This
        runs on every load of the finetune tab.
        """
        candidates = []
        for metadata_file in sorted(Path(session_path).glob("runs/*/metadata.json")):
            try:
                metadata = json.loads(metadata_file.read_text())
            except (OSError, ValueError):
                continue
            job_id = metadata.get("job_id")
            lsf_job_id = metadata.get("lsf_job_id")
            if (
                not job_id
                or job_id in self.jobs
                or not lsf_job_id
                or str(lsf_job_id).startswith("PID:")
                or metadata.get("status") in {s.value for s in TERMINAL_STATUSES}
            ):
                continue
            candidates.append((metadata_file, metadata, job_id, str(lsf_job_id)))
        if not candidates:
            return 0

        reported = jobs_lsf.statuses([lsf_job_id for *_, lsf_job_id in candidates])
        count = 0
        for metadata_file, metadata, job_id, lsf_job_id in candidates:
            if lsf_job_id not in reported:
                continue  # bjobs cannot say; try again next time
            observed = reported[lsf_job_id]
            record = SimpleNamespace(output_dir=metadata_file.parent)
            if observed is None:
                # LSF has forgotten it: it ended long enough ago to be purged,
                # while no dashboard was watching. Asked about again, it never
                # answers. The trainer never prints "done" -- after an
                # iteration it waits for restarts until it is stopped or runs
                # out of walltime -- so a run whose log shows a finished
                # iteration delivered a model, and that is what it is recorded
                # as having done.
                iterations, last = finished_iterations(metadata_file.parent / "training_log.txt")
                if iterations:
                    status = JobStatus.COMPLETED
                    detail = (
                        f"LSF no longer knows job {lsf_job_id}; it had finished "
                        f"{iterations} iteration(s), the last {last}"
                    )
                else:
                    status = JobStatus.FAILED
                    detail = (
                        f"LSF no longer knows job {lsf_job_id}; it ended while no "
                        "dashboard was watching, and how is not known"
                    )
                self._update_metadata(record, status=status.value, status_detail=detail)
                continue
            if observed == LSFJobStatus.COMPLETED:
                # Finished while no dashboard was watching; say so, so it is
                # not asked about again. complete_job does not run for it.
                self._update_metadata(record, status=JobStatus.COMPLETED.value)
                continue
            if observed == LSFJobStatus.FAILED:
                self._update_metadata(record, status=JobStatus.FAILED.value)
                continue
            if observed not in (LSFJobStatus.RUNNING, LSFJobStatus.PENDING):
                continue
            lsf_job = LSFJob(job_id=lsf_job_id, model_name=metadata.get("model_name"))
            params = metadata.get("params") or {}
            output_dir = metadata_file.parent
            try:
                created_at = datetime.fromisoformat(metadata["created_at"])
            except (KeyError, TypeError, ValueError):
                created_at = datetime.now()
            job = FinetuneJob(
                job_id=job_id,
                lsf_job=lsf_job,
                model_name=metadata.get("model_name") or "",
                output_dir=output_dir,
                params=params,
                status=JobStatus.RUNNING if observed == LSFJobStatus.RUNNING else JobStatus.PENDING,
                created_at=created_at,
                log_file=output_dir / "training_log.txt",
                total_epochs=int(params.get("num_epochs") or 10),
                corrections_path=(
                    Path(metadata["corrections_path"]) if metadata.get("corrections_path") else None
                ),
            )
            self.jobs[job_id] = job
            self.logger.info(f"Reattached to job {job_id} (LSF {lsf_job_id}) from {output_dir}")
            self._start_monitor(job)
            count += 1
        return count

    def monitor_job(self, finetune_job: FinetuneJob):
        """
        Background thread for job monitoring.

        Polls LSF status and tails log file to track training progress.
        Triggers completion when job finishes.

        Args:
            finetune_job: The FinetuneJob to monitor
        """
        job_id = finetune_job.job_id
        self.logger.info(f"Monitoring job {job_id}...")

        log = LogTailer(finetune_job.log_file)
        check_interval = 3  # seconds
        persisted_status = finetune_job.status

        try:
            while True:
                # A cancel (or anything else that ended the job) is final. The
                # poll below used to overwrite it: after cancel_job set
                # CANCELLED, LSF reported the kill as EXIT (a local job as
                # return code -15) and the job ended up FAILED.
                if finetune_job.status in TERMINAL_STATUSES:
                    break

                # === Check LSF job status ===

                if finetune_job.lsf_job:
                    if state.on_scheduler_status(finetune_job, finetune_job.lsf_job.get_status()):
                        break

                # === Tail log file for progress updates ===

                if finetune_job.log_file.exists():
                    try:
                        # Whole lines only; see LogTailer.
                        new_content = log.read()
                        if new_content:
                            # Parse for epoch and loss information
                            self._parse_training_progress(finetune_job, new_content)
                            # Parse for inference server ready marker
                            self._parse_inference_server_ready(finetune_job, new_content)

                        # Always check for restart/iteration markers (reads full log).
                        # This must run every cycle, not just when there's new content,
                        # because the marker may have been at the end of the previous
                        # chunk and we need to detect it even if no new output follows.
                        self._parse_training_restart(finetune_job, new_content)
                    except Exception as e:
                        self.logger.debug(f"Error reading log file: {e}")

                if finetune_job.status != persisted_status:
                    persisted_status = finetune_job.status
                    self._update_metadata(
                        finetune_job, status=persisted_status.value,
                        inference_server_url=finetune_job.inference_server_url,
                    )

                # Sleep before next check
                time.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Error monitoring job {job_id}: {e}")
            if finetune_job.status not in TERMINAL_STATUSES:
                finetune_job.status = JobStatus.FAILED

        finally:
            # === Post-completion actions ===

            if finetune_job.status == JobStatus.COMPLETED:
                try:
                    self.complete_job(finetune_job)
                except Exception as e:
                    self.logger.error(f"Error in post-completion for job {job_id}: {e}")
                    finetune_job.status = JobStatus.FAILED
            else:
                # A job that failed after training -- its server would not
                # start, say -- still produced a model; record what it was.
                self._read_trainer_outputs(finetune_job)

            self._update_metadata(
                finetune_job,
                status=finetune_job.status.value,
                finetuned_model_name=finetune_job.finetuned_model_name,
                model_yaml_path=str(finetune_job.model_yaml_path) if finetune_job.model_yaml_path else None,
            )
            self.logger.info(f"Stopped monitoring job {job_id}. Final status: {finetune_job.status.value}")

    def _parse_training_progress(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log content for training progress (epoch, loss).

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        # One loss per epoch, from the per-epoch summary line ("Epoch X/Y -
        # Loss: Z"), so the plot has one point per epoch and the loss is
        # always the one belonging to the epoch shown beside it. Per-batch
        # lines are deliberately not read: pairing them with an epoch is
        # fiddly, and what made the display look stuck was tee's buffering,
        # not the reporting interval.
        #
        # "Starting epoch N of M" is read too, so the epoch counter advances
        # as soon as an epoch begins rather than when it ends.
        for cur, total in markers.EPOCH_START_RE.findall(log_content):
            finetune_job.current_epoch = int(cur)
            finetune_job.total_epochs = int(total)

        summary_matches = markers.EPOCH_SUMMARY_RE.findall(log_content)
        if summary_matches:
            cur, total, loss = summary_matches[-1]
            finetune_job.current_epoch = max(
                finetune_job.current_epoch, int(cur)
            )
            finetune_job.total_epochs = int(total)
            try:
                finetune_job.latest_loss = float(loss)
            except ValueError:
                pass

    def _parse_inference_server_ready(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log for CELLMAP_FLOW_SERVER_IP marker and tell the listeners
        the job's inference server is up.

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        if finetune_job.inference_server_ready:
            return

        # Look for the standard server IP marker (same one start_hosts() uses)
        matches = markers.SERVER_URL_RE.findall(log_content)
        if not matches:
            return

        server_url = matches[-1]
        finetune_job.inference_server_url = server_url
        finetune_job.inference_server_ready = True
        self.logger.info(f"Finetuned inference server detected at {server_url}")

        try:
            # Read the FULL log file to find TRAINING_ITERATION_COMPLETE marker.
            # This marker is printed BEFORE the server starts, so it's typically
            # in an earlier log chunk than the server IP marker.
            full_log = finetune_job.log_file.read_text()
            iter_matches = markers.ITERATION_COMPLETE_RE.findall(full_log)
            if iter_matches:
                model_name = iter_matches[-1]
            else:
                model_name = f"{finetune_job.model_name}_finetuned"
            self._read_trainer_outputs(finetune_job, set_name=False)
        except Exception as e:
            self.logger.error(f"Could not read the served model's name from {finetune_job.log_file}: {e}",
                              exc_info=True)
            return

        self._listeners.notify("on_server_ready", finetune_job, server_url, model_name)
        # Whatever the listeners managed (see FinetuneJobListener), and so
        # that the iteration it serves is not announced again.
        finetune_job.finetuned_model_name = model_name

    def _parse_training_restart(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log for RESTARTING_TRAINING and TRAINING_ITERATION_COMPLETE markers
        to handle iterative training restarts.

        On RESTARTING_TRAINING: reset training progress counters.
        On TRAINING_ITERATION_COMPLETE: tell the listeners the iteration's model.

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        # Status markers, in the order they were printed: a restart that
        # follows a divergence in the same chunk leaves the job running, and
        # the reverse leaves it waiting.
        for marker in _STATUS_MARKER_RE.findall(log_content):
            state.on_status_marker(finetune_job, marker)

        # Check for iteration complete marker - tell the listeners.
        # Read full log in case the marker was in a previous chunk.
        try:
            full_log = finetune_job.log_file.read_text()
        except Exception:
            full_log = log_content
        iter_matches = markers.ITERATION_COMPLETE_RE.findall(full_log)
        # Only process new iteration-complete markers (ignore ones already handled).
        # After a restart, _processed_iteration_count stays at the old count so
        # previously-seen markers don't re-trigger inference_server_ready or
        # the listeners.
        if len(iter_matches) > finetune_job._processed_iteration_count:
            finetune_job._processed_iteration_count = len(iter_matches)

            # For in-process restarts, the inference server usually stays on the same
            # URL and does not emit a fresh CELLMAP_FLOW_SERVER_IP marker. Mark the
            # server as ready once we see a completed training iteration if URL exists.
            if finetune_job.inference_server_url:
                finetune_job.inference_server_ready = True

            new_model_name = iter_matches[-1]
            self._read_trainer_outputs(finetune_job, set_name=False)
            if new_model_name != finetune_job.finetuned_model_name:
                self.logger.info(f"New training iteration complete: {new_model_name}")
                self._listeners.notify("on_iteration_complete", finetune_job, new_model_name)
                # Whatever the listeners managed -- without a server no layer
                # is added -- show the new name, and don't retry every poll.
                finetune_job.finetuned_model_name = new_model_name

    def _read_trainer_outputs(self, finetune_job: FinetuneJob, set_name: bool = True):
        """Take the latest iteration's model name and serving YAML from the log.

        The trainer prints "FINETUNED_MODEL_YAML: <path>" and then
        "TRAINING_ITERATION_COMPLETE: <name>" for every iteration it
        finishes. ``set_name=False`` leaves finetuned_model_name alone, for the
        monitor, whose listeners use the old name (see FinetuneJobListener).
        """
        try:
            log_text = finetune_job.log_file.read_text()
        except OSError:
            return
        name, yaml_path = trainer_outputs_from_log(log_text)
        if set_name and name:
            finetune_job.finetuned_model_name = name
        if yaml_path:
            finetune_job.model_yaml_path = Path(yaml_path)

    def complete_job(self, finetune_job: FinetuneJob):
        """
        Post-training actions after job completes successfully.

        1. Verify adapter files exist
        2. Take the model name and serving YAML the trainer reported
        3. Update job status and metadata

        Args:
            finetune_job: The completed job

        Raises:
            RuntimeError: If adapter files missing or registration fails
        """
        job_id = finetune_job.job_id
        self.logger.info(f"Running post-completion for job {job_id}...")

        # === Verify the training export exists ===

        export = finetune_export_kwargs(finetune_job.output_dir, finetune_job.params)
        if "weights_path" in export:
            # Full finetune (--lora-r 0): a single state dict, no adapter dir.
            weights_file = Path(export["weights_path"])
            if not weights_file.exists():
                raise RuntimeError(
                    f"Training completed but full-finetune weights not found: {weights_file}"
                )
            self.logger.info(f"Verified full-finetune weights exist: {weights_file}")
        else:
            adapter_path = Path(export["lora_adapter_path"])

            # Check for adapter model (supports both .bin and .safetensors formats)
            adapter_model_bin = adapter_path / "adapter_model.bin"
            adapter_model_safetensors = adapter_path / "adapter_model.safetensors"

            if not (adapter_model_bin.exists() or adapter_model_safetensors.exists()):
                raise RuntimeError(
                    f"Training completed but adapter model not found. "
                    f"Checked: {adapter_model_bin} and {adapter_model_safetensors}"
                )

            adapter_config_file = adapter_path / "adapter_config.json"
            if not adapter_config_file.exists():
                raise RuntimeError(
                    f"Training completed but adapter config not found: {adapter_config_file}"
                )

            self.logger.info(f"Verified LoRA adapter files exist in {adapter_path}")

        # === The name and YAML the trainer gave the result ===
        #
        # This used to make up its own: the job's creation time instead of
        # the iteration's, and a sanitized model name instead of the
        # trainer's. So the YAML it looked for never existed, it always
        # generated a second one (with the dashboard's current norms rather
        # than the training ones), and metadata.json named a model that
        # neither the viewer layer nor the registered config did. The trainer
        # prints both, and is the only thing that knows them.
        self._read_trainer_outputs(finetune_job)
        finetuned_model_name = finetune_job.finetuned_model_name
        yaml_path = finetune_job.model_yaml_path
        if yaml_path is None:
            self.logger.warning(
                f"Job {job_id}: the trainer reported no serving YAML (see its "
                f"log for why); the weights are in {finetune_job.output_dir}."
            )
        else:
            self.logger.info(f"Serving YAML for {finetuned_model_name}: {yaml_path}")

        # === Update metadata file with completion info ===

        metadata_file = finetune_job.output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["completed_at"] = datetime.now().isoformat()
            metadata["status"] = "COMPLETED"
            metadata["finetuned_model_name"] = finetuned_model_name
            metadata["model_yaml_path"] = str(yaml_path) if yaml_path else None
            metadata["final_epoch"] = finetune_job.current_epoch
            metadata["final_loss"] = finetune_job.latest_loss

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Updated metadata file: {metadata_file}")

        self.logger.info(f"Job {job_id} completed successfully!")

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        if job_id not in self.jobs:
            self.logger.error(f"Job {job_id} not found")
            return False

        finetune_job = self.jobs[job_id]

        if finetune_job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            self.logger.warning(f"Job {job_id} already finished with status {finetune_job.status}")
            return False

        self.logger.info(f"Cancelling job {job_id}...")

        if finetune_job.lsf_job:
            try:
                finetune_job.cancel_requested = True
                finetune_job.lsf_job.kill()
                finetune_job.status = JobStatus.CANCELLED
                self.logger.info(f"Successfully cancelled job {job_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error cancelling job {job_id}: {e}")
                return False
        else:
            self.logger.error(f"No LSF job associated with {job_id}")
            return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific job.

        Args:
            job_id: Job ID to query

        Returns:
            Dictionary with job status details, or None if not found
        """
        if job_id not in self.jobs:
            return None

        finetune_job = self.jobs[job_id]
        result = finetune_job.to_dict()
        result["loss"] = result.pop("latest_loss", None)
        result["progress_percent"] = (
            finetune_job.current_epoch / finetune_job.total_epochs * 100
        ) if finetune_job.total_epochs > 0 else 0
        return result

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of all jobs with their status.

        Returns:
            List of job status dictionaries
        """
        # A snapshot: monitor threads and rehydration add jobs concurrently.
        return [
            status for status in (self.get_job_status(job_id) for job_id in list(self.jobs))
            if status is not None
        ]

    def get_job_logs(self, job_id: str) -> Optional[str]:
        """
        Get full log content for a job.

        Args:
            job_id: Job ID

        Returns:
            Log file content as string, or None if not found
        """
        if job_id not in self.jobs:
            return None

        finetune_job = self.jobs[job_id]

        if not finetune_job.log_file.exists():
            return "Log file not yet created..."

        try:
            with open(finetune_job.log_file, "r") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
            return f"Error reading log file: {e}"

    def get_job(self, job_id: str) -> Optional[FinetuneJob]:
        """
        Get a FinetuneJob object by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            FinetuneJob object, or None if not found
        """
        return self.jobs.get(job_id)

    def restart_finetuning_job(
        self,
        job_id: str,
        updated_params: Optional[Dict[str, Any]] = None
    ) -> FinetuneJob:
        """
        Restart training on the same GPU via control endpoint.

        Primary path sends an HTTP restart request to the running
        inference server in the same process as the training loop.
        Falls back to file signal if control endpoint is unavailable.

        Args:
            job_id: ID of job to restart
            updated_params: Dict of updated training parameters

        Returns:
            Same FinetuneJob object (updated in-place)

        Raises:
            ValueError: If job not found or not in a restartable state
        """
        restart_t0 = time.perf_counter()

        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]

        if not state.can_restart(job):
            raise ValueError(
                f"Job {job_id} is in state {job.status.value} - can only restart a "
                f"job that is waiting for a restart (its training iteration has "
                f"finished or diverged)"
            )

        signal_data = {
            "restart": True,
            "timestamp": datetime.now().isoformat(),
            "params": updated_params or {}
        }

        # 1. Send restart request to running inference server (primary path)
        signal_write_mode = "http_control"
        write_t0 = time.perf_counter()
        http_error = None
        if job.inference_server_url:
            try:
                control_url = job.inference_server_url.rstrip("/") + "/__control__/restart"
                restart_token = read_restart_token(job.output_dir)
                if restart_token is None:
                    raise RuntimeError(f"No restart token in {job.output_dir}")
                headers = {TOKEN_HEADER: restart_token}
                response = requests.post(control_url, json=signal_data, headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                if not data.get("success", False):
                    raise RuntimeError(data.get("error", "Unknown restart control failure"))
                self.logger.info(f"Sent restart request via HTTP control endpoint: {control_url}")
            except Exception as e:
                http_error = e
                self.logger.warning(f"HTTP restart control failed for job {job_id}: {e}")
        else:
            http_error = RuntimeError("No inference_server_url for HTTP restart control")

        # 2. Fallback to signal file if HTTP control endpoint is unavailable
        if http_error is not None:
            signal_write_mode = "file_signal_fallback"
            signal_file = job.output_dir / "restart_signal.json"
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            self.logger.info(f"Wrote fallback restart signal to {signal_file}")
        write_elapsed = time.perf_counter() - write_t0

        # 3. Reset training progress (keep inference server info)
        state.start_iteration(job)

        # 4. Update stored params
        if updated_params:
            job.params.update(updated_params)

        total_elapsed = time.perf_counter() - restart_t0
        self.logger.info(
            f"Restart signal timings for job {job_id}: "
            f"write={write_elapsed:.2f}s "
            f"mode={signal_write_mode} total={total_elapsed:.2f}s"
        )
        self.logger.info(f"Job {job_id} restart request sent, waiting for CLI to pick it up")

        return job
