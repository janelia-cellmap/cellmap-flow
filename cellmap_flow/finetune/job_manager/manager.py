"""FinetuneJobManager: the finetune jobs a dashboard follows.

The finetune routes call it, as the session's ``finetune_job_manager``. It
holds the jobs by job id, and their listeners, and puts the package's
modules together:

- ``submit_finetuning_job``: check the request and work out what to run
  (submit), make the run's directory and its record (persistence), launch
  the trainer, and follow the job;
- ``rehydrate_session``: follow again the jobs an earlier dashboard left
  running (persistence.rehydrate);
- ``monitor_job``: what each job's monitor thread runs (monitor);
- ``restart_finetuning_job`` (restart) and ``cancel_job``;
- ``get_job_status``, ``list_jobs``, ``get_job_logs``, ``get_job``;
- ``add_listener``, ``remove_listener`` (listener).

Stopping a job early is between the finetune route and the trainer: the
route writes the run's ``stop_signal.json``.
"""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from cellmap_flow.finetune.job_manager import monitor, persistence, restart, submit
from cellmap_flow.finetune.job_manager.listener import Listeners
from cellmap_flow.finetune.job_manager.state import FinetuneJob, JobStatus
from cellmap_flow.jobs.site import current_site
from cellmap_flow.serving.restart_token import write_restart_token

logger = logging.getLogger(__name__)


class FinetuneJobManager:
    """
    Orchestrate finetuning jobs from submission to completion.

    Manages the full lifecycle:
    1. Validation and job submission to LSF
    2. Background monitoring of training progress, told to the listeners
    3. Post-training checks, and the job's record in its metadata.json
    4. Restarts and cancellation
    """

    def __init__(self):
        """Initialize the job manager."""
        self.jobs: Dict[str, FinetuneJob] = {}
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

        logger.info(f"Output directory: {output_dir}")

        # === Build training command ===

        channels, input_voxel_size, output_voxel_size = submit.model_settings(
            model_config, self._made_up_warned
        )

        # Extract data path for inference server if auto-serve is enabled
        serve_data_path = None
        if auto_serve:
            try:
                serve_data_path = submit.extract_data_path_from_corrections(corrections_path)
                logger.info(f"Extracted dataset path for inference: {serve_data_path}")
            except Exception as e:
                logger.warning(f"Could not extract dataset path from corrections: {e}")
                logger.warning("Auto-serve will be disabled")
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

        logger.info(f"Training command: {cli_command}")

        # === Save job metadata ===

        metadata = persistence.submission_metadata(
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
        persistence.write_metadata(output_dir, metadata)

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
        persistence.update_metadata(
            output_dir, lsf_job_id=finetune_job.to_dict()["lsf_job_id"],
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
        logger.info(f"Started monitoring thread for job {finetune_job.job_id}")

    def rehydrate_session(self, session_path) -> int:
        """Pick up the jobs of a session that are still alive on the cluster.

        Jobs live only in this process's memory, so after a dashboard restart
        a running job disappeared: it could not be seen or cancelled, and
        with auto-serve it waited for a restart until walltime. Each job's
        metadata.json now records its LSF job id and status; a run whose
        recorded status is not final and that bjobs still reports as pending
        or running is monitored again, which also tells the listeners again
        once the log shows its server. persistence.rehydrate says which runs
        those are, with one bjobs call per session, and records the others
        that have ended as final. Returns how many jobs were picked up.
        """
        jobs = persistence.rehydrate(session_path, known=self.jobs)
        for job in jobs:
            self.jobs[job.job_id] = job
            logger.info(f"Reattached to job {job.job_id} (LSF {job.lsf_job.job_id}) from {job.output_dir}")
            self._start_monitor(job)
        return len(jobs)

    def monitor_job(self, finetune_job: FinetuneJob):
        """Follow ``finetune_job`` until it ends (monitor.monitor_job); its monitor thread runs this."""
        monitor.monitor_job(finetune_job, self._listeners)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False

        finetune_job = self.jobs[job_id]

        if finetune_job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            logger.warning(f"Job {job_id} already finished with status {finetune_job.status}")
            return False

        logger.info(f"Cancelling job {job_id}...")

        if finetune_job.lsf_job:
            try:
                finetune_job.cancel_requested = True
                finetune_job.lsf_job.kill()
                finetune_job.status = JobStatus.CANCELLED
                logger.info(f"Successfully cancelled job {job_id}")
                return True
            except Exception as e:
                logger.error(f"Error cancelling job {job_id}: {e}")
                return False
        else:
            logger.error(f"No LSF job associated with {job_id}")
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
            logger.error(f"Error reading log file: {e}")
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
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        restart.request_restart(job, updated_params)
        return job
