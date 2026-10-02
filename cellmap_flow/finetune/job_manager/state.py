"""A finetune job's record, its statuses, and what moves a job between them.

A job moves on:
- what the scheduler (LSF, or the local process) says of it, which the
  monitor asks every few seconds: ``on_scheduler_status``. A job the
  scheduler says has completed becomes COMPLETED only once the monitor has
  found its export (monitor.complete_job), and FAILED if it has not;
- the status markers the trainer prints in its log: ``on_status_marker``;
- the dashboard: a restart, which only a job that ``can_restart`` takes
  (``start_iteration``), and a cancel (FinetuneJobManager.cancel_job).

Once a job's status is final (``TERMINAL_STATUSES``), neither the scheduler
nor the log moves it again.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.spec import JobStatus as LSFJobStatus

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a finetuning job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    # Alive and idle: an iteration finished (and is being served) or
    # diverged, and the trainer is waiting for a restart request.
    WAITING_FOR_RESTART = "WAITING_FOR_RESTART"


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


@dataclass
class FinetuneJob:
    """Track a finetuning job with metadata, status, and training progress.

    Manages lifecycle from submission through completion, including inference
    server state.
    """
    job_id: str
    lsf_job: Optional[LSFJob]
    model_name: str
    output_dir: Path
    params: Dict[str, Any]
    status: JobStatus
    created_at: datetime
    log_file: Path
    finetuned_model_name: Optional[str] = None
    model_yaml_path: Optional[Path] = None
    current_epoch: int = 0
    total_epochs: int = 10
    latest_loss: Optional[float] = None
    inference_server_url: Optional[str] = None
    inference_server_ready: bool = False
    # The corrections directory the job trains on. The restart route reads it
    # to refresh the manifest; it was never set, so a restart silently kept
    # the old patches_per_epoch, rehearsal fraction, input norm and
    # postprocessing even though the confirm dialog showed the new ones.
    corrections_path: Optional[Path] = None
    # Set by cancel_job before it kills the job, so the monitor reports the
    # exit that follows as CANCELLED rather than FAILED.
    cancel_requested: bool = False
    _processed_iteration_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Get LSF job ID or local PID
        lsf_job_id = None
        if self.lsf_job:
            if hasattr(self.lsf_job, 'job_id'):
                lsf_job_id = self.lsf_job.job_id
            elif hasattr(self.lsf_job, 'process'):
                lsf_job_id = f"PID:{self.lsf_job.process.pid}"

        return {
            "job_id": self.job_id,
            "lsf_job_id": lsf_job_id,
            "model_name": self.model_name,
            "output_dir": str(self.output_dir),
            "params": self.params,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "log_file": str(self.log_file),
            "finetuned_model_name": self.finetuned_model_name,
            "model_yaml_path": str(self.model_yaml_path) if self.model_yaml_path else None,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "latest_loss": self.latest_loss,
            "inference_server_url": self.inference_server_url,
            "inference_server_ready": self.inference_server_ready,
            "corrections_path": str(self.corrections_path) if self.corrections_path else None,
        }


def on_scheduler_status(job: FinetuneJob, lsf_status) -> Optional[JobStatus]:
    """Move ``job`` on what its scheduler says of it; once it has ended, how.

    ``lsf_status`` is a jobs.spec.JobStatus. Returns None while the job
    lives, else the status it ended with. A job whose cancel was asked for
    has been cancelled however its end is reported. RUNNING moves a pending
    job on and leaves one waiting for a restart waiting; PENDING makes it
    pending.

    COMPLETED is returned but not set: the finetune tab stops polling, and
    the log stream says done, at the first final status they see, so the
    job keeps its status until the monitor has checked what it exported.
    """
    if job.cancel_requested and lsf_status in (
        LSFJobStatus.COMPLETED, LSFJobStatus.FAILED, LSFJobStatus.KILLED
    ):
        job.status = JobStatus.CANCELLED
        return job.status
    if lsf_status == LSFJobStatus.RUNNING:
        if job.status == JobStatus.PENDING:
            logger.info(f"Job {job.job_id} started running")
            job.status = JobStatus.RUNNING
    elif lsf_status == LSFJobStatus.PENDING:
        job.status = JobStatus.PENDING
    elif lsf_status == LSFJobStatus.COMPLETED:
        logger.info(f"Job {job.job_id} completed according to LSF")
        return JobStatus.COMPLETED
    elif lsf_status == LSFJobStatus.FAILED:
        logger.error(f"Job {job.job_id} failed according to LSF")
        job.status = JobStatus.FAILED
        return job.status
    elif lsf_status == LSFJobStatus.KILLED:
        logger.warning(f"Job {job.job_id} was killed")
        job.status = JobStatus.CANCELLED
        return job.status
    return None


def on_status_marker(job: FinetuneJob, marker: str) -> None:
    """Move ``job`` on one of the trainer's status markers
    (markers.STATUS_MARKER_RE), unless its status is final."""
    if job.status in TERMINAL_STATUSES:
        return
    if marker == "TRAINING_DIVERGED":
        # Training produced NaN/Inf loss. The trainer then waits for a
        # restart, or exits if nothing is served yet (LSF then says so).
        logger.warning(f"Training diverged for job {job.job_id}")
        job.status = JobStatus.WAITING_FOR_RESTART
        job.latest_loss = None
    elif marker == "WAITING_FOR_RESTART":
        job.status = JobStatus.WAITING_FOR_RESTART
    else:  # RESTARTING_TRAINING: reset progress
        logger.info(f"Training restart detected for job {job.job_id}")
        start_iteration(job)


def can_restart(job: FinetuneJob) -> bool:
    """Whether ``job``'s trainer can take a restart request.

    Only a trainer that is alive and waiting can take a restart. A
    COMPLETED job has exited: nothing reads the request, and the monitor
    that would have seen it through has stopped, so it used to sit at
    RUNNING forever. WAITING_FOR_RESTART also covers a later iteration
    that diverged: its server is up but not marked ready, and the
    restart the user needed was refused. A RUNNING job whose server is
    up is a trainer that predates the WAITING_FOR_RESTART marker.
    """
    waiting = job.status == JobStatus.WAITING_FOR_RESTART
    serving = job.status == JobStatus.RUNNING and job.inference_server_ready
    return waiting or serving


def start_iteration(job: FinetuneJob) -> None:
    """A new training iteration starts: its progress from zero, and its server
    not ready again until it completes. The server's URL is kept."""
    job.current_epoch = 0
    job.latest_loss = None
    job.status = JobStatus.RUNNING
    job.inference_server_ready = False
