"""Asking a finetune job to train again, or to stop training early.

The trainer's inference server, in the same process as its training loop,
takes ``POST <server>/__control__/restart`` and ``.../__control__/stop`` with
the job's restart token (written into the run's directory at submit;
serving.restart_token) in the ``X-Restart-Token`` header. A restart's body is
``{"restart": true, "timestamp", "params"}``, ``params`` being the settings
to change; a stop's ``{"requested_at", "reason"}``. When the server cannot
be reached, or has no URL yet -- it starts once the first iteration is
trained -- the same body goes into the run's ``restart_signal.json`` or
``stop_signal.json``, which the trainer also watches (finetune.session_loop).
It is written whole or not at all (``finetune.json_files.write_json_atomically``): the
trainer reads it as soon as it exists, and ends the job if a restart's is
not JSON. Jobs outlive dashboard upgrades, so both ways are a protocol: a
trainer from before the stop endpoint answers it 404, and gets the file.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from cellmap_flow.finetune.job_manager import state
from cellmap_flow.finetune.job_manager.state import FinetuneJob, JobStatus
from cellmap_flow.finetune.json_files import write_json_atomically
from cellmap_flow.serving.restart_token import TOKEN_HEADER, read_restart_token

logger = logging.getLogger(__name__)


def _send(job: FinetuneJob, action: str, signal_data: Dict[str, Any]) -> str:
    """Hand ``signal_data`` to ``job``'s trainer: POSTed to its server's
    ``/__control__/<action>``, else written to the run's
    ``<action>_signal.json``. How it went: "http_control" or
    "file_signal_fallback"."""
    if job.inference_server_url:
        try:
            control_url = job.inference_server_url.rstrip("/") + f"/__control__/{action}"
            restart_token = read_restart_token(job.output_dir)
            if restart_token is None:
                raise RuntimeError(f"No restart token in {job.output_dir}")
            headers = {TOKEN_HEADER: restart_token}
            response = requests.post(control_url, json=signal_data, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            if not data.get("success", False):
                raise RuntimeError(data.get("error", f"Unknown {action} control failure"))
            logger.info(f"Sent {action} request via HTTP control endpoint: {control_url}")
            return "http_control"
        except Exception as e:
            logger.warning(f"HTTP {action} control failed for job {job.job_id}: {e}")

    signal_file = job.output_dir / f"{action}_signal.json"
    write_json_atomically(signal_file, signal_data)
    logger.info(f"Wrote fallback {action} signal to {signal_file}")
    return "file_signal_fallback"


def request_restart(job: FinetuneJob, updated_params: Optional[Dict[str, Any]] = None) -> None:
    """Ask ``job``'s trainer to train again, with ``updated_params`` changed.

    The job is then RUNNING a new iteration from epoch 0, its server not
    ready until the iteration completes, and ``job.params`` updated.
    Raises ValueError, and sends nothing, unless the job can take a restart
    (state.can_restart).
    """
    restart_t0 = time.perf_counter()

    if not state.can_restart(job):
        raise ValueError(
            f"Job {job.job_id} is in state {job.status.value} - can only restart a "
            f"job that is waiting for a restart (its training iteration has "
            f"finished or diverged)"
        )

    # A stop that came as the last iteration's training ended by itself is
    # not for the next one. The trainer keeps every stop it finds, so that
    # one asked for while it sets up is not lost; this one goes first. (One
    # sent over HTTP the trainer drops itself, when the restart reaches it.)
    (job.output_dir / "stop_signal.json").unlink(missing_ok=True)

    signal_data = {
        "restart": True,
        "timestamp": datetime.now().isoformat(),
        "params": updated_params or {}
    }
    write_t0 = time.perf_counter()
    signal_write_mode = _send(job, "restart", signal_data)
    write_elapsed = time.perf_counter() - write_t0

    # Reset training progress (keep inference server info)
    state.start_iteration(job)

    # Update stored params
    if updated_params:
        job.params.update(updated_params)

    total_elapsed = time.perf_counter() - restart_t0
    logger.info(
        f"Restart signal timings for job {job.job_id}: "
        f"write={write_elapsed:.2f}s "
        f"mode={signal_write_mode} total={total_elapsed:.2f}s"
    )
    logger.info(f"Job {job.job_id} restart request sent, waiting for CLI to pick it up")


def request_stop(job: FinetuneJob) -> None:
    """Ask ``job``'s trainer to stop training after the epoch it is in (Stop Early).

    It then exports what it has trained and serves it, as at the end of any
    iteration, and waits for a restart. Raises ValueError, and sends
    nothing, unless a training loop is running (the RUNNING job whose
    server is not serving a finished iteration; the browser's backend
    refuses the same).
    """
    if job.status != JobStatus.RUNNING or job.inference_server_ready:
        raise ValueError(f"Job {job.job_id} is not training (its status is {job.status.value})")
    mode = _send(job, "stop", {"requested_at": datetime.now().isoformat(), "reason": "user_requested_stop_early"})
    logger.info(f"Job {job.job_id} stop requested ({mode})")
