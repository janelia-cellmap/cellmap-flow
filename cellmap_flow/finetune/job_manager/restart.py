"""Asking a finetune job that waits for a restart to train again.

The trainer's inference server, in the same process as its training loop,
takes ``POST <server>/__control__/restart`` with the job's restart token
(written into the run's directory at submit; serving.restart_token) in the
``X-Restart-Token`` header. The body is ``{"restart": true, "timestamp",
"params"}``, ``params`` being the settings to change. When the server cannot
be reached, or has no URL yet, the same body goes into the run's
``restart_signal.json``, which the trainer also watches (finetune.session_loop).
It is written whole or not at all (``finetune.json_files.write_json_atomically``): the
trainer reads it as soon as it exists, and ends the job if it is not JSON.
Jobs outlive dashboard upgrades, so both ways are a protocol.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from cellmap_flow.finetune.job_manager import state
from cellmap_flow.finetune.job_manager.state import FinetuneJob
from cellmap_flow.finetune.json_files import write_json_atomically
from cellmap_flow.serving.restart_token import TOKEN_HEADER, read_restart_token

logger = logging.getLogger(__name__)


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
            logger.info(f"Sent restart request via HTTP control endpoint: {control_url}")
        except Exception as e:
            http_error = e
            logger.warning(f"HTTP restart control failed for job {job.job_id}: {e}")
    else:
        http_error = RuntimeError("No inference_server_url for HTTP restart control")

    # 2. Fallback to signal file if HTTP control endpoint is unavailable
    if http_error is not None:
        signal_write_mode = "file_signal_fallback"
        signal_file = job.output_dir / "restart_signal.json"
        write_json_atomically(signal_file, signal_data)
        logger.info(f"Wrote fallback restart signal to {signal_file}")
    write_elapsed = time.perf_counter() - write_t0

    # 3. Reset training progress (keep inference server info)
    state.start_iteration(job)

    # 4. Update stored params
    if updated_params:
        job.params.update(updated_params)

    total_elapsed = time.perf_counter() - restart_t0
    logger.info(
        f"Restart signal timings for job {job.job_id}: "
        f"write={write_elapsed:.2f}s "
        f"mode={signal_write_mode} total={total_elapsed:.2f}s"
    )
    logger.info(f"Job {job.job_id} restart request sent, waiting for CLI to pick it up")
