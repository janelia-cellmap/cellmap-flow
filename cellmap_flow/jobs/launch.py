"""Starting the inference servers the CLIs and the dashboard ask for.

The policy over ``jobs.lsf`` and ``jobs.local``: ``start_hosts`` runs a
server on LSF or on this machine, tries the other GPU queues when one does
not start it, waits for its address and records the job in ``g.jobs``;
``install_cleanup_handlers`` kills the recorded jobs on Ctrl+C.

Two deployment settings are read here, once, at import:

- ``SERVER_COMMAND``, how a server is started on a compute node, from
  ``CELLMAP_FLOW_SERVER_COMMAND``. Callers read it from this module when they
  build a command (serving.launch), never a copy of it.
- ``SERVER_LOG_DIR``, where job logs go; tests and deployments point it
  elsewhere, and ``jobs.spec.default_log_dir`` reads it at submit time.

``cellmap_flow.globals`` is imported only when a job is started or killed,
so importing this module does not load the dashboard's state.
"""

import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from cellmap_flow.jobs import lsf
# Module globals, looked up when a job is started, so tests can replace them
# here.
from cellmap_flow.jobs.local import run as run_locally
from cellmap_flow.jobs.lsf import BsubTimeoutError, LSFJob
from cellmap_flow.jobs.lsf import available as is_bsub_available
from cellmap_flow.jobs.queues import candidates as gpu_queue_candidates
from cellmap_flow.jobs.ready import READY_ENV, ready_path
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import Job, JobSpec, JobStartError, JobStatus

logger = logging.getLogger(__name__)

# How an inference server is started on a compute node; serving/launch.py
# adds `--model <entry> -d <data>`. A deployment whose environment is not on
# PATH sets this: Fileglancer's pixi checkout uses "pixi run cellmap_flow
# serve" (see pixi.toml's activation env), so the server runs from the
# lockfile's environment. The value before 0.3.0, "... cellmap_flow_server",
# still works: that command takes --model too.
SERVER_COMMAND = os.environ.get("CELLMAP_FLOW_SERVER_COMMAND", "cellmap_flow serve")
SERVER_LOG_DIR = Path(os.path.expanduser("~/.cellmap_flow/server_logs"))

# The site's numbers; jobs/site.py says why each is what it is.
_SITE = current_site()
PENDING_FALLBACK_SECONDS = _SITE.pending_fallback_seconds
STARTUP_TIMEOUT_SECONDS = _SITE.startup_timeout_seconds


def _logged(error: Exception) -> Exception:
    """Log ``error`` and hand it back to be raised.

    start_hosts often runs on a dashboard thread whose exceptions reach only
    stderr, while log records also reach the dashboard's log panel.
    """
    logger.error(str(error))
    return error


def cleanup_handler(signum: int, frame) -> None:
    """
    Signal handler for graceful shutdown.
    Kills all tracked jobs, then exits with the conventional 128 + signal
    status, so a stopped run does not report success.
    """
    from cellmap_flow.globals import g

    logger.warning(f"Received signal {signum}. Cleaning up jobs...")
    for job in list(g.jobs):
        logger.info(f"Killing job: {job.model_name}")
        try:
            job.kill()
        except Exception as e:
            logger.error(f"Could not kill job {job.model_name}: {e}")
    sys.exit(128 + signum)


def install_cleanup_handlers() -> bool:
    """Kill the tracked jobs on Ctrl+C or SIGTERM. Returns whether installed.

    For entry points that launch jobs, called from their main thread. This
    used to happen when the module was imported, which set the handlers for
    every importer, and raised ValueError when the first import happened off
    the main thread (the lazy g.finetune_job_manager, a dashboard request).
    """
    if threading.current_thread() is not threading.main_thread():
        logger.debug("Not on the main thread; leaving signal handlers alone")
        return False
    signal.signal(signal.SIGINT, cleanup_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_handler)  # Handle termination
    return True


def submit_bsub_job(
    command: str,
    queue: str = _SITE.default_queue,
    charge_group: Optional[str] = None,
    job_name: str = "my_job",
    num_gpus: int = 1,
    num_cpus: int = 4,
    walltime: Optional[str] = None,
    log_dir: Optional[Path] = None,
    bsub_timeout: Optional[float] = lsf.BSUB_TIMEOUT_SECONDS,
    env: Optional[dict] = None,
) -> LSFJob:
    """
    Submit a shell command to LSF using bsub; see cellmap_flow.jobs.lsf.submit.

    Args:
        command: Shell command to execute (run with ``bash -c``)
        queue: LSF queue name
        charge_group: Project/chargeback group for billing
        job_name: Name for the job
        num_gpus: Number of GPUs to request
        num_cpus: Number of CPUs to request
        walltime: LSF run limit ("HH:MM" or minutes); None leaves the
            queue default
        log_dir: Directory for the job's <name>_<jobid>.log (stdout and
            stderr); defaults to SERVER_LOG_DIR
        bsub_timeout: Seconds to wait for bsub to answer, or None to wait as
            long as it takes (an over-ratio request is held for minutes
            before bsub returns)
        env: Variables to set in the job's environment, on top of this
            process's (which LSF copies into the job anyway)

    Returns:
        LSFJob object for the submitted job

    Raises:
        subprocess.CalledProcessError: If job submission fails
        BsubTimeoutError: bsub did not answer and no new job with this name
            can be found. The caller must not simply submit again.
    """
    spec = JobSpec(
        name=job_name,
        shell=command,
        queue=queue,
        charge_group=charge_group,
        gpus=num_gpus,
        cpus=num_cpus,
        walltime=walltime,
        # Read here, at call time, so that pointing SERVER_LOG_DIR elsewhere
        # takes effect.
        log_dir=Path(log_dir) if log_dir is not None else SERVER_LOG_DIR,
        env=env,
    )
    return lsf.submit(spec, bsub_timeout=bsub_timeout)


def start_hosts(
    command: str,
    queue: str = _SITE.default_queue,
    charge_group: Optional[str] = None,
    job_name: str = "example_job",
    use_https: bool = False,
    wait_for_host: bool = True,
    walltime: Optional[str] = None,
    cycle_queues: Optional[bool] = None,
    local: bool = False,
) -> Job:
    """
    Start a server job either via bsub or locally.

    It runs on this machine only when bsub is not installed, or when
    ``local`` asks for it. When bsub is there and every submission fails,
    this raises: the machine running the CLI or dashboard is often a login
    or submit node, which is no place for a GPU server.

    Args:
        command: Command to execute
        queue: LSF queue name (for bsub)
        charge_group: Project for billing (for bsub)
        job_name: Name for the job
        use_https: Whether to use HTTPS (adds cert/key flags)
        wait_for_host: Whether to wait for host information before returning
        walltime: LSF run limit ("HH:MM" or minutes); defaults to g.walltime
        cycle_queues: Try other GPU queues when the requested one is busy or
            closed. Defaults to g.cycle_gpu_queues, which defaults to True.
        local: Run on this machine even if bsub is available.

    Returns:
        Job object (LSFJob or LocalJob) with job information. ``job.queue``
        is the queue it landed on. The globals are left alone: the queue the
        job fell back to is not what the next submission should ask for.

    Raises:
        JobStartError: bsub is available but no queue accepted the job.
    """
    from cellmap_flow.globals import g

    # An explicit argument wins; otherwise whatever the dashboard or yaml set;
    # otherwise the shared default. Never None, or the job silently inherits
    # the queue's two hours.
    if walltime is None:
        walltime = getattr(g, "walltime", None) or _SITE.default_walltime

    # Same precedence as walltime: explicit argument, then the dashboard/yaml
    # setting, then the default. Cycling is on by default because a job that
    # starts on a different GPU queue beats one that never starts.
    if cycle_queues is None:
        cycle_queues = getattr(g, "cycle_gpu_queues", True)

    # Add HTTPS flags if needed
    if use_https:
        command = f"{command} --certfile=host.cert --keyfile=host.key"

    job: Job

    if local:
        logger.info("Running locally, as requested")
    elif is_bsub_available():
        logger.info("Using bsub for job submission")
        candidates = gpu_queue_candidates(queue, cycle=cycle_queues)
        logger.info(f"Queue order: {' -> '.join(candidates)}")
        submit_errors = []
        for index, candidate in enumerate(candidates):
            # A new file for every submission, so a job that started late on
            # a queue given up on cannot hand its address to the next one.
            # A server too old to know the variable ignores it, and
            # wait_for_host reads its marker through bpeek instead.
            ready_file = ready_path(SERVER_LOG_DIR, job_name)
            try:
                job = submit_bsub_job(
                    command,
                    candidate,
                    charge_group,
                    job_name=f"{job_name}",
                    walltime=walltime,
                    env={READY_ENV: str(ready_file)},
                )
            except BsubTimeoutError:
                # Not a refusal: the job may still appear on this queue, and
                # trying the next one would leave two of it.
                raise
            except Exception as e:
                logger.error(f"Failed to submit bsub job to {candidate}: {e}")
                submit_errors.append(f"{candidate}: {e}")
                continue
            job.queue = candidate

            if not wait_for_host:
                g.jobs.append(job)
                return job

            # Give an unstarted job less patience while there is somewhere
            # else to try, and the full wait once this is the last option.
            more_to_try = index < len(candidates) - 1
            host = job.wait_for_host(
                timeout=PENDING_FALLBACK_SECONDS if more_to_try else 300
            )
            # observed_status(), not get_status(): the latter falls back to
            # self.status, which starts out RUNNING, so an unreadable bjobs
            # would look like "it started" and stop the fallback exactly when
            # LSF is flaky. Unknown is treated as still queued -- the job has
            # produced no host in PENDING_FALLBACK_SECONDS, so there is
            # nothing to lose by trying elsewhere.
            observed = None if host else job.observed_status()

            # Started but still loading its model: that is not a queue
            # problem, so wait on this job rather than trying elsewhere.
            if observed == JobStatus.RUNNING:
                host = job.wait_for_host(timeout=STARTUP_TIMEOUT_SECONDS)
                observed = None if host else job.observed_status()

            if host:
                if candidate != queue:
                    logger.warning(
                        f"Running on {candidate}, not the requested {queue}: "
                        f"{index} earlier queue(s) did not start the job"
                    )
                else:
                    logger.info(f"Running on {candidate}")
                g.jobs.append(job)
                return job

            # Only a job that never started is a queue problem. One that ran
            # and crashed will crash the same way everywhere else, so fail
            # now instead of burning through every queue reproducing it.
            #
            # A job with no host is not returned as if it were ready: there
            # is no server for the viewer to point at (it would build a
            # zarr://None/... layer), and nothing updates it later. One that
            # may still be alive is killed rather than left to bill.
            if observed == JobStatus.RUNNING:
                job.kill()
                raise _logged(JobStartError(
                    f"Job {job.job_id} for {job_name} on {candidate} ran for "
                    f"{STARTUP_TIMEOUT_SECONDS}s without reporting a server "
                    f"address and has been killed; see {job.log_file}"
                ))
            if observed is not None and observed != JobStatus.PENDING:
                raise _logged(JobStartError(
                    f"Job {job.job_id} for {job_name} on {candidate} ended "
                    f"({observed.value}) without reporting a server address; "
                    f"see {job.log_file}"
                ))

            if more_to_try:
                logger.warning(
                    f"Job {job.job_id} has not started on {candidate} after "
                    f"{PENDING_FALLBACK_SECONDS}s; killing it and trying "
                    f"{candidates[index + 1]}"
                )
                job.kill()
            else:
                job.kill()
                raise _logged(JobStartError(
                    f"Job {job.job_id} for {job_name} did not start on "
                    f"{' or '.join(candidates)}; it has been killed"
                ))

        raise _logged(JobStartError(
            f"No GPU queue accepted {job_name}: "
            + ("; ".join(submit_errors) or "no queue to submit to")
        ))
    else:
        logger.info("bsub not available, running locally")

    job = run_locally(command, job_name)

    if wait_for_host and not job.wait_for_host():
        job.kill()
        raise _logged(JobStartError(
            f"The local server for {job_name} did not report its address; "
            f"see {getattr(job, 'log_file', None)}"
        ))

    g.jobs.append(job)
    return job
