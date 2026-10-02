"""Starting the inference servers the CLIs and the dashboard ask for.

The policy over ``jobs.lsf`` and ``jobs.local``: ``start_hosts`` runs a
server on LSF or on this machine, tries the other GPU queues when one does
not start it, waits for its address and records the job in
``started_jobs()``; ``install_cleanup_handlers`` kills the recorded jobs,
and any still starting, on Ctrl+C, SIGTERM or SIGHUP.

Two deployment settings are read here, once, at import:

- ``SERVER_COMMAND``, how a server is started on a compute node, from
  ``CELLMAP_FLOW_SERVER_COMMAND``. Callers read it from this module when they
  build a command (serving.launch), never a copy of it.
- ``SERVER_LOG_DIR``, where job logs go; tests and deployments point it
  elsewhere, and ``jobs.spec.default_log_dir`` reads it at submit time.

The job settings a start falls back on (the run limit, whether to try
other queues) are ``jobs.settings.launcher_settings()``, read when a job is
started.
"""

import contextlib
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
from cellmap_flow.jobs.settings import launcher_settings
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


# The jobs start_hosts started in this process, oldest first. Read it
# through started_jobs(), which looks it up at call time, so tests can give
# each test its own list here.
_started: list = []

# Jobs submitted but not in started_jobs() yet: queued, or loading their
# model, while start_hosts waits for an address. They are not in that list
# because everything that reads it (the dashboard's layers, the server
# check) takes a job there to have a server. cleanup_handler kills these
# too: a Ctrl+C in the minutes a job takes to start used to leave it
# running, billing its GPU, with nothing left that knew about it.
_starting: set = set()
_starting_lock = threading.Lock()

# The last start of each model name that failed, by name: the dashboard
# lists it, so the traceback of a server that died on startup stays on the
# page. It dropped out of the starting jobs the moment it failed, and with
# it the log being read. A new start of the same name replaces it; only the
# newest few are kept, as each costs a bpeek on every Job Logs refresh.
_failed: dict = {}
FAILED_STARTS_KEPT = 5


@contextlib.contextmanager
def _while_starting(job):
    """Count ``job`` among the starting jobs until the block is left, and
    among the failed starts when it is left by an exception."""
    with _starting_lock:
        _starting.add(job)
        _failed.pop(getattr(job, "model_name", None), None)
    try:
        yield
    except BaseException:
        with _starting_lock:
            _failed[getattr(job, "model_name", None)] = job
            while len(_failed) > FAILED_STARTS_KEPT:
                del _failed[next(iter(_failed))]
        raise
    finally:
        with _starting_lock:
            _starting.discard(job)


def starting_jobs() -> list:
    """The jobs submitted but not yet serving (waiting in their queue, or for
    their server to come up), for the dashboard to show as starting."""
    with _starting_lock:
        return [job for job in _starting if job not in _started]


def failed_starts() -> list:
    """The last failed start of each model name not started again since,
    for the dashboard to keep showing with its log."""
    with _starting_lock:
        return [job for job in _failed.values() if job not in _started and job not in _starting]


def started_jobs() -> list:
    """The jobs this process started: the live list, which start_hosts
    appends to, cleanup_handler kills from, and the dashboard lists (its
    Session.jobs replaces the contents rather than the list)."""
    return _started


def cleanup_handler(signum: int, frame) -> None:
    """
    Signal handler for graceful shutdown.
    Kills all tracked jobs, then exits with the conventional 128 + signal
    status, so a stopped run does not report success.
    """
    logger.warning(f"Received signal {signum}. Cleaning up jobs...")
    with _starting_lock:
        starting = [job for job in _starting if job not in started_jobs()]
    for job in list(started_jobs()) + starting:
        logger.info(f"Killing job: {job.model_name}")
        try:
            job.kill()
        except Exception as e:
            logger.error(f"Could not kill job {job.model_name}: {e}")
    sys.exit(128 + signum)


def install_cleanup_handlers() -> bool:
    """Kill the tracked jobs on Ctrl+C, SIGTERM or SIGHUP. Returns whether installed.

    SIGHUP is what a dashboard gets when its terminal goes: the window is
    closed, the connection to it drops, or the interactive LSF session it
    runs in ends. Unhandled, it ended the process and left every server it
    had started running.

    For entry points that launch jobs, called from their main thread. This
    used to happen when the module was imported, which set the handlers for
    every importer, and raised ValueError when the first import happened off
    the main thread (the lazy Session.finetune_job_manager, a dashboard request).
    """
    if threading.current_thread() is not threading.main_thread():
        logger.debug("Not on the main thread; leaving signal handlers alone")
        return False
    signal.signal(signal.SIGINT, cleanup_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_handler)  # Handle termination
    signal.signal(signal.SIGHUP, cleanup_handler)  # The terminal went away
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
        walltime: LSF run limit ("HH:MM" or minutes); defaults to the
            launcher settings' walltime, then the site's
        cycle_queues: Try other GPU queues when the requested one is busy or
            closed. Defaults to the launcher settings' cycle_gpu_queues,
            which defaults to True.
        local: Run on this machine even if bsub is available.

    Returns:
        Job object (LSFJob or LocalJob) with job information. ``job.queue``
        is the queue it landed on. The settings are left alone: the queue
        the job fell back to is not what the next submission should ask for.

    Raises:
        JobStartError: bsub is available but no queue accepted the job.
    """
    # An explicit argument wins; otherwise whatever the dashboard or yaml set;
    # otherwise the shared default. Never None, or the job silently inherits
    # the queue's two hours.
    if walltime is None:
        walltime = launcher_settings().walltime or _SITE.default_walltime

    # Same precedence as walltime: explicit argument, then the dashboard/yaml
    # setting, whose default is True. Cycling is on by default because a job
    # that starts on a different GPU queue beats one that never starts.
    if cycle_queues is None:
        cycle_queues = launcher_settings().cycle_gpu_queues

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
                started_jobs().append(job)
                return job

            with _while_starting(job):
                started = _wait_on_queue(job, job_name, queue, candidate, index, candidates)
            if started is not None:
                return started

        raise _logged(JobStartError(
            f"No GPU queue accepted {job_name}: "
            + ("; ".join(submit_errors) or "no queue to submit to")
        ))
    else:
        logger.info("bsub not available, running locally")

    job = run_locally(command, job_name)

    with _while_starting(job):
        if wait_for_host and not job.wait_for_host():
            job.kill()
            raise _logged(JobStartError(
                f"The local server for {job_name} did not report its address; "
                f"see {getattr(job, 'log_file', None)}"
            ))
        started_jobs().append(job)
    return job


def _wait_on_queue(job, job_name, queue, candidate, index, candidates):
    """Wait for ``job``, submitted to ``candidate``, the ``index``-th of
    ``candidates``, to report its address.

    Returns it, recorded in started_jobs(), once it has. Returns None after
    killing it when it never started and there is another queue to try.
    Raises JobStartError, after killing it if it may be alive, otherwise.
    """
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
        started_jobs().append(job)
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
        return None
    job.kill()
    raise _logged(JobStartError(
        f"Job {job.job_id} for {job_name} did not start on "
        f"{' or '.join(candidates)}; it has been killed"
    ))
