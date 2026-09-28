"""
Utilities for job submission and management across different execution environments.

Supports:
- LSF (bsub) cluster jobs
- Local process execution
- Extensible to cloud providers and other cluster types
"""

import os
import re
import subprocess
import shlex
import logging
import sys
import signal
import tempfile
import time
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod
from enum import Enum

from cellmap_flow.globals import g
from cellmap_flow.utils.web_utils import IP_PATTERN

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SECURITY = "http"
DEFAULT_QUEUE = "gpu_h100"
DEFAULT_CHARGE_GROUP = "cellmap"
SERVER_COMMAND = "cellmap_flow_server"
SERVER_LOG_DIR = Path(os.path.expanduser("~/.cellmap_flow/server_logs"))


def _tail(path: Path, max_chars: int = 4000) -> Optional[str]:
    """Read the tail of a log file, for surfacing crash output. Returns None if unreadable/empty.

    Seeks to the end rather than reading the whole file. An inference server
    log can run to hundreds of megabytes, and this is called from the polling
    loop in ``wait_for_host`` while the caller is blocked.

    Reads 4 bytes per requested character so a multi-byte sequence split at
    the seek boundary still leaves at least ``max_chars`` intact; the leading
    partial character decodes to a replacement char and is sliced off.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_chars * 4), os.SEEK_SET)
            raw = f.read()
    except OSError:
        return None
    content = raw.decode("utf-8", errors="replace").strip()
    if not content:
        return None
    return content[-max_chars:]


def _log_stem(job_name: str) -> str:
    """A filesystem-safe stem for this job's log file.

    ``job_name`` arrives from model names, YAML and HuggingFace repo ids, so
    it can carry a path separator or ``..``. Interpolated straight into a
    path, that writes the log outside SERVER_LOG_DIR -- or, more quietly,
    makes the path we read back afterwards differ from the one we handed
    bsub, so a crashed job looks like it produced no output at all.

    Both the ``-o`` pattern and the path reconstructed after submission are
    built from this one value, so they cannot drift apart.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name).strip("._-")
    return stem or "job"


class JobStatus(Enum):
    """Enumeration of possible job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class JobStartError(RuntimeError):
    """A job was asked for and no usable one came of it."""


class Job(ABC):
    """
    Abstract base class for jobs across different execution environments.
    
    Subclasses should implement:
    - kill(): Terminate the job
    - get_status(): Get current job status
    - wait_for_host(): Wait for and extract host information
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.status = JobStatus.RUNNING
        self.host: Optional[str] = None
        # The queue the job was actually submitted to, which queue cycling
        # can make different from the one requested. None for local jobs.
        self.queue: Optional[str] = None
    
    @abstractmethod
    def kill(self) -> None:
        """Terminate the job."""
        pass
    
    @abstractmethod
    def get_status(self) -> JobStatus:
        """Get the current status of the job."""
        pass
    
    @abstractmethod
    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """
        Wait for the job to provide host information.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Host URL if found, None otherwise
        """
        pass
    
    def peek(self, max_chars: int = 4000) -> Optional[str]:
        """The tail of this job's own output, or None if it cannot be read."""
        return None

    def is_running(self) -> bool:
        """Check if the job is currently running."""
        return self.status == JobStatus.RUNNING


class LocalJob(Job):
    """Job running as a local subprocess.

    The process leads its own session (see run_locally), so kill() can take
    down everything under it -- a ``bash -c`` wrapper, the python it runs,
    a ``tee`` -- rather than only the direct child. Its output goes to
    ``log_file``, which is where the host marker is read from.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        model_name: Optional[str] = None,
        log_file: Optional[Path] = None,
    ):
        super().__init__(model_name)
        self.process = process
        self.log_file = Path(log_file) if log_file else None

    def _own_group(self) -> Optional[int]:
        """The process group to signal, or None to signal only the process.

        Only a process that leads its own group is signalled as a group: one
        that shares ours would take this process down with it.
        """
        try:
            pgid = os.getpgid(self.process.pid)
        except (ProcessLookupError, OSError):
            return None
        return pgid if pgid == self.process.pid else None

    def _signal(self, sig) -> None:
        group = self._own_group()
        try:
            if group is not None:
                os.killpg(group, sig)
            else:
                self.process.send_signal(sig)
        except ProcessLookupError:
            pass

    def _group_alive(self, group: Optional[int]) -> bool:
        if group is None:
            return self.process.poll() is None
        try:
            os.killpg(group, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def kill(self) -> None:
        """Terminate the local process and everything it started."""
        if self.process is None or self.process.poll() is not None:
            logger.warning("Local job is not running.")
            return

        logger.info(f"Killing local process group {self.process.pid}")
        group = self._own_group()
        try:
            self._signal(signal.SIGTERM)
            deadline = time.monotonic() + 5
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The leader exiting does not mean the rest of the group has:
            # give the others the remainder of the grace period, then force.
            while self._group_alive(group) and time.monotonic() < deadline:
                time.sleep(0.1)
            if self._group_alive(group):
                logger.warning("Process didn't terminate, killing forcefully")
                if group is not None:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self.process.kill()
                self.process.wait()
        except Exception as e:
            logger.error(f"Error killing process: {e}")
        finally:
            self.status = JobStatus.KILLED

    def get_status(self) -> JobStatus:
        """Get current status by checking process state."""
        if self.process is None:
            return JobStatus.FAILED

        returncode = self.process.poll()
        if returncode is None:
            return JobStatus.RUNNING
        elif returncode == 0:
            return JobStatus.COMPLETED
        else:
            return JobStatus.FAILED

    def peek(self, max_chars: int = 4000) -> Optional[str]:
        """The tail of the job's log file."""
        return self.log_file and _tail(self.log_file, max_chars)

    def wait_for_host(self, timeout: int = 180) -> Optional[str]:
        """
        Watch the process's log file for host information.

        Args:
            timeout: Maximum time to wait in seconds (default 180s for model loading)

        Returns:
            Host URL if found, None otherwise
        """
        if self.host:
            return self.host

        logger.info(f"Monitoring local process {self.process.pid} for host information...")
        deadline = time.monotonic() + timeout
        position = 0
        # Only the recent tail is kept between reads: enough to hold a marker
        # that straddles two reads, without holding the whole log in memory.
        carry = ""

        while True:
            exited = self.process.poll() is not None
            chunk = b""
            if self.log_file is not None:
                try:
                    with self.log_file.open("rb") as f:
                        f.seek(position)
                        chunk = f.read()
                        position = f.tell()
                except OSError:
                    pass
            text = carry + chunk.decode("utf-8", errors="replace")
            host = extract_host_from_output(text)
            if host:
                self.host = host
                logger.info(f"Found host: {host}")
                return host
            carry = text[-4096:]

            # Checked before the read above, so output written just before
            # exiting has already been searched.
            if exited:
                logger.error(
                    f"Process exited prematurely with code {self.process.returncode}"
                )
                self.status = JobStatus.FAILED
                crash_output = self.peek()
                if crash_output:
                    logger.error(f"Local job log output ({self.log_file}):\n{crash_output}")
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.5, remaining))

        logger.warning(f"Could not extract host from local process after {timeout}s")
        return None


class LSFJob(Job):
    """Job submitted to LSF cluster via bsub."""
    
    def __init__(
        self,
        job_id: str,
        model_name: Optional[str] = None,
        log_file: Optional[Path] = None,
    ):
        super().__init__(model_name)
        self.job_id = job_id
        self.log_file = log_file
        # Set by get_status() to say whether bjobs actually answered; see
        # observed_status().
        self._bjobs_answered = False
    
    def kill(self) -> None:
        """Terminate the LSF job using bkill."""
        logger.info(f"Killing LSF job {self.job_id}")
        try:
            result = subprocess.run(
                ["bkill", self.job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.status = JobStatus.KILLED
                logger.info(f"Successfully killed job {self.job_id}")
            else:
                logger.error(f"Failed to kill job {self.job_id}: {result.stderr}")
        except Exception as e:
            logger.error(f"Error killing LSF job {self.job_id}: {e}")
    
    def peek(self, max_chars: int = 4000) -> Optional[str]:
        """The job's own output, so nobody has to ssh in and run bpeek.

        A running job's output has not been flushed to the ``-o`` file yet --
        LSF writes that at the end -- so bpeek is the only way to see it live.
        Once the job is gone bpeek has nothing, and the file is the only
        record. Try them in that order.
        """
        try:
            result = subprocess.run(
                ["bpeek", self.job_id], capture_output=True, text=True, timeout=10
            )
            output = (result.stdout or "").strip()
            if output:
                return output[-max_chars:]
        except Exception as e:
            logger.debug(f"bpeek {self.job_id} failed: {e}")
        return self.log_file and _tail(self.log_file, max_chars)

    def observed_status(self) -> Optional[JobStatus]:
        """The status bjobs actually reported, or None if it could not say.

        get_status() falls back to self.status, which starts out RUNNING. That
        is fine for display but wrong for decisions: a job that never started
        reads as RUNNING the moment bjobs is unreadable, so a caller asking
        "did this actually leave the queue?" would be told yes. Callers that
        need the difference use this instead.
        """
        reported = self.get_status()
        return reported if self._bjobs_answered else None

    def get_status(self) -> JobStatus:
        """Query LSF for job status using bjobs."""
        self._bjobs_answered = False
        try:
            result = subprocess.run(
                ["bjobs", "-noheader", self.job_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Job not found, likely completed or killed
                return self.status
            
            output = result.stdout.strip()
            if not output:
                return self.status
            
            self._bjobs_answered = True
            # Parse bjobs output (format: JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME)
            fields = output.split()
            if len(fields) >= 3:
                stat = fields[2]
                if stat == "RUN":
                    return JobStatus.RUNNING
                elif stat == "PEND":
                    return JobStatus.PENDING
                elif stat == "DONE":
                    return JobStatus.COMPLETED
                elif stat == "EXIT":
                    return JobStatus.FAILED
            
            return self.status
        except Exception as e:
            logger.debug(f"Error checking LSF job status: {e}")
            return self.status
    
    def _log_crash_output(self) -> None:
        crash_output = self.log_file and _tail(self.log_file)
        if crash_output:
            logger.error(
                f"Job {self.job_id} log output ({self.log_file}):\n{crash_output}"
            )

    def wait_for_host(self, timeout: int = 300) -> Optional[str]:
        """
        Monitor LSF job output using bpeek to extract host information.

        ``timeout`` is wall-clock time, including however long bjobs and
        bpeek take to answer.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            Host URL if found, None otherwise
        """
        if self.host:
            return self.host

        logger.info(f"Monitoring LSF job {self.job_id} for host information...")

        # Model load dominates this wait -- weights off /nrs, a torch.export,
        # sometimes a HuggingFace fetch -- and it is the part people ask about
        # when a submit "takes a while". Report it rather than leaving the gap
        # between submission and the first chunk unaccounted for.
        wait_started = time.time()
        deadline = time.monotonic() + timeout

        # When the job entered PENDING (cleared once it leaves), and the total
        # time it has spent pending, which is never reset -- otherwise a job
        # that queued 5.5s reports "0s of it queued". Each warning is logged
        # once, the first time the job has been pending that long.
        pending_since = None
        total_pending = 0.0
        pending_warnings = [
            (30, "Queue may be busy or resources unavailable."),
            (60, "Consider checking queue status or resource availability."),
            (120, f"This is unusually long. You may want to check with 'bjobs {self.job_id}'"),
        ]
        # bpeek returns everything the job has written so far, every time, so
        # an error line would otherwise be re-logged on every poll.
        reported_errors = set()
        max_reported_errors = 20

        def pause():
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.5, remaining))

        while time.monotonic() < deadline:
            try:
                current_status = self.get_status()
                answered = self._bjobs_answered
                now = time.monotonic()

                if current_status == JobStatus.PENDING:
                    if pending_since is None:
                        pending_since = now
                    pending_for = now - pending_since
                    while pending_warnings and pending_for >= pending_warnings[0][0]:
                        _, advice = pending_warnings.pop(0)
                        logger.warning(
                            f"Job {self.job_id} pending for {pending_for:.0f}s. {advice}"
                        )
                elif pending_since is not None:
                    total_pending += now - pending_since
                    logger.info(
                        f"Job {self.job_id} started after {now - pending_since:.0f}s "
                        f"in pending state"
                    )
                    pending_since = None

                # Only bjobs can say the job is over. An empty bpeek on its
                # own is also what a busy mbatchd looks like.
                if answered and current_status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    logger.warning(
                        f"Job {self.job_id} ended ({current_status.value}) without "
                        f"reporting a host"
                    )
                    self.status = current_status
                    self._log_crash_output()
                    return None

                result = subprocess.run(
                    ["bpeek", self.job_id],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                output = result.stdout
                error = result.stderr

                # Check if job hasn't started yet
                if f"Job <{self.job_id}> : Not yet started." in error:
                    logger.debug(f"Job {self.job_id} not yet started. Waiting...")
                    pause()
                    continue

                if not output and result.returncode != 0:
                    logger.debug(
                        f"bpeek {self.job_id} gave nothing ({error.strip()}); "
                        f"bjobs says {current_status.value}, still waiting"
                    )
                    pause()
                    continue

                # Try to extract host
                if output:
                    host = extract_host_from_output(output)
                    if host:
                        self.host = host
                        if pending_since is not None:
                            total_pending += time.monotonic() - pending_since
                        logger.info(
                            f"Found host: {host} "
                            f"({time.time() - wait_started:.0f}s after submission, "
                            f"{total_pending:.0f}s of it queued)"
                        )
                        return host

                    # Check for errors, reporting each line once
                    for line in output.splitlines():
                        if (
                            "error" in line.lower()
                            and line not in reported_errors
                            and len(reported_errors) < max_reported_errors
                        ):
                            reported_errors.add(line)
                            logger.error(f"Error in job {self.job_id} output: {line}")

                pause()

            except subprocess.TimeoutExpired:
                logger.debug(f"Timeout waiting for job {self.job_id} output")
                pause()
            except Exception as e:
                logger.error(f"Error monitoring job {self.job_id}: {e}")
                break

        logger.warning(f"Timeout waiting for host from job {self.job_id}")
        self._log_crash_output()
        return None


def extract_host_from_output(output: str) -> Optional[str]:
    """
    Extract host/URL from command output using configured patterns.
    
    Args:
        output: String output to search
        
    Returns:
        Host URL if found, None otherwise
    """
    if not output:
        return None
    
    try:
        if IP_PATTERN[0] in output and IP_PATTERN[1] in output:
            host = output.split(IP_PATTERN[0])[1].split(IP_PATTERN[1])[0]
            return host
    except (IndexError, AttributeError) as e:
        logger.debug(f"Could not extract host: {e}")
    
    return None


def cleanup_handler(signum: int, frame) -> None:
    """
    Signal handler for graceful shutdown.
    Kills all tracked jobs before exiting.
    """
    logger.warning(f"Received signal {signum}. Cleaning up jobs...")
    for job in g.jobs:
        logger.info(f"Killing job: {job.model_name}")
        job.kill()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, cleanup_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup_handler)  # Handle termination


# How long a job may sit PENDING before we give up on that queue and try
# another. Long enough that a queue which is merely busy still gets used,
# short enough that nobody watches a spinner while 9000 jobs clear ahead of
# them on a queue that was never going to start.
# LSF's own default run limit on the GPU queues is 120 minutes, and we never
# passed -W, so every inference server was killed two hours in -- while the
# Fileglancer app job that spawns them asks for 8 hours, so the dashboard
# outlived its own servers by six. Match the session: 8 hours, overridable
# per-yaml, per-submission, or from the dashboard. The queues allow up to
# 20160 minutes (14 days).
DEFAULT_WALLTIME = "08:00"


def _walltime_arg(walltime):
    """``["-W", value]`` for bsub, or ``[]`` when there is nothing to set.

    LSF accepts ``[hour:]minute``, so both "08:00" and "480" are valid and
    mean the same thing. Anything else would make bsub reject the whole
    submission, so an unparseable value is dropped with a warning rather than
    taking the job down with it -- the queue default still applies.
    """
    if walltime in (None, "", False):
        return []
    text = str(walltime).strip()
    if re.fullmatch(r"\d+(:\d{1,2})?", text):
        return ["-W", text]
    logger.warning(
        f"Ignoring unusable walltime {walltime!r}; expected minutes (480) or "
        "hours:minutes (08:00). Falling back to the queue default."
    )
    return []


PENDING_FALLBACK_SECONDS = 180

# Once a job is running, how long it gets to report its host. Loading the
# model dominates this -- weights off /nrs, a torch.export, sometimes a
# HuggingFace download -- and none of it is a reason to try another queue.
STARTUP_TIMEOUT_SECONDS = 300


def gpu_queue_candidates(preferred, cycle=True):
    """The queue to try first, then the others worth falling back to.

    With ``cycle=False`` the requested queue is the only candidate: the job
    waits for it however long that takes, rather than being moved to whatever
    is free. Some work is pinned to a queue on purpose -- a benchmark that
    must run on one GPU model, or a charge group only valid on one queue --
    and silently landing somewhere else is worse than waiting.

    Ordered by what LSF says is actually free rather than by a fixed list, so
    the first fallback is the one most likely to start now. Fallback queues
    that are not accepting work are dropped: they take submissions and never
    run them, which is indistinguishable from a very slow job.

    The requested queue is kept whatever LSF says about it, but demoted to
    last if LSF says it is not accepting work, so a closed request does not
    cost a full pending timeout before anything else is tried.

    When LSF cannot be queried at all, the fixed GPU list is used unfiltered.
    """
    from cellmap_flow.utils.lsf_queues import GPU_QUEUES, gpu_queue_availability

    candidates = [preferred] if preferred else []

    if not cycle:
        logger.info(
            f"Queue cycling disabled; using {preferred or 'the default queue'} "
            f"only, and waiting for it."
        )
        return candidates
    all_gpu = [q for q, _, _ in GPU_QUEUES]

    try:
        info = gpu_queue_availability()
    except Exception as e:
        logger.debug(f"Could not read queue availability: {e}")
        info = {}

    if not info.get("available"):
        return candidates + [q for q in all_gpu if q != preferred]

    others = [
        q for q in info["queues"]
        if q["queue"] != preferred and q.get("accepting")
    ]
    # Most free GPUs first; break ties on the shorter pending queue.
    others.sort(key=lambda q: (-(q.get("gpus_free") or 0), q.get("pending") or 0))

    # The order is not arbitrary and the reason is worth seeing -- especially
    # now that these records reach the dashboard's log panel. A queue that was
    # skipped is more interesting than one that was kept.
    for q in info["queues"]:
        state = "skipped, not accepting work" if not q.get("accepting") else (
            "requested" if q["queue"] == preferred else "fallback"
        )
        logger.info(f"  {q['queue']}: {q.get('description') or 'no detail'} [{state}]")

    # If LSF says the requested queue is not accepting work, try it last
    # rather than first. Trying it first costs PENDING_FALLBACK_SECONDS of
    # dead wait on a queue that LSF has already said will not start the job.
    # It stays on the list -- a queue can reopen, and the request should still
    # be honoured if nothing else works -- just not ahead of queues that can
    # run it now. A queue LSF says nothing about (a yaml naming gpu_l4) is
    # unknown, not closed, and keeps its place at the front.
    requested = next(
        (q for q in info["queues"] if q["queue"] == preferred), None
    )
    if requested is not None and not requested.get("accepting") and others:
        logger.warning(
            f"{preferred} is not accepting work ({requested.get('description')}); "
            f"trying it last and starting with {others[0]['queue']}"
        )
        return [q["queue"] for q in others] + [preferred]

    return candidates + [q["queue"] for q in others]


def is_bsub_available() -> bool:
    """Check if bsub command is available in the system PATH."""
    try:
        result = subprocess.run(
            ["which", "bsub"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return bool(result.stdout)
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"bsub not available: {e}")
        return False


def submit_bsub_job(
    command: str,
    queue: str = DEFAULT_QUEUE,
    charge_group: Optional[str] = None,
    job_name: str = "my_job",
    num_gpus: int = 1,
    num_cpus: int = 4,
    walltime: Optional[str] = None,
) -> LSFJob:
    """
    Submit a job to LSF cluster using bsub.
    
    Args:
        command: Shell command to execute
        queue: LSF queue name
        charge_group: Project/chargeback group for billing
        job_name: Name for the job
        num_gpus: Number of GPUs to request
        num_cpus: Number of CPUs to request
        
    Returns:
        LSFJob object for the submitted job
        
    Raises:
        subprocess.CalledProcessError: If job submission fails
    """
    SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # %J is substituted by LSF with the actual job ID once assigned.
    log_stem = _log_stem(job_name)
    log_pattern = SERVER_LOG_DIR / f"{log_stem}_%J.log"

    bsub_command = ["bsub", "-J", job_name, "-o", str(log_pattern)]

    if charge_group:
        bsub_command += ["-P", charge_group]

    bsub_command += [
        "-q", queue,
        "-gpu", f"num={num_gpus}",
        "-n", str(num_cpus),
    ]
    bsub_command += _walltime_arg(walltime)
    bsub_command += ["bash", "-c", command]

    logger.info(f"Submitting bsub job: {' '.join(bsub_command)}")

    try:
        result = subprocess.run(
            bsub_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        # Extract job ID from output like "Job <12345> is submitted..."
        job_id = result.stdout.split()[1].strip('<>')
        logger.info(f"Job {job_id} submitted successfully")

        log_file = SERVER_LOG_DIR / f"{log_stem}_{job_id}.log"
        return LSFJob(job_id=job_id, model_name=job_name, log_file=log_file)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e.stderr}")
        if not charge_group:
            logger.error("Hint: You may need to specify a charge group with -P option")
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise


def run_locally(command, name: str, log_file=None) -> LocalJob:
    """
    Run command locally as a subprocess (fallback when bsub unavailable).

    The child writes stdout and stderr to ``log_file`` -- by default a new
    file under SERVER_LOG_DIR -- rather than to pipes. Nothing reads a pipe
    once the host is known, so a child on pipes blocks for good as soon as it
    has logged a pipe buffer's worth (about 64 KB). It also starts in its own
    session, so LocalJob.kill() can signal everything under it.

    Args:
        command: Shell-free command, as a string (split with shlex) or argv list
        name: Job name for tracking
        log_file: Where the output goes; pass os.devnull when the command
            already writes its own log

    Returns:
        LocalJob object with process information
    """
    logger.info(f"Running locally: {command}")

    # Use shlex.split + shell=False to avoid shell-injection on user-controlled
    # command strings. Callers must not rely on shell features (pipes, &&, env
    # expansion) — pass a plain argv-style command.
    args = shlex.split(command) if isinstance(command, str) else list(command)

    if log_file is None:
        SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        fd, log_path = tempfile.mkstemp(
            prefix=f"{_log_stem(name)}_local_", suffix=".log", dir=SERVER_LOG_DIR
        )
        log_handle = os.fdopen(fd, "ab")
    else:
        log_path = log_file
        log_handle = open(log_file, "ab")

    try:
        process = subprocess.Popen(
            args,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            # Otherwise a python child block-buffers a file and the log
            # trails what the process has actually done.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as e:
        logger.error(f"Error starting local process: {e}")
        raise
    finally:
        # The child holds its own copy of the descriptor.
        log_handle.close()

    logger.info(f"Local job {name} (pid {process.pid}) is logging to {log_path}")
    return LocalJob(process=process, model_name=name, log_file=log_path)


def start_hosts(
    command: str,
    queue: str = DEFAULT_QUEUE,
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
    # An explicit argument wins; otherwise whatever the dashboard or yaml set;
    # otherwise the shared default. Never None, or the job silently inherits
    # the queue's two hours.
    if walltime is None:
        walltime = getattr(g, "walltime", None) or DEFAULT_WALLTIME

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
            try:
                job = submit_bsub_job(
                    command,
                    candidate,
                    charge_group,
                    job_name=f"{job_name}",
                    walltime=walltime,
                )
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
                raise JobStartError(
                    f"Job {job.job_id} for {job_name} on {candidate} ran for "
                    f"{STARTUP_TIMEOUT_SECONDS}s without reporting a server "
                    f"address and has been killed; see {job.log_file}"
                )
            if observed is not None and observed != JobStatus.PENDING:
                raise JobStartError(
                    f"Job {job.job_id} for {job_name} on {candidate} ended "
                    f"({observed.value}) without reporting a server address; "
                    f"see {job.log_file}"
                )

            if more_to_try:
                logger.warning(
                    f"Job {job.job_id} has not started on {candidate} after "
                    f"{PENDING_FALLBACK_SECONDS}s; killing it and trying "
                    f"{candidates[index + 1]}"
                )
                job.kill()
            else:
                job.kill()
                raise JobStartError(
                    f"Job {job.job_id} for {job_name} did not start on "
                    f"{' or '.join(candidates)}; it has been killed"
                )

        raise JobStartError(
            f"No GPU queue accepted {job_name}: "
            + ("; ".join(submit_errors) or "no queue to submit to")
        )
    else:
        logger.info("bsub not available, running locally")

    job = run_locally(command, job_name)

    if wait_for_host and not job.wait_for_host():
        job.kill()
        raise JobStartError(
            f"The local server for {job_name} did not report its address; "
            f"see {getattr(job, 'log_file', None)}"
        )

    g.jobs.append(job)
    return job
