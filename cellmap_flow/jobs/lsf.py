"""Submitting jobs to LSF, and asking it about them.

``bsub_argv`` is the one place a bsub command line is built, and ``submit``
the one place one is run: for inference servers, finetune runs, blockwise
workers and the blockwise master alike.
"""

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import (
    Job,
    JobSpec,
    JobStartError,
    JobStatus,
    default_log_dir,
    extract_host_from_output,
    log_stem,
    tail,
)

logger = logging.getLogger(__name__)

BSUB_TIMEOUT_SECONDS = current_site().bsub_timeout_seconds


class BsubTimeoutError(JobStartError):
    """bsub did not answer in time, and the job it may yet create is unknown.

    LSF does not cancel a submission when the bsub client is killed, so the
    job can still appear minutes later. Submitting again would leave two.
    """


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
        return self.log_file and tail(self.log_file, max_chars)

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
                stat = _STATUS_BY_STAT.get(fields[2])
                if stat is not None:
                    return stat

            return self.status
        except Exception as e:
            logger.debug(f"Error checking LSF job status: {e}")
            return self.status

    def _log_crash_output(self) -> None:
        crash_output = self.log_file and tail(self.log_file)
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


# bjobs' STAT column, for the states that say something definite.
_STATUS_BY_STAT = {
    "RUN": JobStatus.RUNNING,
    "PEND": JobStatus.PENDING,
    "DONE": JobStatus.COMPLETED,
    "EXIT": JobStatus.FAILED,
}


def available() -> bool:
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


def walltime_arg(walltime) -> List[str]:
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


_BSUB_JOB_ID = re.compile(r"Job <(\d+)>")


def parse_job_id(output: Optional[str]) -> Optional[str]:
    """The id in bsub's "Job <12345> is submitted to queue <q>." line, or None.

    Searched for rather than taken by position: an esub can print its own
    notice first.
    """
    match = _BSUB_JOB_ID.search(output or "")
    return match.group(1) if match else None


def job_ids_named(job_name: str) -> Optional[set]:
    """Ids of this user's jobs called ``job_name``, in any state.

    None when bjobs could not say, which is different from "there are none".
    """
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-noheader", "-J", job_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        logger.debug(f"bjobs -J {job_name} failed: {e}")
        return None
    # Continuation lines (a multi-host EXEC_HOST) start with a host, not an id.
    ids = {
        line.split()[0]
        for line in (result.stdout or "").splitlines()
        if line.split() and line.split()[0].isdigit()
    }
    if ids or result.returncode == 0:
        return ids
    # "Job <name> is not found" is how bjobs says there are none.
    if "not found" in f"{result.stdout} {result.stderr}".lower():
        return set()
    return None


def _log_dir(spec: JobSpec) -> Path:
    return Path(spec.log_dir) if spec.log_dir is not None else default_log_dir()


def log_pattern(spec: JobSpec) -> Path:
    """The ``-o`` path for this job; LSF puts the job id in place of ``%J``."""
    return _log_dir(spec) / f"{log_stem(spec.name)}_%J.log"


def bsub_argv(spec: JobSpec) -> List[str]:
    """The bsub command line that submits ``spec``. Builds it; runs nothing."""
    argv = ["bsub", "-J", spec.name, "-o", str(log_pattern(spec))]
    if spec.charge_group:
        argv += ["-P", spec.charge_group]
    if spec.queue:
        argv += ["-q", spec.queue]
    if spec.gpus:
        argv += ["-gpu", f"num={spec.gpus}"]
    argv += ["-n", str(spec.cpus)]
    argv += walltime_arg(spec.walltime)
    if spec.shell is not None:
        argv += ["bash", "-c", spec.shell]
    else:
        argv += list(spec.argv)
    return argv


def submit(spec: JobSpec, *, bsub_timeout: Optional[float] = BSUB_TIMEOUT_SECONDS) -> LSFJob:
    """
    Submit ``spec`` to LSF with bsub.

    Args:
        spec: What to run; its ``log_dir`` gets the job's
            <name>_<jobid>.log (stdout and stderr)
        bsub_timeout: Seconds to wait for bsub to answer, or None to wait as
            long as it takes (an over-ratio request is held for minutes
            before bsub returns)

    Returns:
        LSFJob object for the submitted job

    Raises:
        subprocess.CalledProcessError: If job submission fails
        BsubTimeoutError: bsub did not answer and no new job with this name
            can be found. The caller must not simply submit again.
    """
    log_dir = _log_dir(spec)
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = spec.name
    bsub_command = bsub_argv(spec)

    logger.info(f"Submitting bsub job: {' '.join(bsub_command)}")

    # LSF copies the submitting environment into the job, so anything the
    # job needs to be told travels as environment. Left out when there is
    # nothing to add, so the job inherits exactly what this process has.
    run_kwargs = {"env": {**os.environ, **spec.env}} if spec.env else {}

    # Taken before submitting, so that if bsub times out the job it created
    # can be told apart from older jobs with the same name.
    existing = job_ids_named(job_name) if bsub_timeout is not None else None

    try:
        result = subprocess.run(
            bsub_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=bsub_timeout,
            **run_kwargs,
        )
    except subprocess.TimeoutExpired as e:
        now = job_ids_named(job_name)
        new_ids = (now - existing) if (now is not None and existing is not None) else set()
        if len(new_ids) != 1:
            error = BsubTimeoutError(
                f"bsub did not answer within {bsub_timeout}s and no new "
                f"job named {job_name} can be identified; LSF may still create "
                f"it. Check `bjobs -a -J {job_name}` before submitting again."
            )
            logger.error(str(error))
            raise error from e
        job_id = new_ids.pop()
        logger.warning(f"bsub timed out, but job {job_id} ({job_name}) was submitted")
    except subprocess.CalledProcessError as e:
        logger.error(f"Job submission failed: {e.stderr}")
        if not spec.charge_group:
            logger.error("Hint: You may need to specify a charge group with -P option")
        raise
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise
    else:
        job_id = parse_job_id(result.stdout) or parse_job_id(result.stderr)
        if job_id is None:
            raise RuntimeError(
                f"bsub exited 0 but printed no job id: {result.stdout.strip()!r}"
            )
        logger.info(f"Job {job_id} submitted successfully")

    log_file = log_dir / f"{log_stem(job_name)}_{job_id}.log"
    return LSFJob(job_id=job_id, model_name=job_name, log_file=log_file)
