"""Running a job as a process on this machine, for when there is no LSF."""

import logging
import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional

from cellmap_flow.jobs.spec import (
    Job,
    JobStatus,
    default_log_dir,
    extract_host_from_output,
    log_stem,
    tail,
)

logger = logging.getLogger(__name__)


class LocalJob(Job):
    """Job running as a local subprocess.

    The process leads its own session (see run), so kill() can take down
    everything under it -- a ``bash -c`` wrapper, the python it runs, a
    ``tee`` -- rather than only the direct child. Its output goes to
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
        return self.log_file and tail(self.log_file, max_chars)

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


def run(
    command, name: str, log_file=None, env: Optional[Mapping[str, str]] = None
) -> LocalJob:
    """
    Run command locally as a subprocess (fallback when bsub unavailable).

    The child writes stdout and stderr to ``log_file`` -- by default a new
    file under the log directory (spec.default_log_dir) -- rather than to
    pipes. Nothing reads a pipe once the host is known, so a child on pipes
    blocks for good as soon as it has logged a pipe buffer's worth (about
    64 KB). It also starts in its own session, so LocalJob.kill() can signal
    everything under it.

    Args:
        command: Shell-free command, as a string (split with shlex) or argv list
        name: Job name for tracking
        log_file: Where the output goes; pass os.devnull when the command
            already writes its own log
        env: Added to this process's environment for the child

    Returns:
        LocalJob object with process information
    """
    logger.info(f"Running locally: {command}")

    # Use shlex.split + shell=False to avoid shell-injection on user-controlled
    # command strings. Callers must not rely on shell features (pipes, &&, env
    # expansion) — pass a plain argv-style command.
    args = shlex.split(command) if isinstance(command, str) else list(command)

    if log_file is None:
        log_dir = default_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        fd, log_path = tempfile.mkstemp(
            prefix=f"{log_stem(name)}_local_", suffix=".log", dir=log_dir
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
            env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
        )
    except Exception as e:
        logger.error(f"Error starting local process: {e}")
        raise
    finally:
        # The child holds its own copy of the descriptor.
        log_handle.close()

    logger.info(f"Local job {name} (pid {process.pid}) is logging to {log_path}")
    return LocalJob(process=process, model_name=name, log_file=log_path)
