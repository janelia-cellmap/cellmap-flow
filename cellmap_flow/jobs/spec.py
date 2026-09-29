"""What a job is, whichever backend runs it.

``JobSpec`` says what to run and with what resources; a backend
(``jobs.lsf`` or ``jobs.local``) turns it into a running ``Job``. The rest
is shared by both: the statuses, the host marker a server prints, a log
file's tail, and a filesystem-safe stem for its name.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import urlparse

from cellmap_flow.utils.web_utils import IP_PATTERN

logger = logging.getLogger(__name__)

#: A template for the address viewers use for an inference server, for
#: servers that are reached through a reverse proxy. For example
#: ``https://proxy.example.org/inf-{port}``; ``{url}`` is the address the
#: server reported, ``{host}`` its host and ``{port}`` its port. Unset, the
#: reported address is used as it is.
SERVER_URL_TEMPLATE_ENV = "CELLMAP_FLOW_SERVER_URL_TEMPLATE"


@dataclass(frozen=True)
class JobSpec:
    """One job to submit.

    Exactly one of ``argv`` and ``shell`` is given. ``argv`` is run as it
    stands. ``shell`` is a command line run by ``bash -c``, which is what
    servers, finetune runs and blockwise workers have always been given; the
    finetune command needs it, since it sets LD_LIBRARY_PATH and pipes
    through tee.

    ``queue`` None asks for none (``-q`` is left out, so LSF's default
    queue), and ``gpus`` 0 asks for no GPU. ``walltime`` is LSF's
    ``[hours:]minutes``; None leaves the queue's default. ``log_dir`` None is
    ``bsub_utils.SERVER_LOG_DIR``, read when the job is submitted. ``env`` is
    added to this process's environment, which the job inherits either way.
    """

    name: str
    argv: Optional[Tuple[str, ...]] = None
    shell: Optional[str] = None
    queue: Optional[str] = None
    charge_group: Optional[str] = None
    gpus: int = 1
    cpus: int = 4
    walltime: Optional[str] = None
    log_dir: Optional[Path] = None
    env: Optional[Mapping[str, str]] = None

    def __post_init__(self):
        if (self.argv is None) == (self.shell is None):
            raise ValueError("A JobSpec needs exactly one of argv and shell")
        if self.argv is not None:
            argv = tuple(str(a) for a in self.argv)
            if not argv:
                raise ValueError("A JobSpec's argv cannot be empty")
            object.__setattr__(self, "argv", argv)


def default_log_dir() -> Path:
    """Where job logs go unless the spec says otherwise.

    ``bsub_utils.SERVER_LOG_DIR``, read at call time rather than copied: it
    is the one setting of it (tests and deployments point it elsewhere), and
    importing bsub_utils at module level here would pull in
    ``cellmap_flow.globals``.
    """
    from cellmap_flow.utils import bsub_utils

    return Path(bsub_utils.SERVER_LOG_DIR)


class JobStatus(Enum):
    """Enumeration of possible job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class JobStartError(RuntimeError):
    """A job was asked for and no usable one came of it."""


def tail(path: Path, max_chars: int = 4000) -> Optional[str]:
    """Read the tail of a log file, for surfacing crash output. Returns None if unreadable/empty.

    Seeks to the end rather than reading the whole file. An inference server
    log can run to hundreds of megabytes, and this is called from the polling
    loop in ``wait_for_host`` while the caller is blocked.

    Reads 4 bytes per requested character so a multi-byte sequence split at
    the seek boundary still leaves at least ``max_chars`` intact; the leading
    partial character decodes to a replacement char and is sliced off.
    """
    try:
        with Path(path).open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_chars * 4), os.SEEK_SET)
            raw = f.read()
    except OSError:
        return None
    content = raw.decode("utf-8", errors="replace").strip()
    if not content:
        return None
    return content[-max_chars:]


def log_stem(job_name: str) -> str:
    """A filesystem-safe stem for this job's log file.

    ``job_name`` arrives from model names, YAML and HuggingFace repo ids, so
    it can carry a path separator or ``..``. Interpolated straight into a
    path, that writes the log outside the log directory -- or, more quietly,
    makes the path we read back afterwards differ from the one we handed
    bsub, so a crashed job looks like it produced no output at all.

    Both the ``-o`` pattern and the path reconstructed after submission are
    built from this one value, so they cannot drift apart.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name).strip("._-")
    return stem or "job"


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
            # The newest marker: a requeued LSF job's output still holds its
            # earlier run's address, and the first marker is that dead one.
            tail = output.rsplit(IP_PATTERN[0], 1)[1]
            if IP_PATTERN[1] not in tail:
                return None  # its closing half is not written yet
            return tail.split(IP_PATTERN[1])[0]
    except (IndexError, AttributeError) as e:
        logger.debug(f"Could not extract host: {e}")

    return None


def public_server_url(url: Optional[str]) -> Optional[str]:
    """The address to use for a server that reported ``url``: ``url`` itself,
    or ``url`` put through $CELLMAP_FLOW_SERVER_URL_TEMPLATE when that is
    set (see SERVER_URL_TEMPLATE_ENV). Read at call time.
    """
    template = os.environ.get(SERVER_URL_TEMPLATE_ENV)
    if not template or not url:
        return url
    parsed = urlparse(url)
    try:
        public = template.format(url=url, host=parsed.hostname or "", port=parsed.port or "")
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Ignoring ${SERVER_URL_TEMPLATE_ENV}={template!r}: {e!r}")
        return url
    logger.info(f"Server at {url} is used as {public}")
    return public


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
