"""The file a server writes once it knows its own address.

A launcher learns where a server is from the ``CELLMAP_FLOW_SERVER_IP(...)``
marker the server prints. On LSF that means polling bpeek, and every bpeek
is a request to mbatchd for the job's whole output so far. A launcher that
sets ``READY_ENV`` in the job's environment is told instead: the server
writes ``{"url", "host", "pid", "job_id"}`` to that path as it prints the
marker, and the launcher reads the file before it asks LSF anything.

It is an environment variable, not a command-line flag, so that a server
from before this existed simply ignores it. The launcher then finds no file
and reads the marker through bpeek, as it always did.

The file says what the marker says: the server knows its URL. It is written
just before the server binds its port, not after, so it does not promise
that the first request will be answered.
"""

import json
import logging
import os
import secrets
import socket
from pathlib import Path
from typing import Optional

from cellmap_flow.jobs.spec import exists_now, log_stem

logger = logging.getLogger(__name__)

READY_ENV = "CELLMAP_FLOW_READY_FILE"


def ready_path(log_dir, job_name: str) -> Path:
    """A fresh path for one submission of ``job_name`` to write its address to.

    Different on every call: a job resubmitted on another queue, or the same
    model launched twice, must never read an earlier job's address.
    """
    return Path(log_dir) / f"{log_stem(job_name)}_{secrets.token_hex(4)}.ready"


def write_ready_file(url: str) -> Optional[Path]:
    """Write ``url`` where the launcher asked, if it asked. Never raises.

    Written to a temporary name and renamed into place, so a reader sees
    either no file or a whole one. Returns the path written, or None when
    ``READY_ENV`` is not set or the file could not be written; the printed
    marker still says the same thing.
    """
    target = os.environ.get(READY_ENV)
    if not target:
        return None
    payload = {"url": url, "host": socket.gethostname(), "pid": os.getpid()}
    # LSF sets this in every job. It lets the launcher tell this job's file
    # from one some other job left at the same path.
    if os.environ.get("LSB_JOBID"):
        payload["job_id"] = os.environ["LSB_JOBID"]
    path = Path(target)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"Could not write the ready file {path}: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    return path


def read_ready_file(path, job_id: Optional[str] = None) -> Optional[str]:
    """The URL in a ready file, or None if there is no usable one yet.

    With ``job_id``, a file written by a different LSF job is not usable.
    """
    if not exists_now(path):
        return None
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    url = data.get("url")
    if not isinstance(url, str) or not url:
        return None
    written_by = data.get("job_id")
    if job_id is not None and written_by is not None and str(written_by) != str(job_id):
        return None
    return url
