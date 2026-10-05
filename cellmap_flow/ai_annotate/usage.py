"""A per-user daily cap on hosted-model calls, kept across dashboard restarts.

Each call to a hosted model costs money, and a stuck key or a runaway loop
in the browser could make hundreds. The config's ``daily_call_limit`` caps
the calls one user makes in a local calendar day. The count lives in the
user's home (``~/.cellmap_flow/ai_annotate_usage.json``), not in the session,
so restarting the dashboard does not reset it, and two dashboards the same
user runs share it.

A call is counted before it is made, so a failed call counts too: it may
still have been billed.
"""

import contextlib
import json
import logging
import os
import tempfile
import threading
from datetime import date
from pathlib import Path

from cellmap_flow.ai_annotate.errors import AIAnnotateError

try:
    import fcntl
except ImportError:  # Not on Windows; the in-process lock still applies.
    fcntl = None

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def usage_path():
    """``~/.cellmap_flow/ai_annotate_usage.json``."""
    return Path.home() / ".cellmap_flow" / "ai_annotate_usage.json"


@contextlib.contextmanager
def _locked(path):
    """Hold the count's lock: a thread lock for this process, and an
    exclusive ``flock`` on a sidecar file for other dashboards of the same user."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if fcntl is None:
            yield
            return
        with open(path.with_name(path.name + ".lock"), "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _read(path, today):
    """Calls recorded for ``today``; 0 when the file is missing, from another
    day, or unreadable (an unreadable file is logged, then started afresh)."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return 0
    except (OSError, ValueError):
        logger.warning("AI-annotate usage file %s is unreadable; starting today's count from 0", path)
        return 0
    if not isinstance(data, dict) or data.get("date") != today.isoformat():
        return 0
    calls = data.get("calls")
    return calls if isinstance(calls, int) and calls >= 0 else 0


def _write(path, today, calls):
    """Write the count atomically: to a temporary file in the same directory,
    then renamed over the old one, so a crash mid-write leaves the old count
    rather than a truncated file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump({"date": today.isoformat(), "calls": calls}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def check_and_count(limit, *, path=None, today=None):
    """Count one call against today's ``limit`` and return the calls used, this one included.

    Raises ``AIAnnotateError("limit")`` without counting when today's calls
    already reached the limit. ``path`` and ``today`` are for tests.
    """
    path = Path(path) if path is not None else usage_path()
    today = today or date.today()
    with _locked(path):
        used = _read(path, today)
        if used >= limit:
            raise AIAnnotateError(
                "limit",
                f"The daily limit of {limit} AI-annotate calls is used up; it resets at midnight. "
                "Raise daily_call_limit in the AI-annotate config to allow more.",
            )
        _write(path, today, used + 1)
    return used + 1


def calls_today(*, path=None, today=None):
    """Calls counted so far today."""
    path = Path(path) if path is not None else usage_path()
    today = today or date.today()
    return _read(path, today)
