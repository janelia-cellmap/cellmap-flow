"""A record of every AI annotation: what was sent where, and what became of it.

Each event is one JSON line in ``<corrections_dir>/ai_annotate_log.jsonl``,
next to the volumes it concerns, so whoever reviews a dataset's annotations
can see which labels a hosted model drew, with which prompt, and that data
left the building for it. A line holds the time, the user, the event and
the fields the caller gives, nothing else: callers pass ids, the provider
and model, the plane and counts, never a credential. As a second guard each
line goes through ``secrets.redact`` before it is written.
"""

import getpass
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_NAME = "ai_annotate_log.jsonl"
EVENTS = ("requested", "staged", "failed", "resent", "accepted", "rejected")

# One line per write, from one thread at a time: two requests finishing
# together must not interleave their lines.
_lock = threading.Lock()


def log_path(corrections_dir) -> Path:
    """The audit log of the volumes in ``corrections_dir``."""
    return Path(corrections_dir) / LOG_NAME


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # no login name (a container without one): still record the event
        return "unknown"


def record(corrections_dir, event: str, **fields) -> None:
    """Append ``{"time", "user", "event", **fields}`` to the audit log.

    ``event`` is one of ``EVENTS``. Values that are not JSON (numpy numbers,
    paths) are written as their ``str``. Raises OSError when the log cannot
    be written: the caller decides whether that stops the action.
    """
    from cellmap_flow.ai_annotate.secrets import redact

    if event not in EVENTS:
        raise ValueError(f"Unknown audit event {event!r}; expected one of {', '.join(EVENTS)}")
    clash = {"time", "user", "event"} & set(fields)
    if clash:
        raise ValueError(f"Audit fields may not be named {', '.join(sorted(clash))}")
    entry = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "user": _user(),
        "event": event,
        **fields,
    }
    line = redact(json.dumps(entry, default=str, sort_keys=False)) + "\n"
    path = log_path(corrections_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(line)
