"""Report which GPU queues are actually usable, for the dashboard's picker.

Picking a queue blind is how a job ends up behind nine thousand pending ones:
at the time this was written ``gpu_h200`` had 9224 jobs pending while
``gpu_h100`` had none, and nothing in the UI said so. LSF already knows --
``bqueues`` for the backlog, ``bhosts`` for whether the nodes behind the queue
are up at all -- so ask it, cache the answer, and let the UI show it next to
each option.

Host state matters separately from queue state. A queue reports ``Open:Active``
while its nodes are individually closed: ``closed_Adm`` when an admin has taken
them out for maintenance, ``closed_Full`` when they are merely saturated. Those
mean different things to someone deciding where to submit -- the first will not
clear on its own -- so they are counted separately rather than lumped into one
"busy" number.

Nothing here raises. A dashboard running somewhere without LSF should show a
queue picker that still works, just without the annotations.
"""

import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# The only GPU types we offer. Janelia has others (gpu_l4, gpu_rtx8000), but
# they are not what anyone wants for inference or finetuning here, and a
# shorter list is easier to choose from than a complete one.
GPU_QUEUES = (
    # (queue, label, LSF host group backing it)
    ("gpu_h100", "H100", "h100s"),
    ("gpu_h200", "H200", "h200s"),
    ("gpu_a100", "A100", "a100s"),
)

CACHE_TTL_SECONDS = 60
_COMMAND_TIMEOUT_SECONDS = 15

_cache = {"fetched_at": 0.0, "payload": None}
_cache_lock = threading.Lock()


def _run(args):
    """Run an LSF query, returning stdout or None. Never raises."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=_COMMAND_TIMEOUT_SECONDS
        )
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"{args[0]} failed: {e}")
        return None
    if result.returncode != 0:
        logger.debug(f"{' '.join(args)} exited {result.returncode}: {result.stderr}")
        return None
    return result.stdout


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_bqueues(text):
    """{queue: {status, pending, running}} from ``bqueues`` tabular output.

    Columns are NAME PRIO STATUS MAX JL/U JL/P JL/H NJOBS PEND RUN SUSP.
    """
    queues = {}
    for line in (text or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 11:
            continue
        queues[fields[0]] = {
            "status": fields[2],
            "pending": _int(fields[8]),
            "running": _int(fields[9]),
        }
    return queues


def _parse_bhosts_status(text):
    """Count hosts by state from ``bhosts -w``.

    ``-w`` is load-bearing: without it LSF collapses every closed_* state to a
    bare "closed", which loses the admin/full distinction that is the whole
    point of looking.
    """
    counts = {"total": 0, "ok": 0, "closed_admin": 0, "closed_full": 0, "other": 0}
    for line in (text or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        status = fields[1]
        counts["total"] += 1
        if status == "ok":
            counts["ok"] += 1
        elif status == "closed_Adm":
            counts["closed_admin"] += 1
        elif status == "closed_Full":
            counts["closed_full"] += 1
        else:
            # unavail, unreach, closed_LIM, closed_Excl, ...
            counts["other"] += 1
    return counts


def _parse_bhosts_gpu(text):
    """(free, total) GPUs from ``bhosts -gpu``.

    Rows after the first for a given host omit the host name, so columns are
    counted from the right: ... NJOBS RUN SUSP RSV.
    """
    free = total = 0
    for line in (text or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        njobs = _int(fields[-4])
        if njobs is None:
            continue
        total += 1
        if njobs == 0:
            free += 1
    return free, total


def _describe(entry):
    """A short human summary, so the UI doesn't have to build one."""
    if not entry["open"]:
        return f"closed ({entry['status']})"
    parts = []
    if entry["gpus_total"]:
        parts.append(f"{entry['gpus_free']}/{entry['gpus_total']} GPUs free")
    if entry["pending"] is not None:
        parts.append(f"{entry['pending']} queued")
    down = entry["hosts_closed_admin"]
    if down:
        parts.append(f"{down} node{'' if down == 1 else 's'} down for maintenance")
    return ", ".join(parts) if parts else "available"


def _collect():
    bqueues_out = _run(["bqueues"] + [q for q, _, _ in GPU_QUEUES])
    if bqueues_out is None:
        return {
            "available": False,
            "reason": "LSF is not reachable from the dashboard host",
            "queues": [
                {"queue": q, "label": label, "open": True, "description": ""}
                for q, label, _ in GPU_QUEUES
            ],
        }

    parsed = _parse_bqueues(bqueues_out)
    queues = []
    for queue, label, host_group in GPU_QUEUES:
        info = parsed.get(queue, {})
        status = info.get("status", "unknown")
        hosts = _parse_bhosts_status(_run(["bhosts", "-w", host_group]))
        gpus_free, gpus_total = _parse_bhosts_gpu(_run(["bhosts", "-gpu", host_group]))
        entry = {
            "queue": queue,
            "label": label,
            "status": status,
            # "Open:Active" is the only state that actually accepts work;
            # "Open:Inact" takes submissions and never starts them.
            "open": status.startswith("Open"),
            "accepting": status == "Open:Active",
            "pending": info.get("pending"),
            "running": info.get("running"),
            "hosts_total": hosts["total"],
            "hosts_ok": hosts["ok"],
            "hosts_closed_admin": hosts["closed_admin"],
            "hosts_closed_full": hosts["closed_full"],
            "gpus_free": gpus_free,
            "gpus_total": gpus_total,
        }
        entry["description"] = _describe(entry)
        queues.append(entry)

    return {"available": True, "queues": queues}


def gpu_queue_availability(force=False):
    """Current GPU queue availability, cached for ``CACHE_TTL_SECONDS``.

    The UI polls this about once a minute; the cache means several open
    dashboards don't multiply into an LSF query storm, since the answer is the
    same for all of them.
    """
    now = time.time()
    with _cache_lock:
        fresh = (
            _cache["payload"] is not None
            and not force
            and now - _cache["fetched_at"] < CACHE_TTL_SECONDS
        )
        if fresh:
            payload = dict(_cache["payload"])
            payload["age_seconds"] = round(now - _cache["fetched_at"], 1)
            return payload

    payload = _collect()

    with _cache_lock:
        _cache["payload"] = payload
        _cache["fetched_at"] = time.time()

    payload = dict(payload)
    payload["age_seconds"] = 0.0
    return payload
