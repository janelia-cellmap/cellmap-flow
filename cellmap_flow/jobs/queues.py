"""Which GPU queues are actually usable, and the order to try them in.

gpu_queue_availability() feeds the dashboard's queue picker; candidates()
is the order start_hosts submits in when queue cycling is on.

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

from cellmap_flow.jobs.site import current_site

logger = logging.getLogger(__name__)

# (queue, label, LSF host group backing it), for the queues offered; see
# SiteProfile.gpu_queues for why these three.
GPU_QUEUES = current_site().gpu_queues

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


def _query_bqueues():
    """{queue: {status, pending, running}} for GPU_QUEUES, or None if LSF is unreachable.

    One call for all of them first. bqueues exits non-zero when any named
    queue does not exist, so a single retired or renamed queue used to make
    the whole picker say LSF was unreachable; on failure, ask per queue and
    keep whichever answer.
    """
    names = [q for q, _, _ in GPU_QUEUES]
    out = _run(["bqueues"] + names)
    if out is not None:
        return _parse_bqueues(out)
    parsed = {}
    answered = False
    for name in names:
        out = _run(["bqueues", name])
        if out is None:
            continue
        answered = True
        parsed.update(_parse_bqueues(out))
    return parsed if answered else None


def _collect():
    parsed = _query_bqueues()
    if parsed is None:
        return {
            "available": False,
            "reason": "LSF is not reachable from the dashboard host",
            "queues": [
                {"queue": q, "label": label, "open": True, "description": ""}
                for q, label, _ in GPU_QUEUES
            ],
        }

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


def candidates(preferred, cycle=True):
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
    first = [preferred] if preferred else []

    if not cycle:
        logger.info(
            f"Queue cycling disabled; using {preferred or 'the default queue'} "
            f"only, and waiting for it."
        )
        return first
    all_gpu = [q for q, _, _ in GPU_QUEUES]

    try:
        info = gpu_queue_availability()
    except Exception as e:
        logger.debug(f"Could not read queue availability: {e}")
        info = {}

    if not info.get("available"):
        return first + [q for q in all_gpu if q != preferred]

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
    # rather than first. Trying it first costs pending_fallback_seconds of
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

    return first + [q["queue"] for q in others]

