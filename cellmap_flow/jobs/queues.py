"""Which GPU queues are actually usable, and the order to try them in.

gpu_queue_availability() feeds the dashboard's queue pickers; candidates()
is the order a job is tried on the queues it may run on, which start_hosts
and the finetune job manager submit in; queue_list() reads those queues
wherever they are given (a YAML, a command line, the saved settings, a
request).

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
"busy" number. Neither has a free GPU, however idle its GPUs are: only the
idle GPUs of hosts that are "ok" are counted as free.

Nothing here raises. A dashboard running somewhere without LSF should show a
queue picker that still works, just without the annotations.
"""

import logging
import re
import subprocess
import threading
import time

from cellmap_flow.jobs.site import current_site

logger = logging.getLogger(__name__)

# (queue, label, LSF host group backing it), for the queues offered; see
# SiteProfile.gpu_queues for why these.
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
    """{host: state} from ``bhosts -w``.

    ``-w`` is load-bearing: without it LSF collapses every closed_* state to a
    bare "closed", which loses the admin/full distinction that is the whole
    point of looking.
    """
    states = {}
    for line in (text or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 2:
            states[fields[0]] = fields[1]
    return states


def _count_hosts(states):
    """How many of ``states``' hosts are in each state that matters here."""
    counts = {"total": len(states), "ok": 0, "closed_admin": 0, "closed_full": 0, "other": 0}
    for status in states.values():
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
    """{host: (idle, total)} GPUs from ``bhosts -gpu``, idle being those no
    job holds.

    A host's first row names it and the rows of its other GPUs leave the
    name blank (they start with spaces), so each of those belongs to the
    last host named; the columns are counted from the right: ... NJOBS RUN
    SUSP RSV.
    """
    gpus = {}
    host = None
    for line in (text or "").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        if not line[0].isspace():
            host = fields[0]
        njobs = _int(fields[-4])
        if host is None or njobs is None:
            continue
        idle, total = gpus.get(host, (0, 0))
        gpus[host] = (idle + (njobs == 0), total + 1)
    return gpus


def _free_gpus(states, gpus):
    """(free, total) GPUs of a host group: free are the idle GPUs of the
    hosts whose state is "ok".

    An idle GPU on a closed host cannot take a job: closed_Adm is a node out
    for maintenance, closed_Full one whose job slots are all taken. Counting
    every idle GPU had the picker say "8/80 GPUs free" for gpu_h100 when all
    eight were on one closed_Adm node, so the queue looked freest and jobs
    were sent to it to wait out the pending timeout.
    """
    free = sum(idle for host, (idle, _) in gpus.items() if states.get(host) == "ok")
    return free, sum(total for _, total in gpus.values())


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
        states = _parse_bhosts_status(_run(["bhosts", "-w", host_group]))
        hosts = _count_hosts(states)
        gpus_free, gpus_total = _free_gpus(states, _parse_bhosts_gpu(_run(["bhosts", "-gpu", host_group])))
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


def queue_list(value):
    """The queues ``value`` names, as a list, each once, in its order.

    A job may run on several GPU queues (the dashboard's pickers tick them),
    so wherever a queue is read it is a list; one name is a list of one, so
    a YAML's ``queue: gpu_h200`` and a command line's ``--queue gpu_h200``
    read as they always did. A string may name several, separated by commas
    or spaces, as a text field holds them (the pipeline builder's Queue).
    None and "" name none: [].

    Raises:
        ValueError: ``value`` is neither a name nor a list of names.
    """
    if value is None:
        return []
    if isinstance(value, str):
        names = re.split(r"[\s,]+", value)
    elif isinstance(value, (list, tuple)) and all(isinstance(name, str) for name in value):
        names = [name.strip() for name in value]
    else:
        raise ValueError(f"A queue is a name or a list of names, not {value!r}")
    return list(dict.fromkeys(name for name in names if name))


def candidates(queues, cycle=True):
    """The order to try ``queues``, the queues a job may run on, in.

    ``queues`` is read by queue_list; none named is every GPU queue
    offered (the site's default_queues).

    With ``cycle=False`` the first of them is the only candidate: the job
    waits for it however long that takes, rather than being moved to
    whatever is free. Some work is pinned to a queue on purpose -- a
    benchmark that must run on one GPU model, or a charge group only valid
    on one queue -- and silently landing somewhere else is worse than
    waiting.

    Otherwise every one of them and no other, ordered by what LSF says is
    free right now: the most free GPUs first, ties broken on the shorter
    pending queue. So the job goes first to the queue likeliest to start it,
    and each fallback after a pending timeout is the next likeliest of the
    ones asked for; a queue that was not asked for is never tried, however
    free it is.

    A queue LSF says is not accepting work goes last: it takes submissions
    and never runs them, which is indistinguishable from a very slow job, so
    trying it before the others would cost a pending timeout of dead wait.
    It stays on the list -- a queue can reopen, and it was asked for -- just
    not ahead of queues that can run the job now. A queue LSF says nothing
    about (a YAML naming gpu_l4) is unknown, not closed: it comes after the
    ones known to accept work and before the closed ones.

    When LSF cannot be queried at all, ``queues`` are tried in their order.
    """
    queues = queue_list(queues) or current_site().default_queues

    if not cycle:
        logger.info(f"Queue cycling disabled; using {queues[0]} only, and waiting for it.")
        return queues[:1]

    try:
        info = gpu_queue_availability()
    except Exception as e:
        logger.debug(f"Could not read queue availability: {e}")
        info = {}

    if not info.get("available"):
        return queues

    known = {q["queue"]: q for q in info["queues"]}
    accepting = [known[q] for q in queues if q in known and known[q].get("accepting")]
    accepting.sort(key=lambda q: (-(q.get("gpus_free") or 0), q.get("pending") or 0))
    unknown = [q for q in queues if q not in known]
    closed = [q for q in queues if q in known and not known[q].get("accepting")]

    # The order is not arbitrary and the reason is worth seeing -- especially
    # now that these records reach the dashboard's log panel. A queue that was
    # skipped is more interesting than one that was kept.
    for q in info["queues"]:
        state = "not asked for" if q["queue"] not in queues else (
            "not accepting work, tried last" if not q.get("accepting") else "asked for"
        )
        logger.info(f"  {q['queue']}: {q.get('description') or 'no detail'} [{state}]")

    return [q["queue"] for q in accepting] + unknown + closed
