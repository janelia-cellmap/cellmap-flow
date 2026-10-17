"""Shift+R over the viewer: restart the finetune job the Finetune tab shows, with the tab's settings.

It does what the Training Status card's Restart button does, without the
button's confirm (pressing the key is a deliberate act), from the viewer,
where the user paints. The form's values are in the page, not here, so the
key does not restart the job itself. Its action checks the job and counts
the press against it; the job's status answer, which the Finetune tab polls
every 3 s while it follows a job (job-monitor.js's startStatusPolling),
carries the count as ``restart_requests``; and the tab, seeing the count go
up, sends the restart as Restart does. So the key acts within one poll, up
to 3 s, and the restart route then syncs MinIO as for the button.

The job is the one the tab follows, the last whose status it asked for,
which is the one its card shows. The key counts a press only for a job that
can take a restart (state.can_restart, which is when the card offers
Restart), and otherwise says why not, in neuroglancer's status bar, as
ai_annotate's Shift+G does: there is no job, or it is still running, or it
has finished. A press while an earlier one waits for its restart is ignored,
saying so, as the button is disabled while its request runs.

The status bar says what will happen at the press; then what the restart
route answered (``taken`` as it starts, ``answered`` once it has), or, when
no restart came within PENDING_S (the dashboard's page is closed, or it
refused, saying why in the tab), that none did. Each of those goes after
SHOW_S.

Shift+R pressed with the focus in the dashboard's page, and the browser
mode's key (web/src/backend/routes/finetune.ts binds it in the page), go
straight to job-monitor.js, which says why not in the tab itself.
"""

import math
import threading
import time

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.job_manager.state import can_restart
from cellmap_flow.viewer import keys

# The neuroglancer action and the key bound to it ("shift+keyr" in
# neuroglancer's spelling; KEYBINDING is how messages name it). Neuroglancer
# binds every Shift+letter to a layer's tool of that letter; the paint layer's
# tools are A and F, and ai_annotate's Shift+G takes G the same way.
ACTION = "finetune-restart"
KEY = "shift+keyr"
KEYBINDING = "Shift+R"

# The key this module's messages use in neuroglancer's status bar.
STATUS_KEY = "finetune"

# How long a press waits for the tab's restart before it is taken as lost: a
# few of the tab's 3 s polls, and the restart route's MinIO sync. A press
# after that counts again.
PENDING_S = 15.0
# How long an answer, a refusal or a lost press stays in the status bar.
SHOW_S = 10.0
# When a press was made, once the restart route has taken it: it is on its way
# for as long as the route runs (a MinIO sync may outlast PENDING_S), and
# never lost.
TAKEN = math.inf

NO_JOB = "Finetune: no job to restart. Start one on the Finetune tab."
NOT_WAITING = f"Finetune: {KEYBINDING} restarts a job that is waiting for a restart; {{job_id}} is {{status}}."
ALREADY_ASKED = "Finetune: the restart of {job_id} is already on its way; this press is ignored."
RESTARTING = "Finetune: restarting {job_id} with the Finetune tab's settings..."
NOT_RESTARTED = "Finetune: {job_id} was not restarted; the Finetune tab says why (is the dashboard open?)."


def _state():
    """The session's Shift+R state (``Session.restart_key``)."""
    return get_session().restart_key


def _later(seconds, function, *args):
    """Run ``function(*args)`` in ``seconds``, on a daemon timer. Tests replace this."""
    timer = threading.Timer(seconds, function, args)
    timer.daemon = True
    timer.start()


def _say(viewer, message):
    """Show ``message`` in neuroglancer's status bar (None clears it); never raises."""
    keys.say(viewer, STATUS_KEY, message)


def _show(viewer, message):
    """Show ``message`` in the status bar for SHOW_S, unless another replaces it sooner."""
    keys.say_for(viewer, STATUS_KEY, message, SHOW_S)


def ensure_binding():
    """Bind Shift+R to the restart action in the session's viewer, once per viewer.

    As ai_annotate's Shift+G (viewer.keys.bind_key): a new dataset makes a
    new viewer, which is bound the first time one of the routes that call
    this answers after it. They are the job list, which the Finetune tab
    asks for on every page load (and connect.js reloads the page after a new
    dataset), and a job's status, which it polls while it follows a job.
    """
    keys.bind_key(get_session().viewer, ACTION, KEY, _on_restart_key)


def follow(job_id):
    """The tab asked for ``job_id``'s status: that is the job its card shows, which Shift+R restarts."""
    _state()["followed"] = job_id


def with_requests(status):
    """A job's status answer with ``restart_requests``: the presses that asked for its restart."""
    return {**status, "restart_requests": _state()["requests"].get(status["job_id"], 0)}


def _on_restart_key(action_state):
    """The Shift+R action: count the press against the job the tab follows, saying so.

    Runs on neuroglancer's event loop, which every chunk request also waits
    on. Unlike ai_annotate's Shift+G it needs no thread: it only looks at
    the job manager's records in memory and sets a status message, which
    neuroglancer's own examples do from an action. The restart itself, which
    syncs MinIO, is the tab's request.
    """
    viewer = get_session().viewer
    message, press = _press()
    if press is None:
        _show(viewer, message)
        return
    # Until the restart route answers it (answered), or PENDING_S passes.
    _say(viewer, message)
    _later(PENDING_S, _unanswered, *press)


def _press():
    """Count a press of Shift+R, if the job the tab follows can take a restart.

    Returns the status bar's message and, for a press that counts, the job
    id and when it was made; else None.
    """
    state = _state()
    jobs = getattr(get_session().finetune_job_manager, "jobs", None) or {}
    with state["lock"]:
        job_id = state["followed"]
        job = jobs.get(job_id) if job_id is not None else None
        if job is None:
            return NO_JOB, None
        if not can_restart(job):
            return NOT_WAITING.format(job_id=job_id, status=job.status.value), None
        now = time.monotonic()
        asked_at = state["asked"].get(job_id)
        if asked_at is not None and now - asked_at < PENDING_S:
            return ALREADY_ASKED.format(job_id=job_id), None
        state["asked"][job_id] = now
        state["requests"][job_id] = state["requests"].get(job_id, 0) + 1
        return RESTARTING.format(job_id=job_id), (job_id, now)


def _unanswered(job_id, asked_at):
    """PENDING_S after a press: if no restart has answered it, say that none came."""
    state = _state()
    with state["lock"]:
        if state["asked"].get(job_id) != asked_at:
            return
        del state["asked"][job_id]
    _show(get_session().viewer, NOT_RESTARTED.format(job_id=job_id))


def taken(job_id):
    """The restart route has started on ``job_id``'s restart: a press waiting on it is no longer lost."""
    state = _state()
    with state["lock"]:
        if job_id in state["asked"]:
            state["asked"][job_id] = TAKEN


def answered(job_id, success, message):
    """The restart route answered for ``job_id``: if a press waits on it, say how in the status bar.

    A restart from the button answers a press too: the job was restarted,
    which is what the press asked for, and the tab, seeing the count go up
    while the job runs, does not send another.
    """
    state = _state()
    with state["lock"]:
        if state["asked"].pop(job_id, None) is None:
            return
    _show(get_session().viewer,
          f"Finetune: {message}" if success else f"Finetune: could not restart {job_id}: {message}")
