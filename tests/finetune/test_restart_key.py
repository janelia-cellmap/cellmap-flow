"""Shift+R over the viewer (routes/finetune/restart_key.py): which presses count, how the count
reaches the Finetune tab, and when the status bar says something.

A press is made as neuroglancer makes it, by invoking the viewer's action.
The page's side (job-monitor.js sending the restart when the count goes up)
is not run here; the restart route is, as the page would call it. Timers do
not run by themselves: ``keyed.timers`` holds them, and a test fires them.
What the status bar says is restart_key's own wording; these tests check
only whether it says something, and when that changes.
"""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.routes.finetune import restart_key
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.viewer import keys


@pytest.fixture
def keyed(dashboard, viewer, make_job, tmp_path, monkeypatch):
    """The dashboard with a viewer and a job manager holding job "j" (RUNNING; ``keyed.job.status``
    changes it), its MinIO sync and the trainer's restart faked, and the page loaded (it has asked
    for the job list, which binds the key). ``press()`` presses Shift+R and returns the status
    bar's message; ``status()`` is j's status answer (which makes it the job the tab follows);
    ``timers`` the timers set, as (seconds, function, args), and ``fire(seconds)`` fires those."""
    from cellmap_flow.dashboard.routes.finetune import common, training
    from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager

    monkeypatch.setattr(common, "USER_PREFS_FILE", str(tmp_path / "user_prefs.json"))
    manager = FinetuneJobManager()
    job = make_job()
    manager.jobs[job.job_id] = job
    monkeypatch.setattr(get_session(), "finetune_job_manager", manager)
    monkeypatch.setattr(training, "sync_all_annotations_from_minio", lambda force=True: 0)
    monkeypatch.setattr(manager, "restart_finetuning_job", lambda job_id, updated_params: job)
    world = SimpleNamespace(job=job, timers=[], dashboard=dashboard)
    monkeypatch.setattr(keys, "later", lambda seconds, function, *args: world.timers.append(
        (seconds, function, args)))

    def message():
        return get_session().viewer.config_state.state.status_messages.get(restart_key.STATUS_KEY)

    def press():
        get_session().viewer.actions.invoke(restart_key.ACTION, {})
        return message()

    def status():
        answer = dashboard.get(f"/api/finetune/job/{job.job_id}/status")
        assert answer.status_code == 200, answer.get_json()
        return answer.get_json()

    def fire(seconds):
        """Fire the timers set for ``seconds``, as they would fire then."""
        due = [timer for timer in world.timers if timer[0] == seconds]
        world.timers = [timer for timer in world.timers if timer[0] != seconds]
        for _, function, args in due:
            function(*args)

    world.message, world.press, world.status, world.fire = message, press, status, fire
    assert dashboard.get("/api/finetune/jobs").get_json()["success"]
    return world


def _waiting(keyed):
    from cellmap_flow.finetune.job_manager.state import JobStatus

    keyed.job.status = JobStatus.WAITING_FOR_RESTART
    keyed.status()


def test_the_job_list_binds_shift_r_once_per_viewer(keyed):
    """The tab asks for the list on every page load, so the key is there before any job is, and a
    new dataset's viewer (the page reloads after it) gets it too."""
    from neuroglancer.viewer_base import ViewerBase

    first = get_session().viewer
    assert keyed.dashboard.get("/api/finetune/jobs").get_json()["success"]  # loaded again
    assert first.config_state.state.input_event_bindings.viewer["shift+keyr"] == restart_key.ACTION
    assert first.actions._action_handlers[restart_key.ACTION] == {restart_key._on_restart_key}

    second = get_session().viewer = ViewerBase()
    keyed.status()  # a viewer made while the tab polls is bound by the poll
    assert second.config_state.state.input_event_bindings.viewer["shift+keyr"] == restart_key.ACTION
    assert second.actions._action_handlers[restart_key.ACTION] == {restart_key._on_restart_key}


@pytest.mark.parametrize("serving", [
    pytest.param(False, id="waiting for a restart"),
    # state.can_restart's rule, the one the card offers Restart by: a trainer
    # from before the WAITING_FOR_RESTART marker waits while RUNNING with its server up.
    pytest.param(True, id="running with its server up"),
])
def test_a_press_counts_against_a_job_that_can_take_a_restart(keyed, serving):
    """The count is in the job's status answer, which the tab polls, and in the job list."""
    if serving:
        keyed.job.inference_server_ready = True
        keyed.status()
    else:
        _waiting(keyed)
    assert keyed.status()["restart_requests"] == 0

    assert keyed.press() is not None

    assert keyed.status()["restart_requests"] == 1
    listed = keyed.dashboard.get("/api/finetune/jobs").get_json()["jobs"]
    assert [(job["job_id"], job["restart_requests"]) for job in listed] == [("j", 1)]


@pytest.mark.parametrize("status, followed", [
    pytest.param(None, False, id="the tab follows no job"),
    pytest.param("RUNNING", True, id="still training"),
    pytest.param("COMPLETED", True, id="finished"),
])
def test_a_press_for_a_job_that_cannot_take_a_restart_says_so_and_counts_nothing(keyed, status, followed):
    from cellmap_flow.finetune.job_manager.state import JobStatus

    if status is not None:
        keyed.job.status = JobStatus[status]
    if followed:
        keyed.status()

    assert keyed.press() is not None
    assert get_session().restart_key["requests"] == {}

    # It goes after a while, unless something else has been said since.
    keyed.fire(restart_key.SHOW_S)
    assert keyed.message() is None


def test_a_press_while_a_restart_is_pending_counts_nothing(keyed, monkeypatch):
    """A press is pending for PENDING_S, and for as long as the restart route that took it runs
    (its MinIO sync can take longer than PENDING_S). A press nothing answered for PENDING_S is
    taken as lost: the next one counts."""
    _waiting(keyed)
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(restart_key.time, "monotonic", lambda: clock.now)
    pressed = keyed.press()

    clock.now += restart_key.PENDING_S - 1
    assert keyed.press() not in (None, pressed)  # saying it is ignored
    assert keyed.status()["restart_requests"] == 1

    clock.now += 2
    keyed.press()
    assert keyed.status()["restart_requests"] == 2

    during = []

    def slow_restart(job_id, updated_params):
        clock.now += 2 * restart_key.PENDING_S
        before = keyed.message()
        keyed.fire(restart_key.PENDING_S)  # the press is not taken as lost meanwhile
        during.append(keyed.message() == before)
        keyed.press()
        return keyed.job

    monkeypatch.setattr(get_session().finetune_job_manager, "restart_finetuning_job", slow_restart)
    assert keyed.dashboard.post("/api/finetune/job/j/restart", json={}).get_json()["success"]
    assert during == [True]
    assert keyed.status()["restart_requests"] == 2


@pytest.mark.parametrize("form, code", [
    pytest.param({"num_epochs": 5}, 200, id="restarted"),
    pytest.param({"rehearsal_fraction": "abc"}, 400, id="refused"),
])
def test_the_status_bar_says_how_the_restart_answered_a_press_then_clears(keyed, form, code):
    _waiting(keyed)
    pressed = keyed.press()

    assert keyed.dashboard.post("/api/finetune/job/j/restart", json=form).status_code == code

    answered = keyed.message()
    assert answered not in (None, pressed)
    assert get_session().restart_key["asked"] == {}
    # The press is answered, so its PENDING_S check finds nothing to say.
    keyed.fire(restart_key.PENDING_S)
    assert keyed.message() == answered
    keyed.fire(restart_key.SHOW_S)
    assert keyed.message() is None

    # The button's restart, with no key pressed: the viewer is not told.
    assert keyed.dashboard.post("/api/finetune/job/j/restart", json={}).get_json()["success"]
    assert keyed.message() is None


def test_a_press_no_restart_answers_says_so(keyed):
    """The dashboard's page is closed, or it refused (its form held a bad value, which it says)."""
    _waiting(keyed)
    pressed = keyed.press()

    keyed.fire(restart_key.PENDING_S)

    assert keyed.message() not in (None, pressed)
    keyed.press()
    assert keyed.status()["restart_requests"] == 2
