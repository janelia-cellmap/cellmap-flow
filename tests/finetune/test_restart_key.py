"""Shift+R over the viewer (routes/finetune/restart_key.py): what a press counts and says, and how
the count reaches the Finetune tab.

A press is made as neuroglancer makes it, by invoking the viewer's action.
The page's side (job-monitor.js sending the restart when the count goes up)
is not run here; the restart route is, as the page would call it. Timers do
not run by themselves: ``keyed.timers`` holds them, and a test fires them.
"""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.routes.finetune import restart_key
from cellmap_flow.dashboard.state import get_session

RESTARTED = "Restart request sent. No new annotations to pull; training will restart on the same GPU."


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
    monkeypatch.setattr(restart_key, "_later", lambda seconds, function, *args: world.timers.append(
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
    assert get_session().restart_key["bound_for"] is second


def test_a_press_counts_against_the_waiting_job_and_its_status_carries_the_count(keyed):
    _waiting(keyed)
    assert keyed.status()["restart_requests"] == 0

    assert keyed.press() == "Finetune: restarting j with the Finetune tab's settings..."

    assert keyed.status()["restart_requests"] == 1
    listed = keyed.dashboard.get("/api/finetune/jobs").get_json()["jobs"]
    assert [(job["job_id"], job["restart_requests"]) for job in listed] == [("j", 1)]


@pytest.mark.parametrize("status, followed, message", [
    pytest.param(None, False, "Finetune: no job to restart. Start one on the Finetune tab.",
                 id="the tab follows no job"),
    pytest.param("RUNNING", True,
                 "Finetune: Shift+R restarts a job that is waiting for a restart; j is RUNNING.",
                 id="still training"),
    pytest.param("COMPLETED", True,
                 "Finetune: Shift+R restarts a job that is waiting for a restart; j is COMPLETED.",
                 id="finished"),
])
def test_a_press_for_a_job_that_cannot_take_a_restart_says_why_and_counts_nothing(
        keyed, status, followed, message):
    from cellmap_flow.finetune.job_manager.state import JobStatus

    if status is not None:
        keyed.job.status = JobStatus[status]
    if followed:
        keyed.status()

    assert keyed.press() == message
    assert get_session().restart_key["requests"] == {}

    # It goes after a while, unless something else has been said since.
    keyed.fire(restart_key.SHOW_S)
    assert keyed.message() is None


def test_a_running_job_whose_server_is_up_takes_a_press(keyed):
    """state.can_restart's rule, the one the card offers Restart by: a trainer from before the
    WAITING_FOR_RESTART marker waits for a restart while RUNNING with its server up."""
    keyed.job.inference_server_ready = True
    keyed.status()

    assert keyed.press() == "Finetune: restarting j with the Finetune tab's settings..."


def test_a_second_press_before_the_restart_is_ignored_saying_so(keyed, monkeypatch):
    _waiting(keyed)
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(restart_key.time, "monotonic", lambda: clock.now)
    keyed.press()

    clock.now += restart_key.PENDING_S - 1
    assert keyed.press() == "Finetune: the restart of j is already on its way; this press is ignored."
    assert keyed.status()["restart_requests"] == 1

    # A press nothing answered for PENDING_S is taken as lost: the next one counts.
    clock.now += 2
    assert keyed.press() == "Finetune: restarting j with the Finetune tab's settings..."
    assert keyed.status()["restart_requests"] == 2


def test_the_restart_the_tab_sends_is_said_in_the_status_bar(keyed):
    _waiting(keyed)
    keyed.press()

    answer = keyed.dashboard.post("/api/finetune/job/j/restart", json={"num_epochs": 5})

    assert answer.get_json()["success"]
    assert keyed.message() == f"Finetune: {RESTARTED}"
    assert get_session().restart_key["asked"] == {}
    # The press is answered, so its PENDING_S check finds nothing to say.
    keyed.fire(restart_key.PENDING_S)
    assert keyed.message() == f"Finetune: {RESTARTED}"
    keyed.fire(restart_key.SHOW_S)
    assert keyed.message() is None


def test_a_restart_that_outlasts_the_wait_still_answers_the_press(keyed, monkeypatch):
    """The route's MinIO sync can take longer than PENDING_S: the press is not lost meanwhile, and
    another is still ignored."""
    _waiting(keyed)
    keyed.press()
    during = []

    def slow_restart(job_id, updated_params):
        keyed.fire(restart_key.PENDING_S)
        during.append((keyed.message(), keyed.press()))
        return keyed.job

    monkeypatch.setattr(get_session().finetune_job_manager, "restart_finetuning_job", slow_restart)
    assert keyed.dashboard.post("/api/finetune/job/j/restart", json={}).get_json()["success"]

    assert during == [("Finetune: restarting j with the Finetune tab's settings...",
                       "Finetune: the restart of j is already on its way; this press is ignored.")]
    assert keyed.message() == f"Finetune: {RESTARTED}"


def test_a_refused_restart_is_said_in_the_status_bar(keyed):
    _waiting(keyed)
    keyed.press()

    answer = keyed.dashboard.post("/api/finetune/job/j/restart", json={"rehearsal_fraction": "abc"})

    assert answer.status_code == 400
    assert keyed.message() == "Finetune: could not restart j: rehearsal_fraction must be a number between 0 and 1"


def test_a_restart_without_a_press_leaves_the_status_bar_alone(keyed):
    """The button's restart, with no key pressed: the viewer is not told."""
    _waiting(keyed)

    assert keyed.dashboard.post("/api/finetune/job/j/restart", json={}).get_json()["success"]
    assert keyed.message() is None


def test_a_press_no_restart_answers_says_so(keyed):
    """The dashboard's page is closed, or it refused (its form held a bad value, which it says)."""
    _waiting(keyed)
    keyed.press()

    keyed.fire(restart_key.PENDING_S)

    assert keyed.message() == ("Finetune: j was not restarted; the Finetune tab says why "
                               "(is the dashboard open?).")
    assert keyed.press() == "Finetune: restarting j with the Finetune tab's settings..."
    assert keyed.status()["restart_requests"] == 2
