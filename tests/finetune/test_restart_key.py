"""Shift+R over the viewer (routes/finetune/restart_key.py): a press counts against the job the
Finetune tab follows, and the count reaches the tab.

A press is made as neuroglancer makes it, by invoking the viewer's action. The page's side
(job-monitor.js sending the restart when the count goes up) is not run here. Timers do not run.
"""

from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.routes.finetune import restart_key
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.viewer import keys


@pytest.fixture
def keyed(dashboard, viewer, make_job, tmp_path, monkeypatch):
    """The dashboard with a viewer and a job manager holding job "j" (RUNNING; ``keyed.job.status``
    changes it), and the page loaded (it has asked for the job list, which binds the key).
    ``press()`` presses Shift+R and returns the status bar's message; ``status()`` is j's status
    answer (which makes it the job the tab follows)."""
    from cellmap_flow.dashboard.routes.finetune import common
    from cellmap_flow.finetune.job_manager.manager import FinetuneJobManager

    monkeypatch.setattr(common, "USER_PREFS_FILE", str(tmp_path / "user_prefs.json"))
    manager = FinetuneJobManager()
    job = make_job()
    manager.jobs[job.job_id] = job
    monkeypatch.setattr(get_session(), "finetune_job_manager", manager)
    monkeypatch.setattr(keys, "later", lambda seconds, function, *args: None)
    world = SimpleNamespace(job=job, dashboard=dashboard)

    def press():
        get_session().viewer.actions.invoke(restart_key.ACTION, {})
        return get_session().viewer.config_state.state.status_messages.get(restart_key.STATUS_KEY)

    def status():
        answer = dashboard.get(f"/api/finetune/job/{job.job_id}/status")
        assert answer.status_code == 200, answer.get_json()
        return answer.get_json()

    world.press, world.status = press, status
    assert dashboard.get("/api/finetune/jobs").get_json()["success"]
    return world


def test_shift_r_counts_a_restart_request_for_a_job_waiting_for_one(keyed):
    """The job list binds the key; the count is in the job's status answer, which the tab polls,
    and in the job list."""
    from cellmap_flow.finetune.job_manager.state import JobStatus

    bindings = get_session().viewer.config_state.state.input_event_bindings.viewer
    assert bindings["shift+keyr"] == restart_key.ACTION
    keyed.job.status = JobStatus.WAITING_FOR_RESTART
    assert keyed.status()["restart_requests"] == 0

    assert keyed.press() is not None

    assert keyed.status()["restart_requests"] == 1
    listed = keyed.dashboard.get("/api/finetune/jobs").get_json()["jobs"]
    assert [(job["job_id"], job["restart_requests"]) for job in listed] == [("j", 1)]
