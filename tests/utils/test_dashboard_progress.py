"""dashboard.progress.Progress: how long it remembers a load.

What a progress route answers while a long request runs (each phase as the
request reaches it, a 400 without a load_id and a 404 for an unknown one) is
pinned through the routes in tests/finetune/test_finetune_routes.py.
"""

from types import SimpleNamespace

import flask

from cellmap_flow.dashboard import progress


def test_a_load_is_forgotten_once_it_has_gone_unreported_for_its_ttl(monkeypatch):
    """So the entries of loads nobody polls any more do not pile up. The time
    counts from a load's last report, and stale entries go when any load
    reports."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(progress, "time", SimpleNamespace(time=lambda: clock.now))
    tracker = progress.Progress(ttl_seconds=300)
    app = flask.Flask(__name__)

    def poll(load_id):
        with app.app_context():
            answer = tracker.response(load_id)
        response, status = answer if isinstance(answer, tuple) else (answer, 200)
        return status, response.get_json()

    tracker.update("old", phase="reading")
    clock.now = 1100.0
    tracker.update("old", phase="writing")
    clock.now = 1399.0  # 399 s since "old" began, 299 s since it last reported
    tracker.update("new", phase="starting")
    assert poll("old") == (200, {"success": True, "progress": {
        "phase": "writing", "created_at": 1000.0, "updated_at": 1100.0,
    }})

    clock.now = 1401.0  # 301 s since "old" last reported
    tracker.update("new", phase="finished", done=True)
    assert poll("old") == (404, {"success": False, "error": "Unknown load_id old"})
    assert poll("new") == (200, {"success": True, "progress": {
        "phase": "finished", "done": True, "created_at": 1399.0, "updated_at": 1401.0,
    }})
