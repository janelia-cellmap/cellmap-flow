"""The finetune log stream resumes where it left off and says when it is done.

It re-sent the whole log on every connection and ended without a word when
the job did, so the browser's automatic reconnect replayed the whole log --
plus "=== Training COMPLETED ===" -- every few seconds, for as long as the
page was open. Each block now carries its byte offset as the SSE id, a
reconnect resumes from Last-Event-ID (or ?offset=), and the stream closes
with an "event: done".
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from cellmap_flow.finetune.job_manager.state import FinetuneJob, JobStatus


class _Statuses(FinetuneJob):
    """A job whose status runs through a script, one value per read."""

    def __init__(self, script, **kw):
        object.__setattr__(self, "_script", list(script))
        super().__init__(status=self._script[0], **kw)

    @property
    def status(self):
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]

    @status.setter
    def status(self, value):
        pass


def _job(tmp_path, text, statuses=(JobStatus.COMPLETED,)):
    log = tmp_path / "training_log.txt"
    log.write_bytes(text.encode())
    return _Statuses(
        list(statuses), job_id="j", lsf_job=None, model_name="m", output_dir=tmp_path,
        params={}, created_at=datetime.now(), log_file=log,
    )


@pytest.fixture
def client(monkeypatch):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import training
    from cellmap_flow.dashboard.state import get_session

    manager = SimpleNamespace(jobs={}, get_job_logs=lambda job_id: None)
    monkeypatch.setattr(get_session(), "finetune_job_manager", manager)
    monkeypatch.setattr(training.time, "sleep", lambda s: None)
    return app.test_client(), manager


def _events(body):
    return [block for block in body.split("\n\n") if block.strip() and not block.startswith(":")]


def test_blocks_carry_their_offset_and_the_stream_ends_with_done(tmp_path, client):
    http, manager = client
    manager.jobs["j"] = _job(tmp_path, "Epoch 1/2 - Loss: 0.5\nEpoch 2/2 - Loss: 0.4\n")

    events = _events(http.get("/api/finetune/job/j/logs/stream").get_data(as_text=True))

    size = len("Epoch 1/2 - Loss: 0.5\nEpoch 2/2 - Loss: 0.4\n")
    assert events[0] == f"id: {size}\ndata: Epoch 1/2 - Loss: 0.5\ndata: Epoch 2/2 - Loss: 0.4"
    assert events[-1] == "event: done\ndata: COMPLETED"


@pytest.mark.parametrize("how", ["header", "query"])
def test_a_reconnect_resumes_after_what_it_had(tmp_path, client, how):
    http, manager = client
    first = "line one\n"
    manager.jobs["j"] = _job(tmp_path, first + "line two\n")

    if how == "header":
        response = http.get("/api/finetune/job/j/logs/stream", headers={"Last-Event-ID": str(len(first))})
    else:
        response = http.get(f"/api/finetune/job/j/logs/stream?offset={len(first)}")
    body = response.get_data(as_text=True)

    assert "line one" not in body
    assert "data: line two" in body


def test_a_partial_line_waits_until_the_job_is_done(tmp_path, client):
    http, manager = client
    text = "whole\nEpoch 3/10 - Lo"
    manager.jobs["j"] = _job(
        tmp_path, text,
        statuses=[JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.COMPLETED],
    )
    events = _events(http.get("/api/finetune/job/j/logs/stream").get_data(as_text=True))
    assert events[0] == "id: 6\ndata: whole"
    assert events[1] == f"id: {len(text)}\ndata: Epoch 3/10 - Lo"
    assert events[2] == "event: done\ndata: COMPLETED"


def test_an_unknown_job_ends_the_stream(client):
    http, _ = client
    events = _events(http.get("/api/finetune/job/nope/logs/stream").get_data(as_text=True))
    assert events[-1] == "event: done\ndata: NOT_FOUND"


def test_the_log_snapshot_says_where_the_stream_should_start(tmp_path, client):
    http, manager = client
    manager.jobs["j"] = _job(tmp_path, "one\ntwo\nthr")
    data = http.get("/api/finetune/job/j/logs").get_json()
    assert data["logs"] == "one\ntwo\n"
    assert data["offset"] == len("one\ntwo\n")


def test_a_stream_that_breaks_while_the_job_runs_does_not_say_done(tmp_path, client, monkeypatch):
    """The browser must reconnect and resume, not close for good."""
    from cellmap_flow.dashboard.routes.finetune import training

    http, manager = client
    manager.jobs["j"] = _job(tmp_path, "whole\npart", statuses=[JobStatus.RUNNING])

    def broken_sleep(seconds):
        raise OSError("stale NFS handle")

    monkeypatch.setattr(training.time, "sleep", broken_sleep)
    body = http.get("/api/finetune/job/j/logs/stream").get_data(as_text=True)
    assert "event: done" not in body
    assert "data: part" not in body, "a partial line is kept for the reconnect"
