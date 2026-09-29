"""Asking LSF about many jobs at once, and remembering what it has forgotten.

Rehydration runs on every load of the finetune tab, and it asked bjobs about
each unfinished run of each session separately. A run that ended while no
dashboard was watching, and that LSF had since purged, was never recorded as
over -- bjobs says "not found", which read as "could not say" -- so it cost
a bjobs call on every load, for good. Now a session's runs are asked about
in one call, and a job LSF no longer knows is recorded as final.

subprocess.run is a fake throughout; each test checks the calls it saw.
"""

import json
import subprocess
from datetime import datetime

import pytest

from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager, JobStatus
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.spec import JobStatus as LSFJobStatus

BJOBS_OUT = (
    "201   me  RUN   gpu_h100  login1  4*h10u05  finetune_a  Sep 28 10:00\n"
    "      h10u06\n"  # a continuation line: more hosts of the job above
    "202   me  PEND  gpu_h100  login1            finetune_b  Sep 28 10:01\n"
    "203   me  DONE  gpu_h100  login1  h10u07    finetune_c  Sep 28 09:00\n"
    "204   me  EXIT  gpu_h100  login1  h10u08    finetune_d  Sep 28 09:30\n"
    "205   me  USUSP gpu_h100  login1  h10u09    finetune_e  Sep 28 09:40\n"
)


class FakeBjobs:
    def __init__(self, monkeypatch, stdout="", stderr="", returncode=0, error=None):
        self.calls = []
        self.answer = (returncode, stdout, stderr)
        self.error = error
        monkeypatch.setattr(subprocess, "run", self.run)

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        assert argv[:2] == ["bjobs", "-noheader"], argv
        if self.error is not None:
            raise self.error
        returncode, stdout, stderr = self.answer
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def test_one_call_answers_for_every_job(monkeypatch):
    bjobs = FakeBjobs(
        monkeypatch,
        stdout=BJOBS_OUT,
        stderr="Job <206> is not found\n",
        returncode=255,  # bjobs exits non-zero when any job is not found
    )

    reported = jobs_lsf.statuses(["201", 202, "203", "204", "205", "206", "207", "201"])

    assert bjobs.calls == [["bjobs", "-noheader", "201", "202", "203", "204", "205", "206", "207"]]
    assert reported == {
        "201": LSFJobStatus.RUNNING,
        "202": LSFJobStatus.PENDING,
        "203": LSFJobStatus.COMPLETED,
        "204": LSFJobStatus.FAILED,
        # Suspended: alive, as LSFJob.get_status reads it.
        "205": LSFJobStatus.RUNNING,
        # Not found: LSF has forgotten it.
        "206": None,
    }, "207, which bjobs said nothing about, is left out: not known either way"


def test_no_ids_asks_nothing(monkeypatch):
    bjobs = FakeBjobs(monkeypatch)
    assert jobs_lsf.statuses([]) == {}
    assert bjobs.calls == []


@pytest.mark.parametrize("error", [FileNotFoundError("bjobs"), subprocess.TimeoutExpired("bjobs", 10)])
def test_a_bjobs_that_cannot_answer_says_nothing_about_anyone(monkeypatch, error):
    bjobs = FakeBjobs(monkeypatch, error=error)
    assert jobs_lsf.statuses(["201", "202"]) == {}
    assert len(bjobs.calls) == 1


def _run(session, name, lsf_job_id, status="RUNNING"):
    run = session / "runs" / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "job_id": name, "model_name": "m", "created_at": datetime.now().isoformat(),
        "params": {"num_epochs": 5}, "lsf_job_id": lsf_job_id, "status": status,
    }))
    return run


def _status(run):
    return json.loads((run / "metadata.json").read_text())


def test_a_job_lsf_has_forgotten_is_recorded_as_over_and_not_asked_about_again(tmp_path, monkeypatch):
    session = tmp_path / "20260101_120000"
    _run(session, "alive", "301")
    forgotten = _run(session, "forgotten", "302")
    unanswered = _run(session, "unanswered", "303")
    bjobs = FakeBjobs(
        monkeypatch,
        stdout="301 me RUN gpu_h100 login1 h01 finetune_a Sep 28 10:00\n",
        stderr="Job <302> is not found\n",
        returncode=255,
    )
    manager = FinetuneJobManager()
    monkeypatch.setattr(manager, "_start_monitor", lambda job: None)

    assert manager.rehydrate_session(session) == 1

    assert bjobs.calls == [["bjobs", "-noheader", "301", "302", "303"]], "one call for the session"
    assert manager.jobs["alive"].status == JobStatus.RUNNING
    assert _status(forgotten)["status"] == "FAILED"
    assert "302" in _status(forgotten)["status_detail"]
    assert _status(unanswered)["status"] == "RUNNING", "bjobs said nothing: ask again next time"

    # The next load asks only about the run nobody could say anything about.
    manager.rehydrate_session(session)
    assert bjobs.calls[1:] == [["bjobs", "-noheader", "303"]]


def test_a_session_with_nothing_unfinished_asks_nothing(tmp_path, monkeypatch):
    session = tmp_path / "20260101_120000"
    _run(session, "done", "401", status="COMPLETED")
    _run(session, "local", "PID:12", status="RUNNING")
    bjobs = FakeBjobs(monkeypatch)

    assert FinetuneJobManager().rehydrate_session(session) == 0
    assert bjobs.calls == []
