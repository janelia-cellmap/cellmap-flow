"""A dashboard restart no longer loses its finetune jobs or its sessions.

Jobs, sessions and volumes lived only in the dashboard's memory, and
metadata.json did not even record the LSF job id. After a restart a running
job could not be seen or cancelled (with auto-serve it waited for a restart
until walltime), and submitting for the same base path made a new, empty
timestamped session: "Corrections path does not exist".
"""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJob, FinetuneJobManager, JobStatus
from cellmap_flow.utils.bsub_utils import JobStatus as LSFJobStatus, LSFJob


class _Thread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


class _Script:
    cli_name = "script"
    name = "m"
    script_path = "/s.py"


def _session(tmp_path, name="20260101_120000"):
    session = tmp_path / "base" / name
    corrections = session / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))
    (corrections / "_virtual_sources.json").write_text(json.dumps({"kind": "volume_zarr_v1"}))
    return session


def test_submit_records_the_scheduler_id_and_status(tmp_path):
    session = _session(tmp_path)
    with patch.object(fjm, "is_bsub_available", return_value=False), \
         patch.object(fjm, "run_locally", return_value=SimpleNamespace(process=SimpleNamespace(pid=77))), \
         patch.object(fjm.threading, "Thread", _Thread):
        job = FinetuneJobManager().submit_finetuning_job(
            model_config=_Script(), corrections_path=session / "corrections", output_base=session,
        )
    metadata = json.loads((job.output_dir / "metadata.json").read_text())
    assert metadata["lsf_job_id"] == "PID:77"
    assert metadata["status"] == "PENDING"


def test_the_monitor_records_the_final_status(tmp_path, monkeypatch):
    monkeypatch.setattr(fjm.time, "sleep", lambda s: None)
    out = tmp_path / "runs" / "r"
    out.mkdir(parents=True)
    (out / "metadata.json").write_text(json.dumps({"job_id": "j", "status": "PENDING"}))
    replies = iter([LSFJobStatus.RUNNING, LSFJobStatus.FAILED])
    lsf = SimpleNamespace(job_id="5", get_status=lambda: next(replies))
    job = FinetuneJob(
        job_id="j", lsf_job=lsf, model_name="m", output_dir=out, params={},
        status=JobStatus.PENDING, created_at=datetime.now(), log_file=out / "training_log.txt",
    )
    FinetuneJobManager().monitor_job(job)
    assert json.loads((out / "metadata.json").read_text())["status"] == "FAILED"


def _run(session, name, **metadata):
    run = session / "runs" / name
    run.mkdir(parents=True)
    base = {
        "job_id": name, "model_name": "m", "created_at": datetime.now().isoformat(),
        "corrections_path": str(session / "corrections"), "params": {"num_epochs": 5},
    }
    base.update(metadata)
    (run / "metadata.json").write_text(json.dumps(base))
    return run


def test_jobs_still_on_the_cluster_are_picked_up_again(tmp_path, monkeypatch):
    session = _session(tmp_path)
    _run(session, "alive", lsf_job_id="101", status="WAITING_FOR_RESTART")
    done = _run(session, "done_meanwhile", lsf_job_id="102", status="RUNNING")
    _run(session, "finished", lsf_job_id="103", status="COMPLETED")
    _run(session, "local", lsf_job_id="PID:4", status="RUNNING")
    _run(session, "before_this_change", status="RUNNING")  # no lsf_job_id recorded
    observed = {"101": LSFJobStatus.RUNNING, "102": LSFJobStatus.FAILED}
    asked = []

    def fake_observed(self):
        asked.append(self.job_id)
        return observed.get(self.job_id)

    monkeypatch.setattr(LSFJob, "observed_status", fake_observed)
    manager = FinetuneJobManager()
    monkeypatch.setattr(manager, "_start_monitor", lambda job: None)

    assert manager.rehydrate_session(session) == 1

    job = manager.jobs["alive"]
    assert job.lsf_job.job_id == "101"
    assert job.status == JobStatus.RUNNING
    assert job.corrections_path == session / "corrections"
    assert job.total_epochs == 5
    assert sorted(asked) == ["101", "102"], "finished and local runs are not asked about"
    assert json.loads((done / "metadata.json").read_text())["status"] == "FAILED"
    # Idempotent: a job already known is not attached twice.
    assert manager.rehydrate_session(session) == 0


def test_the_jobs_list_looks_in_the_saved_output_path(tmp_path, monkeypatch):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import common, training
    from cellmap_flow.globals import g

    session = _session(tmp_path)
    seen = []

    class _Manager:
        jobs = {}

        def rehydrate_session(self, path):
            seen.append(path)

        def list_jobs(self):
            return []

    monkeypatch.setattr(g, "finetune_job_manager", _Manager(), raising=False)
    monkeypatch.setattr(common, "load_user_prefs", lambda: {"outputPath": str(tmp_path / "base")})
    with app.test_request_context():
        assert training.list_finetuning_jobs_response().get_json()["success"]
    assert seen == [str(session)]


def test_submit_after_a_restart_finds_the_session_on_disk(tmp_path):
    from cellmap_flow.dashboard.routes.finetune.common import resolve_finetune_session

    older = _session(tmp_path, "20260101_120000")
    (tmp_path / "base" / "20260102_090000").mkdir()  # newer, but nothing to train on
    session, corrections = resolve_finetune_session(str(tmp_path / "base"))
    assert session == older
    assert corrections == older / "corrections"


@pytest.fixture(autouse=True)
def _no_sessions(monkeypatch):
    from cellmap_flow.dashboard import finetune_utils

    monkeypatch.setattr(finetune_utils, "output_sessions", {})
    from cellmap_flow.dashboard.routes.finetune import common

    monkeypatch.setattr(common, "output_sessions", finetune_utils.output_sessions, raising=False)
