"""Job-manager bookkeeping that went wrong without anything failing loudly.

- A restart read job.corrections_path, which FinetuneJob did not have, so it
  never refreshed the manifest: new patches_per_epoch, rehearsal fraction,
  input norm and postprocessing were dropped though the dialog showed them.
- The monitor overwrote CANCELLED with FAILED on its next poll.
- A local run's LocalJob has no job_id, so adding its viewer layer raised.
- A layer was added before the server existed, with source zarr://None/...
- Every job billed the "cellmap" charge group whatever the dashboard used.
"""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJob, FinetuneJobManager, JobStatus
from cellmap_flow.globals import g
from cellmap_flow.utils.bsub_utils import JobStatus as LSFJobStatus


class _Thread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


class _Script:
    cli_name = "script"
    name = "m"
    script_path = "/s.py"


def _corrections(tmp_path):
    corrections = tmp_path / "session" / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))
    (corrections / "_virtual_sources.json").write_text(json.dumps({
        "kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr", "patches_per_epoch": None,
    }))
    return corrections


def _submit(manager, corrections, **kw):
    with patch.object(fjm, "is_bsub_available", return_value=False), \
         patch.object(fjm, "run_locally", return_value=SimpleNamespace(process=SimpleNamespace(pid=1))), \
         patch.object(fjm.threading, "Thread", _Thread):
        return manager.submit_finetuning_job(
            model_config=_Script(), corrections_path=corrections,
            output_base=corrections.parent, **kw,
        )


def test_the_job_records_its_corrections_and_a_restart_refreshes_their_manifest(tmp_path, monkeypatch):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import training

    corrections = _corrections(tmp_path)
    manager = FinetuneJobManager()
    job = _submit(manager, corrections)
    assert job.corrections_path == corrections

    monkeypatch.setattr(training, "sync_all_annotations_from_minio", lambda force=True: 0)
    monkeypatch.setattr(manager, "restart_finetuning_job", lambda job_id, updated_params: job)
    monkeypatch.setattr(g, "finetune_job_manager", manager, raising=False)
    with app.test_request_context():
        response = training.restart_finetuning_job_response(job.job_id, {"patches_per_epoch": 7})
    assert response.get_json()["success"]
    manifest = json.loads((corrections / "_virtual_sources.json").read_text())
    assert manifest["patches_per_epoch"] == 7


class _KilledJob:
    """An LSF job that, once killed, reports EXIT."""

    job_id = "12345"

    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True

    def get_status(self):
        return LSFJobStatus.FAILED if self.killed else LSFJobStatus.RUNNING


def _job(tmp_path, lsf_job=None, **kw):
    out = tmp_path / "runs" / "r"
    out.mkdir(parents=True, exist_ok=True)
    return FinetuneJob(
        job_id="j", lsf_job=lsf_job, model_name="m", output_dir=out, params={},
        status=JobStatus.RUNNING, created_at=datetime.now(), log_file=out / "training_log.txt", **kw,
    )


def test_a_cancelled_job_stays_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(fjm.time, "sleep", lambda s: None)
    manager = FinetuneJobManager()
    job = _job(tmp_path, lsf_job=_KilledJob())
    manager.jobs[job.job_id] = job

    assert manager.cancel_job(job.job_id)
    manager.monitor_job(job)

    assert job.status == JobStatus.CANCELLED


def test_a_cancel_that_races_the_poll_is_still_a_cancel(tmp_path, monkeypatch):
    monkeypatch.setattr(fjm.time, "sleep", lambda s: None)
    job = _job(tmp_path, lsf_job=_KilledJob())
    job.cancel_requested = True
    job.lsf_job.killed = True  # LSF already says EXIT; status not yet CANCELLED

    FinetuneJobManager().monitor_job(job)

    assert job.status == JobStatus.CANCELLED


def test_a_local_run_gets_its_viewer_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "viewer", None, raising=False)
    monkeypatch.setattr(g, "jobs", [], raising=False)
    local = SimpleNamespace(process=SimpleNamespace(pid=99))  # a LocalJob: no job_id
    job = _job(tmp_path, lsf_job=local, inference_server_url="http://node:8000")

    FinetuneJobManager()._add_finetuned_neuroglancer_layer(job, "m_finetuned_1")

    assert [j.job_id for j in g.jobs] == ["local"]


def test_no_layer_is_added_before_the_server_is_up(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "viewer", None, raising=False)
    monkeypatch.setattr(g, "jobs", [], raising=False)
    monkeypatch.setattr(g, "models_config", [], raising=False)
    job = _job(tmp_path)
    job.log_file.write_text("TRAINING_ITERATION_COMPLETE: m_finetuned_1\n")

    FinetuneJobManager()._parse_training_restart(job, "")

    assert g.jobs == [], "no job with host=None"
    assert job.finetuned_model_name == "m_finetuned_1"


def test_the_dashboards_charge_group_is_billed(tmp_path, monkeypatch):
    from cellmap_flow.dashboard.app import app
    from cellmap_flow.dashboard.routes.finetune import training

    corrections = _corrections(tmp_path)
    captured = {}

    class _Manager:
        jobs = {}

        def submit_finetuning_job(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="j", output_dir=Path(tmp_path), lsf_job=None)

    monkeypatch.setattr(g, "finetune_job_manager", _Manager(), raising=False)
    monkeypatch.setattr(g, "models_config", [SimpleNamespace(name="m")], raising=False)
    monkeypatch.setattr(g, "charge_group", "my_lab", raising=False)
    with app.test_request_context():
        response = training.submit_finetuning_response(
            {"model_name": "m", "corrections_path": str(corrections), "output_type": "binary"}
        )
    assert response.get_json()["success"], response.get_json()
    assert captured["charge_group"] == "my_lab"


@pytest.fixture(autouse=True)
def _quiet_viewer_shader(monkeypatch):
    monkeypatch.setattr(FinetuneJobManager, "_finetuned_shader", lambda self, url: "")


def test_a_local_run_does_not_keep_a_second_copy_of_its_log(tmp_path):
    """The command tees to training_log.txt; run_locally's own log is not needed."""
    import os
    from unittest.mock import MagicMock

    local = MagicMock(return_value=SimpleNamespace(process=SimpleNamespace(pid=1)))
    with patch.object(fjm, "is_bsub_available", return_value=False), \
         patch.object(fjm, "run_locally", local), \
         patch.object(fjm.threading, "Thread", _Thread):
        FinetuneJobManager().submit_finetuning_job(
            model_config=_Script(), corrections_path=_corrections(tmp_path),
            output_base=tmp_path / "session",
        )
    assert local.call_args.kwargs["log_file"] == os.devnull
