"""A job waiting for a restart says so, and can be restarted.

After an iteration diverged the trainer waited for a restart, but the job
manager only allowed one once the inference server was marked ready -- which
a restart resets and only a completed iteration sets again -- so a later
iteration that diverged could never be restarted. Restart was also allowed on
COMPLETED, where the trainer has exited: the request went nowhere and the job
showed RUNNING for ever.
"""

import json
from datetime import datetime

import pytest

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJob, FinetuneJobManager, JobStatus
from cellmap_flow.utils.bsub_utils import JobStatus as LSFJobStatus


def _job(tmp_path, status=JobStatus.RUNNING, lsf_job=None, **kw):
    out = tmp_path / "runs" / "r"
    out.mkdir(parents=True, exist_ok=True)
    return FinetuneJob(
        job_id="j", lsf_job=lsf_job, model_name="m", output_dir=out, params={},
        status=status, created_at=datetime.now(), log_file=out / "training_log.txt", **kw,
    )


@pytest.mark.parametrize("chunk, expected", [
    ("Epoch 3/10 - Loss: nan\nTRAINING_DIVERGED\nWAITING_FOR_RESTART\n", JobStatus.WAITING_FOR_RESTART),
    ("TRAINING_ITERATION_COMPLETE: m_1\nWAITING_FOR_RESTART\n", JobStatus.WAITING_FOR_RESTART),
    ("WAITING_FOR_RESTART\nRESTARTING_TRAINING\n", JobStatus.RUNNING),
    ("RESTARTING_TRAINING\nTRAINING_DIVERGED\n", JobStatus.WAITING_FOR_RESTART),
])
def test_the_last_status_marker_in_the_log_decides(tmp_path, chunk, expected):
    job = _job(tmp_path)
    FinetuneJobManager()._parse_training_restart(job, chunk)
    assert job.status == expected


def test_lsf_running_does_not_undo_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(fjm.time, "sleep", lambda s: None)
    seen = []

    class _Lsf:
        job_id = "1"
        calls = 0

        def get_status(self):
            seen.append(job.status)
            self.calls += 1
            return LSFJobStatus.RUNNING if self.calls < 3 else LSFJobStatus.FAILED

    job = _job(tmp_path, status=JobStatus.WAITING_FOR_RESTART, lsf_job=_Lsf())
    FinetuneJobManager().monitor_job(job)
    assert seen[:3] == [JobStatus.WAITING_FOR_RESTART] * 3


def test_a_waiting_job_without_a_server_is_restarted_through_the_signal_file(tmp_path):
    manager = FinetuneJobManager()
    job = _job(tmp_path, status=JobStatus.WAITING_FOR_RESTART)
    manager.jobs[job.job_id] = job

    manager.restart_finetuning_job(job.job_id, {"learning_rate": 5e-5})

    signal = json.loads((job.output_dir / "restart_signal.json").read_text())
    assert signal["params"] == {"learning_rate": 5e-5}
    assert job.status == JobStatus.RUNNING


def test_a_completed_job_cannot_be_restarted(tmp_path):
    manager = FinetuneJobManager()
    job = _job(tmp_path, status=JobStatus.COMPLETED, inference_server_ready=True)
    manager.jobs[job.job_id] = job
    with pytest.raises(ValueError):
        manager.restart_finetuning_job(job.job_id, {})
    assert not (job.output_dir / "restart_signal.json").exists()


def test_the_trainer_announces_that_it_is_waiting(tmp_path, capsys):
    from cellmap_flow.finetune.finetune_cli import RestartController, _wait_for_restart_signal

    controller = RestartController()
    controller.request_restart({"params": {}})
    assert _wait_for_restart_signal(None, restart_controller=controller) is not None
    assert "WAITING_FOR_RESTART" in capsys.readouterr().out
