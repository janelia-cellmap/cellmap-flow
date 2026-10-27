"""A finetune job's control endpoints obey only the job manager that launched it.

The job's inference server listens on every interface, and a restart can
change what the job trains on, so an unauthenticated POST used to be enough to
retrain someone's model with arbitrary settings, including other paths (what a
restart may change is test_finetune_cli's). Stop Early takes the same way.
"""

import os
import stat

import pytest
from starlette.testclient import TestClient

from cellmap_flow.finetune.session_loop import JobControl
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
from cellmap_flow.serving.restart_token import (
    TOKEN_HEADER,
    read_or_create_restart_token,
    read_restart_token,
    write_restart_token,
)

SCRIPT_TEST = os.path.join(os.path.dirname(__file__), os.pardir, "script_test")
RAW = os.path.join(SCRIPT_TEST, "dummy.zarr", "raw")


def test_the_token_is_private_and_made_once(tmp_path):
    assert read_restart_token(tmp_path) is None
    token = read_or_create_restart_token(tmp_path)  # a run from before tokens gets one on demand
    assert read_or_create_restart_token(tmp_path) == read_restart_token(tmp_path) == token and len(token) >= 32
    assert stat.S_IMODE(os.stat(tmp_path / "restart_token").st_mode) == 0o600

    # A new token over a file others could read: rewritten in place, it was
    # readable until the chmod, and for good through a handle opened before.
    os.chmod(tmp_path / "restart_token", 0o644)
    with open(tmp_path / "restart_token") as earlier_reader:
        new = write_restart_token(tmp_path)
        assert earlier_reader.read() == token != new == read_restart_token(tmp_path)
    assert stat.S_IMODE(os.stat(tmp_path / "restart_token").st_mode) == 0o600
    assert os.listdir(tmp_path) == ["restart_token"]


def test_a_restart_needs_the_jobs_token():
    model_config = ScriptModelConfig(script_path=os.path.join(SCRIPT_TEST, "fake_model_script.py"))
    with pytest.raises(ValueError, match="restart_token"):
        ChunkmirageServer(RAW, model_config, restart_callback=lambda payload: True)

    trainer = JobControl()
    server = ChunkmirageServer(RAW, model_config, restart_callback=trainer.request_restart,
                               restart_token="the-job-token")
    client = TestClient(server.app)
    signal = {"params": {"learning_rate": 5e-4}}
    for headers in ({}, {TOKEN_HEADER: "guess"}):
        assert client.post("/__control__/restart", json=signal, headers=headers).status_code == 401
    assert trainer.take_restart() is None
    assert client.post("/__control__/restart", json=signal, headers={TOKEN_HEADER: "the-job-token"}).status_code == 200
    assert trainer.take_restart()["params"] == signal["params"]


def test_stop_early_reaches_a_serving_job_over_its_control_endpoint(make_job, monkeypatch):
    """As a restart does. It used to come only as stop_signal.json, which the
    trainer's host saw up to a minute late (NFS caches the miss), and the run
    trained on for several epochs. From iteration 2 on, the job's server is up
    while it trains."""
    from cellmap_flow.finetune.job_manager import restart

    job = make_job(inference_server_url="http://gpu1:9000/")
    control = JobControl()
    server = ChunkmirageServer(RAW, ScriptModelConfig(script_path=os.path.join(SCRIPT_TEST, "fake_model_script.py")),
                               restart_callback=control.request_restart, stop_callback=control.request_stop,
                               restart_token=write_restart_token(job.output_dir))
    client = TestClient(server.app)
    monkeypatch.setattr(restart.requests, "post",
                        lambda url, timeout, **kwargs: client.post(url.removeprefix("http://gpu1:9000"), **kwargs))

    restart.request_stop(job)
    assert control.take_stop() is not None
    assert not (job.output_dir / "stop_signal.json").exists()
