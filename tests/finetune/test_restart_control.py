"""A finetune job's restart endpoint obeys only the job manager that launched it.

The job's inference server listens on every interface, and a restart can
change what the job trains on, so an unauthenticated POST used to be enough to
retrain someone's model with arbitrary settings, including other paths (what a
restart may change is test_finetune_cli's).
"""

import os
import stat

import pytest

from cellmap_flow.finetune.finetune_cli import RestartController
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.restart_token import TOKEN_HEADER, read_or_create_restart_token, read_restart_token

SCRIPT_TEST = os.path.join(os.path.dirname(__file__), os.pardir, "script_test")
RAW = os.path.join(SCRIPT_TEST, "dummy.zarr", "raw")


def test_the_token_is_private_and_made_once(tmp_path):
    assert read_restart_token(tmp_path) is None
    token = read_or_create_restart_token(tmp_path)  # a run from before tokens gets one on demand
    assert read_or_create_restart_token(tmp_path) == read_restart_token(tmp_path) == token and len(token) >= 32
    assert stat.S_IMODE(os.stat(tmp_path / "restart_token").st_mode) == 0o600


def test_a_restart_needs_the_jobs_token():
    model_config = ScriptModelConfig(script_path=os.path.join(SCRIPT_TEST, "fake_model_script.py"))
    with pytest.raises(ValueError, match="restart_token"):
        CellMapFlowServer(RAW, model_config, restart_callback=lambda payload: True)

    trainer = RestartController()
    server = CellMapFlowServer(RAW, model_config, restart_callback=trainer.request_restart,
                               restart_token="the-job-token")
    client = server.app.test_client()
    signal = {"params": {"learning_rate": 5e-4}}
    for headers in ({}, {TOKEN_HEADER: "guess"}):
        assert client.post("/__control__/restart", json=signal, headers=headers).status_code == 401
    assert trainer.get_if_triggered() is None
    assert client.post("/__control__/restart", json=signal, headers={TOKEN_HEADER: "the-job-token"}).status_code == 200
    assert trainer.get_if_triggered()["params"] == signal["params"]
