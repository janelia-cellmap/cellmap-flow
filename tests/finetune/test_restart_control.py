"""A finetune job's restart endpoint only obeys the job manager that launched it.

The job's inference server listens on every interface, and a restart can
change what the job trains on, so an unauthenticated POST used to be enough
to retrain someone's model with arbitrary settings, including other paths.
"""

import argparse
import json
import os
import stat

import pytest

from cellmap_flow.finetune.finetune_cli import _apply_restart_params
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.restart_token import (
    TOKEN_HEADER,
    read_or_create_restart_token,
    read_restart_token,
    write_restart_token,
)

SCRIPT_TEST = os.path.join(os.path.dirname(__file__), os.pardir, "script_test")
SCRIPT = os.path.join(SCRIPT_TEST, "fake_model_script.py")
RAW = os.path.join(SCRIPT_TEST, "dummy.zarr", "raw")


def test_token_file_is_private_and_round_trips(tmp_path):
    token = write_restart_token(tmp_path)
    assert stat.S_IMODE(os.stat(tmp_path / "restart_token").st_mode) == 0o600
    assert len(token) >= 32
    assert read_restart_token(tmp_path) == token


def test_a_run_without_a_token_gets_one_on_demand(tmp_path):
    assert read_restart_token(tmp_path) is None
    token = read_or_create_restart_token(tmp_path)
    assert read_or_create_restart_token(tmp_path) == token


@pytest.fixture(scope="module")
def model_config():
    return ScriptModelConfig(script_path=SCRIPT)


def test_restart_needs_the_jobs_token(model_config):
    received = []
    server = CellMapFlowServer(
        RAW,
        model_config,
        restart_callback=lambda payload: received.append(payload) or True,
        restart_token="the-job-token",
    )
    client = server.app.test_client()
    signal = {"params": {"learning_rate": 5e-4}}

    assert client.post("/__control__/restart", json=signal).status_code == 401
    wrong = client.post("/__control__/restart", json=signal, headers={TOKEN_HEADER: "guess"})
    assert wrong.status_code == 401
    assert received == []

    ok = client.post("/__control__/restart", json=signal, headers={TOKEN_HEADER: "the-job-token"})
    assert ok.status_code == 200
    assert received == [signal]


def test_restart_control_cannot_be_enabled_without_a_token(model_config):
    with pytest.raises(ValueError, match="restart_token"):
        CellMapFlowServer(RAW, model_config, restart_callback=lambda payload: True)


def test_a_restart_changes_training_settings_but_not_paths(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"params": {"learning_rate": 1e-4, "corrections": "/original"}})
    )
    args = argparse.Namespace(
        output_dir=str(tmp_path),
        corrections="/original",
        serve_data_path="/data",
        learning_rate=1e-4,
    )
    _apply_restart_params(
        args,
        {
            "params": {
                "learning_rate": 5e-4,
                "corrections": "/somewhere/else",
                "serve_data_path": "/other/data",
            }
        },
    )
    assert args.learning_rate == 5e-4
    assert args.corrections == "/original"
    assert args.serve_data_path == "/data"
    recorded = json.loads((tmp_path / "metadata.json").read_text())["params"]
    assert recorded == {"learning_rate": 5e-4, "corrections": "/original"}
