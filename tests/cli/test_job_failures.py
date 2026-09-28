"""A model whose server never came up is a failure, not "ready".

start_hosts used to return a job with host=None; cellmap_flow_yaml then
logged "Job for X is ready" and the viewer got a zarr://None/... layer.
"""

import logging
import os

import pytest
from click.testing import CliRunner

from cellmap_flow.cli import cli as cli_module
from cellmap_flow.cli import yaml_cli
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.utils import neuroglancer_utils
from cellmap_flow.utils.bsub_utils import JobStartError

HERE = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(HERE, "script_test", "fake_model_script.py")
RAW = os.path.join(HERE, "script_test", "dummy.zarr", "raw")


@pytest.fixture
def viewer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        neuroglancer_utils,
        "generate_neuroglancer_url",
        lambda path, wrap_raw=True: calls.append(path),
    )
    return calls


def _start_hosts_failing_for(*bad):
    def fake(command, queue=None, charge_group=None, job_name=None, **_):
        if job_name in bad:
            raise JobStartError(f"{job_name} never reported a server address")
        return object()

    return fake


def _models(*names):
    return [ScriptModelConfig(script_path=SCRIPT, name=n) for n in names]


def test_only_models_that_started_are_reported_ready(monkeypatch, caplog, viewer):
    monkeypatch.setattr(yaml_cli, "start_hosts", _start_hosts_failing_for("bad"))
    with caplog.at_level(logging.INFO, logger="cellmap_flow.cli.yaml_cli"):
        yaml_cli.run_multiple(_models("good", "bad"), RAW, "grp", "gpu_h100")

    assert "Job for good is ready" in caplog.text
    assert "Job for bad is ready" not in caplog.text
    assert "Failed to start job for bad" in caplog.text
    assert viewer == [RAW], "the models that did start still get a viewer"


def test_no_model_starting_is_an_error_not_an_empty_dashboard(monkeypatch, viewer):
    monkeypatch.setattr(yaml_cli, "start_hosts", _start_hosts_failing_for("a", "b"))
    with pytest.raises(JobStartError):
        yaml_cli.run_multiple(_models("a", "b"), RAW, "grp", "gpu_h100")
    assert viewer == []


def test_cellmap_flow_exits_non_zero_with_the_reason(monkeypatch, viewer):
    monkeypatch.setattr(cli_module, "start_hosts", _start_hosts_failing_for("m"))
    result = CliRunner().invoke(
        cli_module.cli, ["script", "--script-path", SCRIPT, "--name", "m", "-d", RAW]
    )
    assert result.exit_code != 0
    assert "m never reported a server address" in result.output
    assert viewer == []
    assert g.jobs == []
