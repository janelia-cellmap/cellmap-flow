"""`cellmap_flow infer <type>`: what it launches, and how it fails.

The options themselves, and `run` (its alias before 0.3.0), are pinned in
test_cli_surface; the commands they build in tests/utils/test_launch_command.
"""

import os

import pytest
import yaml
from click.testing import CliRunner

import cellmap_flow.globals as G
from cellmap_flow.cli import infer
from cellmap_flow.cli.main import cli
from cellmap_flow.globals import g
from cellmap_flow.dashboard.services import startup
from cellmap_flow.jobs.spec import JobStartError

HERE = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(HERE, "script_test", "fake_model_script.py")
RAW = os.path.join(HERE, "script_test", "dummy.zarr", "raw")
PER_TYPE = ["infer", "script", "--script-path", SCRIPT, "-d", RAW]


@pytest.mark.parametrize("argv, queue", [
    pytest.param(PER_TYPE, "gpu_a100", id="saved"),
    pytest.param(PER_TYPE + ["-q", "gpu_h200"], "gpu_h200", id="explicit-q-wins"),
])
def test_without_q_the_saved_queue_is_used_and_kept(monkeypatch, tmp_path, argv, queue):
    """-q defaulted to gpu_h100, unlike -P, and that was then saved: running a
    model without -q replaced the queue chosen in the dashboard or a YAML."""
    launched = []
    monkeypatch.setattr(G, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    monkeypatch.setattr(infer, "start_hosts", lambda command, queue, project, name: launched.append(queue))
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path: None)
    g.queue = "gpu_a100"
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert launched == [queue]
    if queue == "gpu_a100":
        assert yaml.safe_load((tmp_path / "server_config.yaml").read_text())["queue"] == "gpu_a100"


def test_server_check_runs_one_chunk_through_the_model():
    """It called _chunk_impl with six arguments against five, and crashed
    before touching the model, for every type."""
    result = CliRunner().invoke(cli, PER_TYPE + ["--server-check"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Server check passed" in result.output


def test_a_server_that_never_came_up_exits_non_zero_with_the_reason(monkeypatch):
    """start_hosts returned a job with no host, and the viewer got zarr://None/..."""
    viewers = []

    def fail(command, queue=None, project=None, name=None, **_):
        raise JobStartError(f"{name} never reported a server address")

    monkeypatch.setattr(infer, "start_hosts", fail)
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda *a, **k: viewers.append(a))
    result = CliRunner().invoke(cli, PER_TYPE + ["--name", "m"])
    assert result.exit_code != 0 and "m never reported a server address" in result.output
    assert viewers == [] and g.jobs == []
