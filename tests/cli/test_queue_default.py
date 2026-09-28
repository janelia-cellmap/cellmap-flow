"""`cellmap_flow <type>` without -q uses the saved queue, and keeps it saved.

-q defaulted to a hard-coded gpu_h100, unlike -P, and the value was then
saved: running a model without -q silently replaced the queue the user had
chosen in the dashboard or a YAML.
"""

import os

import pytest
import yaml
from click.testing import CliRunner

import cellmap_flow.globals as G
from cellmap_flow.cli import cli as cli_module
from cellmap_flow.globals import g
from cellmap_flow.utils import neuroglancer_utils

HERE = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(HERE, "script_test", "fake_model_script.py")
RAW = os.path.join(HERE, "script_test", "dummy.zarr", "raw")


@pytest.fixture
def launched(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(G, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    monkeypatch.setattr(
        cli_module,
        "start_hosts",
        lambda command, queue, project, name: calls.append(queue),
    )
    monkeypatch.setattr(neuroglancer_utils, "generate_neuroglancer_url", lambda path: None)
    return calls


@pytest.mark.parametrize(
    "argv",
    [
        ["script", "--script-path", SCRIPT, "-d", RAW],
        ["run", "-m", "script", "-c", f"script_path={SCRIPT}", "-d", RAW],
    ],
    ids=["per-type", "run"],
)
def test_the_saved_queue_is_used_and_kept(launched, tmp_path, argv):
    g.queue = "gpu_a100"

    result = CliRunner().invoke(cli_module.cli, argv)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert launched == ["gpu_a100"]
    saved = yaml.safe_load((tmp_path / "server_config.yaml").read_text())
    assert saved["queue"] == "gpu_a100"


def test_an_explicit_queue_still_wins(launched):
    g.queue = "gpu_a100"
    result = CliRunner().invoke(
        cli_module.cli, ["script", "--script-path", SCRIPT, "-d", RAW, "-q", "gpu_h200"]
    )
    assert result.exit_code == 0, result.output
    assert launched == ["gpu_h200"]
