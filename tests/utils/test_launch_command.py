"""Every launcher builds its server command through serving.launch.

SERVER_COMMAND is read when a command is built and split into words, so a
deploy's override (fileglancer's "pixi run cellmap_flow_server") reaches
every launcher; cellmap_flow and cellmap_flow_yaml used to copy it at import,
and the dashboard quoted it as one word. The data path, with a space in it,
goes through the one data_path + scale rule: the YAML's `scale: s3` next to
a path ending in s3 used to become .../s3/s3.
"""

import shlex

import pytest
import zarr
from click.testing import CliRunner

from cellmap_flow.cli import cli as cli_module
from cellmap_flow.cli import yaml_cli
from cellmap_flow.globals import g
from cellmap_flow.models import run
from cellmap_flow.models.models_config import HuggingFaceModelConfig, ScriptModelConfig
from cellmap_flow.serving import launch
from cellmap_flow.utils import bsub_utils
from cellmap_flow.utils.bsub_utils import JobStartError

def _cli(*argv):
    result = CliRunner().invoke(cli_module.cli, list(argv))
    assert result.exit_code == 0, result.output + repr(result.exception)


LAUNCHERS = {
    "cellmap_flow-type": (
        lambda data: _cli("script", "--script-path", "/s.py", "--name", "m", "-d", data),
        ["script", "--script-path", "/s.py", "--name", "m"],
    ),
    "cellmap_flow-run": (
        lambda data: _cli("run", "-m", "script", "-c", "script_path=/s.py", "-c", "name=m", "-d", data),
        ["script", "--script-path", "/s.py", "--name", "m"],
    ),
    "cellmap_flow_yaml": (
        lambda data: yaml_cli.run_multiple([ScriptModelConfig(script_path="/s.py", name="m", scale="s3")], data, "grp", "q"),
        ["script", "--script-path", "/s.py", "--name", "m", "--scale", "s3"],
    ),
    "dashboard-catalog": (
        lambda data: run.run_model("/models/mito v2", "mito", "blob"),
        ["cellmap", "--folder-path", "/models/mito v2", "--name", "mito"],
    ),
    "dashboard-huggingface": (
        lambda data: run.run_hf_model("cellmap/mito-v1", "mito v1", "blob"),
        ["huggingface", "--repo", "cellmap/mito-v1", "--name", "mito_v1"],
    ),
}


@pytest.fixture
def launched(monkeypatch, tmp_path):
    """The commands the launchers submit, and the data path they are given."""
    from cellmap_flow.utils import neuroglancer_utils

    data = str(tmp_path / "my raw.zarr" / "s3")
    zarr.open_group(str(tmp_path / "my raw.zarr"), mode="w").create_dataset("s3", shape=(4, 4, 4), dtype="u1")
    commands = []

    def started(command, *args, **kwargs):
        commands.append(command)
        return object()

    def refused(command, *args, **kwargs):  # the dashboard logs it and stops before the viewer
        commands.append(command)
        raise JobStartError("recorded")

    monkeypatch.setattr(bsub_utils, "SERVER_COMMAND", "pixi run cellmap_flow_server")
    monkeypatch.setattr(cli_module, "start_hosts", started)
    monkeypatch.setattr(yaml_cli, "start_hosts", started)
    monkeypatch.setattr(run, "start_hosts", refused)
    monkeypatch.setattr(neuroglancer_utils, "generate_neuroglancer_url", lambda path, wrap_raw=True: None)
    monkeypatch.setattr(type(g), "save_server_config", lambda self: None)
    g.dataset_path = data
    return commands, data


@pytest.mark.parametrize("launcher", list(LAUNCHERS))
def test_every_launcher_submits_the_split_server_command(launched, launcher):
    commands, data = launched
    launch_it, model_argv = LAUNCHERS[launcher]
    launch_it(data)
    assert [shlex.split(c) for c in commands] == [["pixi", "run", "cellmap_flow_server", *model_argv, "-d", data]]


def test_a_command_from_type_and_arguments_is_the_config_s_own(monkeypatch):
    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", lambda self: {})
    config = HuggingFaceModelConfig(repo="cellmap/mito", name="m v1")
    argv = launch.server_argv_for("huggingface", {"repo": "cellmap/mito", "name": "m v1"}, "/d/my raw.zarr")
    assert argv == launch.server_argv(config, "/d/my raw.zarr") == shlex.split(bsub_utils.SERVER_COMMAND) + [
        "huggingface", "--repo", "cellmap/mito", "--name", "m v1", "-d", "/d/my raw.zarr",
    ]
    assert launch.server_command(config, "/d/my raw.zarr") == shlex.join(argv)
