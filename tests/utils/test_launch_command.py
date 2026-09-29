"""Every launcher builds its server command through serving.launch.

SERVER_COMMAND is read when a command is built and split into words, so a
deploy's override (fileglancer's "pixi run cellmap_flow_server") reaches
every launcher. cellmap_flow and cellmap_flow_yaml copied it at import; the
dashboard's launchers are checked in test_dashboard_launch_failures.
"""

import shlex

import pytest
from click.testing import CliRunner

from cellmap_flow.cli import cli as cli_module
from cellmap_flow.cli import yaml_cli
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import HuggingFaceModelConfig, ScriptModelConfig
from cellmap_flow.serving import launch
from cellmap_flow.utils import bsub_utils

DATA = "/d/my raw.zarr"


@pytest.fixture
def launched(monkeypatch):
    from cellmap_flow.utils import neuroglancer_utils

    commands = []

    def record(command, *args, **kwargs):
        commands.append(command)
        return object()

    monkeypatch.setattr(bsub_utils, "SERVER_COMMAND", "pixi run cellmap_flow_server")
    monkeypatch.setattr(cli_module, "start_hosts", record)
    monkeypatch.setattr(yaml_cli, "start_hosts", record)
    monkeypatch.setattr(
        neuroglancer_utils, "generate_neuroglancer_url", lambda path, wrap_raw=True: None
    )
    monkeypatch.setattr(type(g), "save_server_config", lambda self: None)
    return commands


@pytest.mark.parametrize(
    "argv",
    [
        ["script", "--script-path", "/s.py", "--name", "m", "-d", DATA],
        ["run", "-m", "script", "-c", "script_path=/s.py", "-c", "name=m", "-d", DATA],
        None,  # cellmap_flow_yaml
    ],
    ids=["cellmap_flow-type", "cellmap_flow-run", "cellmap_flow_yaml"],
)
def test_a_multi_word_server_command_reaches_every_launcher(launched, argv):
    if argv is None:
        yaml_cli.run_multiple([ScriptModelConfig(script_path="/s.py", name="m")], DATA, "grp", "q")
    else:
        result = CliRunner().invoke(cli_module.cli, argv)
        assert result.exit_code == 0, result.output + repr(result.exception)

    assert [shlex.split(c) for c in launched] == [
        ["pixi", "run", "cellmap_flow_server", "script", "--script-path", "/s.py", "--name", "m", "-d", DATA]
    ]


def test_a_command_from_type_and_arguments_is_the_config_s_own(monkeypatch):
    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", lambda self: {})
    config = HuggingFaceModelConfig(repo="cellmap/mito", name="m v1")

    argv = launch.server_argv_for("huggingface", {"repo": "cellmap/mito", "name": "m v1"}, DATA)

    assert argv == launch.server_argv(config, DATA)
    assert argv == shlex.split(bsub_utils.SERVER_COMMAND) + [
        "huggingface", "--repo", "cellmap/mito", "--name", "m v1", "-d", DATA,
    ]
    assert launch.server_command(config, DATA) == shlex.join(argv)
