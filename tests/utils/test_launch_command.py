"""Every launcher builds its server command through serving.launch.

It is SERVER_COMMAND, then `--model <the model's launch entry, as JSON>`
and the data path. SERVER_COMMAND is read when a command is built and split
into words, so a deploy's override (fileglancer's "pixi run cellmap_flow
serve") reaches every launcher; cellmap_flow and cellmap_flow_yaml used to
copy it at import, and the dashboard quoted it as one word. The data path,
with a space in it, goes through the one data_path + scale rule: the YAML's
`scale: s3` next to a path ending in s3 used to become .../s3/s3.
"""

import json
import shlex
import subprocess
import sys

import pytest
import zarr
from click.testing import CliRunner

from cellmap_flow.cli import infer, yaml_cli
from cellmap_flow.cli.main import cli
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.dashboard.services import launch as dashboard_launch
from cellmap_flow.models.models_config import HuggingFaceModelConfig, ScriptModelConfig
from cellmap_flow.jobs import launch as jobs_launch
from cellmap_flow.jobs.settings import LauncherSettings
from cellmap_flow.jobs.spec import JobSpec, JobStartError
from cellmap_flow.serving import launch

def _cli(*argv):
    result = CliRunner().invoke(cli, list(argv))
    assert result.exit_code == 0, result.output + repr(result.exception)


# launcher: (launch it on the data path, the model entry it passes, the flags after the data path)
LAUNCHERS = {
    "cellmap_flow-infer": (
        lambda data: _cli("infer", "script", "--script-path", "/s.py", "--name", "m", "-d", data),
        {"type": "script", "script_path": "/s.py", "name": "m"},
        ["--resample"],
    ),
    "cellmap_flow-infer-no-resample": (
        lambda data: _cli("infer", "script", "--script-path", "/s.py", "--name", "m", "-d", data, "--no-resample"),
        {"type": "script", "script_path": "/s.py", "name": "m"},
        ["--no-resample"],
    ),
    "cellmap_flow-run": (
        lambda data: _cli("run", "-m", "script", "-c", "script_path=/s.py", "-c", "name=m", "-d", data),
        {"type": "script", "script_path": "/s.py", "name": "m"},
        ["--resample"],
    ),
    "cellmap_flow-yaml": (
        lambda data: yaml_cli.run_multiple([ScriptModelConfig(script_path="/s.py", name="m", scale="s3")], data, "grp", "q"),
        {"type": "script", "script_path": "/s.py", "name": "m", "scale": "s3"},
        ["--resample"],
    ),
    # The YAML's resample: false.
    "cellmap_flow-yaml-no-resample": (
        lambda data: yaml_cli.run_multiple([ScriptModelConfig(script_path="/s.py", name="m")], data, "grp", "q",
                                           resample=False),
        {"type": "script", "script_path": "/s.py", "name": "m"},
        ["--no-resample"],
    ),
    "dashboard-catalog": (
        lambda data: dashboard_launch.run_model("/models/mito v2", "mito", "blob"),
        {"type": "cellmap", "folder_path": "/models/mito v2", "name": "mito"},
        ["--resample"],
    ),
    "dashboard-huggingface": (
        lambda data: dashboard_launch.run_hf_model("cellmap/mito-v1", "mito v1", "blob"),
        {"type": "huggingface", "repo": "cellmap/mito-v1", "name": "mito_v1"},
        ["--resample"],
    ),
}


@pytest.fixture
def launched(monkeypatch, tmp_path):
    """The commands the launchers submit, and the data path they are given."""
    from cellmap_flow.dashboard.services import startup

    data = str(tmp_path / "my raw.zarr" / "s3")
    zarr.open_group(str(tmp_path / "my raw.zarr"), mode="w").create_dataset("s3", shape=(4, 4, 4), dtype="u1")
    commands = []

    def started(command, *args, **kwargs):
        commands.append(command)
        return object()

    def refused(command, *args, **kwargs):  # the dashboard logs it and stops before the viewer
        commands.append(command)
        raise JobStartError("recorded")

    monkeypatch.setattr(jobs_launch, "SERVER_COMMAND", "pixi run cellmap_flow serve")
    monkeypatch.setattr(infer, "start_hosts", started)
    monkeypatch.setattr(yaml_cli, "start_hosts", started)
    monkeypatch.setattr(dashboard_launch, "start_hosts", refused)
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path, wrap_raw=True: None)
    monkeypatch.setattr(LauncherSettings, "save", lambda self: None)
    get_session().dataset_path = data
    return commands, data


@pytest.mark.parametrize("launcher", list(LAUNCHERS))
def test_every_launcher_submits_the_split_server_command(launched, launcher):
    commands, data = launched
    launch_it, entry, *rest = LAUNCHERS[launcher]
    flags = rest[0] if rest else []
    launch_it(data)
    (argv,) = [shlex.split(c) for c in commands]
    assert argv[:4] == ["pixi", "run", "cellmap_flow", "serve"] and argv[4:8:2] == ["--model", "-d"]
    assert (json.loads(argv[5]), argv[7], argv[8:]) == (entry, data, flags)


def test_a_command_from_type_and_arguments_is_the_config_s_own(monkeypatch):
    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", lambda self: {})
    config = HuggingFaceModelConfig(repo="cellmap/mito", name="m v1")
    argv = launch.server_argv_for("huggingface", {"repo": "cellmap/mito", "name": "m v1"}, "/d/my raw.zarr")
    assert argv == launch.server_argv(config, "/d/my raw.zarr") == shlex.split(jobs_launch.SERVER_COMMAND) + [
        "--model", '{"type":"huggingface","repo":"cellmap/mito","name":"m v1"}', "-d", "/d/my raw.zarr",
        "--resample",
    ]
    assert shlex.split(launch.server_command(config, "/d/my raw.zarr")) == argv


@pytest.mark.parametrize("launcher", list(LAUNCHERS))
def test_every_launcher_s_command_reaches_the_server_whole_through_lsf(launched, launcher, monkeypatch, tmp_path):
    """LSF runs a job's `bash -c` line wrapped in single quotes of its own, so
    a single-quoted --model JSON came out as "type:huggingface" (2026-10-01)."""
    commands, data = launched
    launch_it, entry, *rest = LAUNCHERS[launcher]
    printer = tmp_path / "print_argv.py"
    printer.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    monkeypatch.setattr(jobs_launch, "SERVER_COMMAND", f"{sys.executable} {printer}")
    launch_it(data)
    (command,) = commands
    as_lsf_runs_it = subprocess.run(["sh", "-c", f"bash -c '{command}'"], capture_output=True, text=True, check=True)
    model, entry_json, d, path, *flags = json.loads(as_lsf_runs_it.stdout)
    assert (model, json.loads(entry_json), d, path, flags) == ("--model", entry, "-d", data, rest[0] if rest else [])


def test_a_job_s_shell_line_with_a_single_quote_is_refused():
    with pytest.raises(ValueError, match="single quote"):
        JobSpec(name="j", shell="cellmap_flow serve --model '{}'")
