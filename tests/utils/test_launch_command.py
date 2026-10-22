"""Every launcher starts its server through serving.launch.launch.

The command is SERVER_COMMAND, then `--model <the model's launch entry, as
JSON>` and the data path. SERVER_COMMAND is read when a command is built and
split into words, so a deploy's override (fileglancer's "pixi run
cellmap_flow serve") reaches every launcher; cellmap_flow and
cellmap_flow_yaml used to copy it at import, and the dashboard quoted it as
one word. The data path, with a space in it, goes through the one data_path
+ scale rule: the YAML's `scale: s3` next to a path ending in s3 used to
become .../s3/s3.
"""

import json
import shlex
import subprocess
import sys

import pytest
import zarr
from click.testing import CliRunner

from cellmap_flow.cli import yaml_cli
from cellmap_flow.cli.main import cli
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.dashboard.services import launch as dashboard_launch
from cellmap_flow.models.models_config import HuggingFaceModelConfig, ScriptModelConfig
from cellmap_flow.jobs import launch as jobs_launch
from cellmap_flow.jobs.settings import LauncherSettings
from cellmap_flow.jobs.spec import JobSpec

def _cli(*argv):
    result = CliRunner().invoke(cli, list(argv))
    assert result.exit_code == 0, result.output + repr(result.exception)


# One launcher of each kind: the CLI, a YAML and the dashboard.
# launcher: (launch it on the data path, the model entry it passes, the flags after the data path)
LAUNCHERS = {
    "cellmap_flow-infer-no-resample": (
        lambda data: _cli("infer", "script", "--script-path", "/s.py", "--name", "m", "-d", data, "--no-resample"),
        {"type": "script", "script_path": "/s.py", "name": "m"},
        ["--no-resample"],
    ),
    "cellmap_flow-yaml": (
        lambda data: yaml_cli.run_multiple([ScriptModelConfig(script_path="/s.py", name="m", scale="s3")], data, "grp", "q"),
        {"type": "script", "script_path": "/s.py", "name": "m", "scale": "s3"},
        ["--resample"],
    ),
    "dashboard-huggingface": (
        lambda data: dashboard_launch.run_model(HuggingFaceModelConfig(repo="cellmap/mito-v1", name="mito_v1"), "blob"),
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

    monkeypatch.setattr(jobs_launch, "SERVER_COMMAND", "pixi run cellmap_flow serve")
    monkeypatch.setattr(jobs_launch, "start_hosts", started)
    monkeypatch.setattr(dashboard_launch, "_show", lambda job, st_data: None)
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path, wrap_raw=True: None)
    monkeypatch.setattr(LauncherSettings, "save", lambda self: None)
    get_session().dataset_path = data
    return commands, data


@pytest.mark.parametrize("launcher", list(LAUNCHERS))
def test_each_launcher_submits_the_split_server_command(launched, launcher):
    """Split as a local run splits it (jobs.local.run): the deploy's
    SERVER_COMMAND as words, then the model's entry and the data path whole."""
    commands, data = launched
    launch_it, entry, flags = LAUNCHERS[launcher]
    launch_it(data)
    (argv,) = [shlex.split(c) for c in commands]
    assert argv[:4] == ["pixi", "run", "cellmap_flow", "serve"] and argv[4:8:2] == ["--model", "-d"]
    assert (json.loads(argv[5]), argv[7], argv[8:]) == (entry, data, flags)


@pytest.mark.parametrize("launcher", list(LAUNCHERS))
def test_each_launcher_s_command_reaches_the_server_whole_through_lsf(launched, launcher, monkeypatch, tmp_path):
    """LSF runs a job's `bash -c` line wrapped in single quotes of its own, so
    a single-quoted --model JSON came out as "type:huggingface" (2026-10-01)."""
    commands, data = launched
    launch_it, entry, flags = LAUNCHERS[launcher]
    printer = tmp_path / "print_argv.py"
    printer.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    monkeypatch.setattr(jobs_launch, "SERVER_COMMAND", f"{sys.executable} {printer}")
    launch_it(data)
    (command,) = commands
    as_lsf_runs_it = subprocess.run(["sh", "-c", f"bash -c '{command}'"], capture_output=True, text=True, check=True)
    model, entry_json, d, path, *passed = json.loads(as_lsf_runs_it.stdout)
    assert (model, json.loads(entry_json), d, path, passed) == ("--model", entry, "-d", data, flags)


def test_a_job_s_shell_line_with_a_single_quote_is_refused():
    with pytest.raises(ValueError, match="single quote"):
        JobSpec(name="j", shell="cellmap_flow serve --model '{}'")
