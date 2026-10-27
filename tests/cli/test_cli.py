"""`cellmap_flow infer <type>`: what it launches, and how it fails; and the
commands before 0.3.0, which still run.

`run`, its alias before 0.3.0, launches the same way; the commands they build
are checked in tests/utils/test_launch_command.
"""

import os

import pytest
import yaml
from click.testing import CliRunner

from cellmap_flow.cli import aliases
from cellmap_flow.cli.main import cli
from cellmap_flow.dashboard.services import startup
from cellmap_flow.jobs import settings
from cellmap_flow.jobs.launch import started_jobs
from cellmap_flow.jobs.spec import JobStartError

HERE = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(HERE, "script_test", "fake_model_script.py")
RAW = os.path.join(HERE, "script_test", "dummy.zarr", "raw")
PER_TYPE = ["infer", "script", "--script-path", SCRIPT, "-d", RAW]
RUN = ["run", "-m", "script", "-c", f"script_path={SCRIPT}", "-d", RAW]  # the alias before 0.3.0


@pytest.mark.parametrize("argv, queue, project", [
    pytest.param(PER_TYPE, ["gpu_a100"], "saved_grp", id="saved"),
    pytest.param(RUN, ["gpu_a100"], "saved_grp", id="saved-through-the-run-alias"),
    # One queue is a list of one.
    pytest.param(PER_TYPE + ["-q", "gpu_h200", "-P", "grp"], ["gpu_h200"], "grp", id="explicit-q-and-P-win"),
])
def test_without_q_or_P_the_saved_queue_and_project_are_used_and_kept(monkeypatch, tmp_path, argv, queue, project):
    """-q defaulted to gpu_h100, unlike -P, and that was then saved: running a
    model without -q replaced the queue chosen in the dashboard or a YAML."""
    launched = []
    monkeypatch.setattr(settings, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    monkeypatch.setattr("cellmap_flow.jobs.launch.start_hosts",
                        lambda command, queue, charge_group, job_name, **_: launched.append((queue, charge_group)))
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path: None)
    settings.launcher_settings().queue = ["gpu_a100"]
    settings.launcher_settings().charge_group = "saved_grp"
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert launched == [(queue, project)]
    saved = yaml.safe_load((tmp_path / "server_config.yaml").read_text())
    assert (saved["queue"], saved["charge_group"]) == (queue, project)


def test_server_check_runs_one_chunk_through_the_model():
    """The check once asked the server for a chunk with one argument too
    many, and crashed before touching the model, for every type."""
    result = CliRunner().invoke(cli, PER_TYPE + ["--server-check"])
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Server check passed" in result.output


def test_a_server_that_never_came_up_exits_non_zero_with_the_reason(monkeypatch):
    """start_hosts returned a job with no host, and the viewer got zarr://None/..."""
    viewers = []

    def fail(command, job_name=None, **_):
        raise JobStartError(f"{job_name} never reported a server address")

    monkeypatch.setattr("cellmap_flow.jobs.launch.start_hosts", fail)
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda *a, **k: viewers.append(a))
    result = CliRunner().invoke(cli, PER_TYPE + ["--name", "m"])
    assert result.exit_code != 0 and "m never reported a server address" in result.output
    assert viewers == [] and started_jobs() == []


def test_an_old_console_script_says_what_replaces_it_and_runs_it(capsys):
    """An environment installed before 0.3.0 still has `cellmap_flow_yaml`
    and the other console scripts (cli/aliases.py) until the release after."""
    with pytest.raises(SystemExit) as exited:
        aliases.yaml(["--help"])
    out, err = capsys.readouterr()
    assert exited.value.code == 0 and "cellmap_flow yaml" in err
    assert out.startswith("Usage: cellmap_flow yaml ")
