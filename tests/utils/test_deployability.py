"""cellmap-flow stays launchable the way Fileglancer launches it.

runnables.yaml is the manifest Fileglancer reads. It runs `cellmap_flow`
subcommands through `pixi run`, learns the dashboard's URL from the file named by
SERVICE_URL_PATH, and bills the models picked in the dashboard to the job's
LSF project. These used to live on a separate deploy branch, which is how
they drifted from main.
"""

import importlib
import os
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import click
import yaml

import cellmap_flow
from cellmap_flow.globals import g

ROOT = Path(__file__).resolve().parents[2]


def _command(words):
    """The click command that ``words``, a console script and its subcommands, runs."""
    from cellmap_flow.cli import main

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    script, *subcommands = words
    module, attr = scripts[script].split(":")
    command = getattr(importlib.import_module(module), attr)
    if command is main.main:  # the cellmap_flow script: main() runs the group
        command = main.cli
    assert isinstance(command, click.Command), words
    for name in subcommands:
        command = command.get_command(click.Context(command), name)
        assert command is not None, words
    return command


def test_the_manifest_runs_installed_scripts_with_flags_they_accept():
    manifest = yaml.safe_load((ROOT / "runnables.yaml").read_text())
    assert manifest["version"] == cellmap_flow.__version__
    assert any(r.startswith("pixi") for r in manifest["requirements"])

    for runnable in manifest["runnables"]:
        pixi, run, *words = runnable["command"].split()
        assert (pixi, run) == ("pixi", "run"), runnable["command"]
        command = _command(words)
        script = runnable["command"]
        options = {opt for p in command.params for opt in p.opts}
        positionals = [p for p in command.params if isinstance(p, click.Argument)]
        for param in runnable["parameters"]:
            if "flag" in param:
                assert param["flag"] in options, (script, param["flag"])
            else:
                assert positionals, (script, param["name"])


def test_the_dashboard_writes_its_url_where_fileglancer_looks(tmp_path, monkeypatch):
    import werkzeug.serving

    from cellmap_flow.dashboard import app as dashboard

    class FakeServer:
        def __init__(self, host, port, wsgi_app, threaded=False, **kwargs):
            assert threaded, "one request at a time would stall the dashboard's polls"
            self.socket = socket.socket()
            self.socket.bind((host, port))  # port 0: the OS picks, as in production

        def serve_forever(self):
            self.socket.close()

    monkeypatch.setattr(werkzeug.serving, "make_server", FakeServer)
    url_file = tmp_path / "service_url"
    monkeypatch.setenv("SERVICE_URL_PATH", str(url_file))

    dashboard.create_and_run_app(neuroglancer_url="http://ng")

    url = url_file.read_text()
    host, port = url.removeprefix("http://").rsplit(":", 1)
    assert host == socket.gethostname() and int(port) > 0


def test_the_server_command_is_read_from_the_environment():
    code = "from cellmap_flow.jobs.launch import SERVER_COMMAND; print(SERVER_COMMAND)"
    env = {**os.environ, "CELLMAP_FLOW_SERVER_COMMAND": "pixi run cellmap_flow_server"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "pixi run cellmap_flow_server"


def test_the_viewer_bills_models_to_the_launching_jobs_project(monkeypatch, tmp_path):
    import neuroglancer
    from click.testing import CliRunner

    from cellmap_flow.cli.main import cli
    from cellmap_flow.dashboard import app as dashboard
    from cellmap_flow.jobs import launch
    from cellmap_flow.viewer import raw

    class FakeViewer:
        def txn(self):
            import contextlib, types
            return contextlib.nullcontext(types.SimpleNamespace(layers={}, dimensions=None))

    monkeypatch.setattr(neuroglancer, "Viewer", FakeViewer)
    raw_layers = []
    monkeypatch.setattr(raw, "get_raw_layer", lambda path: raw_layers.append(path) or "raw")
    monkeypatch.setattr(launch, "install_cleanup_handlers", lambda: True)
    started = []
    monkeypatch.setattr(dashboard, "create_and_run_app", lambda **k: started.append(k))
    monkeypatch.setenv("LSB_PROJECT_NAME", "cellmap-fileglancer")

    result = CliRunner().invoke(cli, ["view", "-d", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert g.charge_group == "cellmap-fileglancer" and started and raw_layers == [str(tmp_path)]

    result = CliRunner().invoke(cli, ["view", "-d", str(tmp_path), "-P", "explicit"])
    assert result.exit_code == 0 and g.charge_group == "explicit"
