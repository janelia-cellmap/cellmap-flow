"""A model that fails to start is reported, and never becomes a layer."""

import contextlib
import logging

import pytest

from cellmap_flow.globals import g
from cellmap_flow.utils.bsub_utils import BsubTimeoutError, JobStartError


class _Viewer:
    def __init__(self):
        self.state = type("State", (), {"layers": {}})()

    @contextlib.contextmanager
    def txn(self):
        yield self.state


@pytest.mark.parametrize("error", [JobStartError("no GPU queue took it"), BsubTimeoutError("bsub hung")])
@pytest.mark.parametrize("launcher", ["run_model", "run_hf_model"])
def test_a_failed_launch_is_logged_not_raised(monkeypatch, caplog, error, launcher):
    import cellmap_flow.models.run as run

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(run, "start_hosts", fail)
    g.viewer = _Viewer()
    g.jobs = []
    g.dataset_path = "/data/raw.zarr"

    target = "/models/mito" if launcher == "run_model" else "cellmap/mito"
    with caplog.at_level(logging.ERROR, logger="cellmap_flow.models.run"):
        getattr(run, launcher)(target, "mito", "blob")

    assert g.jobs == []
    assert g.viewer.state.layers == {}
    assert any(str(error) in r.getMessage() for r in caplog.records)



@pytest.mark.parametrize("launcher", ["run_model", "run_hf_model"])
def test_a_multi_word_server_command_is_not_quoted_as_one_program(monkeypatch, launcher):
    """The fileglancer deploy sets SERVER_COMMAND to "pixi run cellmap_flow_server".
    Quoted as one token, the shell looked for a program by that whole name."""
    import shlex

    import cellmap_flow.models.run as run
    from cellmap_flow.utils import bsub_utils

    commands = []

    def record(command, **kwargs):
        commands.append(command)
        raise JobStartError("recorded")

    monkeypatch.setattr(bsub_utils, "SERVER_COMMAND", "pixi run cellmap_flow_server")
    monkeypatch.setattr(run, "start_hosts", record)
    g.viewer = _Viewer()
    g.jobs = []
    g.dataset_path = "/data/my raw.zarr"

    target = "/models/mito" if launcher == "run_model" else "cellmap/mito"
    getattr(run, launcher)(target, "mito", "blob")

    assert len(commands) == 1
    argv = shlex.split(commands[0])
    assert argv[:3] == ["pixi", "run", "cellmap_flow_server"]
    assert argv[argv.index("-d") + 1] == "/data/my raw.zarr"
