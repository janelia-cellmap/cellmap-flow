"""A model that fails to start is reported, and never becomes a layer."""

import contextlib
import logging

import pytest
from flask import Flask

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


class _Job:
    def __init__(self, name, host):
        self.model_name, self.host = name, host


def test_submit_skips_jobs_that_have_no_host_yet(monkeypatch):
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "get_raw_layer", lambda path: type("Raw", (), {"shader": None})()
    )
    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {})
    g.viewer = _Viewer()
    g.jobs = [_Job("pending", None), _Job("mito", "http://gpu-node:8000")]
    g.dataset_path = "/data/raw.zarr"
    g.shaders, g.shader_controls = {}, {}

    app = Flask(__name__)
    app.register_blueprint(pipeline.pipeline_bp)
    response = app.test_client().post(
        "/api/process", json={"input_norm": [], "postprocess": []}
    )

    assert response.status_code == 200
    assert "pending" not in g.viewer.state.layers
    assert "mito" in g.viewer.state.layers
