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
