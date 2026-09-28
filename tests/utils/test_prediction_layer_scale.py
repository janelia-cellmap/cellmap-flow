"""Prediction layers are drawn at the raw's closest scale, on every path."""

import contextlib

import pytest

from flask import Flask

from cellmap_flow.globals import g


def _dimensions(source):
    if isinstance(source, list):  # a layer's to_json() lists its sources
        source = source[0]
    assert isinstance(source, dict), f"no transform on {source!r}"
    return source["transform"]["outputDimensions"]


def test_override_scales_keep_z_and_x_apart():
    from cellmap_flow.utils.neuroglancer_utils import build_prediction_source

    # z, y, x -- anisotropic, as get_raw_closest_scale returns it.
    source = build_prediction_source("http://h:1", "mito", "blob", (40, 8, 4))
    dims = _dimensions(source)
    assert dims["z"] == [40e-9, "m"]
    assert dims["y"] == [8e-9, "m"]
    assert dims["x"] == [4e-9, "m"]


class _Job:
    model_name = "mito"
    host = "http://gpu-node:8000"


class _Viewer:
    def __init__(self):
        self.state = type("State", (), {"layers": {}})()

    @contextlib.contextmanager
    def txn(self):
        yield self.state


def test_submit_rebuilds_prediction_layers_with_the_scale_override(monkeypatch):
    import cellmap_flow.dashboard.routes.pipeline as pipeline
    import cellmap_flow.utils.neuroglancer_utils as ngu

    monkeypatch.setattr(
        pipeline, "get_raw_layer", lambda path: type("Raw", (), {"shader": None})()
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_model_info",
        lambda host: {"output_voxel_size": [16, 16, 16], "output_class": "unit"},
    )
    monkeypatch.setattr(ngu, "get_raw_closest_scale", lambda path, vs: (24, 12, 12))
    g.viewer = _Viewer()
    g.jobs = [_Job()]
    g.dataset_path = "/data/raw.zarr"
    g.shaders, g.shader_controls = {}, {}

    app = Flask(__name__)
    app.register_blueprint(pipeline.pipeline_bp)
    response = app.test_client().post(
        "/api/process", json={"input_norm": [], "postprocess": []}
    )

    assert response.status_code == 200
    layer = g.viewer.state.layers["mito"]
    dims = _dimensions(layer.to_json()["source"])
    assert dims["z"][0] == pytest.approx(24e-9)
    assert dims["x"][0] == pytest.approx(12e-9)
