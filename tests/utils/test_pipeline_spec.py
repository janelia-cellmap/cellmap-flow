"""The chain's formats that leave the process, and the process's chain state.

A chain reaches inference servers (older or newer than the dashboard) as JSON
in the layer URL, the YAML and blockwise paths as json_data, and the trainer
through current_*_config(). How each op rebuilds from its to_dict() is in
test_ops; the layer URL's bytes, which the browser must compute alike, are
in web/test/parity/args; a legacy json_data dict is run in test_yaml_cli, and
a strict one refused in the blockwise precheck (test_blockwise_processor).
"""

import json

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.launch import started_jobs
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.process_chain import process_chain
from cellmap_flow.serving.protocol import ARGS_KEY, decode_to_json

# --- the dashboard's chain state -------------------------------------------------

# What the Input/Output tabs post: every value a string, name first.
POSTED = {
    "input_norm": [
        {"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255", "invert": "false"},
        {"name": "LambdaNormalizer", "expression": "x*2-1"},
    ],
    "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "0.5"}],
}


def _send(dashboard, url, payload, method="POST"):
    # As a browser sends it: the test client's json= sorts the keys.
    response = dashboard.open(url, method=method, data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200, response.data
    return response.get_json()


def _layer_source(name):
    source = get_session().viewer.state.layers[name].to_json()["source"]
    return source[0] if isinstance(source, list) else source


@pytest.fixture
def submit(dashboard, viewer, ome_pyramid, monkeypatch):
    """Submit from the Input/Output tabs with two jobs, one without a host yet.
    The model writes 16 nm voxels with outputs in [0, 1], over raw at 24, 12, 12 nm."""
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {"output_voxel_size": [16] * 3, "output_class": "unit"})
    get_session().dataset_path = ome_pyramid((((24, 12, 12), None),))
    started_jobs().extend(type("Job", (), {"model_name": name, "host": host})() for name, host in
                          [("mito", "http://gpu:8000"), ("pending", None)])
    return lambda payload=POSTED: _send(dashboard, "/api/pipeline", payload, method="PUT")


def test_submit_keeps_the_posted_chain_as_the_config(submit):
    """Finetune jobs, the manifest and the exported YAML read the config: it
    must be the normalization inference uses."""
    chain = process_chain()
    submit()
    assert list(chain.spec.input_norm) == POSTED["input_norm"]
    assert list(chain.spec.postprocess) == POSTED["postprocess"]
    assert [type(n).__name__ for n in chain.input_norms] == ["MinMaxNormalizer", "LambdaNormalizer"]


def test_submit_names_the_layer_by_the_chains_digest(submit):
    """Stamped with the time, every Submit made neuroglancer refetch and each
    server rebuild its chain (and lose its merger state)."""
    received = submit()
    source = _layer_source("mito")
    blob = decode_to_json(source["url"].split(ARGS_KEY)[1])
    assert blob["input_norm"] == POSTED["input_norm"] and blob["dashboard_url"] == "http://localhost/"
    assert blob["digest"] == received["digest"] == PipelineSpec.from_json_data(POSTED).digest()
    assert "time" not in received
    submit()
    assert _layer_source("mito") == source, "the same settings, the same source"
    submit(dict(POSTED, postprocess=[dict(POSTED["postprocess"][0], threshold="0.6")]))
    assert _layer_source("mito") != source, "a changed parameter, a new one"


def test_prediction_layers_are_drawn_at_the_raws_scale_and_only_with_a_host(submit):
    submit()
    dimensions = _layer_source("mito")["transform"]["outputDimensions"]
    assert (dimensions["z"][0], dimensions["x"][0]) == pytest.approx((24e-9, 12e-9)), "z and x kept apart"
    assert "pending" not in get_session().viewer.state.layers, "a job with no host yet gets no layer"


def test_a_layer_is_a_segmentation_or_an_image_shaded_over_the_outputs_range(submit):
    submit()
    assert get_session().viewer.state.layers["mito"].type == "segmentation", "a threshold's labels"
    submit(dict(POSTED, postprocess=[]))
    mito = get_session().viewer.state.layers["mito"]
    assert mito.type == "image" and "range=[0, 1]" in mito.shader


def test_apply_keeps_the_builders_steps_in_order(dashboard):
    chain = process_chain()
    nodes = {
        "input_normalizers": [
            {"id": "n1", "name": "MinMaxNormalizer", "params": {"min_value": 0, "max_value": 255}},
            {"id": "n2", "name": "LambdaNormalizer", "params": {"expression": "x+1"}},
            {"id": "n3", "name": "LambdaNormalizer", "params": {"expression": "x*2"}},  # the same op twice
        ],
        "postprocessors": [{"id": "p1", "name": "SigmoidPostprocessor"}],
        "models": [{"id": "m1", "name": "mito", "config": {"type": "script", "script_path": "/m.py"}}],
    }
    _send(dashboard, "/api/pipeline/apply", nodes)
    assert list(chain.spec.input_norm) == [
        {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
        {"name": "LambdaNormalizer", "expression": "x+1"},
        {"name": "LambdaNormalizer", "expression": "x*2"},
    ]
    assert [n.expression for n in chain.input_norms[1:]] == ["x+1", "x*2"]
    assert list(chain.spec.postprocess) == [{"name": "SigmoidPostprocessor"}]
    assert get_session().builder_state["normalizers"] == nodes["input_normalizers"]
    assert get_session().builder_model_configs["mito"] == {"type": "script", "script_path": "/m.py"}
