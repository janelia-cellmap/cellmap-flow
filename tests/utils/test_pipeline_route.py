"""The routes that set the dashboard's chain: what each answers, and what the
viewer shows after it.

PUT /api/pipeline sets the chain and redraws the viewer through it; the
dashboard page's Submit and the pipeline builder (2 s after each edit, and
when it is left with a change unsent) both call it. /api/process and
/api/pipeline/apply, the routes those two called before, stay for a release
as its deprecated aliases, answering as they did. How each layer is built is
pinned in test_layer_sources_snapshot; here, what each route answers, the
chain it leaves configured, and the chain the viewer's layers then carry.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving.protocol import split_dataset_url

# The chain configured and drawn before each request.
SHOWN = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], "postprocess": []}
# Submit's body: the Input and Postprocess tabs' steps, every value a string.
SUBMITTED = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"},
                            {"name": "LambdaNormalizer", "expression": "x*2-1"}],
             "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "0.5"}]}
# The builder's body for /api/pipeline/apply: its nodes by type, and its edges.
APPLIED = {
    "input_normalizers": [{"id": "n1", "name": "MinMaxNormalizer", "params": {"min_value": 0, "max_value": 255}},
                          {"id": "n2", "name": "LambdaNormalizer", "params": {"expression": "x*2-1"}}],
    "postprocessors": [{"id": "p1", "name": "ThresholdPostprocessor", "params": {"threshold": 0.5}}],
    "models": [{"id": "m1", "name": "mito", "params": {}}],
    "inputs": [{"id": "i1", "params": {"dataset_path": "/data/raw.zarr"}}],
    "outputs": [],
    "edges": [{"id": "e1", "from": "i1", "to": "n1"}],
}
APPLIED_CHAIN = {"input_norm": [{"min_value": 0, "max_value": 255, "name": "MinMaxNormalizer"},
                                {"expression": "x*2-1", "name": "LambdaNormalizer"}],
                 "postprocess": [{"threshold": 0.5, "name": "ThresholdPostprocessor"}]}
UNKNOWN = {"input_norm": [{"name": "NoSuchNormalizer"}], "postprocess": []}
BAD_THRESHOLD = {"name": "ThresholdPostprocessor", "threshold": "high"}


def _received(body):
    """What Submit answers: the body it was sent, with the dashboard's address and the chain's digest."""
    return {"message": "Data received successfully",
            "received_data": {**body, "dashboard_url": "http://localhost/",
                              "digest": PipelineSpec.from_json_data(body).digest()}}


@pytest.fixture
def call(dashboard, viewer, ome_pyramid, monkeypatch):
    """``call(method, url, body)`` -> (status, the JSON answer or None), over a
    viewer drawing "mito" through SHOWN; "queued" has no host yet."""
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {"output_voxel_size": [16] * 3, "output_class": "unit"})
    get_session().dataset_path = ome_pyramid((((24, 12, 12), None),))
    get_session().jobs = [SimpleNamespace(model_name="mito", host="http://gpu:8000"), SimpleNamespace(model_name="queued", host=None)]

    def call(method, url, body):
        response = dashboard.open(url, method=method, data=json.dumps(body), content_type="application/json")
        return response.status_code, response.get_json(silent=True)

    assert call("PUT", "/api/pipeline", SHOWN)[0] == 200
    return call


def _drawn(name="mito"):
    """The chain the layer's URL carries."""
    source = get_session().viewer.state.layers[name].to_json()["source"]
    source = source[0] if isinstance(source, list) else source
    url = source["url"] if isinstance(source, dict) else source
    return PipelineSpec.from_url_blob(split_dataset_url(url))[0]


# --- PUT /api/pipeline ---------------------------------------------------------

# What the builder sends as its canvas (a missing list is empty), and what is kept.
CANVAS = {"inputs": APPLIED["inputs"], "edges": APPLIED["edges"], "normalizers": APPLIED["input_normalizers"],
          "models": [{"id": "m1", "name": "mito", "params": {}, "config": {"type": "script", "script_path": "/m.py"}}],
          "postprocessors": APPLIED["postprocessors"]}


@pytest.mark.parametrize("builder", [pytest.param(None, id="submit"), pytest.param(CANVAS, id="the-builder")])
def test_put_sets_the_chain_and_redraws_the_layers_through_it(call, builder):
    body = SUBMITTED if builder is None else {**SUBMITTED, "builder": builder}
    canvas_before = get_session().builder_state
    assert call("PUT", "/api/pipeline", body) == (200, {
        "success": True, "pipeline": SUBMITTED, "digest": PipelineSpec.from_json_data(SUBMITTED).digest(),
        "layers": ["mito"]})
    assert get_session().pipeline_spec == PipelineSpec.from_json_data(SUBMITTED)
    assert _drawn() == PipelineSpec.from_json_data(SUBMITTED)
    if builder is None:
        assert get_session().builder_state == canvas_before, "Submit leaves the builder's canvas as it was"
    else:
        assert get_session().builder_state == {**builder, "outputs": []}
        assert get_session().builder_model_configs["mito"] == {"type": "script", "script_path": "/m.py"}


def test_a_put_that_leaves_the_chain_as_drawn_leaves_the_viewer_alone(call):
    """The builder PUTs after every edit, a node dragged too. Each redrew the
    viewer, rebuilding every layer and undoing what the user set there."""
    with get_session().viewer.txn() as s:
        s.layers["mito"].visible = False
        s.layers["data"].opacity = 0.3
    moved = {**CANVAS, "inputs": [{**CANVAS["inputs"][0], "position": {"x": 50, "y": 80}}]}
    assert call("PUT", "/api/pipeline", {**SHOWN, "builder": moved})[1]["layers"] == ["mito"]
    layers = get_session().viewer.state.layers
    assert (layers["mito"].visible, layers["data"].opacity) == (False, 0.3)
    assert get_session().builder_state["inputs"] == moved["inputs"], "the canvas is kept all the same"

    # A model whose server came up since gets its layer.
    get_session().jobs = get_session().jobs + [SimpleNamespace(model_name="nuc", host="http://gpu:8001")]
    assert call("PUT", "/api/pipeline", SHOWN)[1]["layers"] == ["mito", "nuc"]
    assert _drawn("nuc") == PipelineSpec.from_json_data(SHOWN)


@pytest.mark.parametrize(
    "body, error",
    [
        pytest.param({"input_norm": []}, "postprocess: Field required", id="no-postprocess"),
        # The older {Name: {params}} form is read from files, not taken here.
        pytest.param({"input_norm": {"MinMaxNormalizer": {}}, "postprocess": []},
                     "input_norm: Input should be a valid list", id="a-dict-of-steps"),
        pytest.param(UNKNOWN, "Unknown normalizer: NoSuchNormalizer", id="unknown-normalizer"),
        pytest.param({"input_norm": [], "postprocess": [{"name": "NoSuchPostprocessor"}]},
                     "Unknown postprocessor: NoSuchPostprocessor", id="unknown-postprocessor"),
        pytest.param({"input_norm": [{"name": ["MinMaxNormalizer"]}], "postprocess": []},
                     "Unknown normalizer: ['MinMaxNormalizer']", id="a-name-that-is-not-text"),
        pytest.param({"input_norm": [], "postprocess": [BAD_THRESHOLD]}, "could not convert string to float: 'high'",
                     id="a-parameter-its-class-refuses"),
        pytest.param({**SUBMITTED, "builder": {"inputs": "i1"}}, "builder.inputs: Input should be a valid list",
                     id="a-bad-canvas"),
    ],
)
def test_a_refused_put_is_a_400_that_says_why_and_changes_nothing(call, body, error):
    canvas_before = get_session().builder_state
    assert call("PUT", "/api/pipeline", body) == (400, {"success": False, "error": error})
    assert get_session().pipeline_spec == PipelineSpec.from_json_data(SHOWN)
    assert _drawn() == PipelineSpec.from_json_data(SHOWN)
    assert get_session().builder_state == canvas_before


# --- the deprecated aliases ------------------------------------------------------


@pytest.mark.parametrize(
    "url, body, status, answer, chain",
    [
        pytest.param("/api/process", SUBMITTED, 200, _received(SUBMITTED), SUBMITTED, id="submit"),
        pytest.param("/api/pipeline/apply", APPLIED, 200,
                     {"message": "Pipeline applied successfully", "normalizers_applied": 2, "postprocessors_applied": 1},
                     APPLIED_CHAIN, id="apply"),
        pytest.param("/api/pipeline/apply", {**APPLIED, "input_normalizers": [{"id": "n1", "name": "NoSuchNormalizer"}]},
                     400, {"valid": False, "error": "Unknown normalizer: NoSuchNormalizer"}, SHOWN,
                     id="apply-unknown-normalizer"),
    ],
)
def test_each_old_route_answers_as_it_did_and_redraws_as_put_does(call, url, body, status, answer, chain):
    """``chain`` is the chain configured and drawn after the request."""
    canvas_before = get_session().builder_state
    assert call("POST", url, body) == (status, answer)
    assert get_session().pipeline_spec == PipelineSpec.from_json_data(chain)
    assert _drawn() == PipelineSpec.from_json_data(chain)
    if url == "/api/process":
        assert get_session().builder_state == canvas_before, "Submit leaves the builder's canvas as it was"


@pytest.mark.parametrize("url, body", [pytest.param("/api/process", SUBMITTED, id="submit"),
                                       pytest.param("/api/pipeline/apply", APPLIED, id="apply")])
def test_the_old_routes_say_they_are_deprecated(call, dashboard, caplog, url, body):
    with caplog.at_level(logging.WARNING, logger="cellmap_flow.dashboard.routes.pipeline"):
        response = dashboard.post(url, data=json.dumps(body), content_type="application/json")
    assert (response.headers["Deprecation"], response.headers["Link"]) == (
        "@1790726400", '</api/pipeline>; rel="successor-version"')
    assert [record.levelname for record in caplog.records if url in record.getMessage()] == ["WARNING"]
    put = dashboard.put("/api/pipeline", data=json.dumps(SUBMITTED), content_type="application/json")
    assert "Deprecation" not in put.headers


def test_a_step_a_served_model_cannot_take_is_refused_saying_which_and_why(call):
    """CellposeMasksPostprocessor on a Cellpose server serving its probability
    failed on every chunk inside the server: the page showed an empty layer."""
    from cellmap_flow.models.models_config import CellposeModelConfig

    get_session().models_config = [CellposeModelConfig(voxel_size=8, output="probability", name="cellpose_sam_v2")]
    status, answer = call("PUT", "/api/pipeline", {"input_norm": [], "postprocess": [{"name": "CellposeMasksPostprocessor"}]})
    assert status == 400 and "cellpose_sam_v2 serves Cellpose's probability" in answer["error"]
    assert "Output: All channels" in answer["error"] and _drawn() == PipelineSpec.from_json_data(SHOWN)
    get_session().models_config = [CellposeModelConfig(voxel_size=8, output="flows", name="cellpose_sam_v2_flows")]
    assert call("PUT", "/api/pipeline", {"input_norm": [], "postprocess": [{"name": "CellposeMasksPostprocessor"}]})[0] == 200
