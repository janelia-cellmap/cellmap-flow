"""The routes that set the dashboard's chain: what each answers, and what the
viewer shows after it.

The dashboard page's Submit posts its two chains to /api/process. The
pipeline builder posts its nodes to /api/pipeline/apply 2 s after each edit,
and when it is left with a change unsent. How each layer is built is pinned
in test_layer_sources_snapshot; here, what each route answers, the chain it
leaves configured, and the chain the viewer's layers then carry.
"""

import json
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.globals import g
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.utils.web_utils import ARGS_KEY, decode_to_json

# The chain configured and drawn before each request.
SHOWN = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], "postprocess": []}
# Submit's body: the Input and Postprocess tabs' steps, every value a string.
SUBMITTED = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"},
                            {"name": "LambdaNormalizer", "expression": "x*2-1"}],
             "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "0.5"}]}
# The builder's body: its nodes by type, and its edges.
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
def post(dashboard, viewer, ome_pyramid, monkeypatch):
    """``post(url, body)`` -> (status, the JSON answer or None), over a viewer
    drawing "mito" through SHOWN; "queued" has no host yet."""
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {"output_voxel_size": [16] * 3, "output_class": "unit"})
    g.dataset_path = ome_pyramid((((24, 12, 12), None),))
    g.jobs = [SimpleNamespace(model_name="mito", host="http://gpu:8000"), SimpleNamespace(model_name="queued", host=None)]

    def post(url, body):
        response = dashboard.post(url, data=json.dumps(body), content_type="application/json")
        return response.status_code, response.get_json(silent=True)

    assert post("/api/process", SHOWN)[0] == 200
    return post


def _drawn(name="mito"):
    """The chain the layer's URL carries."""
    source = g.viewer.state.layers[name].to_json()["source"]
    source = source[0] if isinstance(source, list) else source
    url = source["url"] if isinstance(source, dict) else source
    blob = decode_to_json(url.split(ARGS_KEY)[1])
    return {key: blob[key] for key in ("input_norm", "postprocess")}


@pytest.mark.parametrize(
    "url, body, status, answer, configured, drawn",
    [
        pytest.param("/api/process", SUBMITTED, 200, _received(SUBMITTED), SUBMITTED, SUBMITTED, id="submit"),
        # An op it does not know is kept in the chain, and skipped where it is built.
        pytest.param("/api/process", UNKNOWN, 200, _received(UNKNOWN), UNKNOWN, UNKNOWN, id="submit-unknown-op"),
        # Flask's 500 page.
        pytest.param("/api/process", {"input_norm": []}, 500, None, SHOWN, SHOWN, id="submit-without-postprocess"),
        pytest.param("/api/process", {"input_norm": [], "postprocess": [BAD_THRESHOLD]}, 500, None, SHOWN, SHOWN,
                     id="submit-bad-parameter"),
        # The chain changes, and the layers keep drawing the old one.
        pytest.param("/api/pipeline/apply", APPLIED, 200,
                     {"message": "Pipeline applied successfully", "normalizers_applied": 2, "postprocessors_applied": 1},
                     APPLIED_CHAIN, SHOWN, id="apply"),
        pytest.param("/api/pipeline/apply", {**APPLIED, "input_normalizers": [{"id": "n1", "name": "NoSuchNormalizer"}]},
                     400, {"valid": False, "error": "Unknown normalizer: NoSuchNormalizer"}, SHOWN, SHOWN,
                     id="apply-unknown-normalizer"),
        pytest.param("/api/pipeline/apply", {**APPLIED, "postprocessors": [{"id": "p1", "name": "NoSuchPostprocessor"}]},
                     400, {"valid": False, "error": "Unknown postprocessor: NoSuchPostprocessor"}, SHOWN, SHOWN,
                     id="apply-unknown-postprocessor"),
        pytest.param("/api/pipeline/apply",
                     {**APPLIED, "postprocessors": [{"id": "p1", "name": "ThresholdPostprocessor",
                                                     "params": {"threshold": "high"}}]},
                     500, {"error": "could not convert string to float: 'high'"}, SHOWN, SHOWN, id="apply-bad-parameter"),
    ],
)
def test_what_each_route_answers_and_the_chain_it_leaves(post, url, body, status, answer, configured, drawn):
    builder_before = get_session().builder_state
    assert post(url, body) == (status, answer)
    assert g.pipeline_spec == PipelineSpec.from_json_data(configured)
    assert _drawn() == drawn
    if url == "/api/process":
        assert get_session().builder_state == builder_before, "Submit leaves the builder's canvas as it was"
