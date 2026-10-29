"""The routes that set the dashboard's chain: what each answers, and what the
viewer shows after it.

PUT /api/pipeline sets the chain and redraws the viewer through it; the
dashboard page's Submit and the pipeline builder (2 s after each edit, and
when it is left with a change unsent) both call it. /api/process and
/api/pipeline/apply, the routes those two called before, stay for a release
as its deprecated aliases. Here, what PUT answers, the chain it leaves
configured, and the chain the viewer's layers then carry; what the
builder's PUTs leave its canvas is test_dashboard_pages'.
"""

import json
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving.protocol import split_dataset_url
from cellmap_flow.viewer.layers import raw_layer

# The chain configured and drawn before each request.
SHOWN = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}], "postprocess": []}
# Apply's body: the Processing tab's steps, every value a string.
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


@pytest.fixture
def call(dashboard, viewer, ome_pyramid, monkeypatch):
    """``call(method, url, body)`` -> (status, the JSON answer or None), over a
    viewer drawing "mito" through SHOWN; "queued" has no host yet."""
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {"output_voxel_size": [16] * 3, "output_class": "unit"})
    get_session().dataset_path = ome_pyramid((((24, 12, 12), None),))
    with viewer.txn() as s:  # every viewer is made with its raw (bootstrap.new_viewer)
        s.layers["data"] = raw_layer(get_session().dataset_path)
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


def test_put_sets_the_chain_and_redraws_the_layers_through_it(call):
    """As the dashboard page's Submit sends it."""
    canvas_before = get_session().builder_state
    assert call("PUT", "/api/pipeline", SUBMITTED) == (200, {
        "success": True, "pipeline": SUBMITTED, "digest": PipelineSpec.from_json_data(SUBMITTED).digest(),
        "layers": ["mito"]})
    assert get_session().pipeline_spec == PipelineSpec.from_json_data(SUBMITTED)
    assert _drawn() == PipelineSpec.from_json_data(SUBMITTED)
    assert get_session().builder_state == canvas_before, "Submit leaves the builder's canvas as it was"


@pytest.mark.parametrize("url, body, chain", [pytest.param("/api/process", SUBMITTED, SUBMITTED, id="submit"),
                                              pytest.param("/api/pipeline/apply", APPLIED, APPLIED_CHAIN, id="apply")])
def test_the_old_routes_still_set_the_chain_and_say_they_are_deprecated(call, dashboard, url, body, chain):
    """A script written against the routes the page called before PUT keeps
    working for a release, and is told what replaced it."""
    response = dashboard.post(url, data=json.dumps(body), content_type="application/json")
    assert response.status_code == 200
    assert response.headers["Deprecation"] and "successor-version" in response.headers["Link"]
    assert get_session().pipeline_spec == PipelineSpec.from_json_data(chain)
    assert _drawn() == PipelineSpec.from_json_data(chain)


def test_what_the_user_set_on_a_layer_survives_a_redraw(call):
    """A redraw rebuilds every layer, and neuroglancer showed each as built: a
    model's additive blending and the raw's opacity were back at their
    defaults when another model's server came up, or the chain changed."""
    viewer = get_session().viewer
    with viewer.txn() as s:
        s.layers["mito"].blend = "additive"
        s.layers["mito"].opacity = 0.6
        s.layers["data"].opacity = 0.3
    get_session().jobs = get_session().jobs + [SimpleNamespace(model_name="nuc", host="http://gpu:8001")]
    assert call("PUT", "/api/pipeline", SHOWN)[1]["layers"] == ["mito", "nuc"]
    layers = viewer.state.layers
    assert (layers["mito"].blend, layers["mito"].opacity, layers["data"].opacity) == ("additive", 0.6, 0.3)

    # A new chain that still gives an image: drawn through it, as the user set it.
    renormalized = {"input_norm": SUBMITTED["input_norm"], "postprocess": []}
    assert call("PUT", "/api/pipeline", renormalized)[0] == 200
    layers = viewer.state.layers
    assert _drawn() == PipelineSpec.from_json_data(renormalized), "drawn through the new chain"
    assert (layers["mito"].blend, layers["mito"].opacity, layers["data"].opacity) == ("additive", 0.6, 0.3)
    assert [layer.name for layer in layers] == ["data", "mito", "nuc"], "each in its place"

    # One that ends in labels makes the layer a segmentation, which has no
    # blending to keep; the raw's opacity stays.
    assert call("PUT", "/api/pipeline", SUBMITTED)[0] == 200
    layers = viewer.state.layers
    assert layers["mito"].type == "segmentation" and layers["data"].opacity == 0.3


def test_the_raw_is_drawn_again_only_through_a_new_input_normalization(call):
    """The raw is shown through the input normalizers, and a raw layer drawn
    again is a new source, whose every chunk the browser fetches again: the
    raw was black for a long while after an Apply that changed only the
    postprocessing."""
    def raw():
        return get_session().viewer.state.layers["data"].to_json()["source"]

    shown = raw()
    postprocessed = {"input_norm": SHOWN["input_norm"], "postprocess": SUBMITTED["postprocess"]}
    assert call("PUT", "/api/pipeline", postprocessed)[0] == 200
    assert _drawn() == PipelineSpec.from_json_data(postprocessed)
    assert raw() == shown
    assert call("PUT", "/api/pipeline", SUBMITTED)[0] == 200
    assert raw() != shown


@pytest.mark.parametrize(
    "body, why",
    [
        pytest.param({"input_norm": []}, "postprocess", id="no-postprocess"),
        pytest.param({"input_norm": [{"name": "NoSuchNormalizer"}], "postprocess": []}, "NoSuchNormalizer",
                     id="unknown-normalizer"),
        pytest.param({"input_norm": [], "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "high"}]},
                     "high", id="a-parameter-its-class-refuses"),
        pytest.param({**SUBMITTED, "builder": {"inputs": "i1"}}, "builder.inputs", id="a-bad-canvas"),
    ],
)
def test_a_refused_put_is_a_400_that_says_why_and_changes_nothing(call, body, why):
    """``why`` is a word of the error: the field or the value refused."""
    canvas_before = get_session().builder_state
    status, answer = call("PUT", "/api/pipeline", body)
    assert (status, answer["success"]) == (400, False) and why in answer["error"], answer
    assert get_session().pipeline_spec == PipelineSpec.from_json_data(SHOWN)
    assert _drawn() == PipelineSpec.from_json_data(SHOWN)
    assert get_session().builder_state == canvas_before


def test_a_step_a_served_model_cannot_take_is_refused_saying_which_and_why(call):
    """CellposeMasksPostprocessor on a Cellpose server serving its probability
    failed on every chunk inside the server: the page showed an empty layer."""
    from cellmap_flow.models.models_config import CellposeModelConfig

    masks = {"input_norm": [], "postprocess": [{"name": "CellposeMasksPostprocessor"}]}
    get_session().models_config = [CellposeModelConfig(voxel_size=8, output="probability", name="cellpose_sam_v2")]
    get_session().jobs = [SimpleNamespace(model_name="cellpose_sam_v2", host=None)]
    status, answer = call("PUT", "/api/pipeline", masks)
    assert status == 400 and "cellpose_sam_v2" in answer["error"] and "probability" in answer["error"], answer
    assert _drawn() == PipelineSpec.from_json_data(SHOWN)
    get_session().models_config.append(CellposeModelConfig(voxel_size=8, output="flows", name="cellpose_sam_v2_flows"))
    # The probability model stopped (unticked on the Models tab): its config stays, but it is not served.
    get_session().jobs = [SimpleNamespace(model_name="cellpose_sam_v2_flows", host=None)]
    assert call("PUT", "/api/pipeline", masks)[0] == 200


def test_the_channel_picker_lists_the_raws_channels_and_refuses_one_past_them(dashboard, ome_pyramid):
    """A light microscope's raw has a channel axis: the Processing tab's
    Channel picks the one every model reads, the chain's ChannelSelector,
    among the channels named as OME-Zarr names them. A pick past them failed
    every read, deep in a model's server."""
    import zarr

    path = ome_pyramid(channels=2)
    zarr.open_group(path, mode="r+").attrs["omero"] = {"channels": [{"label": "DAPI"}, {"label": "GFP"}]}
    get_session().dataset_path = path

    def channels():
        return dashboard.get("/api/raw/channels").get_json()

    assert channels() == {"success": True, "channels": 2, "labels": ["DAPI", "GFP"], "selected": 0}
    body = {"input_norm": [{"name": "ChannelSelector", "channel": "1"}], "postprocess": []}
    assert dashboard.put("/api/pipeline", json=body).status_code == 200
    assert channels()["selected"] == 1
    body["input_norm"][0]["channel"] = "2"
    assert dashboard.put("/api/pipeline", json=body).status_code == 400
    assert channels()["selected"] == 1
