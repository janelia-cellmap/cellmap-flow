"""What the dashboard's two server-rendered pages hand to the browser."""

import json
import re
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.process_chain import process_chain


def _job(name):
    return SimpleNamespace(model_name=name, host="http://node:8000")


def test_rendering_the_index_leaves_the_model_catalog_alone(dashboard):
    catalog_before = {k: dict(v) for k, v in get_session().model_catalog.items()}
    get_session().jobs = [_job("mito_server")]
    html = dashboard.get("/").get_data(as_text=True)
    assert 'value="mito_server"' in html, "the running model is listed on the Models tab"
    assert get_session().model_catalog == catalog_before, "but not added to the catalog everything else reads"


@pytest.mark.parametrize("headers, expected", [
    pytest.param({}, "http://node7:8765/v/abc/", id="direct"),
    pytest.param({"X-Forwarded-Host": "proxy.example.org"}, "http://proxy.example.org/v/abc/", id="proxy"),
    pytest.param({"X-Forwarded-Host": "proxy.example.org", "X-Forwarded-Proto": "https"},
                 "https://proxy.example.org/v/abc/", id="https-proxy"),
])
def test_behind_a_reverse_proxy_the_viewer_is_loaded_through_it(dashboard, headers, expected):
    get_session().neuroglancer_url = "http://node7:8765/v/abc/"
    assert f'<iframe src="{expected}"' in dashboard.get("/", headers=headers).get_data(as_text=True)


def _rows(html, row_class, checkbox_class):
    """(name, checked, {param: value}) for each list row, in page order."""
    rows = []
    for block in re.split(rf'<div class="{row_class} ', html)[1:]:
        checkbox = re.search(rf'<input[^>]*class="form-check-input {checkbox_class}"[^>]*>', block, re.S).group(0)
        rows.append((re.search(r'value="([^"]*)"', checkbox).group(1), re.search(r"\schecked\b", checkbox) is not None,
                     dict(re.findall(r'data-param="([^"]*)"\s*value="([^"]*)"', block))))
    return rows


INPUT_CHAIN = [  # registry order is MinMax before Lambda; this runs Lambda first, and twice
    {"name": "LambdaNormalizer", "expression": "x*2"},
    {"name": "MinMaxNormalizer", "min_value": 10, "max_value": 20},
    {"name": "LambdaNormalizer", "expression": "x-1"},
]
POST_CHAIN = [
    {"name": "ThresholdPostprocessor", "threshold": 0.7},
    {"name": "SigmoidPostprocessor"},
    {"name": "ChannelSelection", "channels": [0, 2]},
]


@pytest.mark.parametrize(
    "kind, chain, listed",
    [
        pytest.param("input", INPUT_CHAIN, [
            ("LambdaNormalizer", True, {"expression": "x*2"}),
            ("MinMaxNormalizer", True, {"min_value": "10.0", "max_value": "20.0", "invert": "False"}),
            ("LambdaNormalizer", True, {"expression": "x-1"}),
        ], id="input-norms"),
        pytest.param("postprocess", POST_CHAIN, [
            ("ThresholdPostprocessor", True, {"threshold": "0.7"}),
            ("SigmoidPostprocessor", True, {}),
            ("ChannelSelection", True, {"channels": "0,2"}),
        ], id="postprocessors"),
    ],
)
def test_the_tabs_list_the_configured_chain_first_in_its_order(dashboard, kind, chain, listed):
    from cellmap_flow.norm.input_normalize import get_normalizations
    from cellmap_flow.post.postprocessors import get_postprocessors

    if kind == "input":
        process_chain().input_norms = get_normalizations(chain)
    else:
        process_chain().postprocess = get_postprocessors(chain)
    html = dashboard.get("/").get_data(as_text=True)
    rows = _rows(html, *(("normalizer-item", "inputNormCheckbox") if kind == "input"
                         else ("postprocessor-item", "postProcessCheckbox")))

    assert rows[:3] == listed
    rest = rows[3:]  # every other op once, unticked, after the chain
    assert rest and not any(checked for _, checked, _ in rest)
    assert not {name for name, _, _ in rest} & {name for name, _, _ in listed}
    # Two rows of the same op must not share element ids.
    ids = re.findall(r'\bid="((?:inputNorm|postProcess)[^"]*)"', html)
    assert len(ids) == len(set(ids))
    if kind == "postprocess":  # what Submit All sends back builds the same step
        (sent,) = get_postprocessors([{"name": "ChannelSelection", **rows[2][2]}])
        assert sent.channels == [0, 2]


def test_with_nothing_configured_every_op_is_listed_once_unticked(dashboard):
    from cellmap_flow.norm.input_normalize import get_input_normalizers

    process_chain().input_norms = []
    rows = _rows(dashboard.get("/").get_data(as_text=True), "normalizer-item", "inputNormCheckbox")
    assert [name for name, _, _ in rows] == [op["name"] for op in get_input_normalizers()]
    assert not any(checked for _, checked, _ in rows)


def _page_data(html):
    return json.loads(re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1))


def _builder_state(dashboard):
    """The pipeline state the builder page starts from."""
    return _page_data(dashboard.get("/pipeline-builder").get_data(as_text=True))["pipeline"]


def _edges(*pairs):
    return [{"id": f"e{i}", "from": a, "to": b} for i, (a, b) in enumerate(pairs, 1)]


# A canvas as the builder's apply leaves it (PUT /api/pipeline's "builder"):
# INPUT -> Lambda -> MinMax -> mito -> Threshold -> OUTPUT, with the INPUT's
# boxes and the OUTPUT's channels, and the chain those nodes stand for.
CANVAS = {
    "inputs": [{"id": "input-1", "position": {"x": 11, "y": 22}, "params": {
        "dataset_path": "/data/raw.zarr", "bounding_boxes": [{"offset": [0, 0, 0], "shape": [8, 8, 8]}]}}],
    "outputs": [{"id": "output-1", "params": {"dataset_path": "/out.zarr", "output_channels": ["mito"]},
                 "position": {"x": 900, "y": 22}}],
    "edges": _edges(("input-1", "norm-1"), ("norm-1", "norm-2"), ("norm-2", "model-1"),
                    ("model-1", "post-1"), ("post-1", "output-1")),
    "normalizers": [
        {"id": "norm-1", "name": "LambdaNormalizer", "params": {"expression": "x*2"}, "position": {"x": 200, "y": 20}},
        {"id": "norm-2", "name": "MinMaxNormalizer", "params": {"min_value": 0, "max_value": 255},
         "position": {"x": 580, "y": 20}},
    ],
    "models": [{"id": "model-1", "name": "mito", "params": {}, "position": {"x": 400, "y": 22}}],
    "postprocessors": [{"id": "post-1", "name": "ThresholdPostprocessor", "params": {"threshold": 0.5},
                        "position": {"x": 600, "y": 20}}],
}
APPLY = {"input_norm": [{"expression": "x*2", "name": "LambdaNormalizer"},
                        {"min_value": 0, "max_value": 255, "name": "MinMaxNormalizer"}],
         "postprocess": [{"threshold": 0.5, "name": "ThresholdPostprocessor"}],
         "builder": CANVAS}
# INPUT -> mito -> OUTPUT: a canvas with no steps, whose chain is empty.
NO_STEPS = {**CANVAS, "normalizers": [], "postprocessors": [],
            "edges": _edges(("input-1", "model-1"), ("model-1", "output-1"))}
APPLY_NO_STEPS = {"input_norm": [], "postprocess": [], "builder": NO_STEPS}
# What Submit on the dashboard page sends (every parameter as its field's
# text): one parameter changed, and then other steps.
SUBMIT_A_PARAMETER = {"input_norm": [{"name": "LambdaNormalizer", "expression": "x*2"},
                                     {"name": "MinMaxNormalizer", "min_value": "0", "max_value": "200"}],
                      "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "0.5"}]}
SUBMIT_OTHER_STEPS = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"},
                                     {"name": "ZScoreNormalizer", "mean": "1", "std": "2"}],
                      "postprocess": []}
LAMBDA, MINMAX = CANVAS["normalizers"]


def _new_ids(state):
    """``state`` with each id the page made up (``<prefix>-<i>-<ms>``) as ``<prefix>-<i>-new``."""
    return json.loads(re.sub(r'"((?:norm|post|model)-\d+)-\d{13}"', r'"\1-new"', json.dumps(state)))


@pytest.mark.parametrize("sent, expected", [
    pytest.param([APPLY], CANVAS, id="the-builders-own-apply"),
    # Any saved node counts as a saved canvas, not only a normalizer.
    pytest.param([APPLY_NO_STEPS], NO_STEPS, id="an-apply-with-no-steps"),
    # A step whose parameters changed keeps its node's id and position.
    pytest.param([APPLY, SUBMIT_A_PARAMETER], {
        **CANVAS,
        "normalizers": [LAMBDA, {**MINMAX, "params": {"min_value": "0", "max_value": "200"}}],
        "postprocessors": [{**CANVAS["postprocessors"][0], "params": {"threshold": "0.5"}}],
    }, id="then-submit-changed-a-parameter"),
    # A new step gets a new node, with no position (the page places it); a
    # step that went takes its node's edges with it.
    pytest.param([APPLY, SUBMIT_OTHER_STEPS], {
        **CANVAS,
        "normalizers": [{**MINMAX, "params": {"min_value": "0", "max_value": "255"}},
                        {"id": "norm-1-new", "name": "ZScoreNormalizer", "params": {"mean": "1", "std": "2"}}],
        "postprocessors": [],
        "edges": [edge for edge in CANVAS["edges"] if edge["id"] == "e3"],  # norm-2 -> model-1
    }, id="then-submit-changed-the-steps"),
    # With nothing applied, the steps as Submit sent them, not as the live
    # ops' to_dict() has them (typed, with every default).
    pytest.param([SUBMIT_A_PARAMETER], {
        "inputs": [], "outputs": [], "edges": [],
        "normalizers": [{"id": "norm-0-new", "name": "LambdaNormalizer", "params": {"expression": "x*2"}},
                        {"id": "norm-1-new", "name": "MinMaxNormalizer",
                         "params": {"min_value": "0", "max_value": "200"}}],
        "models": [{"id": "model-0-new", "name": "other_model", "params": {}}],
        "postprocessors": [{"id": "post-0-new", "name": "ThresholdPostprocessor", "params": {"threshold": "0.5"}}],
    }, id="submit-before-any-apply"),
])
def test_the_builder_opens_on_the_live_chain_and_its_last_canvas(dashboard, sent, expected):
    """The normalizer and postprocessor nodes are the live chain, however it
    was last set; everything else is the builder's last canvas. ``sent`` are
    the PUT /api/pipeline bodies that set the chain, in order."""
    from cellmap_flow.pipeline_spec import PipelineSpec, builder_steps

    get_session().jobs = [_job("other_model")]
    for body in sent:
        assert dashboard.put("/api/pipeline", json=body).status_code == 200
    state = _builder_state(dashboard)

    assert _new_ids(state) == expected
    # So the page's first edit, which sends its nodes' steps, sends the live chain back.
    assert PipelineSpec(builder_steps(state["normalizers"]), builder_steps(state["postprocessors"])) == (
        get_session().pipeline_spec)


def test_before_anything_is_applied_the_builder_starts_from_the_live_chain(dashboard):
    from cellmap_flow.norm.input_normalize import get_normalizations

    process_chain().input_norms = get_normalizations([{"name": "ZScoreNormalizer", "mean": 1, "std": 2}])
    get_session().jobs = [_job("mito")]

    state = _builder_state(dashboard)
    assert [(n["name"], n["params"]) for n in state["normalizers"]] == [("ZScoreNormalizer", {"mean": 1.0, "std": 2.0})]
    assert [m["name"] for m in state["models"]] == ["mito"] and state["inputs"] == state["edges"] == []


class _Configured(SimpleNamespace):
    def to_dict(self):
        return {"type": "script", "script_path": f"/{self.name}.py"}


@pytest.mark.parametrize("applied", [pytest.param(False, id="nothing-applied"), pytest.param(True, id="applied")])
def test_each_model_node_carries_its_config(dashboard, applied):
    """From the configured model of its name, else (nothing applied yet) from
    the YAML the builder imported; the palette lists both kinds of model."""
    get_session().models_config = [_Configured(name="mito")]
    get_session().builder_model_configs = {"nuc": {"type": "script", "script_path": "/imported.py"}}
    get_session().model_catalog = {"catalog": {"er": "/models/er"}}
    if applied:
        get_session().builder_state = {
            **get_session().builder_state,
            "inputs": [{"id": "input-1", "params": {}}],
            "models": [{"id": "model-1", "name": "mito", "params": {}},
                       {"id": "model-2", "name": "nuc", "params": {}, "config": {"type": "given"}}],
        }
    else:
        get_session().jobs = [_job("mito"), _job("nuc"), _job("unknown")]
    html = dashboard.get("/pipeline-builder").get_data(as_text=True)

    models = _builder_state(dashboard)["models"]
    configs = {m["name"]: m.get("config") for m in models}
    assert configs == ({"mito": {"type": "script", "script_path": "/mito.py"}, "nuc": {"type": "given"}} if applied else
                       {"mito": {"type": "script", "script_path": "/mito.py"},
                        "nuc": {"type": "script", "script_path": "/imported.py"}, "unknown": None})
    palette = _page_data(html)["available_models"]
    assert palette == {"catalog/er": {"name": "catalog/er", "category": "catalog", "model_name": "er",
                                      "path": "/models/er"},
                       "mito": {"name": "mito", "type": "script", "script_path": "/mito.py"}}


@pytest.mark.parametrize("page", ["/", "/pipeline-builder"])
def test_both_pages_hand_their_scripts_each_ops_schema(dashboard, page):
    from cellmap_flow.norm.input_normalize import get_input_normalizers
    from cellmap_flow.post.postprocessors import get_postprocessors_list

    schemas = _page_data(dashboard.get(page).get_data(as_text=True))["op_schemas"]
    assert [s["name"] for s in schemas["input_norm"]] == [op["name"] for op in get_input_normalizers()]
    assert [s["name"] for s in schemas["postprocess"]] == [op["name"] for op in get_postprocessors_list()]
    assert all(s["schema"]["type"] == "object" for kind in schemas.values() for s in kind)
