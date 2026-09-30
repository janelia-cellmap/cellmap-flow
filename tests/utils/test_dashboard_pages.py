"""What the dashboard's two server-rendered pages hand to the browser."""

import json
import re
from types import SimpleNamespace

import pytest

from cellmap_flow.globals import g


def _job(name):
    return SimpleNamespace(model_name=name, host="http://node:8000")


def test_rendering_the_index_leaves_the_model_catalog_alone(dashboard):
    catalog_before = {k: dict(v) for k, v in g.model_catalog.items()}
    g.jobs = [_job("mito_server")]
    html = dashboard.get("/").get_data(as_text=True)
    assert 'value="mito_server"' in html, "the running model is listed on the Models tab"
    assert g.model_catalog == catalog_before, "but not added to the catalog everything else reads"


@pytest.mark.parametrize("headers, expected", [
    pytest.param({}, "http://node7:8765/v/abc/", id="direct"),
    pytest.param({"X-Forwarded-Host": "proxy.example.org"}, "http://proxy.example.org/v/abc/", id="proxy"),
    pytest.param({"X-Forwarded-Host": "proxy.example.org", "X-Forwarded-Proto": "https"},
                 "https://proxy.example.org/v/abc/", id="https-proxy"),
])
def test_behind_a_reverse_proxy_the_viewer_is_loaded_through_it(dashboard, headers, expected):
    g.NEUROGLANCER_URL = "http://node7:8765/v/abc/"
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
        g.input_norms = get_normalizations(chain)
    else:
        g.postprocess = get_postprocessors(chain)
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

    g.input_norms = []
    rows = _rows(dashboard.get("/").get_data(as_text=True), "normalizer-item", "inputNormCheckbox")
    assert [name for name, _, _ in rows] == [op["name"] for op in get_input_normalizers()]
    assert not any(checked for _, checked, _ in rows)


def _builder_state(dashboard):
    """The pipeline state the builder page starts from."""
    html = dashboard.get("/pipeline-builder").get_data(as_text=True)
    return {key: json.loads(re.search(rf"^\s*{key}: (.*?),?$", html, re.M).group(1))
            for key in ("inputs", "outputs", "normalizers", "models", "postprocessors", "edges")}


def test_the_builder_keeps_a_saved_pipeline_that_has_no_normalizers(dashboard):
    # What /api/pipeline/apply stores for INPUT -> model -> OUTPUT.
    g.pipeline_inputs = [{"id": "input-1", "position": {"x": 11, "y": 22}, "params": {
        "dataset_path": "/data/raw.zarr", "bounding_boxes": [{"offset": [0, 0, 0], "shape": [8, 8, 8]}]}}]
    g.pipeline_outputs = [{"id": "output-1", "params": {"dataset_path": "/out.zarr"}, "position": {"x": 900, "y": 22}}]
    g.pipeline_models = [{"id": "model-1", "name": "mito", "params": {}, "position": {"x": 400, "y": 22}}]
    g.pipeline_edges = [{"id": "e1", "from": "input-1", "to": "model-1"}, {"id": "e2", "from": "model-1", "to": "output-1"}]
    g.pipeline_normalizers, g.pipeline_postprocessors = [], []
    g.jobs = [_job("other_model")]

    state = _builder_state(dashboard)
    assert (state["inputs"], state["outputs"], state["edges"]) == (g.pipeline_inputs, g.pipeline_outputs, g.pipeline_edges)
    assert [m["id"] for m in state["models"]] == ["model-1"] and state["normalizers"] == []


def test_before_anything_is_applied_the_builder_starts_from_the_live_chain(dashboard):
    from cellmap_flow.norm.input_normalize import get_normalizations

    for attr in ("pipeline_inputs", "pipeline_outputs", "pipeline_edges", "pipeline_normalizers",
                 "pipeline_models", "pipeline_postprocessors"):
        setattr(g, attr, [])
    g.input_norms = get_normalizations([{"name": "ZScoreNormalizer", "mean": 1, "std": 2}])
    g.jobs = [_job("mito")]

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
    g.models_config = [_Configured(name="mito")]
    g.pipeline_model_configs = {"nuc": {"type": "script", "script_path": "/imported.py"}}
    g.model_catalog = {"catalog": {"er": "/models/er"}}
    if applied:
        g.pipeline_inputs = [{"id": "input-1", "params": {}}]
        g.pipeline_models = [{"id": "model-1", "name": "mito", "params": {}},
                             {"id": "model-2", "name": "nuc", "params": {}, "config": {"type": "given"}}]
    else:
        g.jobs = [_job("mito"), _job("nuc"), _job("unknown")]
    html = dashboard.get("/pipeline-builder").get_data(as_text=True)

    models = _builder_state(dashboard)["models"]
    configs = {m["name"]: m.get("config") for m in models}
    assert configs == ({"mito": {"type": "script", "script_path": "/mito.py"}, "nuc": {"type": "given"}} if applied else
                       {"mito": {"type": "script", "script_path": "/mito.py"},
                        "nuc": {"type": "script", "script_path": "/imported.py"}, "unknown": None})
    palette = json.loads(re.search(r"^\s*availableModels: (.*?),?$", html, re.M).group(1))
    assert palette == {"catalog/er": {"name": "catalog/er", "category": "catalog", "model_name": "er",
                                      "path": "/models/er"},
                       "mito": {"name": "mito", "type": "script", "script_path": "/mito.py"}}
