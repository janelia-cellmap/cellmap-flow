"""What the dashboard's two server-rendered pages hand to the browser."""

import json
import re
from types import SimpleNamespace

import pytest

from cellmap_flow.dashboard.app import app
from cellmap_flow.globals import g


@pytest.fixture
def client():
    return app.test_client()


def _job(name):
    return SimpleNamespace(model_name=name, host="http://node:8000")


def test_rendering_the_index_leaves_the_model_catalog_alone(client):
    catalog_before = {k: dict(v) for k, v in g.model_catalog.items()}
    g.jobs = [_job("mito_server")]

    response = client.get("/")

    assert response.status_code == 200
    # The running model is still listed on the Models tab...
    assert 'value="mito_server"' in response.get_data(as_text=True)
    # ...but not added to the catalog everything else reads.
    assert g.model_catalog == catalog_before


# --- Input / Postprocess lists ---------------------------------------------


def _rows(html, row_class, checkbox_class):
    """(name, checked, {param: value}) for each list row, in page order."""
    rows = []
    for block in re.split(rf'<div class="{row_class} ', html)[1:]:
        checkbox = re.search(
            rf'<input[^>]*class="form-check-input {checkbox_class}"[^>]*>', block, re.S
        ).group(0)
        name = re.search(r'value="([^"]*)"', checkbox).group(1)
        checked = re.search(r"\schecked\b", checkbox) is not None
        params = dict(re.findall(r'data-param="([^"]*)"\s*value="([^"]*)"', block))
        rows.append((name, checked, params))
    return rows


def test_the_input_tab_lists_the_configured_chain_first_in_its_order(client):
    from cellmap_flow.norm.input_normalize import get_normalizations

    # Registry order is MinMax before Lambda; this chain runs Lambda first,
    # and uses Lambda twice.
    g.input_norms = get_normalizations([
        {"name": "LambdaNormalizer", "expression": "x*2"},
        {"name": "MinMaxNormalizer", "min_value": 10, "max_value": 20},
        {"name": "LambdaNormalizer", "expression": "x-1"},
    ])

    html = client.get("/").get_data(as_text=True)
    rows = _rows(html, "normalizer-item", "inputNormCheckbox")

    assert rows[:3] == [
        ("LambdaNormalizer", True, {"expression": "x*2"}),
        ("MinMaxNormalizer", True, {"min_value": "10.0", "max_value": "20.0", "invert": "False"}),
        ("LambdaNormalizer", True, {"expression": "x-1"}),
    ]
    # Every other normalizer still appears once, unticked, after the chain.
    rest = rows[3:]
    assert not any(checked for _, checked, _ in rest)
    assert "LambdaNormalizer" not in [name for name, _, _ in rest]
    assert "ZScoreNormalizer" in [name for name, _, _ in rest]
    # Two rows of the same op must not share element ids.
    ids = re.findall(r'\bid="(inputNorm[^"]*)"', html)
    assert len(ids) == len(set(ids))


def test_the_postprocess_tab_lists_the_configured_chain_first_in_its_order(client):
    from cellmap_flow.post.postprocessors import get_postprocessors

    g.postprocess = get_postprocessors([
        {"name": "ThresholdPostprocessor", "threshold": 0.7},
        {"name": "SigmoidPostprocessor"},
    ])

    html = client.get("/").get_data(as_text=True)
    rows = _rows(html, "postprocessor-item", "postProcessCheckbox")

    assert rows[:2] == [
        ("ThresholdPostprocessor", True, {"threshold": "0.7"}),
        ("SigmoidPostprocessor", True, {}),
    ]
    assert not any(checked for _, checked, _ in rows[2:])
    assert [name for name, _, _ in rows].count("SigmoidPostprocessor") == 1


def test_with_nothing_configured_every_op_is_listed_once_unticked(client):
    from cellmap_flow.norm.input_normalize import get_input_normalizers

    g.input_norms = []
    html = client.get("/").get_data(as_text=True)
    rows = _rows(html, "normalizer-item", "inputNormCheckbox")

    assert [name for name, _, _ in rows] == [op["name"] for op in get_input_normalizers()]
    assert not any(checked for _, checked, _ in rows)


# --- Pipeline builder --------------------------------------------------------


def _builder_pipeline(html):
    """The pipeline state the builder page starts from."""
    state = {}
    for key in ("inputs", "outputs", "normalizers", "models", "postprocessors", "edges"):
        match = re.search(rf"^\s*{key}: (.*?),?$", html, re.M)
        state[key] = json.loads(match.group(1))
    return state


def test_the_builder_keeps_a_saved_pipeline_that_has_no_normalizers(client):
    # What /api/pipeline/apply stores for INPUT -> model -> OUTPUT.
    g.pipeline_inputs = [{
        "id": "input-1",
        "params": {"dataset_path": "/data/raw.zarr",
                   "bounding_boxes": [{"offset": [0, 0, 0], "shape": [8, 8, 8]}]},
        "position": {"x": 11, "y": 22},
    }]
    g.pipeline_outputs = [{"id": "output-1", "params": {"dataset_path": "/out.zarr"},
                           "position": {"x": 900, "y": 22}}]
    g.pipeline_models = [{"id": "model-1", "name": "mito", "params": {},
                          "position": {"x": 400, "y": 22}}]
    g.pipeline_edges = [{"id": "e1", "from": "input-1", "to": "model-1"},
                        {"id": "e2", "from": "model-1", "to": "output-1"}]
    g.pipeline_normalizers = []
    g.pipeline_postprocessors = []
    g.jobs = [_job("other_model")]

    state = _builder_pipeline(client.get("/pipeline-builder").get_data(as_text=True))

    assert state["inputs"] == g.pipeline_inputs
    assert state["outputs"] == g.pipeline_outputs
    assert state["edges"] == g.pipeline_edges
    assert [m["id"] for m in state["models"]] == ["model-1"]
    assert state["normalizers"] == []


def test_the_builder_starts_from_the_live_chain_before_anything_was_applied(client):
    from cellmap_flow.norm.input_normalize import get_normalizations

    for attr in ("pipeline_inputs", "pipeline_outputs", "pipeline_edges",
                 "pipeline_normalizers", "pipeline_models", "pipeline_postprocessors"):
        setattr(g, attr, [])
    g.input_norms = get_normalizations([{"name": "ZScoreNormalizer", "mean": 1, "std": 2}])
    g.jobs = [_job("mito")]

    state = _builder_pipeline(client.get("/pipeline-builder").get_data(as_text=True))

    assert [(n["name"], n["params"]) for n in state["normalizers"]] == [
        ("ZScoreNormalizer", {"mean": 1.0, "std": 2.0})
    ]
    assert [m["name"] for m in state["models"]] == ["mito"]
    assert state["inputs"] == [] and state["edges"] == []
