"""What the dashboard's two server-rendered pages hand to the browser."""

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
