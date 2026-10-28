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
    catalog = _page_data(html, "model-catalog-data")
    assert ["User", ["mito_server"]] in [group[:2] for group in catalog["groups"]] and catalog["ticked"] == ["mito_server"], \
        "the running model is listed on the Models tab, ticked"
    assert get_session().model_catalog == catalog_before, "but not added to the catalog everything else reads"


class _ViewerAt(SimpleNamespace):
    """A viewer whose address is ``url``."""

    def __init__(self, url):
        super().__init__(url=url, state=None)

    def __str__(self):
        return self.url


@pytest.mark.parametrize("headers, expected", [
    pytest.param({}, "http://10.0.0.7:8765/v/abc/", id="direct"),
    pytest.param({"X-Forwarded-Host": "proxy.example.org"}, "http://proxy.example.org/v/abc/", id="proxy"),
    pytest.param({"X-Forwarded-Host": "proxy.example.org", "X-Forwarded-Proto": "https"},
                 "https://proxy.example.org/v/abc/", id="https-proxy"),
])
@pytest.mark.parametrize("viewer", ["the-dashboards", "the-box-tools"])
def test_behind_a_reverse_proxy_the_viewer_is_loaded_through_it(dashboard, monkeypatch, viewer, headers, expected):
    if viewer == "the-dashboards":
        get_session().neuroglancer_url = "http://10.0.0.7:8765/v/abc/"
        assert f'<iframe src="{expected}"' in dashboard.get("/", headers=headers).get_data(as_text=True)
    else:
        from cellmap_flow.dashboard.routes import bbx_generator

        monkeypatch.setattr(bbx_generator, "new_viewer", lambda *args, **kwargs: _ViewerAt("http://10.0.0.7:8765/v/abc/"))
        answer = dashboard.post("/api/bbx-generator", json={"dataset_path": "/data/raw.zarr"}, headers=headers)
        assert answer.get_json()["viewer_url"] == expected


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
    """The Processing tab's lists show the chain that runs, ticked and with
    its values, so Apply sends that chain back."""
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


def _page_data(html, island="page-data"):
    return json.loads(re.search(rf'<script type="application/json" id="{island}">(.*?)</script>', html, re.S).group(1))


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
# What Submit on the dashboard page sends (every parameter as its field's
# text): other steps, one of them the canvas's with its parameters changed.
SUBMIT_OTHER_STEPS = {"input_norm": [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "200"},
                                     {"name": "ZScoreNormalizer", "mean": "1", "std": "2"}],
                      "postprocess": []}
MINMAX = CANVAS["normalizers"][1]


def _new_ids(state):
    """``state`` with each id the page made up (``<prefix>-<i>-<ms>``) as ``<prefix>-<i>-new``."""
    return json.loads(re.sub(r'"((?:norm|post|model)-\d+)-\d{13}"', r'"\1-new"', json.dumps(state)))


@pytest.mark.parametrize("sent, expected", [
    pytest.param([APPLY], CANVAS, id="the-builders-own-apply"),
    # A step whose parameters changed keeps its node's id and position; a new
    # step gets a new node, with no position (the page places it); a step
    # that went takes its node's edges with it.
    pytest.param([APPLY, SUBMIT_OTHER_STEPS], {
        **CANVAS,
        "normalizers": [{**MINMAX, "params": {"min_value": "0", "max_value": "200"}},
                        {"id": "norm-1-new", "name": "ZScoreNormalizer", "params": {"mean": "1", "std": "2"}}],
        "postprocessors": [],
        "edges": [edge for edge in CANVAS["edges"] if edge["id"] == "e3"],  # norm-2 -> model-1
    }, id="then-submit-changed-the-steps"),
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


@pytest.mark.parametrize("bsub", [True, False], ids=["cluster", "laptop"])
def test_the_lsf_settings_are_asked_for_only_where_jobs_go_to_lsf(dashboard, monkeypatch, bsub):
    """On a Mac (no bsub) every server runs on the machine itself, so the
    first-run dialog, which cannot be closed without saving queue and charge
    group, only stood in the way; the GPU queue picker chose nothing."""
    from cellmap_flow.dashboard.routes import index_page

    monkeypatch.setattr(index_page, "bsub_available", lambda: bsub)
    html = dashboard.get("/").get_data(as_text=True)
    assert ('id="serverConfigModal"' in html) is bsub
    assert ('id="gpuQueuePicker"' in html) is bsub
    assert ("No cluster here" in html) is not bsub
