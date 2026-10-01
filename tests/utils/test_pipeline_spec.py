"""The chain's formats that leave the process, and the process's chain state.

A chain reaches inference servers (older or newer than the dashboard) as JSON
in the layer URL, the YAML and blockwise paths as json_data, and the trainer
through current_*_config(). How each op rebuilds from its to_dict() is in
test_ops.
"""

import base64
import json

import numpy as np
import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs.launch import started_jobs
from cellmap_flow.norm.input_normalize import EuclideanDistance, LambdaNormalizer, MinMaxNormalizer
from cellmap_flow.pipeline_spec import (
    PipelineSpec,
    builder_steps,
    chain_is_segmentation,
    chain_num_channels,
    normalize_steps,
)
from cellmap_flow.post.postprocessors import (
    AffinityPostprocessor,
    ChannelSelection,
    DefaultPostprocessor,
    LambdaPostprocessor,
    SigmoidPostprocessor,
    SimpleBlockwiseMerger,
    ThresholdPostprocessor,
)
from cellmap_flow.process_chain import process_chain
from cellmap_flow.serving.protocol import ARGS_KEY, decode_to_json

MINMAX = {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}
SHIFT = {"name": "LambdaNormalizer", "expression": "x*2-1"}
THRESHOLD = {"name": "ThresholdPostprocessor", "threshold": 0.5}


def _ordered(steps):
    """Steps with their key order, which a plain == would ignore."""
    return [list(s.items()) if isinstance(s, dict) else s for s in steps]


# --- the layer URL's args blob -------------------------------------------------


@pytest.mark.parametrize(
    "norms, posts, text",
    [
        (
            [MinMaxNormalizer(), LambdaNormalizer("x*2-1")], [],
            '{"input_norm":[{"name":"MinMaxNormalizer","min_value":0.0,"max_value":255.0,'
            '"invert":false},{"name":"LambdaNormalizer","expression":"x*2-1"}],"postprocess":[]}',
        ),
        (
            [], [LambdaPostprocessor("x + 1"), LambdaPostprocessor("x * 10")],
            '{"input_norm":[],"postprocess":[{"name":"LambdaPostprocessor","expression":"x + 1"},'
            '{"name":"LambdaPostprocessor","expression":"x * 10"}]}',
        ),
        (
            [], [AffinityPostprocessor(bias=0.5, neighborhood="[[1, 0, 0], [0, 1, 0]]")],
            '{"input_norm":[],"postprocess":[{"name":"AffinityPostprocessor","bias":0.5,'
            '"neighborhood":"[[1, 0, 0], [0, 1, 0]]"}]}',
        ),
        (
            [], [ChannelSelection("0,2")],
            '{"input_norm":[],"postprocess":[{"name":"ChannelSelection","channels":"0,2"}]}',
        ),
        (
            [EuclideanDistance(anisotropy=8, black_border=False, type="sdf")], [],
            '{"input_norm":[{"name":"EuclideanDistance","anisotropy":8,"black_border":false,'
            '"parallel":5,"type":"sdf","activation":"tanh"}],"postprocess":[]}',
        ),
    ],
    ids=["minmax_lambda", "two_lambdas", "affinity", "channel_selection", "edt"],
)
def test_url_blob_bytes(norms, posts, text):
    """input_norm first; flat {"name", **constructor args} steps with real types.

    Older servers pass every key but "name" to the constructor, so the steps
    must never be nested as {"name", "params"}.
    """
    blob = PipelineSpec.from_steps(norms, posts).to_url_blob()
    assert base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4)).decode() == text


def test_the_blobs_extras_come_after_the_chains():
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    blob = spec.to_url_blob(dashboard_url="http://dash/", digest=spec.digest())
    assert list(decode_to_json(blob)) == ["input_norm", "postprocess", "dashboard_url", "digest"]
    assert PipelineSpec.from_url_blob(blob) == (spec, {"dashboard_url": "http://dash/", "digest": spec.digest()})
    with pytest.raises(ValueError):
        spec.to_url_blob(postprocess=[])  # an extra may not replace a chain


def test_the_digest_is_the_same_in_every_process_and_follows_only_the_content():
    """It names layers, so another process must compute the same one."""
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    assert spec.digest() == "4d284fb937eabaec"
    reordered = {"max_value": 255, "name": "MinMaxNormalizer", "min_value": 0}
    assert PipelineSpec([reordered, SHIFT], [THRESHOLD]).digest() == spec.digest()
    for other in (
        PipelineSpec([SHIFT, MINMAX], [THRESHOLD]),
        PipelineSpec([MINMAX, SHIFT], [dict(THRESHOLD, threshold=0.6)]),
        PipelineSpec([MINMAX, SHIFT, THRESHOLD], []),
    ):
        assert other.digest() != spec.digest()


# --- reading chains ------------------------------------------------------------

LEGACY = {
    "input_norm": {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}, "LambdaNormalizer": {"expression": "x*2-1"}},
    "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}},
}


@pytest.mark.parametrize(
    "read, given, expected",
    [
        pytest.param(normalize_steps, None, [], id="nothing"),
        # The list form is kept as given: values are not coerced.
        pytest.param(normalize_steps, [dict(MINMAX, min_value="0"), "junk"],
                     [[("name", "MinMaxNormalizer"), ("min_value", "0"), ("max_value", 255)], "junk"], id="list"),
        # The legacy dict: a step per key, in order, and the key names the class.
        pytest.param(normalize_steps, {"MinMaxNormalizer": {"name": "X", "min_value": 0}, "SigmoidPostprocessor": None},
                     [[("name", "MinMaxNormalizer"), ("min_value", 0)], [("name", "SigmoidPostprocessor")]],
                     id="legacy-dict"),
        # Builder nodes: {**params, "name"}, the order apply always stored.
        pytest.param(builder_steps, [{"id": 1, "name": "MinMaxNormalizer", "params": {"min_value": 0}},
                                     {"params": {"a": 1}}, "junk", {"name": "SigmoidPostprocessor"}],
                     [[("min_value", 0), ("name", "MinMaxNormalizer")], [("name", "SigmoidPostprocessor")]],
                     id="builder-nodes"),
    ],
)
def test_reading_a_chains_steps(read, given, expected):
    assert _ordered(read(given)) == expected


@pytest.mark.parametrize(
    "json_data, strict, expected",
    [
        pytest.param(json.dumps(LEGACY), True, PipelineSpec([MINMAX, SHIFT], [THRESHOLD]), id="legacy-json-string"),
        # Without strict, a missing or null chain is empty; with it, a mistake
        # the blockwise precheck reports.
        pytest.param({"postprocess": None}, False, PipelineSpec(), id="lenient"),
        pytest.param({"input_norm": []}, True, KeyError, id="no-postprocess"),
        pytest.param({"postprocess": []}, True, KeyError, id="no-input-norm"),
        pytest.param({"input_norm": None, "postprocess": []}, True, ValueError, id="null-chain"),
        pytest.param({"input_norm": [], "postprocess": "SigmoidPostprocessor"}, True, ValueError, id="not-a-chain"),
    ],
)
def test_reading_a_json_data(json_data, strict, expected):
    if isinstance(expected, type):
        with pytest.raises(expected):
            PipelineSpec.from_json_data(json_data, strict=strict)
    else:
        assert PipelineSpec.from_json_data(json_data, strict=strict) == expected


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
    assert _ordered(list(chain.spec.input_norm)) == _ordered(POSTED["input_norm"])
    assert _ordered(list(chain.spec.postprocess)) == _ordered(POSTED["postprocess"])
    assert [type(n).__name__ for n in chain.input_norms] == ["MinMaxNormalizer", "LambdaNormalizer"]


def test_submit_names_the_layer_by_the_chains_digest(submit):
    """Stamped with the time, every Submit made neuroglancer refetch and each
    server rebuild its chain (and lose its merger state)."""
    received = submit()
    source = _layer_source("mito")
    blob = decode_to_json(source["url"].split(ARGS_KEY)[1])
    assert list(blob) == ["input_norm", "postprocess", "dashboard_url", "digest"]
    assert _ordered(blob["input_norm"]) == _ordered(POSTED["input_norm"]) and blob["dashboard_url"] == "http://localhost/"
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


def test_apply_keeps_the_builders_steps_in_order_with_the_name_last(dashboard):
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
    assert _ordered(list(chain.spec.input_norm)) == [
        [("min_value", 0), ("max_value", 255), ("name", "MinMaxNormalizer")],
        [("expression", "x+1"), ("name", "LambdaNormalizer")],
        [("expression", "x*2"), ("name", "LambdaNormalizer")],
    ]
    assert [n.expression for n in chain.input_norms[1:]] == ["x+1", "x*2"]
    assert list(chain.spec.postprocess) == [{"name": "SigmoidPostprocessor"}]
    assert get_session().builder_state["normalizers"] == nodes["input_normalizers"]
    assert get_session().builder_model_configs["mito"] == {"type": "script", "script_path": "/m.py"}


def test_after_a_yaml_boot_the_config_is_the_live_chain():
    # yaml_cli builds the live chain from json_data and leaves the configs empty.
    chain = process_chain()
    chain.input_norm_config, chain.postprocess_config = {}, {}
    chain.input_norms = [MinMaxNormalizer(min_value="0"), LambdaNormalizer("x*2")]
    chain.postprocess = [AffinityPostprocessor(bias=0.5, neighborhood="[[1, 0, 0]]")]
    assert _ordered(list(chain.spec.input_norm)) == [
        [("name", "MinMaxNormalizer"), ("min_value", 0.0), ("max_value", 255.0), ("invert", False)],
        [("name", "LambdaNormalizer"), ("expression", "x*2")],
    ]
    assert list(chain.spec.postprocess) == [{"name": "AffinityPostprocessor", "bias": 0.5, "neighborhood": "[[1, 0, 0]]"}]
    # Per chain: a configured one wins over the live one.
    chain.postprocess_config = [THRESHOLD]
    assert chain.spec == PipelineSpec(list(chain.spec.input_norm), [THRESHOLD])


def test_set_pipeline_writes_all_four_attributes_or_none():
    chain = process_chain()
    merger = SimpleBlockwiseMerger()
    spec = PipelineSpec([MINMAX], [{"name": "SimpleBlockwiseMerger"}])
    chain.set(spec, built=([MinMaxNormalizer()], [merger]))
    assert chain.postprocess[0] is merger, "the stateful instances given are kept"
    assert (chain.input_norm_config, chain.postprocess_config) == ([MINMAX], [{"name": "SimpleBlockwiseMerger"}])
    assert chain.spec == spec
    with pytest.raises(ValueError):
        chain.set(PipelineSpec([SHIFT], [dict(THRESHOLD, threshold="high")]))
    assert chain.postprocess[0] is merger and chain.input_norm_config == [MINMAX]


# --- what a chain outputs ----------------------------------------------------------


@pytest.mark.parametrize(
    "chain, dtype, channels, is_segmentation",
    [
        pytest.param([], np.float16, 9, False, id="nothing-the-models-own"),
        pytest.param([ChannelSelection("0,2")], np.float16, 2, None, id="channels-picked"),
        # The last step that declares a dtype decides: uint64 label ids must
        # not be advertised (or cast) as a sigmoid's float32.
        pytest.param([SigmoidPostprocessor(), AffinityPostprocessor()], np.uint64, 1, True, id="sigmoid-then-affinity"),
        pytest.param([AffinityPostprocessor(), SigmoidPostprocessor()], np.float32, 1, True, id="affinity-then-sigmoid"),
        pytest.param([AffinityPostprocessor(), ChannelSelection("0,0")], np.uint64, 2, True, id="affinity-then-channels"),
        pytest.param([ThresholdPostprocessor(), DefaultPostprocessor()], np.uint8, 9, False, id="threshold-then-uint8"),
    ],
)
def test_what_a_chain_outputs(chain, dtype, channels, is_segmentation):
    process_chain().postprocess = chain
    assert process_chain().output_dtype(np.float16) is dtype
    assert chain_num_channels(chain, 9) == channels
    assert chain_is_segmentation(chain) is is_segmentation
