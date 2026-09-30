"""The chain's formats that leave the process, and the chain state on g.

A chain reaches inference servers (older or newer than the dashboard) as JSON
in the layer URL, the YAML and blockwise paths as json_data, and the trainer
through current_*_config(). How each op rebuilds from its to_dict() is in
test_ops.
"""

import base64
import functools
import gc
import json

import numpy as np
import pytest

from cellmap_flow.globals import current_input_norm_config, current_postprocess_config, g
from cellmap_flow.norm.input_normalize import EuclideanDistance, LambdaNormalizer, MinMaxNormalizer
from cellmap_flow.pipeline_spec import PipelineSpec, builder_steps, chain_num_channels, normalize_steps, op_schemas
from cellmap_flow.post.postprocessors import (
    AffinityPostprocessor,
    ChannelSelection,
    DefaultPostprocessor,
    LambdaPostprocessor,
    SigmoidPostprocessor,
    SimpleBlockwiseMerger,
    ThresholdPostprocessor,
)
from cellmap_flow.utils.web_utils import ARGS_KEY, decode_to_json, get_norms_post_args

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
    blob = get_norms_post_args(norms, posts)
    assert base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4)).decode() == text


def test_the_blobs_extras_and_its_digest():
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    blob = spec.to_url_blob(dashboard_url="http://dash/", digest=spec.digest())
    assert list(decode_to_json(blob)) == ["input_norm", "postprocess", "dashboard_url", "digest"]
    assert PipelineSpec.from_url_blob(blob) == (spec, {"dashboard_url": "http://dash/", "digest": spec.digest()})
    with pytest.raises(ValueError):
        spec.to_url_blob(postprocess=[])
    # It names layers, so it must be the same in every process, and follow the content only.
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

STRICT = functools.partial(PipelineSpec.from_json_data, strict=True)
LEGACY = {
    "input_norm": {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}, "LambdaNormalizer": {"expression": "x*2-1"}},
    "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}},
}


@pytest.mark.parametrize(
    "read, given, expected",
    [
        (normalize_steps, None, []),
        # The list form is kept as given: values are not coerced.
        (normalize_steps, [dict(MINMAX, min_value="0"), "junk"],
         [[("name", "MinMaxNormalizer"), ("min_value", "0"), ("max_value", 255)], "junk"]),
        # The legacy dict: a step per key, in order, and the key names the class.
        (normalize_steps, {"MinMaxNormalizer": {"name": "X", "min_value": 0}, "SigmoidPostprocessor": None},
         [[("name", "MinMaxNormalizer"), ("min_value", 0)], [("name", "SigmoidPostprocessor")]]),
        # Builder nodes: {**params, "name"}, the order apply always stored.
        (builder_steps, [{"id": 1, "name": "MinMaxNormalizer", "params": {"min_value": 0}},
                         {"params": {"a": 1}}, "junk", {"name": "SigmoidPostprocessor"}],
         [[("min_value", 0), ("name", "MinMaxNormalizer")], [("name", "SigmoidPostprocessor")]]),
        (STRICT, json.dumps(LEGACY), PipelineSpec([MINMAX, SHIFT], [THRESHOLD])),
        # Without strict, a missing or null chain is empty; with it, a mistake
        # the blockwise precheck reports.
        (PipelineSpec.from_json_data, {"postprocess": None}, PipelineSpec()),
        (STRICT, {"input_norm": []}, KeyError),
        (STRICT, {"postprocess": []}, KeyError),
        (STRICT, {"input_norm": None, "postprocess": []}, ValueError),
        (STRICT, {"input_norm": [], "postprocess": "SigmoidPostprocessor"}, ValueError),
    ],
    ids=["nothing", "list", "legacy-dict", "builder", "json-data", "lenient", "no-postprocess", "no-input-norm",
         "null-chain", "not-a-chain"],
)
def test_reading_chains(read, given, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            read(given)
    elif isinstance(expected, PipelineSpec):
        assert read(given) == expected
    else:
        assert _ordered(read(given)) == expected


# --- the dashboard's chain state -------------------------------------------------

# What the Input/Output tabs post: every value a string, name first.
POSTED = {
    "input_norm": [
        {"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255", "invert": "false"},
        {"name": "LambdaNormalizer", "expression": "x*2-1"},
    ],
    "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "0.5"}],
}


def _post(dashboard, url, payload):
    # As a browser sends it: the test client's json= sorts the keys.
    response = dashboard.post(url, data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200, response.data
    return response.get_json()


def _layer_source(name):
    source = g.viewer.state.layers[name].to_json()["source"]
    return source[0] if isinstance(source, list) else source


def test_submit_keeps_the_posted_chain_and_names_layers_by_their_digest(dashboard, viewer, ome_pyramid,
                                                                        monkeypatch):
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    # The model writes 16 nm voxels over raw at 24, 12, 12 nm: drawn at the raw's scale.
    info = {"output_voxel_size": [16] * 3, "output_class": "unit"}
    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: info)
    g.dataset_path = ome_pyramid((((24, 12, 12), None),))
    g.jobs = [type("Job", (), {"model_name": name, "host": host})() for name, host in
              [("mito", "http://gpu:8000"), ("pending", None)]]
    received = _post(dashboard, "/api/process", POSTED)["received_data"]

    assert _ordered(current_input_norm_config()) == _ordered(POSTED["input_norm"])
    assert _ordered(current_postprocess_config()) == _ordered(POSTED["postprocess"])
    assert [type(n).__name__ for n in g.input_norms] == ["MinMaxNormalizer", "LambdaNormalizer"]
    source = _layer_source("mito")
    blob = decode_to_json(source["url"].split(ARGS_KEY)[1])
    assert list(blob) == ["input_norm", "postprocess", "dashboard_url", "digest"]
    assert _ordered(blob["input_norm"]) == _ordered(POSTED["input_norm"])
    assert blob["dashboard_url"] == "http://localhost/"
    assert blob["digest"] == received["digest"] == PipelineSpec.from_json_data(POSTED).digest()
    assert "time" not in received
    dimensions = source["transform"]["outputDimensions"]
    assert (dimensions["z"][0], dimensions["x"][0]) == pytest.approx((24e-9, 12e-9))
    assert "pending" not in g.viewer.state.layers, "a job with no host yet gets no layer"

    assert g.viewer.state.layers["mito"].type == "segmentation"  # a threshold's labels

    # The same settings give the same source, a changed parameter a new one.
    _post(dashboard, "/api/process", POSTED)
    assert _layer_source("mito") == source
    _post(dashboard, "/api/process", dict(POSTED, postprocess=[dict(POSTED["postprocess"][0], threshold="0.6")]))
    assert _layer_source("mito") != source
    # Without one it is an image, shaded over the model's output range (0 to 1).
    _post(dashboard, "/api/process", dict(POSTED, postprocess=[]))
    mito = g.viewer.state.layers["mito"]
    assert mito.type == "image" and "range=[0, 1]" in mito.shader


def test_apply_keeps_the_builders_steps_in_order_with_the_name_last(dashboard):
    nodes = {
        "input_normalizers": [
            {"id": "n1", "name": "MinMaxNormalizer", "params": {"min_value": 0, "max_value": 255}},
            {"id": "n2", "name": "LambdaNormalizer", "params": {"expression": "x+1"}},
            {"id": "n3", "name": "LambdaNormalizer", "params": {"expression": "x*2"}},  # the same op twice
        ],
        "postprocessors": [{"id": "p1", "name": "SigmoidPostprocessor"}],
        "models": [{"id": "m1", "name": "mito", "config": {"type": "script", "script_path": "/m.py"}}],
    }
    _post(dashboard, "/api/pipeline/apply", nodes)
    assert _ordered(current_input_norm_config()) == [
        [("min_value", 0), ("max_value", 255), ("name", "MinMaxNormalizer")],
        [("expression", "x+1"), ("name", "LambdaNormalizer")],
        [("expression", "x*2"), ("name", "LambdaNormalizer")],
    ]
    assert [n.expression for n in g.input_norms[1:]] == ["x+1", "x*2"]
    assert current_postprocess_config() == [{"name": "SigmoidPostprocessor"}]
    assert g.pipeline_normalizers == nodes["input_normalizers"]
    assert g.pipeline_model_configs["mito"] == {"type": "script", "script_path": "/m.py"}


def test_the_configured_chain_after_a_yaml_boot_and_set_pipeline():
    # yaml_cli builds the live chain from json_data and leaves the configs
    # empty: the config is the live chain then, every step of it.
    g.input_norm_config, g.postprocess_config = {}, {}
    g.input_norms = [MinMaxNormalizer(min_value="0"), LambdaNormalizer("x*2")]
    g.postprocess = [AffinityPostprocessor(bias=0.5, neighborhood="[[1, 0, 0]]")]
    assert _ordered(current_input_norm_config()) == [
        [("name", "MinMaxNormalizer"), ("min_value", 0.0), ("max_value", 255.0), ("invert", False)],
        [("name", "LambdaNormalizer"), ("expression", "x*2")],
    ]
    assert current_postprocess_config() == [
        {"name": "AffinityPostprocessor", "bias": 0.5, "neighborhood": "[[1, 0, 0]]"}
    ]
    # Per chain: a configured one wins over the live one.
    g.postprocess_config = [THRESHOLD]
    assert g.pipeline_spec == PipelineSpec(current_input_norm_config(), [THRESHOLD])

    # set_pipeline writes the four attributes together, keeping the stateful
    # instances it is given, or none of them.
    merger = SimpleBlockwiseMerger()
    spec = PipelineSpec([MINMAX], [{"name": "SimpleBlockwiseMerger"}])
    g.set_pipeline(spec, built=([MinMaxNormalizer()], [merger]))
    assert g.postprocess[0] is merger
    assert (g.input_norm_config, g.postprocess_config) == ([MINMAX], [{"name": "SimpleBlockwiseMerger"}])
    assert g.pipeline_spec == spec and "pipeline_spec" not in vars(g)  # derived: conftest's restore covers it
    with pytest.raises(ValueError):
        g.set_pipeline(PipelineSpec([SHIFT], [dict(THRESHOLD, threshold="high")]))
    assert g.postprocess[0] is merger and g.input_norm_config == [MINMAX]


# --- what a chain outputs ----------------------------------------------------------


@pytest.mark.parametrize(
    "chain, dtype, channels, is_segmentation",
    [
        ([], np.float16, 9, False),
        ([ChannelSelection("0,2")], np.float16, 2, None),
        # The last step that declares a dtype decides: uint64 label ids must
        # not be advertised (or cast) as a sigmoid's float32.
        ([SigmoidPostprocessor(), AffinityPostprocessor()], np.uint64, 1, True),
        ([AffinityPostprocessor(), SigmoidPostprocessor()], np.float32, 1, True),
        ([AffinityPostprocessor(), ChannelSelection("0,0")], np.uint64, 2, True),
        ([ThresholdPostprocessor(), DefaultPostprocessor()], np.uint8, 9, False),
    ],
)
def test_what_a_chain_outputs(chain, dtype, channels, is_segmentation):
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    g.postprocess = chain
    assert g.get_output_dtype(np.float16) is dtype
    assert chain_num_channels(chain, 9) == channels
    assert pipeline.is_output_segmentation() is is_segmentation


@pytest.mark.parametrize("kind", ["input_norm", "postprocess"])
def test_op_schemas_describe_every_registered_op(kind):
    from cellmap_flow.norm.input_normalize import get_input_normalizers
    from cellmap_flow.post.postprocessors import get_postprocessors_list

    jsonschema = pytest.importorskip("jsonschema")
    gc.collect()  # so no test-local op class disappears between the two listings
    listed = get_input_normalizers() if kind == "input_norm" else get_postprocessors_list()
    schemas = op_schemas(kind)
    json.dumps(schemas)  # it goes into page data
    assert [s["name"] for s in schemas] == [op["name"] for op in listed]
    for entry, op in zip(schemas, listed):
        schema = entry["schema"]
        jsonschema.Draft202012Validator.check_schema(schema)
        assert entry["title"] == schema["title"]
        assert list(schema["properties"]) == list(op["params"])
        assert schema["required"] == [p for p, default in op["params"].items() if default == ""]
