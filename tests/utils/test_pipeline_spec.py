"""The chain's formats that leave the process, and the chain state on g.

A chain reaches inference servers (older or newer than the dashboard) as JSON
in the layer URL, the YAML and blockwise paths as json_data, and the trainer
through current_*_config(). Rebuilding individual ops from their to_dict() is
covered in test_chain_serialization.
"""

import base64
import contextlib
import gc
import json

import numpy as np
import pytest
from flask import Flask

from cellmap_flow.globals import current_input_norm_config, current_postprocess_config, g
from cellmap_flow.norm.input_normalize import EuclideanDistance, LambdaNormalizer, MinMaxNormalizer
from cellmap_flow.pipeline_spec import (
    PipelineSpec,
    builder_steps,
    chain_num_channels,
    normalize_steps,
    op_schemas,
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
from cellmap_flow.utils.serilization_utils import get_process_dataset, get_process_dataset_url
from cellmap_flow.utils.web_utils import ARGS_KEY, decode_to_json, encode_to_str, get_norms_post_args

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


def test_url_blob_extras_follow_the_chains():
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    blob = spec.to_url_blob(dashboard_url="http://dash/", digest=spec.digest())
    assert list(decode_to_json(blob)) == ["input_norm", "postprocess", "dashboard_url", "digest"]
    extras = {"dashboard_url": "http://dash/", "digest": spec.digest()}
    assert PipelineSpec.from_url_blob(blob) == (spec, extras)
    # The server tells this dashboard about new equivalences.
    assert get_process_dataset_url(f"m{ARGS_KEY}{blob}{ARGS_KEY}")[0] == "http://dash/"
    with pytest.raises(ValueError):
        spec.to_url_blob(postprocess=[])


def test_digest_is_stable_and_follows_the_content():
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    # It goes into layer URLs, so it must be the same in every process.
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
    ],
    ids=["nothing", "list", "legacy_dict", "builder"],
)
def test_step_readers(read, given, expected):
    assert _ordered(read(given)) == expected


def test_json_data_forms():
    legacy = {
        "input_norm": {"MinMaxNormalizer": {"min_value": 0, "max_value": 255},
                       "LambdaNormalizer": {"expression": "x*2-1"}},
        "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}},
    }
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    assert PipelineSpec.from_json_data(json.dumps(legacy), strict=True) == spec
    assert list(spec.to_json_data()) == ["input_norm", "postprocess"]
    # Without strict, a missing or null chain is empty.
    assert PipelineSpec.from_json_data({"postprocess": None}) == PipelineSpec()


@pytest.mark.parametrize(
    "json_data, error",
    [
        ({"input_norm": []}, KeyError),
        ({"postprocess": []}, KeyError),
        ({"input_norm": None, "postprocess": []}, ValueError),
        ({"input_norm": [], "postprocess": "SigmoidPostprocessor"}, ValueError),
    ],
)
def test_the_readers_reject_a_json_data_without_both_chains(json_data, error):
    """A misspelt json_data is a mistake the blockwise precheck reports, not an
    empty chain."""
    with pytest.raises(error):
        PipelineSpec.from_json_data(json_data, strict=True)
    with pytest.raises(error):
        get_process_dataset(json_data)
    with pytest.raises(error):
        get_process_dataset_url(f"m{ARGS_KEY}{encode_to_str(json_data)}{ARGS_KEY}")


# --- the dashboard's chain state -------------------------------------------------


@pytest.fixture
def post(monkeypatch):
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(pipeline, "get_raw_layer", lambda path: type("Raw", (), {"shader": None})())
    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {})

    class Viewer:
        state = type("State", (), {"layers": {}})()

        def txn(self):
            return contextlib.nullcontext(self.state)

    g.viewer, g.dataset_path = Viewer(), "/data/raw.zarr"
    g.jobs = [type("Job", (), {"model_name": "mito", "host": "http://gpu:8000"})()]
    g.shaders, g.shader_controls = {}, {}
    g.input_norms, g.postprocess, g.input_norm_config, g.postprocess_config = [], [], {}, {}
    app = Flask(__name__)
    app.register_blueprint(pipeline.pipeline_bp)
    client = app.test_client()

    def post(url, payload):
        # As a browser sends it: the test client's json= sorts the keys.
        response = client.post(url, data=json.dumps(payload), content_type="application/json")
        assert response.status_code == 200, response.data
        return response.get_json()

    return post


def _layer_source():
    return g.viewer.state.layers["mito"].to_json()["source"]


def _layer_blob():
    source = _layer_source()
    source = source[0] if isinstance(source, list) else source
    url = source["url"] if isinstance(source, dict) else source
    return decode_to_json(url.split(ARGS_KEY)[1])


# What the Input/Output tabs post: every value a string, name first.
POSTED = {
    "input_norm": [
        {"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255", "invert": "false"},
        {"name": "LambdaNormalizer", "expression": "x*2-1"},
    ],
    "postprocess": [{"name": "ThresholdPostprocessor", "threshold": "0.5"}],
}


def test_submit_keeps_the_posted_steps_and_names_the_layer_by_digest(post):
    received = post("/api/process", POSTED)["received_data"]
    assert _ordered(current_input_norm_config()) == _ordered(POSTED["input_norm"])
    assert _ordered(current_postprocess_config()) == _ordered(POSTED["postprocess"])
    assert [type(n).__name__ for n in g.input_norms] == ["MinMaxNormalizer", "LambdaNormalizer"]

    blob = _layer_blob()
    assert list(blob) == ["input_norm", "postprocess", "dashboard_url", "digest"]
    assert _ordered(blob["input_norm"]) == _ordered(POSTED["input_norm"])
    assert blob["dashboard_url"] == "http://localhost/"
    assert blob["digest"] == received["digest"] == PipelineSpec.from_json_data(POSTED).digest()
    assert "time" not in received

    # The same settings give the same source, a changed parameter a new one.
    source = _layer_source()
    post("/api/process", POSTED)
    assert _layer_source() == source
    post("/api/process", dict(POSTED, postprocess=[dict(POSTED["postprocess"][0], threshold="0.6")]))
    assert _layer_source() != source


def test_apply_keeps_the_steps_with_the_name_last(post):
    nodes = {
        "input_normalizers": [
            {"id": "n1", "name": "MinMaxNormalizer", "params": {"min_value": 0, "max_value": 255}},
            {"id": "n2", "name": "LambdaNormalizer", "params": {"expression": "x*2-1"}},
        ],
        "postprocessors": [{"id": "p1", "name": "SigmoidPostprocessor"}],
    }
    post("/api/pipeline/apply", nodes)
    assert _ordered(current_input_norm_config()) == [
        [("min_value", 0), ("max_value", 255), ("name", "MinMaxNormalizer")],
        [("expression", "x*2-1"), ("name", "LambdaNormalizer")],
    ]
    assert current_postprocess_config() == [{"name": "SigmoidPostprocessor"}]
    assert g.pipeline_normalizers == nodes["input_normalizers"]


def test_after_a_yaml_boot_the_config_is_the_live_chain():
    # yaml_cli builds the live chain from json_data and leaves the configs empty.
    g.input_norm_config, g.postprocess_config = {}, {}
    g.input_norms, g.postprocess = get_process_dataset({
        "input_norm": [{"name": "MinMaxNormalizer", "min_value": "0"}],
        "postprocess": [{"name": "SigmoidPostprocessor"}],
    })
    assert _ordered(current_input_norm_config()) == [
        [("name", "MinMaxNormalizer"), ("min_value", 0.0), ("max_value", 255.0), ("invert", False)]
    ]
    assert current_postprocess_config() == [{"name": "SigmoidPostprocessor"}]
    # Per chain: a configured one wins over the live one.
    g.postprocess_config = [THRESHOLD]
    assert g.pipeline_spec == PipelineSpec(current_input_norm_config(), [THRESHOLD])


def test_set_pipeline_writes_all_four_attributes_or_none():
    merger = SimpleBlockwiseMerger()
    spec = PipelineSpec([MINMAX], [{"name": "SimpleBlockwiseMerger"}])
    g.set_pipeline(spec, built=([MinMaxNormalizer()], [merger]))
    assert g.postprocess[0] is merger, "the stateful instances given are kept"
    assert (g.input_norm_config, g.postprocess_config) == ([MINMAX], [{"name": "SimpleBlockwiseMerger"}])
    # Derived, so conftest's vars(g) restore covers it.
    assert g.pipeline_spec == spec and "pipeline_spec" not in vars(g)

    with pytest.raises(ValueError):
        g.set_pipeline(PipelineSpec([SHIFT], [dict(THRESHOLD, threshold="high")]))
    assert g.postprocess[0] is merger and g.input_norm_config == [MINMAX]
    assert type(g.input_norms[0]).__name__ == "MinMaxNormalizer"


# --- what a chain outputs ----------------------------------------------------------


@pytest.mark.parametrize(
    "chain, dtype, channels, is_segmentation",
    [
        ([], np.float16, 9, False),
        ([SigmoidPostprocessor()], np.float32, 9, None),
        ([ChannelSelection("0,2")], np.float16, 2, None),
        # The last step that declares a dtype decides: uint64 label ids must
        # not be advertised (or cast) as a sigmoid's float32.
        ([SigmoidPostprocessor(), AffinityPostprocessor()], np.uint64, 1, True),
        ([AffinityPostprocessor(), SigmoidPostprocessor()], np.float32, 1, True),
        ([AffinityPostprocessor(), ChannelSelection("0,0")], np.uint64, 2, True),
        ([ThresholdPostprocessor(), DefaultPostprocessor()], np.uint8, 9, False),
        ([DefaultPostprocessor(), SigmoidPostprocessor()], np.float32, 9, False),
    ],
)
def test_what_a_chain_outputs(chain, dtype, channels, is_segmentation):
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    g.postprocess = chain
    assert g.get_output_dtype(np.float16) is dtype
    assert chain_num_channels(chain, 9) == channels
    assert pipeline.is_output_segmentation() is is_segmentation


# --- op_schemas ----------------------------------------------------------------------


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
