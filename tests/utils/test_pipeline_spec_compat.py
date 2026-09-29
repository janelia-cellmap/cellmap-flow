"""What the chain serializers and the dashboard's chain state produce today.

These pin the formats that leave the process: the args blob in every layer
URL (read by inference servers that may be older or newer than the
dashboard), the JSON the YAML and blockwise paths read, and what
``current_input_norm_config()`` / ``current_postprocess_config()`` hand the
finetune manifest and the exported YAML. The chain code is being folded into
one PipelineSpec; these must keep passing across that, except the one
assertion marked as a deliberate change.
"""

import base64
import contextlib
import json

import pytest
from flask import Flask

from cellmap_flow.globals import (
    current_input_norm_config,
    current_postprocess_config,
    g,
)
from cellmap_flow.norm.input_normalize import (
    EuclideanDistance,
    LambdaNormalizer,
    MinMaxNormalizer,
)
from cellmap_flow.post.postprocessors import (
    AffinityPostprocessor,
    ChannelSelection,
    LambdaPostprocessor,
    ThresholdPostprocessor,
)
from cellmap_flow.utils.serilization_utils import (
    get_process_dataset,
    get_process_dataset_url,
    serialize_norms_posts_to_json,
)
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    decode_to_json,
    encode_to_str,
    get_norms_post_args,
)


def _chains():
    """(id, input_norms, postprocess, the blob's JSON text) for five chains."""
    return [
        (
            "minmax_lambda",
            [MinMaxNormalizer(), LambdaNormalizer("x*2-1")],
            [],
            '{"input_norm":[{"name":"MinMaxNormalizer","min_value":0.0,'
            '"max_value":255.0,"invert":false},'
            '{"name":"LambdaNormalizer","expression":"x*2-1"}],"postprocess":[]}',
        ),
        (
            "two_lambdas",
            [],
            [LambdaPostprocessor("x + 1"), LambdaPostprocessor("x * 10")],
            '{"input_norm":[],"postprocess":['
            '{"name":"LambdaPostprocessor","expression":"x + 1"},'
            '{"name":"LambdaPostprocessor","expression":"x * 10"}]}',
        ),
        (
            "affinity",
            [],
            [
                AffinityPostprocessor(
                    bias=0.5, neighborhood="[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
                )
            ],
            '{"input_norm":[],"postprocess":[{"name":"AffinityPostprocessor",'
            '"bias":0.5,"neighborhood":"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}]}',
        ),
        (
            "channel_selection",
            [],
            [ChannelSelection("0,2")],
            '{"input_norm":[],"postprocess":'
            '[{"name":"ChannelSelection","channels":"0,2"}]}',
        ),
        (
            "edt",
            [EuclideanDistance(anisotropy=8, black_border=False, type="sdf")],
            [ThresholdPostprocessor(threshold=0.25)],
            '{"input_norm":[{"name":"EuclideanDistance","anisotropy":8,'
            '"black_border":false,"parallel":5,"type":"sdf","activation":"tanh"}],'
            '"postprocess":[{"name":"ThresholdPostprocessor","threshold":0.25}]}',
        ),
    ]


CHAINS = _chains()
CHAIN_IDS = [c[0] for c in CHAINS]


def _json_text(blob):
    """The exact JSON a blob carries, so byte-level changes show up."""
    return base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4)).decode()


def _dicts(steps):
    return [s.to_dict() for s in steps]


@pytest.mark.parametrize("_, norms, posts, text", CHAINS, ids=CHAIN_IDS)
def test_url_blob_bytes(_, norms, posts, text):
    blob = get_norms_post_args(norms, posts)
    assert _json_text(blob) == text
    assert list(decode_to_json(blob)) == ["input_norm", "postprocess"]


@pytest.mark.parametrize("_, norms, posts, text", CHAINS, ids=CHAIN_IDS)
def test_url_round_trip_rebuilds_the_same_steps(_, norms, posts, text):
    blob = get_norms_post_args(norms, posts)
    dashboard_url, got_norms, got_posts = get_process_dataset_url(
        f"http://h:1/m{ARGS_KEY}{blob}{ARGS_KEY}"
    )
    assert dashboard_url is None
    assert _dicts(got_norms) == _dicts(norms)
    assert _dicts(got_posts) == _dicts(posts)


def test_url_round_trip_returns_the_dashboard_url():
    blob = encode_to_str(
        {
            "input_norm": [{"name": "MinMaxNormalizer"}],
            "postprocess": [],
            "dashboard_url": "http://dash:5000/",
            "time": 1.5,
        }
    )
    dashboard_url, norms, posts = get_process_dataset_url(f"m{ARGS_KEY}{blob}{ARGS_KEY}")
    assert dashboard_url == "http://dash:5000/"
    assert [type(n).__name__ for n in norms] == ["MinMaxNormalizer"]
    assert posts == []


def test_url_with_one_args_marker_is_rejected():
    with pytest.raises(ValueError, match="Expected two occurrences"):
        get_process_dataset_url(f"m{ARGS_KEY}abc")


@pytest.mark.parametrize("_, norms, posts, text", CHAINS, ids=CHAIN_IDS)
def test_json_round_trip(_, norms, posts, text):
    serialized = serialize_norms_posts_to_json(norms, posts)
    # Same content and order as the blob, with json.dumps' default separators.
    assert serialized == json.dumps(json.loads(text))
    got_norms, got_posts = get_process_dataset(serialized)
    assert _dicts(got_norms) == _dicts(norms)
    assert _dicts(got_posts) == _dicts(posts)


def test_process_dataset_reads_the_list_form():
    norms, posts = get_process_dataset(
        {
            "input_norm": [
                {"name": "LambdaNormalizer", "expression": "x+1"},
                {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 10},
                {"name": "LambdaNormalizer", "expression": "x*3"},
            ],
            "postprocess": [{"name": "ThresholdPostprocessor", "threshold": 0.2}],
        }
    )
    assert _dicts(norms) == [
        {"name": "LambdaNormalizer", "expression": "x+1"},
        {"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 10.0, "invert": False},
        {"name": "LambdaNormalizer", "expression": "x*3"},
    ]
    assert _dicts(posts) == [{"name": "ThresholdPostprocessor", "threshold": 0.2}]


def test_process_dataset_reads_the_legacy_dict_form_in_order():
    norms, posts = get_process_dataset(
        {
            "input_norm": {
                "MinMaxNormalizer": {"min_value": 0, "max_value": 255},
                "LambdaNormalizer": {"expression": "x*2-1"},
            },
            "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}},
        }
    )
    assert _dicts(norms) == [
        {"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False},
        {"name": "LambdaNormalizer", "expression": "x*2-1"},
    ]
    assert _dicts(posts) == [{"name": "ThresholdPostprocessor", "threshold": 0.5}]


# --- the dashboard's chain state ----------------------------------------------


class _Job:
    model_name = "mito"
    host = "http://gpu-node:8000"


class _Viewer:
    def __init__(self):
        self.state = type("State", (), {"layers": {}})()

    @contextlib.contextmanager
    def txn(self):
        yield self.state


@pytest.fixture
def dashboard(monkeypatch):
    import cellmap_flow.dashboard.routes.pipeline as pipeline

    monkeypatch.setattr(
        pipeline, "get_raw_layer", lambda path: type("Raw", (), {"shader": None})()
    )
    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {})
    g.viewer = _Viewer()
    g.jobs = [_Job()]
    g.dataset_path = "/data/raw.zarr"
    g.shaders, g.shader_controls = {}, {}
    g.input_norms, g.postprocess = [], []
    g.input_norm_config, g.postprocess_config = {}, {}

    app = Flask(__name__)
    app.register_blueprint(pipeline.pipeline_bp)
    return app.test_client()


def _layer_blob(layer):
    source = layer.to_json()["source"]
    if isinstance(source, list):
        source = source[0]
    if isinstance(source, dict):
        source = source["url"]
    return decode_to_json(source.split(ARGS_KEY)[1])


def _post(client, url, payload):
    """POST keeping the payload's key order, as a browser does.

    The test client's ``json=`` sorts keys, and the order inside a step is
    part of what is pinned here.
    """
    return client.post(url, data=json.dumps(payload), content_type="application/json")


def _ordered(steps):
    """Steps with their key order, which a plain == would ignore."""
    return [list(step.items()) for step in steps]


# What the Input/Output tabs post: every value a string, name first.
POSTED_NORMS = [
    {"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255", "invert": "false"},
    {"name": "LambdaNormalizer", "expression": "x*2-1"},
]
POSTED_POSTS = [{"name": "ThresholdPostprocessor", "threshold": "0.5"}]


def test_after_process_the_config_is_the_posted_list_verbatim(dashboard):
    response = _post(
        dashboard,
        "/api/process",
        {"input_norm": POSTED_NORMS, "postprocess": POSTED_POSTS},
    )
    assert response.status_code == 200

    assert _ordered(current_input_norm_config()) == _ordered(POSTED_NORMS)
    assert _ordered(current_postprocess_config()) == _ordered(POSTED_POSTS)
    assert [type(n).__name__ for n in g.input_norms] == [
        "MinMaxNormalizer",
        "LambdaNormalizer",
    ]
    assert [type(p).__name__ for p in g.postprocess] == ["ThresholdPostprocessor"]


def test_process_with_empty_chains_falls_back_to_the_live_chain(dashboard):
    response = _post(dashboard, "/api/process", {"input_norm": [], "postprocess": []})
    assert response.status_code == 200
    assert not g.input_norm_config and not g.postprocess_config
    assert current_input_norm_config() == [] and current_postprocess_config() == []
    assert g.input_norms == [] and g.postprocess == []


def test_process_layer_blob_carries_the_posted_chain(dashboard):
    _post(
        dashboard,
        "/api/process",
        {"input_norm": POSTED_NORMS, "postprocess": POSTED_POSTS},
    )
    blob = _layer_blob(g.viewer.state.layers["mito"])

    assert _ordered(blob["input_norm"]) == _ordered(POSTED_NORMS)
    assert _ordered(blob["postprocess"]) == _ordered(POSTED_POSTS)
    assert blob["dashboard_url"] == "http://localhost/"


def test_process_layer_blob_keys(dashboard):
    _post(
        dashboard,
        "/api/process",
        {"input_norm": POSTED_NORMS, "postprocess": POSTED_POSTS},
    )
    blob = _layer_blob(g.viewer.state.layers["mito"])
    # Deliberately changes when the blob swaps its "time" for a digest.
    assert set(blob) == {"input_norm", "postprocess", "dashboard_url", "time"}


def test_after_apply_the_config_puts_the_name_last(dashboard):
    payload = {
        "input_normalizers": [
            {"id": "n1", "name": "MinMaxNormalizer", "params": {"min_value": 0, "max_value": 255}},
            {"id": "n2", "name": "LambdaNormalizer", "params": {"expression": "x*2-1"}},
        ],
        "postprocessors": [
            {"id": "p1", "name": "SigmoidPostprocessor", "params": {}},
            {"id": "p2", "name": "ThresholdPostprocessor"},
        ],
    }
    response = _post(dashboard, "/api/pipeline/apply", payload)
    assert response.status_code == 200
    assert response.get_json()["normalizers_applied"] == 2

    assert _ordered(current_input_norm_config()) == [
        [("min_value", 0), ("max_value", 255), ("name", "MinMaxNormalizer")],
        [("expression", "x*2-1"), ("name", "LambdaNormalizer")],
    ]
    assert _ordered(current_postprocess_config()) == [
        [("name", "SigmoidPostprocessor")],
        [("name", "ThresholdPostprocessor")],
    ]
    assert [type(p).__name__ for p in g.postprocess] == [
        "SigmoidPostprocessor",
        "ThresholdPostprocessor",
    ]
    assert g.pipeline_normalizers == payload["input_normalizers"]
    assert g.pipeline_postprocessors == payload["postprocessors"]


def test_after_yaml_boot_the_config_is_derived_from_the_live_chain():
    # yaml_cli builds the chain from json_data and never touches the *_config
    # attributes.
    g.input_norm_config, g.postprocess_config = {}, {}
    g.input_norms, g.postprocess = get_process_dataset(
        {
            "input_norm": [
                {"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"},
                {"name": "LambdaNormalizer", "expression": "x*2-1"},
            ],
            "postprocess": [{"name": "SigmoidPostprocessor"}],
        }
    )

    assert _ordered(current_input_norm_config()) == [
        [
            ("name", "MinMaxNormalizer"),
            ("min_value", 0.0),
            ("max_value", 255.0),
            ("invert", False),
        ],
        [("name", "LambdaNormalizer"), ("expression", "x*2-1")],
    ]
    assert _ordered(current_postprocess_config()) == [[("name", "SigmoidPostprocessor")]]
