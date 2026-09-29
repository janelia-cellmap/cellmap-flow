"""PipelineSpec: the one reader and writer of the chain's wire forms."""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cellmap_flow.pipeline_spec import (
    INPUT_NORM_KEY,
    POSTPROCESS_KEY,
    PipelineSpec,
    builder_steps,
    normalize_steps,
    split_dataset_url,
)
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
    decode_to_json,
    get_norms_post_args,
    list_cls_to_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

MINMAX = {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}
SHIFT = {"name": "LambdaNormalizer", "expression": "x*2-1"}
THRESHOLD = {"name": "ThresholdPostprocessor", "threshold": 0.5}


def test_keys_are_the_web_utils_strings():
    assert (INPUT_NORM_KEY, POSTPROCESS_KEY) == (INPUT_NORM_DICT_KEY, POSTPROCESS_DICT_KEY)


# --- normalize_steps -----------------------------------------------------------


@pytest.mark.parametrize("empty", [None, {}, [], ()])
def test_nothing_is_an_empty_chain(empty):
    assert normalize_steps(empty) == ()


def test_the_list_form_is_copied_as_given():
    steps = [dict(MINMAX, min_value="0"), SHIFT]
    got = normalize_steps(steps)
    assert got == (steps[0], steps[1])
    assert got[0]["min_value"] == "0", "values are not coerced"
    got[0]["min_value"] = 5
    assert steps[0]["min_value"] == "0", "the caller's steps are not shared"


def test_the_legacy_dict_form_becomes_steps_in_order():
    got = normalize_steps(
        {
            "MinMaxNormalizer": {"min_value": 0, "max_value": 255},
            "LambdaNormalizer": {"expression": "x*2-1"},
            "SigmoidPostprocessor": None,
        }
    )
    assert [list(s.items()) for s in got] == [
        [("name", "MinMaxNormalizer"), ("min_value", 0), ("max_value", 255)],
        [("name", "LambdaNormalizer"), ("expression", "x*2-1")],
        [("name", "SigmoidPostprocessor")],
    ]


def test_in_the_dict_form_the_key_names_the_class():
    # output_probe.suggest_input_norm writes a name inside the params too.
    (step,) = normalize_steps({"MinMaxNormalizer": {"name": "Other", "min_value": 1}})
    assert step == {"name": "MinMaxNormalizer", "min_value": 1}


def test_non_dict_list_elements_are_kept_for_the_reader_to_skip():
    assert normalize_steps([SHIFT, "junk"]) == (SHIFT, "junk")


def test_other_shapes_are_rejected_like_the_op_readers_do():
    with pytest.raises(ValueError, match="Expected dict or list"):
        normalize_steps("MinMaxNormalizer")


# --- builder_steps --------------------------------------------------------------


def test_builder_steps_put_the_name_last():
    nodes = [
        {"id": "n1", "name": "MinMaxNormalizer", "params": {"min_value": 0}},
        {"id": "n2", "name": "SigmoidPostprocessor"},
        {"id": "n3", "name": "LambdaNormalizer", "params": None},
    ]
    assert [list(s.items()) for s in builder_steps(nodes)] == [
        [("min_value", 0), ("name", "MinMaxNormalizer")],
        [("name", "SigmoidPostprocessor")],
        [("name", "LambdaNormalizer")],
    ]


def test_builder_steps_skip_nodes_without_a_name():
    nodes = [{"params": {"a": 1}}, {"name": ""}, "junk", {"name": "SigmoidPostprocessor"}]
    assert builder_steps(nodes) == ({"name": "SigmoidPostprocessor"},)
    assert builder_steps(None) == ()


# --- split_dataset_url ------------------------------------------------------------


def test_split_dataset_url():
    assert split_dataset_url("http://h:1/m") is None
    assert split_dataset_url(f"http://h:1/m{ARGS_KEY}abc{ARGS_KEY}") == "abc"
    with pytest.raises(ValueError, match="Expected two occurrences"):
        split_dataset_url(f"m{ARGS_KEY}abc")
    with pytest.raises(ValueError, match="found 4"):
        split_dataset_url(f"m{ARGS_KEY}a{ARGS_KEY}b{ARGS_KEY}")


# --- PipelineSpec -----------------------------------------------------------------


def test_construction_normalizes_both_chains():
    from_list = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    from_dict = PipelineSpec(
        {"MinMaxNormalizer": {"min_value": 0, "max_value": 255},
         "LambdaNormalizer": {"expression": "x*2-1"}},
        {"ThresholdPostprocessor": {"threshold": 0.5}},
    )
    assert from_list.input_norm == (MINMAX, SHIFT)
    assert from_list == from_dict
    assert PipelineSpec() == PipelineSpec(None, [])


def test_a_spec_is_frozen_and_hands_out_copies():
    spec = PipelineSpec([MINMAX])
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.input_norm = ()
    spec.to_json_data()[INPUT_NORM_KEY][0]["min_value"] = 99
    assert spec.input_norm[0]["min_value"] == 0


def test_from_json_data_accepts_every_shape():
    data = {"input_norm": [MINMAX], "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}}}
    spec = PipelineSpec.from_json_data(data)
    assert spec == PipelineSpec([MINMAX], [THRESHOLD])
    assert PipelineSpec.from_json_data(json.dumps(data)) == spec
    assert PipelineSpec.from_json_data(None).is_empty()
    # A missing chain is an empty one.
    assert PipelineSpec.from_json_data({"postprocess": [THRESHOLD]}).input_norm == ()


def test_strict_json_data_needs_both_chains_as_lists_or_dicts():
    data = {"input_norm": [MINMAX], "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}}}
    assert PipelineSpec.from_json_data(data, strict=True) == PipelineSpec([MINMAX], [THRESHOLD])
    with pytest.raises(KeyError):
        PipelineSpec.from_json_data({"postprocess": []}, strict=True)
    with pytest.raises(ValueError, match="Expected dict or list"):
        PipelineSpec.from_json_data({"input_norm": None, "postprocess": []}, strict=True)
    with pytest.raises(ValueError, match="Expected dict or list"):
        PipelineSpec.from_json_data({"input_norm": [], "postprocess": ()}, strict=True)


def test_to_json_data_has_input_norm_first():
    data = PipelineSpec(postprocess=[THRESHOLD], input_norm=[MINMAX]).to_json_data()
    assert list(data) == ["input_norm", "postprocess"]
    assert data == {"input_norm": [MINMAX], "postprocess": [THRESHOLD]}


def test_from_steps_is_list_cls_to_dict():
    from cellmap_flow.norm.input_normalize import LambdaNormalizer, MinMaxNormalizer
    from cellmap_flow.post.postprocessors import ChannelSelection

    norms = [MinMaxNormalizer(min_value="3"), LambdaNormalizer("x+1")]
    posts = [ChannelSelection("0,2")]
    spec = PipelineSpec.from_steps(norms, posts)
    assert list(spec.input_norm) == list_cls_to_dict(norms)
    assert list(spec.postprocess) == list_cls_to_dict(posts)
    assert PipelineSpec.from_steps() == PipelineSpec()


def test_url_blob_without_extras_is_get_norms_post_args():
    from cellmap_flow.norm.input_normalize import EuclideanDistance
    from cellmap_flow.post.postprocessors import LambdaPostprocessor

    norms = [EuclideanDistance(black_border=False)]
    posts = [LambdaPostprocessor("x + 1"), LambdaPostprocessor("x * 10")]
    assert PipelineSpec.from_steps(norms, posts).to_url_blob() == get_norms_post_args(
        norms, posts
    )


def test_url_blob_round_trip_with_extras():
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    blob = spec.to_url_blob(dashboard_url="http://dash/", digest=spec.digest())
    assert list(decode_to_json(blob)) == [
        "input_norm", "postprocess", "dashboard_url", "digest",
    ]
    got, extras = PipelineSpec.from_url_blob(blob)
    assert got == spec
    assert extras == {"dashboard_url": "http://dash/", "digest": spec.digest()}


def test_extras_cannot_replace_a_chain():
    with pytest.raises(ValueError):
        PipelineSpec().to_url_blob(postprocess=[])


def test_digest_follows_content_not_identity():
    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    assert spec.digest() == PipelineSpec([dict(MINMAX), dict(SHIFT)], [dict(THRESHOLD)]).digest()
    assert len(spec.digest()) == 16 and int(spec.digest(), 16) >= 0
    # The order of a step's own keys does not matter; everything else does.
    reordered = {"max_value": 255, "name": "MinMaxNormalizer", "min_value": 0}
    assert PipelineSpec([reordered, SHIFT], [THRESHOLD]).digest() == spec.digest()
    assert PipelineSpec([SHIFT, MINMAX], [THRESHOLD]).digest() != spec.digest()
    assert PipelineSpec([MINMAX, SHIFT], [dict(THRESHOLD, threshold=0.6)]).digest() != spec.digest()
    assert PipelineSpec([MINMAX, SHIFT], []).digest() != spec.digest()
    # The same chain on the other side is a different pipeline.
    assert PipelineSpec([], [MINMAX]).digest() != PipelineSpec([MINMAX], []).digest()


def test_digest_is_the_same_in_every_process():
    # It goes into layer URLs, so it must not depend on hash seeds and the like.
    assert PipelineSpec([MINMAX, SHIFT], [THRESHOLD]).digest() == "4d284fb937eabaec"


def test_build_makes_new_instances_in_order():
    spec = PipelineSpec([SHIFT, MINMAX, dict(SHIFT, expression="x*3")], [THRESHOLD])
    norms, posts = spec.build()
    assert [type(n).__name__ for n in norms] == [
        "LambdaNormalizer", "MinMaxNormalizer", "LambdaNormalizer",
    ]
    assert [n.expression for n in norms if hasattr(n, "expression")] == ["x*2-1", "x*3"]
    assert [p.threshold for p in posts] == [0.5]
    again, _ = spec.build()
    assert again[0] is not norms[0]


def test_is_empty():
    assert PipelineSpec().is_empty()
    assert not PipelineSpec(postprocess=[THRESHOLD]).is_empty()


def test_importing_the_module_stays_light():
    """No globals (and so no logging config), Flask, viewer, torch or ops."""
    forbidden = [
        "cellmap_flow.globals",
        "flask",
        "neuroglancer",
        "torch",
        "huggingface_hub",
        "peft",
        "cellmap_flow.norm.input_normalize",
        "cellmap_flow.post.postprocessors",
    ]
    code = (
        "import sys, cellmap_flow.pipeline_spec\n"
        f"loaded = [m for m in {forbidden!r} if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    ))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


# --- Flow.pipeline_spec and Flow.set_pipeline ---------------------------------------


def test_the_flow_spec_prefers_the_configured_steps():
    from cellmap_flow.globals import g
    from cellmap_flow.norm.input_normalize import ZScoreNormalizer

    g.input_norms = [ZScoreNormalizer(mean=1)]
    g.input_norm_config = [dict(SHIFT, expression="x*5")]
    g.postprocess, g.postprocess_config = [], []
    assert g.pipeline_spec == PipelineSpec([dict(SHIFT, expression="x*5")], [])


def test_the_flow_spec_falls_back_to_the_live_chain_per_chain():
    from cellmap_flow.globals import g
    from cellmap_flow.post.postprocessors import SigmoidPostprocessor

    g.input_norms, g.input_norm_config = [], [MINMAX]
    g.postprocess, g.postprocess_config = [SigmoidPostprocessor()], {}
    spec = g.pipeline_spec
    assert spec.input_norm == (MINMAX,)
    assert spec.postprocess == ({"name": "SigmoidPostprocessor"},)
    # An old name-keyed config reads as steps.
    g.input_norm_config = {"LambdaNormalizer": {"expression": "x*2-1"}}
    assert g.pipeline_spec.input_norm == (SHIFT,)


def test_the_flow_spec_is_derived_not_stored():
    from cellmap_flow.globals import g

    assert "pipeline_spec" not in vars(g)
    with pytest.raises(AttributeError):
        g.pipeline_spec = PipelineSpec()


def test_set_pipeline_writes_all_four_attributes():
    from cellmap_flow.globals import g

    spec = PipelineSpec([MINMAX, SHIFT], [THRESHOLD])
    g.set_pipeline(spec)
    assert [type(n).__name__ for n in g.input_norms] == [
        "MinMaxNormalizer", "LambdaNormalizer",
    ]
    assert [type(p).__name__ for p in g.postprocess] == ["ThresholdPostprocessor"]
    assert g.input_norm_config == [MINMAX, SHIFT]
    assert g.postprocess_config == [THRESHOLD]
    assert g.pipeline_spec == spec


def test_set_pipeline_keeps_the_instances_it_is_given():
    from cellmap_flow.globals import g
    from cellmap_flow.post.postprocessors import SimpleBlockwiseMerger

    merger = SimpleBlockwiseMerger()
    spec = PipelineSpec.from_steps([], [merger])
    g.set_pipeline(spec, built=([], [merger]))
    assert g.postprocess[0] is merger


def test_a_chain_that_fails_to_build_changes_nothing():
    from cellmap_flow.globals import g

    g.set_pipeline(PipelineSpec([MINMAX], []))
    before = (list(g.input_norms), g.input_norm_config, g.postprocess_config)
    with pytest.raises(ValueError):
        g.set_pipeline(
            PipelineSpec([SHIFT], [{"name": "ThresholdPostprocessor", "threshold": "high"}])
        )
    assert (g.input_norms, g.input_norm_config, g.postprocess_config) == before


@pytest.fixture
def dashboard(monkeypatch):
    from flask import Flask

    import cellmap_flow.dashboard.routes.pipeline as pipeline
    from cellmap_flow.globals import g

    monkeypatch.setattr(
        pipeline, "get_raw_layer", lambda path: type("Raw", (), {"shader": None})()
    )
    monkeypatch.setattr(pipeline, "fetch_model_info", lambda host: {})

    class Viewer:
        state = type("State", (), {"layers": {}})()

        def txn(self):
            import contextlib

            return contextlib.nullcontext(self.state)

    g.viewer, g.jobs, g.dataset_path = Viewer(), [], "/data/raw.zarr"
    g.shaders, g.shader_controls = {}, {}

    written = []
    real = type(g).set_pipeline

    def spy(spec, built=None):
        written.append(spec)
        real(g, spec, built)

    monkeypatch.setattr(g, "set_pipeline", spy)
    app = Flask(__name__)
    app.register_blueprint(pipeline.pipeline_bp)
    return app.test_client(), written


def test_process_writes_through_set_pipeline(dashboard):
    client, written = dashboard
    response = client.post(
        "/api/process", json={"input_norm": [MINMAX], "postprocess": [THRESHOLD]}
    )
    assert response.status_code == 200
    assert written == [PipelineSpec([MINMAX], [THRESHOLD])]


def test_apply_writes_through_set_pipeline(dashboard):
    client, written = dashboard
    response = client.post(
        "/api/pipeline/apply",
        json={
            "input_normalizers": [{"name": "LambdaNormalizer", "params": {"expression": "x+1"}}],
            "postprocessors": [],
        },
    )
    assert response.status_code == 200
    assert written == [PipelineSpec([{"expression": "x+1", "name": "LambdaNormalizer"}])]


@pytest.mark.parametrize(
    "chains", [{"input_norm": []}, {"input_norm": None, "postprocess": []}]
)
def test_process_still_refuses_a_request_without_both_chains(dashboard, chains):
    from cellmap_flow.globals import g

    client, written = dashboard
    g.set_pipeline(PipelineSpec([MINMAX], []))
    client.application.config["PROPAGATE_EXCEPTIONS"] = False
    response = client.post("/api/process", json=chains)
    assert response.status_code == 500
    assert written == [PipelineSpec([MINMAX], [])], "nothing new was written"
    assert g.input_norm_config == [MINMAX]
