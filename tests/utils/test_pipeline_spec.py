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
