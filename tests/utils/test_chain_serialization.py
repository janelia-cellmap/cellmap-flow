"""Normalizer/postprocessor chains must survive the trip through a layer URL.

The chain reaches every inference server as JSON inside the layer URL, so a
step that cannot be rebuilt from its own ``to_dict()`` breaks the layer, and a
serialized form that loses steps or order silently serves a different chain.
"""

import numpy as np
import pytest

from cellmap_flow.norm.input_normalize import (
    ChannelSelector,
    EuclideanDistance,
    InputNormalizer,
    LambdaNormalizer,
    MinMaxNormalizer,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import (
    AffinityPostprocessor,
    ChannelSelection,
    DefaultPostprocessor,
    LambdaPostprocessor,
    MortonSegmentationRelabeling,
    PostProcessor,
    SimpleBlockwiseMerger,
    get_postprocessors,
)
from cellmap_flow.utils.serilization_utils import (
    get_process_dataset,
    get_process_dataset_url,
    serialize_norms_posts_to_json,
)
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    get_norms_post_args,
    list_cls_to_dict,
)


def _rebuild(step):
    d = step.to_dict()
    base = InputNormalizer if isinstance(step, InputNormalizer) else PostProcessor
    (rebuilt,) = (
        get_normalizations([d]) if base is InputNormalizer else get_postprocessors([d])
    )
    return rebuilt


@pytest.mark.parametrize(
    "step",
    [
        AffinityPostprocessor(bias=0.5),
        MortonSegmentationRelabeling(channel=1),
        SimpleBlockwiseMerger(face_erosion_iterations=1),
        ChannelSelection(channels="0,2"),
        EuclideanDistance(anisotropy=8, black_border=False, type="sdf"),
        DefaultPostprocessor(clip_min=0.0, clip_max=1.0),
        MinMaxNormalizer(min_value=10, max_value=200, invert=True),
    ],
    ids=lambda s: type(s).__name__,
)
def test_step_rebuilds_from_its_own_to_dict(step):
    rebuilt = _rebuild(step)
    assert type(rebuilt) is type(step)
    assert rebuilt.to_dict() == step.to_dict()


def test_to_dict_is_the_constructor_arguments_only():
    d = AffinityPostprocessor(bias=0.25).to_dict()
    assert set(d) == {"name", "bias", "neighborhood"}
    # Internal state the constructor adds is not a constructor argument.
    assert "use_exact" not in d and "num_previous_segments" not in d


def test_to_dict_reports_arguments_with_their_real_types():
    # The dashboard forms send strings; the step itself knows the types.
    d = MinMaxNormalizer(min_value="10", max_value="20", invert="false").to_dict()
    assert (d["min_value"], d["max_value"], d["invert"]) == (10.0, 20.0, False)
    # An argument the constructor parses into something else stays as given.
    assert ChannelSelection(channels="0,2").to_dict()["channels"] == "0,2"


def test_rebuilt_steps_compute_the_same_thing():
    data = np.random.default_rng(0).random((3, 6, 6, 6)).astype(np.float32)
    select = ChannelSelection(channels="0,2")
    assert np.array_equal(_rebuild(select)(data), select(data))

    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    edt = EuclideanDistance(anisotropy=4, black_border=False, activation=None)
    np.testing.assert_allclose(_rebuild(edt)(mask), edt(mask))


def test_values_keep_their_types_through_the_url_blob():
    blob = get_norms_post_args(
        [EuclideanDistance(black_border=False)], [ChannelSelection(channels="1")]
    )
    _, norms, posts = get_process_dataset_url(f"m{ARGS_KEY}{blob}{ARGS_KEY}")
    # Stringified, "False" read back as True.
    assert norms[0].black_border is False
    assert posts[0].channels == [1]


def test_two_steps_of_the_same_class_both_survive_in_order():
    posts = [LambdaPostprocessor("x + 1"), LambdaPostprocessor("x * 10")]
    blob = get_norms_post_args([], posts)
    _, _, rebuilt = get_process_dataset_url(f"m{ARGS_KEY}{blob}{ARGS_KEY}")

    assert [p.expression for p in rebuilt] == ["x + 1", "x * 10"]
    x = np.array([1.0], dtype=np.float32)
    out = x
    for p in rebuilt:
        out = p(out)
    assert out[0] == 20.0


def test_serialized_chains_are_ordered_lists():
    steps = list_cls_to_dict([MinMaxNormalizer(), LambdaNormalizer("x*2-1")])
    assert [s["name"] for s in steps] == ["MinMaxNormalizer", "LambdaNormalizer"]

    norms, _ = get_process_dataset(
        serialize_norms_posts_to_json(
            [LambdaNormalizer("x+1"), LambdaNormalizer("x*3")], []
        )
    )
    assert [n.expression for n in norms] == ["x+1", "x*3"]


def test_the_old_dict_form_is_still_read():
    norms, posts = get_process_dataset(
        {
            "input_norm": {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}},
            "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}},
        }
    )
    assert [type(n).__name__ for n in norms] == ["MinMaxNormalizer"]
    assert [type(p).__name__ for p in posts] == ["ThresholdPostprocessor"]


def test_current_config_fallback_keeps_every_step():
    from cellmap_flow.globals import (
        current_input_norm_config,
        current_postprocess_config,
        g,
    )

    g.input_norm_config = {}
    g.postprocess_config = {}
    g.input_norms = [LambdaNormalizer("x+1"), LambdaNormalizer("x*2")]
    g.postprocess = [AffinityPostprocessor(bias=0.5)]

    norms = get_normalizations(current_input_norm_config())
    assert [n.expression for n in norms] == ["x+1", "x*2"]
    (affinity,) = get_postprocessors(current_postprocess_config())
    assert affinity.bias == 0.5


def test_pipeline_apply_keeps_repeated_steps():
    from flask import Flask

    from cellmap_flow.dashboard.routes.pipeline import pipeline_bp
    from cellmap_flow.globals import g

    app = Flask(__name__)
    app.register_blueprint(pipeline_bp)
    payload = {
        "input_normalizers": [
            {"name": "LambdaNormalizer", "params": {"expression": "x+1"}},
            {"name": "LambdaNormalizer", "params": {"expression": "x*2"}},
        ],
        "postprocessors": [],
    }
    response = app.test_client().post("/api/pipeline/apply", json=payload)

    assert response.status_code == 200
    assert [n.expression for n in g.input_norms] == ["x+1", "x*2"]
    assert [n.expression for n in get_normalizations(g.input_norm_config)] == [
        "x+1",
        "x*2",
    ]
