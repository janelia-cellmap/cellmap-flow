"""The normalizers and postprocessors, one by one.

Each op reaches an inference server as its to_dict() inside a layer URL (the
wire form is pinned in test_pipeline_spec), so it must rebuild from it; the
rest are fixes to what individual ops computed, and the whitelist that Lambda
expressions from URLs are held to.
"""

import pickle
import threading
import time

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
    FillHolesPostprocessor,
    LabelPostprocessor,
    LambdaPostprocessor,
    MortonSegmentationRelabeling,
    SimpleBlockwiseMerger,
    get_postprocessors,
)
from cellmap_flow.norm.safe_expression import compile_expression

MASK = np.zeros((8, 8, 8), np.uint8)
MASK[2:6, 2:6, 2:6] = 1


@pytest.mark.parametrize(
    "op, data",
    [
        pytest.param(AffinityPostprocessor(bias=0.5), None, id="AffinityPostprocessor"),
        pytest.param(MortonSegmentationRelabeling(channel=1), None, id="MortonSegmentationRelabeling"),
        pytest.param(SimpleBlockwiseMerger(face_erosion_iterations=1), None, id="SimpleBlockwiseMerger"),
        pytest.param(ChannelSelection(channels="0,2"), np.random.default_rng(0).random((3, 6, 6, 6)).astype(np.float32), id="ChannelSelection"),
        pytest.param(EuclideanDistance(anisotropy=4, black_border=False, activation=None), MASK, id="EuclideanDistance"),
        pytest.param(DefaultPostprocessor(clip_min=0.0, clip_max=1.0), np.linspace(-1, 2, 64, dtype=np.float32), id="DefaultPostprocessor"),
        pytest.param(MinMaxNormalizer(min_value=10, max_value=200, invert=True), None, id="MinMaxNormalizer"),
    ],
)
def test_every_op_rebuilds_from_its_own_to_dict(op, data):
    build = get_normalizations if isinstance(op, InputNormalizer) else get_postprocessors
    (rebuilt,) = build([op.to_dict()])
    assert type(rebuilt) is type(op) and rebuilt.to_dict() == op.to_dict()
    if data is not None:
        np.testing.assert_allclose(rebuilt(data), op(data))


def test_the_dashboards_strings_become_the_ops_own_types():
    # The forms send every value as a string; the op knows its types.
    assert MinMaxNormalizer(min_value="10", max_value="20", invert="false").to_dict() == {
        "name": "MinMaxNormalizer", "min_value": 10.0, "max_value": 20.0, "invert": False,
    }
    assert EuclideanDistance(black_border="False").black_border is False


def test_channel_selection_takes_what_a_yaml_gives_and_the_url_keeps_it_as_given():
    assert [ChannelSelection(c).channels for c in ([0, 2], 1, "0,2")] == [[0, 2], [1], [0, 2]]
    assert ChannelSelection("0,2").to_dict()["channels"] == "0,2"


def test_steps_without_a_dtype_keep_their_inputs_rather_than_float64():
    data = np.arange(16, dtype=np.uint8).reshape(2, 2, 2, 2)
    assert ChannelSelector(0)(data).dtype == ChannelSelection("0")(data).dtype == np.uint8


def test_euclidean_distance_honours_its_parameters():
    mask = np.zeros((9, 9, 9), np.uint8)
    mask[2:7, 2:7, 2:7] = 1
    assert EuclideanDistance(activation=None, black_border=False)(mask).max() == pytest.approx(150.0)  # 3 voxels in at 50
    assert EuclideanDistance(activation="tanh")(mask).max() <= 1.0
    signed = EuclideanDistance(type="sdf", activation=None)(mask)
    assert signed.min() < 0 < signed.max()


def test_labels_do_not_wrap_above_255_and_leave_the_input_alone():
    data = np.zeros((1, 3, 30, 30), np.uint8)
    data[0, ::2, ::2, ::2] = 1  # 2 x 15 x 15 = 450 isolated voxels
    labels = LabelPostprocessor(channel=0)(data, chunk_corner=(0, 0, 0), chunk_num_voxels=data[0].size)
    assert (labels.dtype, labels.max(), data.max()) == (np.uint32, 450, 1)


def test_each_chunks_ids_are_offset_by_its_morton_index():
    blob = np.zeros((1, 4, 4, 4), np.uint8)
    blob[0, 1:3, 1:3, 1:3] = 1
    assert MortonSegmentationRelabeling()(blob, chunk_corner=(1, 0, 0), chunk_num_voxels=np.int64(64)).max() == 1 + 64


@pytest.mark.parametrize("shape", [pytest.param((6, 6, 6), id="zyx"), pytest.param((2, 6, 6, 6), id="with-channels")])
def test_fill_holes_fills_what_a_blob_encloses(shape):
    logits = np.full(shape, -1.0, np.float32)
    logits[..., 1:5, 1:5, 1:5] = 1.0
    logits[..., 2, 2, 2] = -1.0  # an enclosed hole
    expected = np.zeros(shape, np.uint8)
    expected[..., 1:5, 1:5, 1:5] = 1
    np.testing.assert_array_equal(FillHolesPostprocessor(threshold="0")(logits), expected)


def test_affinities_are_agglomerated_from_probabilities_or_uint8():
    affs = np.full((3, 8, 8, 8), 0.9, np.float32)
    affs[:, :, :, 4] = 0.05  # a wall of repulsive edges down the middle
    kwargs = dict(chunk_num_voxels=512, chunk_corner=(0, 0, 0))
    post = AffinityPostprocessor(bias=0.5)
    from_float = post(affs, **kwargs)
    from_uint8 = AffinityPostprocessor(bias=0.5)((affs * 255).astype(np.uint8), **kwargs)
    assert len(np.unique(from_float[from_float > 0])) == 2, "two halves either side of the wall"
    assert np.array_equal(from_float > 0, from_uint8 > 0) and len(np.unique(from_float)) == len(np.unique(from_uint8))
    assert len(post.neighborhood) == 9, "three channels of input did not shrink it"


@pytest.mark.parametrize("output_class, chain, level", [
    pytest.param("unbounded", ["SigmoidPostprocessor", "AffinityPostprocessor"], "ok", id="probabilities"),
    pytest.param("unit", ["AffinityPostprocessor"], "ok", id="a-model-giving-probabilities"),
    pytest.param("unbounded", ["AffinityPostprocessor"], "warn", id="logits"),
])
def test_the_model_advice_agrees_with_what_affinities_accept(output_class, chain, level):
    from cellmap_flow.utils import output_probe

    output_class = {"unbounded": output_probe.UNBOUNDED, "unit": output_probe.UNIT}[output_class]
    assert output_probe.review_postprocess(output_class, chain, out_channels=3, model_name="aff")["level"] == level


def test_the_merger_survives_concurrent_chunks():
    """The server runs one merger from every request thread: iterating
    keys_to_skip while another thread added to it raised "Set changed size
    during iteration". The set yields slowly so the threads interleave."""

    class SlowSet(set):
        def __iter__(self):
            for item in super().__iter__():
                time.sleep(0.001)
                yield item

    merger = SimpleBlockwiseMerger()
    merger.keys_to_skip = SlowSet()
    errors = []

    def serve(corner):
        try:
            merger(np.ones((1, 4, 4, 4), np.uint64), chunk_corner=corner)
        except Exception as e:  # noqa: BLE001 - any failure fails the test
            errors.append(e)

    for _ in range(3):  # neighbouring chunks, so every call matches faces
        threads = [threading.Thread(target=serve, args=((z, y, 0),)) for z in range(4) for y in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert not errors, errors[0]
    assert len(merger.keys_to_skip) > 0


def test_the_merger_pickles_with_its_equivalences():
    """Its libraries are imported lazily (see test_import_hygiene), which
    must not break pickling it."""
    merger = SimpleBlockwiseMerger(face_erosion_iterations=1)
    merger(np.ones((1, 4, 4, 4), np.uint64), chunk_corner=(0, 0, 0))
    merger.equivalences.union(1, 2)
    assert pickle.loads(pickle.dumps(merger)).equivalences_json() == merger.equivalences_json()


# --- Lambda expressions, which arrive in layer URLs -------------------------------

X = np.array([[-1.0, 0.25], [0.75, 2.0]], dtype=np.float32)


@pytest.mark.parametrize(
    "expression, expected",
    [
        pytest.param("x*2-1", X * 2 - 1, id="affine"),
        pytest.param("np.clip(x, 0, 1)", np.clip(X, 0, 1), id="clip"),
        pytest.param("(x > 0.5).astype(np.uint8)", (X > 0.5).astype(np.uint8), id="threshold-to-uint8"),
        pytest.param("x.astype('uint8')", X.astype("uint8"), id="astype-string"),
        pytest.param("np.where(x > 0, x, 0)", np.where(X > 0, X, 0), id="where"),
        pytest.param("x[..., 0]", X[..., 0], id="index"),
        pytest.param("x ** 2 + -x", X**2 - X, id="power-and-negation"),
        pytest.param("abs(x) / np.max(abs(x))", np.abs(X) / np.abs(X).max(), id="abs-and-max"),
        pytest.param("x if x.ndim == 2 else x * 0", X, id="conditional"),
        pytest.param("np.clip(x, a_min=0, a_max=None)", np.clip(X, 0, None), id="keyword-arguments"),
    ],
)
def test_numpy_math_on_x_is_allowed(expression, expected):
    np.testing.assert_array_equal(compile_expression(expression)(X), expected)
    for op in (LambdaNormalizer, LambdaPostprocessor):
        np.testing.assert_array_equal(op(expression)(X), expected)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("__import__('os').system('true')", id="import"),
        pytest.param("open('/etc/passwd').read()", id="open-a-file"),
        pytest.param("x.__class__.__mro__", id="dunder-attribute"),
        pytest.param("().__class__.__bases__[0].__subclasses__()", id="subclasses-escape"),
        pytest.param("np.load('/tmp/anything.npy', allow_pickle=True)", id="np-load-pickle"),
        pytest.param("getattr(x, 'shape')", id="getattr"),
        pytest.param("exec('1')", id="exec"),
        pytest.param("(lambda: 1)()", id="lambda"),
        pytest.param("[y for y in x]", id="comprehension"),
        pytest.param("{**{}}", id="dict-unpacking"),
        pytest.param("x ** 10**10", id="huge-power"),
        pytest.param("x ** x", id="power-of-x"),
        pytest.param("x.astype('U1000')", id="huge-string-dtype"),
        pytest.param("x = 1", id="assignment"),
        pytest.param("x" * 600, id="too-long"),
    ],
)
def test_anything_else_is_refused_before_it_runs(expression):
    """Anything reaching builtins, dunders or unbounded work would let a URL
    run code on, or stall, an inference server."""
    with pytest.raises(ValueError):
        compile_expression(expression)
    for op in (LambdaNormalizer, LambdaPostprocessor):
        with pytest.raises(ValueError):
            op(expression)
