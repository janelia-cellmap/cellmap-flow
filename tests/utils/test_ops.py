"""The normalizers and postprocessors, one by one.

Each op reaches an inference server as its to_dict() inside a layer URL, so
it must rebuild from it; the rest are fixes to what individual ops computed,
and the whitelist that Lambda expressions from URLs are held to. What the ops
the browser ports (labels, affinities, Morton ids, the merger, the norms)
compute for fixed inputs is in web/test/fixtures, which test_fixtures_fresh
checks against this code.
"""

import threading
import time

import numpy as np
import pytest

from cellmap_flow.norm.input_normalize import (
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


def test_euclidean_distance_honours_its_parameters():
    mask = np.zeros((9, 9, 9), np.uint8)
    mask[2:7, 2:7, 2:7] = 1
    assert EuclideanDistance(activation=None, black_border=False)(mask).max() == pytest.approx(150.0)  # 3 voxels in at 50
    assert EuclideanDistance(activation="tanh")(mask).max() <= 1.0
    signed = EuclideanDistance(type="sdf", activation=None)(mask)
    assert signed.min() < 0 < signed.max()


@pytest.mark.parametrize("shape", [pytest.param((6, 6, 6), id="zyx"), pytest.param((2, 6, 6, 6), id="with-channels")])
def test_fill_holes_fills_what_a_blob_encloses(shape):
    logits = np.full(shape, -1.0, np.float32)
    logits[..., 1:5, 1:5, 1:5] = 1.0
    logits[..., 2, 2, 2] = -1.0  # an enclosed hole
    expected = np.zeros(shape, np.uint8)
    expected[..., 1:5, 1:5, 1:5] = 1
    np.testing.assert_array_equal(FillHolesPostprocessor(threshold="0")(logits), expected)


@pytest.mark.parametrize("output_class, chain, names, level", [
    pytest.param("unbounded", ["SigmoidPostprocessor", "AffinityPostprocessor"], {"model_name": "aff"}, "ok",
                 id="probabilities"),
    pytest.param("unit", ["AffinityPostprocessor"], {"model_name": "aff"}, "ok", id="a-model-giving-probabilities"),
    pytest.param("unbounded", ["AffinityPostprocessor"], {"model_name": "aff"}, "warn", id="logits"),
    pytest.param("unbounded", ["AffinityPostprocessor"], {"channels_names": "affinities"}, "warn",
                 id="logits-from-a-model-whose-one-channel-name-is-a-string"),
    # The sigmoid a start or the ranking set shows them, a threshold of the
    # user's too; the segmentation is the user's choice, not a fault.
    pytest.param("unbounded", ["SigmoidPostprocessor", "ThresholdPostprocessor"], {"model_name": "aff"}, "optional",
                 id="a-segmentation-is-optional"),
    # The threshold leaves the watershed 0s and 1s, and no objects.
    pytest.param("unbounded", ["SigmoidPostprocessor", "ThresholdPostprocessor", "AffinityPostprocessor"],
                 {"model_name": "aff"}, "warn", id="a-threshold-ahead-of-the-watershed"),
])
def test_the_model_advice_agrees_with_what_affinities_accept(output_class, chain, names, level):
    from cellmap_flow.serving import probe

    output_class = {"unbounded": probe.UNBOUNDED, "unit": probe.UNIT}[output_class]
    assert probe.review_postprocess(output_class, chain, out_channels=3, **names)["level"] == level


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
        pytest.param("np.max(x, axis=0, keepdims=True)", np.max(X, axis=0, keepdims=True), id="reduction-keywords"),
        pytest.param("np.full_like(x, 3, np.uint8)", np.full_like(X, 3, np.uint8), id="full-like-with-a-dtype"),
        pytest.param("np.stack([x, x])", np.stack([X, X]), id="a-literal-list"),
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
        pytest.param("np.concatenate([x] * 10**5)", id="repeated-list"),
        pytest.param("np.stack((x,) * 10**4)", id="repeated-tuple"),
        pytest.param("np.full_like(x, 0, shape=(10**5, 10**5))", id="shape-keyword"),
        pytest.param("np.zeros_like(x, None, None, True, (10**5, 10**5))", id="positional-shape"),
        pytest.param("np.max(x, out=x)", id="other-keyword"),
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
