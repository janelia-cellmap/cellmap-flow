"""Generate the parity fixtures the TypeScript ports are tested against.

The browser mode re-implements pieces of the Python package in TypeScript:
the input normalizers (``web/src/infer/norms.ts`` from
``cellmap_flow/norm/input_normalize.py``), the postprocessors
(``web/src/infer/post.ts`` from ``cellmap_flow/post/postprocessors.py``, with
the segmentation steps in ``affinity.ts`` and ``morton.ts``, which also port
``cellmap_flow/post/segment.py``'s ``mutex_watershed`` and pymorton's
interleave) and the layer URL's args blob (``web/src/shared/args.ts`` from
``cellmap_flow/pipeline_spec.py`` and ``serving/protocol.py``). A shared link
works across the two modes only if these agree byte for byte, so the vitest
suite in ``web/test/parity`` checks each port against what the *Python classes
themselves* produce, written here as JSON, rather than against a reading of
the Python code.

Run it with a Python that has cellmap_flow's dependencies, from anywhere::

    .pixi/envs/default/bin/python web/test/fixtures/make_fixtures.py          # regenerate
    .pixi/envs/default/bin/python web/test/fixtures/make_fixtures.py --check  # CI: stale?

``--check`` regenerates into a temporary directory and compares with the
committed files (floats with a small tolerance, since numpy's float32
transcendentals may differ by an ulp between machines; everything else
exactly), exiting 1 when the Python has changed and the fixtures have not.
``tests/web/test_fixtures_fresh.py`` runs that in pytest.

Every fixture is deterministic (a fixed seed, recorded in the file) and the
script is idempotent. Each JSON carries a ``_format`` key describing it;
arrays are ``{"dtype", "shape", "data"}`` with ``data`` the flat C-order
values, exactly what ``new Float32Array(data)`` takes.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The worktree's own package, ahead of any installed copy: the fixtures must
# describe the code beside them.
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from cellmap_flow.pipeline_spec import (  # noqa: E402
    PipelineSpec,
    chain_is_segmentation,
    chain_num_channels,
    chain_output_dtype,
)
from cellmap_flow.post import segment  # noqa: E402
from cellmap_flow.post.postprocessors import (  # noqa: E402
    AffinityPostprocessor,
    DefaultPostprocessor,
    ThresholdPostprocessor,
)
from cellmap_flow.serving.protocol import ARGS_KEY  # noqa: E402
from cellmap_flow.viewer.layers import prediction_url  # noqa: E402

FIXTURES = ("norms.json", "post.json", "args.json", "affinity.json", "morton.json")

# The dashboard a browser-mode blob names, and the model/host the layer URL
# fixture uses. Any values do; they only have to be the same on both sides.
DASHBOARD_URL = "http://localhost:5173/"
HOST = "host"
MODEL = "model"

# How far a value that a uint8 step truncates may sit from an integer, and a
# value that a threshold compares may sit from the threshold. The TypeScript
# may compute in float64 where numpy uses float32, so a value closer than this
# could truncate or compare differently for a reason that is not a bug; the
# input seed is chosen so that no fixture value is that close.
TRUNCATION_MARGIN = 1e-4
THRESHOLD_MARGIN = 1e-5


def array_record(arr: np.ndarray) -> dict:
    """``{"dtype", "shape", "data"}``; ``data`` flat, C order, plain Python numbers."""
    arr = np.ascontiguousarray(arr)
    return {"dtype": arr.dtype.name, "shape": list(arr.shape), "data": arr.ravel().tolist()}


def packed_record(arr: np.ndarray) -> dict:
    """``{"dtype", "shape", "base64"}``: the C-order little-endian bytes, base64.

    For a block too large to list value by value (the indented JSON spends
    some 30 bytes on each); the tests decode it into the typed array of
    ``dtype`` directly, so the values are bit-exact.
    """
    arr = np.ascontiguousarray(arr)
    return {"dtype": arr.dtype.name, "shape": list(arr.shape), "base64": base64.b64encode(arr.astype(arr.dtype.newbyteorder("<")).tobytes()).decode("ascii")}


def uint64_record(arr: np.ndarray) -> dict:
    """``{"dtype": "uint64", "shape", "data"}`` with the values as decimal strings.

    A JSON number is a double in JavaScript and loses digits above 2^53;
    Morton-offset ids on a large volume pass that, so they travel as text
    and the tests read them with ``BigInt(text)``.
    """
    arr = np.ascontiguousarray(arr)
    assert arr.dtype == np.uint64, f"uint64 expected, got {arr.dtype}"
    return {"dtype": "uint64", "shape": list(arr.shape), "data": [str(int(v)) for v in arr.ravel()]}


def dtype_name(dtype) -> str:
    return np.dtype(dtype).name


# --- norms.json ---------------------------------------------------------------

NORM_CASES = [
    # (id, steps). Parameters are given both as numbers and as the strings
    # the dashboard's forms send, to pin the parsing in the port.
    ("minmax_defaults", [{"name": "MinMaxNormalizer"}]),
    ("minmax_0_255", [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}]),
    ("minmax_strings", [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"}]),
    ("minmax_invert_bool", [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255, "invert": True}]),
    ("minmax_invert_string", [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255", "invert": "true"}]),
    # The Python lower-cases the string: "True" (a YAML's) inverts, "false" does not.
    ("minmax_invert_capitalised", [{"name": "MinMaxNormalizer", "invert": "True"}]),
    ("minmax_invert_false_string", [{"name": "MinMaxNormalizer", "invert": "false"}]),
    # A window inside the data's range, so the clip changes more than the out-of-range values.
    ("minmax_50_200", [{"name": "MinMaxNormalizer", "min_value": 50, "max_value": 200}]),
    ("zscore_defaults", [{"name": "ZScoreNormalizer"}]),
    ("zscore_128_64", [{"name": "ZScoreNormalizer", "mean": 128, "std": 64}]),
    ("zscore_strings", [{"name": "ZScoreNormalizer", "mean": "128", "std": "64"}]),
    ("lambda_2x_minus_1", [{"name": "LambdaNormalizer", "expression": "2*x-1"}]),
    ("lambda_x_over_255", [{"name": "LambdaNormalizer", "expression": "x/255"}]),
    ("lambda_parenthesised", [{"name": "LambdaNormalizer", "expression": "(x-128)/128"}]),
    # The chain the browser mode's models declare (onnx-web-models manifests).
    (
        "chain_default_browser",
        [
            {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
            {"name": "LambdaNormalizer", "expression": "2*x-1"},
        ],
    ),
]


def norm_input(seed: int) -> np.ndarray:
    """A 3x4x5 float32 block of 0..255 values with out-of-range and edge values in it."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0, 255, size=(3, 4, 5)).astype(np.float32)
    data[0, 0, :5] = [-5, 300, 0, 255, 127.5]
    return data


def make_norms() -> dict:
    data = norm_input(0)
    cases = []
    for case_id, steps in NORM_CASES:
        normalizers = PipelineSpec(steps, ()).build()[0]
        assert len(normalizers) == len(steps), f"{case_id}: a step was not built"
        out = data.copy()
        for normalizer in normalizers:
            out = normalizer(out)
        assert out.dtype == np.float32, f"{case_id}: normalizers return float32, got {out.dtype}"
        cases.append({"id": case_id, "steps": steps, "output": array_record(out)})
    return {
        "_format": (
            "Input normalizers, from cellmap_flow/norm/input_normalize.py. `input` is one float32 "
            "block (z, y, x); each case applies its `steps` (in order, as PipelineSpec builds them) "
            "to a copy of it and records the Python result in `output`. Arrays are {dtype, shape, "
            "data} with data flat in C order. Generated by make_fixtures.py; do not edit."
        ),
        "seed": 0,
        "input": array_record(data),
        "cases": cases,
    }


# --- post.json ----------------------------------------------------------------

POST_CASES = [
    # (id, steps). The model output the steps run on is float32 logits, [C, z, y, x].
    ("empty", []),
    ("sigmoid", [{"name": "SigmoidPostprocessor"}]),
    ("default_defaults", [{"name": "DefaultPostprocessor"}]),
    ("default_variant", [{"name": "DefaultPostprocessor", "clip_min": -2, "clip_max": 2, "bias": 2, "multiplier": 63.75}]),
    ("default_strings", [{"name": "DefaultPostprocessor", "clip_min": "-1", "clip_max": "1", "bias": "1", "multiplier": "127.5"}]),
    # (x + 1) * 200 reaches 400: numpy's astype(uint8) keeps the low byte of
    # the truncated value (400 -> 144), on x86 and on ARM alike for positive
    # values. Negative overflow is platform-dependent (x86 wraps, ARM gives 0),
    # so no fixture has one.
    ("default_wraps", [{"name": "DefaultPostprocessor", "bias": 1, "multiplier": 200}]),
    ("threshold_default", [{"name": "ThresholdPostprocessor"}]),
    ("threshold_string", [{"name": "ThresholdPostprocessor", "threshold": "0.3"}]),
    ("threshold_zero", [{"name": "ThresholdPostprocessor", "threshold": 0}]),
    ("channels_0", [{"name": "ChannelSelection", "channels": "0"}]),
    ("channels_0_2", [{"name": "ChannelSelection", "channels": "0,2"}]),
    ("channels_2_0", [{"name": "ChannelSelection", "channels": "2,0"}]),
    ("lambda_x2", [{"name": "LambdaPostprocessor", "expression": "x*2"}]),
    ("lambda_1_minus_x", [{"name": "LambdaPostprocessor", "expression": "1-x"}]),
    ("lambda_parenthesised", [{"name": "LambdaPostprocessor", "expression": "(x+1)/2"}]),
    ("chain_sigmoid_threshold", [{"name": "SigmoidPostprocessor"}, {"name": "ThresholdPostprocessor"}]),
    ("chain_sigmoid_default", [{"name": "SigmoidPostprocessor"}, {"name": "DefaultPostprocessor"}]),
    # is_segmentation is the last step's that says: Threshold says True, ChannelSelection says nothing.
    ("chain_threshold_channels", [{"name": "ThresholdPostprocessor"}, {"name": "ChannelSelection", "channels": "0"}]),
    # ... and Default says False, overriding Threshold's True.
    ("chain_threshold_default", [{"name": "ThresholdPostprocessor"}, {"name": "DefaultPostprocessor"}]),
    ("chain_channels_sigmoid", [{"name": "ChannelSelection", "channels": "0,2"}, {"name": "SigmoidPostprocessor"}]),
]
POST_MODEL_DTYPE = "float32"
POST_MODEL_CHANNELS = 3


def post_input(seed: int) -> np.ndarray:
    """A [3, 2, 3, 4] float32 block of logits in about -6..6, with the boundary values in it."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(-6, 6, size=(3, 2, 3, 4)).astype(np.float32)
    # Exactly 0 (sigmoid 0.5, the default threshold's boundary: `>` is false),
    # the default threshold itself, the clip bounds, and the extremes.
    data[0, 0, 0, :4] = [0.0, 0.5, -1.0, 1.0]
    data[0, 0, 1, :2] = [6.0, -6.0]
    return data


def _margin_ok(values: np.ndarray, targets: np.ndarray, margin: float) -> bool:
    """No value within ``margin`` of its target unless it is exactly on it.

    Exact hits (a clipped value, an input placed on the boundary) compare
    the same in float32 and float64; near misses may not.
    """
    distance = np.abs(values.astype(np.float64) - targets)
    return bool(np.all((distance == 0) | (distance > margin)))


def _safe_for_truncation(step, entering: np.ndarray) -> bool:
    """Whether ``step``'s result on ``entering`` is robust to float32/float64 differences."""
    if isinstance(step, DefaultPostprocessor):
        value = (np.clip(entering.astype(np.float64), step.clip_min, step.clip_max) + step.bias) * step.multiplier
        return _margin_ok(value, np.round(value), TRUNCATION_MARGIN)
    if isinstance(step, ThresholdPostprocessor):
        return _margin_ok(entering, np.full(entering.shape, step.threshold), THRESHOLD_MARGIN)
    return True


def run_post_chain(steps: list, data: np.ndarray):
    """``(output, safe)``: the chain applied as the server applies it, and
    whether every truncating step saw values clear of its boundaries."""
    processors = PipelineSpec((), steps).build()[1]
    assert len(processors) == len(steps), "a step was not built"
    out = data
    safe = True
    for processor in processors:
        safe = safe and _safe_for_truncation(processor, out)
        # The server passes the chunk's corner and voxel count for the steps
        # that key ids on them; these steps ignore them (SerializableInterface.process).
        out = processor(out, chunk_corner=(0, 0, 0), chunk_num_voxels=int(np.prod(data.shape[1:])))
    return out, safe, processors


def make_post() -> dict:
    # The first seed whose values keep clear of every truncation boundary.
    for seed in range(100):
        data = post_input(seed)
        results = [(case_id, steps, *run_post_chain(steps, data)) for case_id, steps in POST_CASES]
        if all(safe for _, _, _, safe, _ in results):
            break
    else:
        raise RuntimeError("no seed under 100 keeps the post fixture clear of truncation boundaries")
    cases = []
    for case_id, steps, out, _, processors in results:
        is_segmentation = chain_is_segmentation(processors)
        cases.append(
            {
                "id": case_id,
                "steps": steps,
                "output": array_record(out),
                "chain_output_dtype": dtype_name(chain_output_dtype(processors, POST_MODEL_DTYPE)),
                "chain_num_channels": chain_num_channels(processors, POST_MODEL_CHANNELS),
                "chain_is_segmentation": is_segmentation,
            }
        )
    return {
        "_format": (
            "Postprocessors, from cellmap_flow/post/postprocessors.py, applied channel-first "
            "[C, z, y, x] as the server applies them to a model's output. `input` is one float32 "
            "block of logits; each case applies its `steps` in order and records the Python result "
            "in `output` (dtype included: uint8 steps truncate toward zero and wrap), plus "
            "pipeline_spec.chain_output_dtype(steps, model_dtype), chain_num_channels(steps, "
            "model_channels) and chain_is_segmentation(steps) (true/false/null = does not say). "
            "Arrays are {dtype, shape, data} with data flat in C order. Generated by "
            "make_fixtures.py; do not edit."
        ),
        "seed": seed,
        "model_dtype": POST_MODEL_DTYPE,
        "model_channels": POST_MODEL_CHANNELS,
        "input": array_record(data),
        "cases": cases,
    }


# --- args.json ----------------------------------------------------------------

ARGS_CASES = [
    # (id, input_norm, postprocess, known_difference). `known_difference`
    # names a way the Python and JavaScript JSON serializers differ on this
    # input, so the parity test records it instead of failing: both sides
    # still decode each other's blob to the same chain.
    ("empty", [], [], None),
    (
        "default_browser",
        [
            {"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255},
            {"name": "LambdaNormalizer", "expression": "2*x-1"},
        ],
        [{"name": "SigmoidPostprocessor"}],
        None,
    ),
    (
        "dashboard_strings",
        [{"name": "MinMaxNormalizer", "min_value": "0", "max_value": "255"}],
        [{"name": "ThresholdPostprocessor", "threshold": "0.5"}],
        None,
    ),
    (
        "two_lambdas",
        [],
        [
            {"name": "LambdaPostprocessor", "expression": "x*2"},
            {"name": "LambdaPostprocessor", "expression": "x-1"},
        ],
        None,
    ),
    (
        "legacy_dict",
        {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}},
        {"DefaultPostprocessor": {"clip_min": -1, "clip_max": 1}},
        None,
    ),
    (
        "floats_fractional",
        [{"name": "MinMaxNormalizer", "min_value": 0.5, "max_value": 254.5}],
        [{"name": "DefaultPostprocessor", "bias": 1.5, "multiplier": 127.5}],
        None,
    ),
    (
        "floats_integral",
        [{"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0}],
        [],
        "Python's json renders an integral float as 0.0; JavaScript's JSON.stringify renders 0",
    ),
    (
        "bools",
        [{"name": "MinMaxNormalizer", "invert": True}, {"name": "MinMaxNormalizer", "invert": False}],
        [],
        None,
    ),
    (
        "non_ascii",
        [{"name": "LambdaNormalizer", "expression": "x/255  # µm"}],
        [],
        "Python's json escapes non-ASCII as \\uXXXX (ensure_ascii); JSON.stringify writes UTF-8",
    ),
]


def make_args() -> dict:
    cases = []
    for case_id, input_norm, postprocess, known_difference in ARGS_CASES:
        spec = PipelineSpec(input_norm, postprocess)
        digest = spec.digest()
        extras = {"dashboard_url": DASHBOARD_URL, "digest": digest}
        blob = spec.to_url_blob(**extras)
        cases.append(
            {
                "id": case_id,
                # As given to PipelineSpec: a list of steps, or the legacy
                # {Name: {params}} dict, which it normalizes to the list.
                "given": {"input_norm": input_norm, "postprocess": postprocess},
                "chain": spec.to_json_data(),
                "extras": extras,
                "blob": blob,
                "blob_without_extras": spec.to_url_blob(),
                "digest": digest,
                # What the digest hashes: sorted keys, minified (pipeline_spec.PipelineSpec.digest).
                "digest_text": json.dumps(spec.to_json_data(), sort_keys=True, separators=(",", ":"), default=str),
                "layer_url": prediction_url(HOST, MODEL, blob, engine="chunkmirage"),
                "known_difference": known_difference,
            }
        )
    return {
        "_format": (
            "The layer URL's args blob, from cellmap_flow/pipeline_spec.py and serving/protocol.py. "
            "Each case gives the chain as PipelineSpec received it (`given`) and as it normalizes it "
            "(`chain`, PipelineSpec.to_json_data()), the blob of the chain plus `extras` "
            "(PipelineSpec.to_url_blob(**extras): minified JSON, input_norm then postprocess then the "
            "extras in order, URL-safe base64 without padding), the blob without extras, "
            "PipelineSpec.digest() with the text it hashes, and the layer URL "
            "viewer.layers.prediction_url(host, model, blob, engine='chunkmirage'). "
            "`known_difference`, when set, says why the JavaScript blob/digest will not match this "
            "one byte for byte. Generated by make_fixtures.py; do not edit."
        ),
        "args_key": ARGS_KEY,
        "dashboard_url": DASHBOARD_URL,
        "host": HOST,
        "model": MODEL,
        "cases": cases,
    }


# --- affinity.json ------------------------------------------------------------

SHORT_NEIGHBORHOOD = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
# The class's own default literal (nine offsets, several lines, trailing
# commas), passed explicitly in one case so the port's literal parser sees it.
DEFAULT_NEIGHBORHOOD_LITERAL = inspect.signature(AffinityPostprocessor).parameters["neighborhood"].default
# A chunk off the origin, so the Morton increment is exercised:
# pymorton.interleave(2, 1, 3) == 46.
AFFINITY_CORNER = (2, 1, 3)
AFFINITY_CASES = [
    # (id, input, step parameters). Without `neighborhood` the step uses its
    # default (0.5); `bias` as a number and as the form's string.
    ("short_bias_default", "logits_3", {"neighborhood": SHORT_NEIGHBORHOOD}),
    ("short_bias_0.3", "logits_3", {"neighborhood": SHORT_NEIGHBORHOOD, "bias": 0.3}),
    ("short_bias_string", "logits_3", {"neighborhood": SHORT_NEIGHBORHOOD, "bias": "0.3"}),
    # The default neighborhood needs a block of 10 or more voxels along each
    # axis: mwatershed panics on an offset (9) past an axis's length.
    ("default_bias_default", "logits_9", {}),
    ("default_literal_bias_0.3", "logits_9", {"neighborhood": DEFAULT_NEIGHBORHOOD_LITERAL, "bias": 0.3}),
    # Integer input (DefaultPostprocessor's 0..255) is scaled by 1/255; three
    # offsets for two channels, cut to the channels.
    ("uint8_bias_0.3", "bytes_2", {"neighborhood": "[[0, 0, 1], [0, 1, 0], [1, 0, 0]]", "bias": 0.3}),
]
# How far an edge's |affinity - bias| must sit from every other's (the
# mutex watershed takes the edges in that order, and mwatershed breaks a tie
# as its unstable sort happens to) and from 0 (mwatershed splits there, the
# port does nothing; the partition is the same, but keep the fixtures clear
# of it), and a fragment's mean affinity from the bias (the port sums in
# the same order as numpy, but a different order would still be right).
EDGE_MARGIN = 1e-9
MEAN_MARGIN = 1e-6


def affinity_inputs(seed: int) -> dict:
    """The blocks the affinity cases segment, by name.

    ``logits_3``: a [3, 6, 7, 8] float32 block of values in [-3, 3], one
    channel per unit offset. ``logits_9``: [9, 10, 10, 10], one channel per
    default offset, the smallest block mwatershed takes with an offset of 9.
    ``bytes_2``: a [2, 2, 3, 4] uint8 block of distinct byte values, as
    DefaultPostprocessor's output would be, small enough for its bytes to be
    distinct (equal bytes would tie).
    """
    rng = np.random.default_rng(seed)
    logits_3 = rng.uniform(-3, 3, size=(3, 6, 7, 8)).astype(np.float32)
    logits_9 = rng.uniform(-3, 3, size=(9, 10, 10, 10)).astype(np.float32)
    byte_values = rng.permutation(256)[: 2 * 2 * 3 * 4].astype(np.uint8).reshape(2, 2, 3, 4)
    return {"logits_3": logits_3, "logits_9": logits_9, "bytes_2": byte_values}


def affinity_edge_weights(affinities: np.ndarray, offsets: list, bias: float) -> np.ndarray:
    """|affinity - bias| of every edge mwatershed builds: per channel, the
    voxels whose partner at the channel's offset lies inside the block."""
    weights = []
    for channel, offset in enumerate(offsets[: affinities.shape[0]]):
        index = tuple(slice(max(0, -o), min(n, n - o)) for o, n in zip(offset, affinities.shape[1:]))
        weights.append(np.abs(affinities[channel][index].astype(np.float64) - bias).ravel())
    return np.concatenate(weights)


def affinity_case_is_safe(affinities: np.ndarray, offsets: list, bias: float) -> bool:
    """No tie or zero among the edge weights, no fragment mean near the bias."""
    import mwatershed

    weights = np.sort(affinity_edge_weights(affinities, offsets, bias))
    if weights.size and (weights[0] <= EDGE_MARGIN or np.min(np.diff(weights)) <= EDGE_MARGIN):
        return False
    scaled = affinities.astype(np.float64)
    segmentation = mwatershed.agglom(scaled - bias, [list(o) for o in offsets[: affinities.shape[0]]])
    mean_affinity = np.mean(scaled, axis=0)
    for fragment in np.unique(segmentation[segmentation > 0]):
        if abs(float(mean_affinity[segmentation == fragment].mean()) - bias) <= MEAN_MARGIN:
            return False
    return True


def run_affinity_case(steps: list, data: np.ndarray):
    """``(processor, output, segmentation, safe)``: the step built as the
    server builds it and applied to ``data`` at AFFINITY_CORNER, plus
    segment.mutex_watershed's labels alone (before the Morton increment)."""
    processor = PipelineSpec((), steps).build()[1][0]
    affinities = data / 255.0 if np.issubdtype(data.dtype, np.integer) else data
    safe = affinity_case_is_safe(affinities, processor.neighborhood, processor.bias)
    segmentation = segment.mutex_watershed(affinities, processor.neighborhood, processor.bias)
    chunk_num_voxels = int(np.prod(data.shape[1:]))
    output = processor(data, chunk_corner=AFFINITY_CORNER, chunk_num_voxels=chunk_num_voxels)
    return processor, output, segmentation, safe


def make_affinity() -> dict:
    # The first seed whose values keep every case clear of ties and margins.
    for seed in range(200):
        inputs = affinity_inputs(seed)
        results = [(case_id, name, params, *run_affinity_case([{"name": "AffinityPostprocessor", **params}], inputs[name])) for case_id, name, params in AFFINITY_CASES]
        if all(safe for *_, safe in results):
            break
    else:
        raise RuntimeError("no seed under 200 keeps the affinity fixture clear of ties and margins")
    cases = []
    for case_id, name, params, processor, output, segmentation, _ in results:
        steps = [{"name": "AffinityPostprocessor", **params}]
        assert output.dtype == np.uint64 and output.shape == (1, *inputs[name].shape[1:]), case_id
        assert segmentation.max() > 0 and (segmentation == 0).any(), f"{case_id}: objects and background both expected"
        cases.append(
            {
                "id": case_id,
                "steps": steps,
                "input": name,
                "bias": processor.bias,
                "neighborhood": processor.neighborhood,
                "chunk_corner": list(AFFINITY_CORNER),
                "chunk_num_voxels": int(np.prod(inputs[name].shape[1:])),
                "morton": pymorton_interleave(AFFINITY_CORNER),
                "segmentation": uint64_record(segmentation),
                "output": uint64_record(output),
                "chain_output_dtype": dtype_name(chain_output_dtype([processor], POST_MODEL_DTYPE)),
                "chain_num_channels": chain_num_channels([processor], inputs[name].shape[0]),
                "chain_is_segmentation": chain_is_segmentation([processor]),
            }
        )
    return {
        "_format": (
            "AffinityPostprocessor, from cellmap_flow/post/postprocessors.py, and segment.mutex_watershed "
            "from cellmap_flow/post/segment.py. `inputs` are blocks [C, z, y, x] by name (a float block as "
            "{dtype, shape, base64}: its little-endian C-order bytes); each case applies "
            "its one step (built by PipelineSpec) to the named input as the server does, with chunk_corner "
            "and chunk_num_voxels, and records `output` (uint64 [1, z, y, x], ids offset by chunk_num_voxels "
            "* pymorton.interleave(*chunk_corner) = `morton`), and `segmentation`: segment.mutex_watershed "
            "(affinities, neighborhood, bias) alone, uint64 [z, y, x], labels 1..n by first appearance, 0 "
            "background, before the offset (integer input scaled by 1/255 first, as the step does). `bias` "
            "and `neighborhood` are the step's parsed parameters. uint64 values are decimal strings. The "
            "seed keeps every edge weight distinct and nonzero and every fragment mean away from the bias, "
            "so the result depends on no tie. Generated by make_fixtures.py; do not edit."
        ),
        "seed": seed,
        # The float blocks packed (a thousand voxels times nine channels), the byte block listed.
        "inputs": {name: packed_record(block) if block.dtype == np.float32 else array_record(block) for name, block in inputs.items()},
        "cases": cases,
    }


# --- morton.json --------------------------------------------------------------

# Corners whose pymorton values the unit tests check, including ones past 10
# bits: pymorton's interleave keeps 10 bits per coordinate (its first mask),
# so 1024 is 0 and (1023, 1023, 1023) is 2^30 - 1.
MORTON_CORNERS = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (5, 0, 7), (2, 1, 3), (7, 7, 7),
    (100, 200, 300), (1023, 0, 0), (1024, 0, 0), (1023, 1023, 1023), (1025, 1, 2),
]
MORTON_CASES = [
    # (id, input, step parameters, chunk_corner, chunk_num_voxels)
    ("origin", "labels_uint8", {}, (0, 0, 0), 1000),
    ("corner_1_2_3", "labels_uint8", {}, (1, 2, 3), 1000),
    ("corner_5_0_7_uint32", "labels_uint32", {}, (5, 0, 7), 4096),
    # The channel from the form's string; the other channel is cast but not moved.
    ("channel_1_string", "labels_uint8", {"channel": "1"}, (1, 2, 3), 1000),
    # Float ids truncate toward zero (astype(uint64)) before the test for nonzero.
    ("float_input", "labels_float32", {}, (1, 2, 3), 1000),
]


def pymorton_interleave(corner) -> int:
    import pymorton

    return int(pymorton.interleave(*corner))


def morton_inputs(seed: int) -> dict:
    """Small [2, 2, 3, 4] label blocks with zeros in them: uint8 ids 0..4,
    uint32 ids up to 4e9 (past 2^31), and float32 values 0, 1, 2.7, 300.2."""
    rng = np.random.default_rng(seed)
    uint8_labels = rng.integers(0, 5, size=(2, 2, 3, 4), dtype=np.uint8)
    uint32_labels = rng.choice(np.array([0, 1, 7, 123456, 4_000_000_000], dtype=np.uint32), size=(2, 2, 3, 4))
    float_labels = rng.choice(np.array([0.0, 1.0, 2.7, 300.2], dtype=np.float32), size=(2, 2, 3, 4))
    return {"labels_uint8": uint8_labels, "labels_uint32": uint32_labels, "labels_float32": float_labels}


def make_morton() -> dict:
    seed = 0
    inputs = morton_inputs(seed)
    cases = []
    for case_id, name, params, corner, chunk_num_voxels in MORTON_CASES:
        steps = [{"name": "MortonSegmentationRelabeling", **params}]
        processor = PipelineSpec((), steps).build()[1][0]
        output = processor(inputs[name], chunk_corner=corner, chunk_num_voxels=chunk_num_voxels)
        assert output.dtype == np.uint64 and output.shape == inputs[name].shape, case_id
        cases.append(
            {
                "id": case_id,
                "steps": steps,
                "input": name,
                "chunk_corner": list(corner),
                "chunk_num_voxels": chunk_num_voxels,
                "morton": pymorton_interleave(corner),
                "output": uint64_record(output),
                "chain_output_dtype": dtype_name(chain_output_dtype([processor], POST_MODEL_DTYPE)),
                "chain_num_channels": chain_num_channels([processor], inputs[name].shape[0]),
                "chain_is_segmentation": chain_is_segmentation([processor]),
            }
        )
    return {
        "_format": (
            "MortonSegmentationRelabeling, from cellmap_flow/post/postprocessors.py. `inputs` are label blocks "
            "[C, z, y, x] by name; each case applies its one step (built by PipelineSpec) to the named input "
            "with chunk_corner and chunk_num_voxels, as the server does, and records `output` (uint64, same "
            "shape: the chosen channel's nonzero ids plus chunk_num_voxels * pymorton.interleave(*chunk_corner) "
            "= `morton`). `interleave` lists pymorton.interleave(z, y, x) for the corners in `corner`. uint64 "
            "values are decimal strings. Generated by make_fixtures.py; do not edit."
        ),
        "seed": seed,
        "inputs": {name: array_record(block) for name, block in inputs.items()},
        "interleave": [{"corner": list(corner), "value": pymorton_interleave(corner)} for corner in MORTON_CORNERS],
        "cases": cases,
    }


# --- writing and checking -----------------------------------------------------

MAKERS = {"norms.json": make_norms, "post.json": make_post, "args.json": make_args, "affinity.json": make_affinity, "morton.json": make_morton}


def write_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, make in MAKERS.items():
        text = json.dumps(make(), indent=2, ensure_ascii=False) + "\n"
        (out_dir / name).write_text(text, encoding="utf-8")
        size = len(text.encode("utf-8"))
        assert size < 200_000, f"{name} is {size} bytes; keep fixtures small enough to commit"
        print(f"wrote {out_dir / name} ({size} bytes)")


def differences(path: str, committed, fresh, out: list) -> None:
    """Append a line per difference between two loaded fixtures.

    Floats compare with a tolerance (float32 transcendentals may differ by an
    ulp between machines; a stale fixture differs by far more); everything
    else exactly, including the types and the key order of dicts.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        if list(committed) != list(fresh):
            out.append(f"{path}: keys {list(committed)} != {list(fresh)}")
            return
        for key in committed:
            differences(f"{path}.{key}", committed[key], fresh[key], out)
    elif isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            out.append(f"{path}: {len(committed)} items != {len(fresh)}")
            return
        for i, (a, b) in enumerate(zip(committed, fresh)):
            differences(f"{path}[{i}]", a, b, out)
    elif isinstance(committed, float) and isinstance(fresh, float):
        if not math.isclose(committed, fresh, rel_tol=1e-6, abs_tol=1e-6):
            out.append(f"{path}: {committed!r} != {fresh!r}")
    elif type(committed) is not type(fresh) or committed != fresh:
        out.append(f"{path}: {committed!r} != {fresh!r}")


def check_fixtures(committed_dir: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fresh_dir = Path(tmp)
        write_fixtures(fresh_dir)
        problems: list[str] = []
        for name in FIXTURES:
            committed_path = committed_dir / name
            if not committed_path.exists():
                problems.append(f"{name}: missing")
                continue
            committed = json.loads(committed_path.read_text(encoding="utf-8"))
            fresh = json.loads((fresh_dir / name).read_text(encoding="utf-8"))
            differences(name, committed, fresh, problems)
    if problems:
        print(f"{len(problems)} difference(s) between the committed fixtures and the Python code:")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(f"Regenerate with: python {Path(__file__).relative_to(ROOT)}")
        return 1
    print("fixtures are up to date")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE, help="directory to write into (default: beside this script)")
    parser.add_argument("--check", action="store_true", help="regenerate to a temporary directory and compare; exit 1 if stale")
    args = parser.parse_args(argv)
    if args.check:
        return check_fixtures(args.out)
    write_fixtures(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
