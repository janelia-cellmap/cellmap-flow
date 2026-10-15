"""Generate the parity fixtures the TypeScript ports are tested against.

The browser mode re-implements three pieces of the Python package in
TypeScript: the input normalizers (``web/src/infer/norms.ts`` from
``cellmap_flow/norm/input_normalize.py``), the postprocessors
(``web/src/infer/post.ts`` from ``cellmap_flow/post/postprocessors.py``) and
the layer URL's args blob (``web/src/shared/args.ts`` from
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
from cellmap_flow.post.postprocessors import DefaultPostprocessor, ThresholdPostprocessor  # noqa: E402
from cellmap_flow.serving.protocol import ARGS_KEY  # noqa: E402
from cellmap_flow.viewer.layers import prediction_url  # noqa: E402

FIXTURES = ("norms.json", "post.json", "args.json")

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


# --- writing and checking -----------------------------------------------------

MAKERS = {"norms.json": make_norms, "post.json": make_post, "args.json": make_args}


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
