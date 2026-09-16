"""Infer what postprocessing a model's raw output needs, from the output itself.

A model's output activation is observable: feed it extreme inputs and see
whether the result stays bounded. That tells us what activation is *already*
baked into the weights, which is enough to catch the common misconfigurations
(a SigmoidPostprocessor stacked on a model that already ends in a sigmoid, or a
tanh-range model served without the rescale it needs).

Deliberately does NOT try to decide what unbounded output *means*. Raw logits
and signed-distance predictions are both unbounded and want opposite treatment,
and nothing in the tensor distinguishes them -- see ``UNBOUNDED`` below.
"""

import logging

logger = logging.getLogger(__name__)

UNIT = "unit"  # [0, 1] -- sigmoid/softmax already applied
SIGNED_UNIT = "signed_unit"  # [-1, 1] -- tanh already applied
UNBOUNDED = "unbounded"  # no output activation: logits, distances, affinities...


def classify_output_range(lo: float, hi: float) -> str:
    """Bucket an observed (min, max) output range into an activation class."""
    # Small tolerance: a saturating activation can land a hair outside its
    # asymptote in float32.
    eps = 1e-4
    if lo >= -eps and hi <= 1.0 + eps:
        return UNIT
    if lo >= -1.0 - eps and hi <= 1.0 + eps:
        return SIGNED_UNIT
    return UNBOUNDED


def looks_like_affinities(out_channels=None, model_name="", channels_names=None) -> bool:
    """Heuristic: does this model emit affinities rather than a single map?

    Affinity models have one output channel per neighborhood offset (3 for
    nearest-neighbour, 9 with longer range), so a channel count of 3+ is the
    structural tell. Require a naming hint as well, since a multi-class
    semantic model also has several channels and wants entirely different
    handling.
    """
    if not out_channels or int(out_channels) < 3:
        return False
    haystack = " ".join(
        [str(model_name or "")] + [str(c) for c in (channels_names or [])]
    ).lower()
    return any(tok in haystack for tok in ("aff", "affinit"))


def suggest_affinity_chain(output_class: str) -> list:
    """The postprocessing an affinity model needs, in order.

    AffinityPostprocessor divides its input by 255 -- it is written to sit
    downstream of DefaultPostprocessor, which maps [-1,1] to uint8 0-255. Feed
    it probabilities in [0,1] directly and the affinities come out ~250x too
    small, so every edge reads as weakly attractive and the mutex watershed
    degenerates. The rescale step is therefore mandatory, not cosmetic.
    """
    chain = []
    if output_class == UNBOUNDED:
        chain.append("SigmoidPostprocessor")  # logits -> probabilities
    # [0,1] -> 0-255, the range AffinityPostprocessor expects
    chain.append("DefaultPostprocessor")
    chain.append("AffinityPostprocessor")
    return chain


# Offsets for the mutex watershed, longest-range last: AffinityPostprocessor
# truncates to the model's channel count, so the order decides which offsets a
# 3-channel model actually gets.
NEIGHBORHOOD_OFFSETS = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [3, 0, 0], [0, 3, 0], [0, 0, 3],
    [9, 0, 0], [0, 9, 0], [0, 0, 9],
]

# What each activation class means as a value range. UNBOUNDED stays None:
# logits and distances share it and imply different parameters.
_CLASS_RANGE = {UNIT: (0.0, 1.0), SIGNED_UNIT: (-1.0, 1.0)}


def suggest_postprocess_params(chain, output_class, out_channels=None) -> dict:
    """Parameter values for a suggested chain, from the range flowing into each step.

    The default parameters assume each step sees the range its most common
    predecessor produces, which stops being true as soon as the chain differs.
    The costly case is DefaultPostprocessor after a sigmoid: its defaults clip
    to [-1, 1] and map that to 0-255, so probabilities in [0, 1] land in
    [127.5, 255] -- half the range. AffinityPostprocessor then divides by 255
    and subtracts its bias, and a probability of 0.0003 (a strongly repulsive
    edge) comes out at -0.002 instead of -0.5. The mutex watershed is left with
    essentially no repulsive edges and merges everything.

    Returns ``{postprocessor_name: {param: value}}``, containing only the steps
    whose defaults are wrong for this chain.
    """
    params = {}
    rng = _CLASS_RANGE.get(output_class)  # None while the range is unknown

    for name in chain:
        if name == "SigmoidPostprocessor":
            rng = (0.0, 1.0)
        elif name == "DefaultPostprocessor":
            if rng is not None:
                lo, hi = rng
                params[name] = {
                    "clip_min": lo,
                    "clip_max": hi,
                    "bias": -lo + 0.0,  # avoid rendering "-0.0" in the form
                    "multiplier": 255.0 / (hi - lo),
                }
            rng = (0.0, 255.0)
        elif name == "ThresholdPostprocessor":
            # Unbounded means logits here: the decision boundary is 0, not the
            # 0.5 that only makes sense once a sigmoid has been applied.
            threshold = 0.0 if rng is None else (rng[0] + rng[1]) / 2.0
            params[name] = {"threshold": threshold}
            rng = (0.0, 1.0)
        elif name == "AffinityPostprocessor":
            # It divides by 255 internally, so it wants 0-255 in and works in
            # [0, 1]; the mutex watershed needs the midpoint subtracted to get
            # the signed affinities it is defined on.
            affinity = {"bias": 0.5}
            if out_channels:
                offsets = NEIGHBORHOOD_OFFSETS[: int(out_channels)]
                affinity["neighborhood"] = str(offsets)
            params[name] = affinity
            rng = None  # labels from here on

    return params


def review_postprocess(
    output_class: str,
    postprocess_names,
    out_channels=None,
    model_name="",
    channels_names=None,
) -> dict:
    """Compare an observed output class against the configured postprocessors.

    Returns ``{"level", "message", "suggest"}`` where ``level`` is one of
    "ok" / "suggest" / "warn", and ``suggest`` is a list of postprocessor names
    to apply (empty when the current configuration is already sensible).

    ``level == "warn"`` means the current chain is actively wrong, not merely
    unconventional.
    """
    names = list(postprocess_names or [])
    has_sigmoid = "SigmoidPostprocessor" in names
    has_default = "DefaultPostprocessor" in names

    def verdict(level, message, suggest):
        """Attach parameter values to whatever chain we are proposing."""
        return {
            "level": level,
            "message": message,
            "suggest": suggest,
            "params": suggest_postprocess_params(suggest, output_class, out_channels),
        }

    if looks_like_affinities(out_channels, model_name, channels_names):
        if "AffinityPostprocessor" in names:
            if not has_default:
                return verdict(
                    "warn",
                    (
                        "AffinityPostprocessor divides its input by 255, so it "
                        "needs DefaultPostprocessor ahead of it to rescale "
                        "[0,1] to 0-255. Without that the affinities are ~250x "
                        "too small and the watershed collapses to one segment."
                    ),
                    suggest_affinity_chain(output_class),
                )
            return verdict("ok", "Affinity chain looks complete.", [])
        return verdict(
            "suggest",
            (
                f"This model has {out_channels} output channels and an affinity "
                "name, so it probably predicts affinities. The chain below "
                "converts them to a segmentation."
            ),
            suggest_affinity_chain(output_class),
        )

    if output_class == UNIT:
        if has_sigmoid:
            return verdict(
                "warn",
                (
                    "This model's output is already bounded to [0, 1], so it "
                    "ends in a sigmoid. SigmoidPostprocessor would apply a "
                    "second one, flattening the contrast."
                ),
                [n for n in names if n != "SigmoidPostprocessor"],
            )
        return verdict(
            "ok",
            "Model output is already in [0, 1]; no activation needed.",
            [],
        )

    if output_class == SIGNED_UNIT:
        if has_sigmoid:
            return verdict(
                "warn",
                (
                    "This model's output is bounded to [-1, 1] (a tanh head). "
                    "A sigmoid on top compresses it to roughly [0.27, 0.73], "
                    "which looks washed out. DefaultPostprocessor is the usual "
                    "choice for this range."
                ),
                ["DefaultPostprocessor"],
            )
        if not has_default:
            return verdict(
                "suggest",
                (
                    "Model output is in [-1, 1] (a tanh head). "
                    "DefaultPostprocessor rescales that to 0-255 for display."
                ),
                ["DefaultPostprocessor"],
            )
        return verdict("ok", "DefaultPostprocessor suits this [-1, 1] output.", [])

    # UNBOUNDED: logits and signed distances are indistinguishable here, so
    # suggest only when nothing at all is configured, and say why it's a guess.
    if not names:
        return verdict(
            "suggest",
            (
                "Model output is unbounded, so it has no activation. If these "
                "are logits, SigmoidPostprocessor converts them to "
                "probabilities. If they are distances or affinities, leave this "
                "alone -- the output range cannot tell the two apart."
            ),
            ["SigmoidPostprocessor"],
        )
    return verdict("ok", "Model output is unbounded; current postprocessing left as set.", [])


# --- input normalization -----------------------------------------------------
#
# Unlike the output side, the input scale a model expects leaves no signature in
# its weights -- it is a training-time convention. So this does not infer
# anything; it reads a declared convention and falls back to the raw dtype.

DACAPO_FRAMEWORKS = ("dacapo",)

# Models published by the CellMap project. Their metadata.json does not always
# say "dacapo" -- cellmap/mito-aff-unet-setup-16 reports a bare "torch" -- but
# the collection is trained on EM rescaled to [-1, 1]. The repo agrees with
# itself on this: the commented-out default in globals.py, every example yaml,
# and the training input_norm used for the finetuning runs all use
# MinMax(0,255) then x*2-1.
# The HF org prefix must be anchored: every model on this filesystem lives
# under /nrs/cellmap/models/, including other groups' (saalfeldlab's fly
# models), so a bare "cellmap/" substring matches far too much.
CELLMAP_HF_ORG = "cellmap/"
CELLMAP_MODEL_DIR = "/models/cellmap/"


def _is_cellmap_model(source) -> bool:
    src = str(source or "")
    return src.startswith(CELLMAP_HF_ORG) or CELLMAP_MODEL_DIR in src


def _to_unit_range(lo, hi) -> dict:
    """A MinMaxNormalizer mapping [lo, hi] onto [0, 1]."""
    return {
        "MinMaxNormalizer": {
            "name": "MinMaxNormalizer",
            "min_value": lo,
            "max_value": hi,
            "invert": False,
        }
    }


SHIFT_TO_SIGNED = {"name": "LambdaNormalizer", "expression": "x*2-1"}


def suggest_input_norm(
    framework=None, raw_dtype="uint8", declared=None, source=None, data_range=None
) -> dict:
    """Propose an input_norm config for a model.

    Priority:
      1. ``declared`` -- an explicit input_norm in the model's own metadata.
         Nothing writes this today, but honour it when it appears.
      2. The training framework's convention. DaCapo trains on [-1, 1], hence
         the extra ``x*2-1`` on top of the 0-1 rescale.
      3. ``source`` -- where the model came from. A CellMap model follows the
         same [-1, 1] convention whatever its metadata calls the framework.
      4. Nothing known about the model: assume [-1, 1] anyway, because it is
         what everything else in this repo trains on, and mark it low
         confidence so callers can present it as the assumption it is. Getting
         the scale wrong is not subtle -- the model sees inputs far outside its
         training range and predicts noise -- so a stated assumption beats
         silently feeding it raw uint8.

    ``data_range`` is the actual (min, max) of the raw data. Prefer it over
    ``raw_dtype``: uint8 data spans 0-255, but float data can span anything and
    its dtype says nothing useful.

    Returns ``{"input_norm": {...}, "order": [...], "reason", "confidence"}``.

    ``order`` is redundant with the dict's insertion order but not with what
    survives transport: Flask's jsonify sorts keys, which would silently turn
    MinMax-then-Lambda into Lambda-then-MinMax -- i.e. apply ``x*2-1`` to raw
    uint8 before the rescale. Callers must use ``order``.
    """
    if declared:
        return {
            "input_norm": declared,
            "order": list(declared.keys()),
            "reason": "declared in the model's own metadata",
            "confidence": "high",
        }

    if data_range:
        lo, hi = (float(v) for v in data_range)
        span = f"{raw_dtype or 'the raw data'} spans [{lo:.6g}, {hi:.6g}]"
    else:
        try:
            import numpy as np

            info = np.iinfo(np.dtype(raw_dtype))
            lo, hi = float(info.min), float(info.max)
            span = f"{raw_dtype} spans [{lo:.6g}, {hi:.6g}]"
        except Exception:
            lo, hi = 0.0, 255.0
            span = "assuming 8-bit data in [0, 255] (could not read the dtype)"

    norm = _to_unit_range(lo, hi)

    fw = str(framework or "").lower()
    if any(tok in fw for tok in DACAPO_FRAMEWORKS):
        norm["LambdaNormalizer"] = dict(SHIFT_TO_SIGNED)
        return {
            "input_norm": norm,
            "order": list(norm.keys()),
            "reason": (
                f"framework is '{framework}'; DaCapo models are trained on "
                f"inputs in [-1, 1], and {span}"
            ),
            "confidence": "medium",
        }

    if _is_cellmap_model(source):
        norm["LambdaNormalizer"] = dict(SHIFT_TO_SIGNED)
        return {
            "input_norm": norm,
            "order": list(norm.keys()),
            "reason": (
                "this is a CellMap model; the collection is trained on inputs "
                f"in [-1, 1], and {span}"
            ),
            "confidence": "medium",
        }

    norm["LambdaNormalizer"] = dict(SHIFT_TO_SIGNED)
    return {
        "input_norm": norm,
        "order": list(norm.keys()),
        "reason": (
            "nothing declares how this model was trained, so assuming the "
            f"[-1, 1] range everything else here uses; {span}"
        ),
        "confidence": "low",
    }
