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

    if looks_like_affinities(out_channels, model_name, channels_names):
        if "AffinityPostprocessor" in names:
            if not has_default:
                return {
                    "level": "warn",
                    "message": (
                        "AffinityPostprocessor divides its input by 255, so it "
                        "needs DefaultPostprocessor ahead of it to rescale "
                        "[0,1] to 0-255. Without that the affinities are ~250x "
                        "too small and the watershed collapses to one segment."
                    ),
                    "suggest": suggest_affinity_chain(output_class),
                }
            return {"level": "ok", "message": "Affinity chain looks complete.", "suggest": []}
        return {
            "level": "suggest",
            "message": (
                f"This model has {out_channels} output channels and an affinity "
                "name, so it probably predicts affinities. The chain below "
                "converts them to a segmentation."
            ),
            "suggest": suggest_affinity_chain(output_class),
        }

    if output_class == UNIT:
        if has_sigmoid:
            return {
                "level": "warn",
                "message": (
                    "This model's output is already bounded to [0, 1], so it "
                    "ends in a sigmoid. SigmoidPostprocessor would apply a "
                    "second one, flattening the contrast."
                ),
                "suggest": [n for n in names if n != "SigmoidPostprocessor"],
            }
        return {
            "level": "ok",
            "message": "Model output is already in [0, 1]; no activation needed.",
            "suggest": [],
        }

    if output_class == SIGNED_UNIT:
        if has_sigmoid:
            return {
                "level": "warn",
                "message": (
                    "This model's output is bounded to [-1, 1] (a tanh head). "
                    "A sigmoid on top compresses it to roughly [0.27, 0.73], "
                    "which looks washed out. DefaultPostprocessor is the usual "
                    "choice for this range."
                ),
                "suggest": ["DefaultPostprocessor"],
            }
        if not has_default:
            return {
                "level": "suggest",
                "message": (
                    "Model output is in [-1, 1] (a tanh head). "
                    "DefaultPostprocessor rescales that to 0-255 for display."
                ),
                "suggest": ["DefaultPostprocessor"],
            }
        return {"level": "ok", "message": "DefaultPostprocessor suits this [-1, 1] output.", "suggest": []}

    # UNBOUNDED: logits and signed distances are indistinguishable here, so
    # suggest only when nothing at all is configured, and say why it's a guess.
    if not names:
        return {
            "level": "suggest",
            "message": (
                "Model output is unbounded, so it has no activation. If these "
                "are logits, SigmoidPostprocessor converts them to "
                "probabilities. If they are distances or affinities, leave this "
                "alone -- the output range cannot tell the two apart."
            ),
            "suggest": ["SigmoidPostprocessor"],
        }
    return {"level": "ok", "message": "Model output is unbounded; current postprocessing left as set.", "suggest": []}


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


def suggest_input_norm(
    framework=None, raw_dtype="uint8", declared=None, source=None
) -> dict:
    """Propose an input_norm config for a model.

    Priority:
      1. ``declared`` -- an explicit input_norm in the model's own metadata.
         Nothing writes this today, but honour it when it appears.
      2. The training framework's convention. DaCapo trains on [-1, 1], hence
         the extra ``x*2-1`` on top of the 0-1 rescale.
      3. ``source`` -- where the model came from. A CellMap model follows the
         same [-1, 1] convention whatever its metadata calls the framework.
      4. The raw dtype alone, giving [0, 1].

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

    try:
        import numpy as np

        info = np.iinfo(np.dtype(raw_dtype))
        lo, hi = float(info.min), float(info.max)
    except Exception:
        lo, hi = 0.0, 255.0

    norm = {
        "MinMaxNormalizer": {
            "name": "MinMaxNormalizer",
            "min_value": lo,
            "max_value": hi,
            "invert": False,
        }
    }

    fw = str(framework or "").lower()
    if any(tok in fw for tok in DACAPO_FRAMEWORKS):
        norm["LambdaNormalizer"] = {
            "name": "LambdaNormalizer",
            "expression": "x*2-1",
        }
        return {
            "input_norm": norm,
            "order": list(norm.keys()),
            "reason": (
                f"framework is '{framework}'; DaCapo models are trained on "
                f"inputs in [-1, 1], so {raw_dtype} is rescaled to [0,1] then "
                "shifted with x*2-1"
            ),
            "confidence": "medium",
        }

    if _is_cellmap_model(source):
        norm["LambdaNormalizer"] = {
            "name": "LambdaNormalizer",
            "expression": "x*2-1",
        }
        return {
            "input_norm": norm,
            "order": list(norm.keys()),
            "reason": (
                "this is a CellMap model; the collection is trained on inputs "
                f"in [-1, 1], so {raw_dtype} is rescaled to [0,1] then shifted "
                "with x*2-1"
            ),
            "confidence": "medium",
        }

    return {
        "input_norm": norm,
        "order": list(norm.keys()),
        "reason": (
            f"no framework declared; defaulting to a plain {raw_dtype} rescale "
            "to [0, 1]. Check this against how the model was trained."
        ),
        "confidence": "low",
    }
