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


def review_postprocess(output_class: str, postprocess_names) -> dict:
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
