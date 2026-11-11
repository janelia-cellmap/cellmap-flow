"""The model advice's rule for what follows a model's output, written once:
the postprocessing a model's output wants, from what that output is (its
activation class, portable/label_it.py) and what its channels are, and
whether the chain configured gives it (``review_postprocess``); and what a
model's start sets on a chain with none (``start_postprocess``), the
advice's suggestion for that empty chain. The server's model advice
(dashboard/routes/model_advice.py, through serving/probe.py) and Submit
(dashboard/routes/pipeline.py set_declared_postprocess) call it in their
process, and the browser's Submit (web/src/backend/serving.ts) through
calls.py.
"""

from typing import Optional

from cellmap_flow.portable.label_it import BYTE_UNIT, SIGNED_UNIT, UNBOUNDED, UNIT


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
    # A string is one name, as models.geometry.channel_names_of reads it:
    # iterated, "affinities" became the letters "a f f i ...", with no "aff".
    if isinstance(channels_names, str):
        channels_names = [channels_names]
    haystack = " ".join(
        [str(model_name or "")] + [str(c) for c in (channels_names or [])]
    ).lower()
    return any(tok in haystack for tok in ("aff", "affinit"))


def suggest_affinity_chain(output_class: str) -> list:
    """The postprocessing an affinity model needs, in order.

    AffinityPostprocessor takes affinities in [0, 1], or the uint8 0-255 that
    DefaultPostprocessor produces (integer input is divided by 255). The chain
    suggested here goes through DefaultPostprocessor, the long-standing
    convention; a sigmoid feeding AffinityPostprocessor directly works too.
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
_CLASS_RANGE = {UNIT: (0.0, 1.0), BYTE_UNIT: (0.0, 255.0), SIGNED_UNIT: (-1.0, 1.0)}


def suggest_postprocess_params(chain, output_class, out_channels=None, bias=None, offsets=None) -> dict:
    """Parameter values for a suggested chain, from the range flowing into each step.

    The default parameters assume each step sees the range its most common
    predecessor produces, which stops being true as soon as the chain differs.
    The costly case is DefaultPostprocessor after a sigmoid: its defaults clip
    to [-1, 1] and map that to 0-255, so probabilities in [0, 1] land in
    [127.5, 255] -- half the range. AffinityPostprocessor then divides by 255
    and subtracts its bias, and a probability of 0.0003 (a strongly repulsive
    edge) comes out at -0.002 instead of -0.5. The mutex watershed is left with
    essentially no repulsive edges and merges everything.

    ``bias`` is the AffinityPostprocessor's: the cut-off the ranking fitted
    to the paint for the model (its row's threshold, a probability), or
    None for the midpoint, 0.5. ``offsets`` are its neighborhood when the
    model declares them (``ModelConfig.declared_affinity_offsets``), else
    NEIGHBORHOOD_OFFSETS cut to its channels.

    Returns ``{postprocessor_name: {param: value}}``, containing only the steps
    whose defaults are wrong for this chain, and the AffinityPostprocessor's
    bias and neighborhood always.
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
            # It works in [0, 1] (integer 0-255 input is divided by 255); the
            # mutex watershed needs a cut subtracted to get the signed
            # affinities it is defined on: the midpoint, or the cut-off the
            # ranking fitted, where its score called a painted pair joined,
            # so the objects are cut where the score read the model.
            affinity = {"bias": 0.5 if bias is None else float(bias)}
            if offsets:
                affinity["neighborhood"] = str([list(offset) for offset in offsets])
            elif out_channels:
                affinity["neighborhood"] = str(NEIGHBORHOOD_OFFSETS[: int(out_channels)])
            params[name] = affinity
            rng = None  # labels from here on
        elif name == "ArgmaxPostprocessor":
            # Background where a voxel's likeliest class is below a
            # probability of 0.5, in the output's units (127.5 of a
            # probability x 255); its best class's everywhere while the range
            # is not known.
            if rng is not None:
                params[name] = {"threshold": (rng[0] + rng[1]) / 2.0}
            rng = None  # labels from here on

    return params


def review_postprocess(
    output_class: Optional[str],
    postprocess_names,
    out_channels=None,
    model_name="",
    channels_names=None,
    evidence="",
    threshold=None,
    affinity_offsets=None,
) -> dict:
    """Compare a model's output class against the configured postprocessors.

    Returns ``{"level", "message", "suggest", "params"}`` where ``level`` is
    one of "ok" / "optional" / "suggest" / "warn" / "unknown", ``suggest`` is
    a list of postprocessor names to apply (empty when the current
    configuration is already sensible), and ``params`` their values
    (suggest_postprocess_params).

    ``level == "warn"`` means the current chain is actively wrong, not merely
    unconventional. ``"optional"`` offers a chain the model's output shows
    right without: an affinity model's into a segmentation, once its
    affinities show as probabilities. The page shows it as a quiet line
    with its Apply, not a suggestion's banner, which over the sigmoid a
    model's start or the ranking set (portable/rank.py needed_postprocess,
    an affinity model's logits' too) reads as though that were wrong. A
    threshold is no fault either: the ranking sets one at the probability
    the paint is fitted best at, an affinity model's too, so its layer shows
    the painted pairs' decision the score read. Ahead of an
    AffinityPostprocessor it is a "warn": the watershed then has no
    affinities to join objects by.

    ``evidence`` (``activation_evidence``'s) is what says the class, which
    each message gives. ``output_class`` None, when neither the network nor
    its output on noise says what the output is, is "unknown", with what
    the probe saw.

    ``threshold`` is the one the ranking fitted to the paint for this model
    (its row's, Session.rank_fits), or None when it was not ranked: an
    affinity chain suggested cuts its objects there (the watershed's bias),
    where the score called a painted pair joined, and says so; at 0.5
    without one.

    ``affinity_offsets``, the ones the model declares
    (``ModelConfig.declared_affinity_offsets``), make it an affinity model
    whatever its channels are named, and are the watershed's neighborhood.
    """
    names = list(postprocess_names or [])
    said = f" ({evidence})" if evidence else ""
    has_sigmoid = "SigmoidPostprocessor" in names
    has_default = "DefaultPostprocessor" in names
    has_threshold = "ThresholdPostprocessor" in names
    # The affinity the watershed joins neighbours above, as the messages say it.
    cut = "0.5" if threshold is None else f"{threshold} (fitted to your paint)"

    def verdict(level, message, suggest):
        """Attach parameter values to whatever chain we are proposing."""
        return {
            "level": level,
            "message": message,
            "suggest": suggest,
            "params": suggest_postprocess_params(suggest, output_class, out_channels, threshold, affinity_offsets),
        }

    if output_class is None:
        return verdict(
            "unknown",
            (
                f"Neither this model's network nor its output on noise says what its output is{said}. "
                "Set its postprocessing yourself: SigmoidPostprocessor if it gives logits, none if probabilities."
            ),
            [],
        )

    if affinity_offsets or looks_like_affinities(out_channels, model_name, channels_names):
        if "AffinityPostprocessor" in names:
            # Whether the affinities reach it in [0, 1], or the 0-255 of a
            # DefaultPostprocessor.
            scaled = has_default or has_sigmoid or output_class in (UNIT, BYTE_UNIT)
            # The watershed joins neighbours by how far their affinity is
            # above its bias: a threshold ahead of it leaves 0s and 1s, which
            # it reads as DefaultPostprocessor's 0-255 and divides by 255, and
            # no edge reaches the bias. The chain is suggested back without
            # it, or as the whole affinity chain when nothing else ahead puts
            # the affinities in [0, 1].
            ahead = names.index("AffinityPostprocessor")
            if "ThresholdPostprocessor" in names[:ahead]:
                rest = [n for i, n in enumerate(names) if i > ahead or n != "ThresholdPostprocessor"]
                return verdict(
                    "warn",
                    (
                        "A ThresholdPostprocessor comes before the "
                        "AffinityPostprocessor, so the watershed gets 0s and 1s "
                        "instead of affinities and finds no objects: take the "
                        f"threshold out, and neighbours join above {cut}."
                    ),
                    rest if scaled else suggest_affinity_chain(output_class),
                )
            if not scaled:
                return verdict(
                    "warn",
                    (
                        "AffinityPostprocessor needs affinities in [0, 1] (or "
                        "the 0-255 that DefaultPostprocessor produces), but "
                        f"this model's output is not bounded to [0, 1]{said} and "
                        "nothing ahead of it rescales it, so the watershed "
                        "works on the wrong values."
                    ),
                    suggest_affinity_chain(output_class),
                )
            return verdict("ok", "Affinity chain looks complete.", [])
        # The chain into a segmentation, offered over whatever shows the
        # affinities: the ranking's row among them, the sigmoid and the
        # threshold fitted to the paint (only the threshold on a tanh's or a
        # sigmoid's output). It replaces the threshold, which would leave the
        # watershed 0s and 1s, and joins neighbours above the same cut-off.
        # Unless the activation needs setting first, which the checks below
        # suggest or warn of: the sigmoid logits with nothing set need, a
        # second one taken away.
        if not ((output_class == UNBOUNDED and not names) or (output_class in (UNIT, BYTE_UNIT, SIGNED_UNIT) and has_sigmoid)):
            return verdict(
                "optional",
                (
                    f"its {out_channels} channels look like affinities, which this "
                    f"chain turns into a segmentation, joined above {cut}."
                ),
                suggest_affinity_chain(output_class),
            )

    if output_class in (UNIT, BYTE_UNIT):
        # A probability: in [0, 1], or x 255 in a uint8 export's whole numbers.
        what = "in [0, 1], a sigmoid's" if output_class == UNIT else "a probability x 255, in [0, 255]"
        if has_sigmoid:
            return verdict(
                "warn",
                (
                    f"This model's output is already {what}{said}. "
                    "SigmoidPostprocessor would apply a second one, flattening "
                    "the contrast."
                ),
                [n for n in names if n != "SigmoidPostprocessor"],
            )
        # A uint8 export's channels, when there are several, are classes,
        # each a probability x 255 (cellmap_models writes a semantic model's
        # so: a class's summed softmax, model_export.mia_model), which
        # exclude each other: their segmentation is each voxel's likeliest
        # class, background where none reaches 0.5. Offered as the affinity
        # chain is: the probabilities show right without it.
        if output_class == BYTE_UNIT and not names and out_channels and int(out_channels) > 1:
            return verdict(
                "optional",
                (
                    f"Its {out_channels} channels are classes, each {what}{said}: ArgmaxPostprocessor "
                    "labels each voxel with its likeliest class, background where none reaches 0.5."
                ),
                ["ArgmaxPostprocessor"],
            )
        return verdict(
            "ok",
            f"Model output is already {what}{said}; no activation needed.",
            [],
        )

    if output_class == SIGNED_UNIT:
        if has_sigmoid:
            return verdict(
                "warn",
                (
                    f"This model's output is bounded to [-1, 1], a tanh's{said}. "
                    "A sigmoid on top compresses it to roughly [0.27, 0.73], "
                    "which looks washed out. DefaultPostprocessor is the usual "
                    "choice for this range."
                ),
                ["DefaultPostprocessor"],
            )
        if not (has_default or has_threshold):
            return verdict(
                "suggest",
                (
                    f"Model output is in [-1, 1], a tanh's{said}. "
                    "DefaultPostprocessor rescales that to 0-255 for display."
                ),
                ["DefaultPostprocessor"],
            )
        return verdict("ok", "DefaultPostprocessor suits this [-1, 1] output." if has_default
                       else "Model output is in [-1, 1]; the threshold is left as set.", [])

    # UNBOUNDED: logits and signed distances are indistinguishable here, so
    # suggest only when nothing at all is configured, and say why it's a guess.
    if not names:
        return verdict(
            "suggest",
            (
                f"Model output has no activation{said}. If these are logits "
                "(an affinity model's are), SigmoidPostprocessor converts them "
                "to probabilities. If they are distances, leave this alone -- "
                "neither the network's last op nor the output's range can tell "
                "the two apart."
            ),
            ["SigmoidPostprocessor"],
        )
    return verdict("ok", f"Model output has no activation{said}; current postprocessing left as set.", [])


def start_postprocess(output_class: Optional[str], out_channels=None, model_name="", channels_names=None,
                      affinity_offsets=None) -> list:
    """The postprocessing a model's start sets on a chain with none, from what
    its output is (``output_class``, as the model declares it): the steps
    ``review_postprocess`` suggests for an empty chain, each ``{"name",
    **params}`` with the values it gives, so that what a start sets and
    what the model advice suggests are one rule. [] when ``output_class`` is
    None, nothing saying what the output is, and when the output shows
    right as it is (one probability).

    What that is, by the output: logits (UNBOUNDED) a sigmoid, an affinity
    model's too; affinities in [0, 1] or x 255 the chain into a
    segmentation (DefaultPostprocessor, then the watershed at the offsets
    the model declares, ``affinity_offsets``, else the nearest neighbours
    cut to its channels); several channels of probabilities x 255 (classes)
    their argmax, background below 0.5; a tanh's [-1, 1]
    DefaultPostprocessor's 0-255.
    """
    if output_class is None:
        return []
    review = review_postprocess(output_class, [], out_channels, model_name, channels_names,
                                affinity_offsets=affinity_offsets)
    return [{"name": name, **review["params"].get(name, {})} for name in review["suggest"]]
