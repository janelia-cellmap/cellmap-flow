"""How well a model's prediction already matches the paint: the score the
Finetune tab ranks models by, to pick the one to start finetuning from.

The one implementation both modes run: the server dashboard
(dashboard/routes/finetune/rank_routes.py) in its process, the browser
(web/src/backend/routes/rank.ts) in chunkmirage's Pyodide workers through
portable/calls.py; numpy and the standard library only. In short:

- A model is scored on the targets its own finetune trains on, over the
  painted voxels: an object or not on each channel of a binary model, the
  painted pairs of an affinity model.
- The score is the balanced accuracy, the mean of the share of object
  targets called object and of background targets called background. A
  binary model's painted voxels next to one of the other class are not
  counted (rough strokes are forgiven along their edges).
- It is read at the model's own cut-off: the logit, among CUTOFFS, with the
  best Dice of its calls against the painted objects (the nearest to 0
  among equals; 0 when only one class is painted). The score at 0 is given
  beside it.
- A binary model answers with its best channel, by name and index
  (``channel``, ``channel_index``): the one the Finetune tab's Use this
  model makes the training form's Channel (select_channel) for a model of
  several, and Fit to Paint when none is picked. An affinity model is
  scored on its channels together, and answers neither.
- A model is read through its own input normalization (what its start
  sets: its declaration, or its kind's convention) and through the
  session's when that is another (``norm_tries``), and the better score is
  kept (``best_try``, its own among equals). Read through none, or another
  model's, a model's score is noise.
- Where the raw under the paint is narrower than NARROW_WINDOW of the
  model's MinMaxNormalizer's range, the model is scored again with the
  normalizer stretched to the raw's 1st to 99th percentiles (``windowed``),
  kept only when it scores WINDOW_MARGIN better.
- The row of the ranking (``score_row``) is the score kept, with what it
  was read at in the units its controls take: the cut-off as Seed from
  Prediction's threshold (a probability) and the raw's range read as 0 to
  1 (the stretched window, or the normalizer's own range), and the
  normalizer steps it was read through (``input_norm``), which Use this
  model and Fit to Paint make the session's, with what the other channels
  and normalizations matched (``channel_matches``, ``input_norm_others``):
  why this one. Both modes answer it as it is, so their tables agree.
- Beside it, the postprocessing the model's layer needs to show a good
  prediction (``needed_postprocess``): the sigmoid its output needs and the
  threshold fitted to the paint, which they set too, as a model's start
  sets the sigmoid alone.

The class-balanced BCE with label smoothing RANK_SMOOTHING, the finetune's
default loss, is given beside the score: about 0.69 is guessing, about 0.20
the paint fitted exactly.
"""

import math
from typing import List, Optional, Sequence

import numpy as np

from cellmap_flow.portable.label_it import SIGNED_UNIT, UNBOUNDED, UNIT

# The most painted blocks one score reads, spread over the paint.
RANK_BLOCKS = 32
RANK_SMOOTHING = 0.1
# The cut-offs a score is read at: logits from -6 to 6 in quarters, 0 among them.
CUTOFFS = np.arange(49) / 4 - 6
NARROW_WINDOW = 0.25
WINDOW_MARGIN = 0.03


def spread(items: Sequence, count: int) -> list:
    """``count`` of ``items``, spread evenly over them in their order; all of them when there are no more."""
    if len(items) <= count:
        return list(items)
    return [items[int((j + 0.5) * len(items) // count)] for j in range(count)]


def empty_tallies(channels: int) -> List[dict]:
    """One channel's agreement with the paint per channel: the targets
    counted, their logits binned against CUTOFFS (bin b: above b of them),
    and the loss summed, apart for object and background targets."""
    return [
        {"objects": 0, "object_bins": np.zeros(len(CUTOFFS) + 1, np.int64), "object_loss": 0.0,
         "background": 0, "background_bins": np.zeros(len(CUTOFFS) + 1, np.int64), "background_loss": 0.0}
        for _ in range(channels)
    ]


def paint_edges(labels: np.ndarray) -> np.ndarray:
    """The painted voxels of ``labels`` (z, y, x; 0 unpainted, 1 background,
    2 and up objects) that touch a painted voxel of the other class along z,
    y or x: the strokes' edges, which a binary model's score leaves out."""
    painted, objects = labels > 0, labels > 1
    edges = np.zeros(labels.shape, bool)
    for axis in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        differ = painted[a] & painted[b] & (objects[a] != objects[b])
        edges[a] |= differ
        edges[b] |= differ
    return edges


def tally(tallies: List[dict], target: np.ndarray, mask: np.ndarray, logits: np.ndarray,
          skip: Optional[np.ndarray] = None) -> None:
    """Add one painted block to ``tallies``: ``target`` and ``mask`` (C, z, y, x)
    as the finetune's target transform makes them, ``logits`` the model's on
    the same voxels (NaN where it was not read), and ``skip`` (z, y, x) the
    voxels not to count (paint_edges), or None."""
    high, low = 1 - RANK_SMOOTHING / 2, RANK_SMOOTHING / 2
    for c, t in enumerate(tallies):
        counted = (mask[c] > 0) & ~np.isnan(logits[c])
        if skip is not None:
            counted &= ~skip
        z = logits[c][counted].astype(np.float64)
        is_object = target[c][counted] > 0.5
        bins = np.clip(np.ceil((z - CUTOFFS[0]) * 4), 0, len(CUTOFFS)).astype(np.int64)
        # BCE on the logit against the smoothed target: softplus(z) - y z.
        softplus = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
        t["objects"] += int(is_object.sum())
        t["background"] += int((~is_object).sum())
        t["object_bins"] += np.bincount(bins[is_object], minlength=len(CUTOFFS) + 1)
        t["background_bins"] += np.bincount(bins[~is_object], minlength=len(CUTOFFS) + 1)
        t["object_loss"] += float((softplus - high * z)[is_object].sum())
        t["background_loss"] += float((softplus - low * z)[~is_object].sum())


def rank_score(tallies: List[dict], channels: Sequence[str], affinity: bool) -> Optional[dict]:
    """The score of ``tallies``: an affinity model's over its channels
    together (``channel`` and ``channel_index`` None), a binary model's best
    channel's (by the score, then the lower loss), ``channel`` its name in
    ``channels`` ("channel <i>" past them) and ``channel_index`` its index,
    as select_channel takes it; with ``channel_matches``, every channel's
    ``{channel, channel_index, matches}``, the best first, which says why
    that one (organelle close behind mito, say). None when nothing was
    counted."""
    if affinity:
        all_ = {k: sum(t[k] for t in tallies) for k in tallies[0]}
        return _score_of(all_, None, None)
    best, each = None, []
    for c, t in enumerate(tallies):
        s = _score_of(t, channels[c] if c < len(channels) else f"channel {c}", c)
        if s is None:
            continue
        each.append(s)
        if best is None or s["matches"] > best["matches"] or (s["matches"] == best["matches"] and s["loss"] < best["loss"]):
            best = s
    if best is None:
        return None
    others = sorted((s for s in each if s is not best), key=lambda s: -s["matches"])
    return {**best, "channel_matches": [{k: s[k] for k in ("channel", "channel_index", "matches")} for s in [best, *others]]}


def _score_of(t: dict, channel: Optional[str], channel_index: Optional[int]) -> Optional[dict]:
    if not t["objects"] and not t["background"]:
        return None

    def at(k):
        found = int(t["object_bins"][k + 1:].sum())
        wrong = int(t["background_bins"][k + 1:].sum())
        found_share = found / t["objects"] if t["objects"] else None
        right = 1 - wrong / t["background"] if t["background"] else None
        rates = [r for r in (found_share, right) if r is not None]
        dice = 2 * found / max(found + t["objects"] + wrong, 1)
        return found_share, right, sum(rates) / len(rates), dice

    zero = int(np.flatnonzero(CUTOFFS == 0)[0])
    k = zero
    if t["objects"] and t["background"]:
        for j in range(len(CUTOFFS)):
            better = at(j)[3] - at(k)[3]
            if better > 1e-12 or (better > -1e-12 and abs(CUTOFFS[j]) < abs(CUTOFFS[k])):
                k = j
    found_share, right, matches, _ = at(k)
    losses = [l for l in (t["object_loss"] / t["objects"] if t["objects"] else None,
                          t["background_loss"] / t["background"] if t["background"] else None) if l is not None]
    return {
        "matches": matches, "objects_found": found_share, "background_right": right,
        "cutoff": float(CUTOFFS[k]), "matches_at_zero": at(zero)[2],
        "loss": sum(losses) / len(losses),
        "scored_voxels": t["objects"] + t["background"], "object_voxels": t["objects"], "background_voxels": t["background"],
        "channel": channel, "channel_index": channel_index,
    }


def _min_max(norm: Sequence[dict]) -> Optional[tuple]:
    """``(index, low, high)`` of the MinMaxNormalizer among ``norm``
    (normalizer specs) and the raw's range it maps to 0 to 1 (its class's
    defaults, 0 and 255, where a spec leaves them out); None without one."""
    k = next((i for i, s in enumerate(norm) if s.get("name") == "MinMaxNormalizer"), None)
    if k is None:
        return None
    return k, float(norm[k].get("min_value", 0)), float(norm[k].get("max_value", 255))


def windowed(norm: Sequence[dict], lo: float, hi: float) -> Optional[List[dict]]:
    """``norm`` (normalizer specs, ``{"name": ..., ...}``) with its
    MinMaxNormalizer set to the raw's window [lo, hi], or None: no
    MinMaxNormalizer, no window (lo = hi), or one not narrower than
    NARROW_WINDOW of the normalizer's own range."""
    found = _min_max(norm)
    if found is None or not hi > lo:
        return None
    k, low, high = found
    if not hi - lo < NARROW_WINDOW * (high - low):
        return None
    return [dict(s, min_value=float(lo), max_value=float(hi)) if i == k else dict(s) for i, s in enumerate(norm)]


def _value(v):
    """A normalizer parameter as a number or a bool where it reads as one: the
    session's chain keeps what the page sent ("0", "false"), a model's
    declaration numbers."""
    if isinstance(v, str):
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            return float(v)
        except ValueError:
            return v
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def same_steps(a: Sequence[dict], b: Sequence[dict]) -> bool:
    """Whether two normalizer chains are the same steps with the same
    parameters, read as values (``_value``: "0" is 0)."""
    return len(a) == len(b) and all(
        x.keys() == y.keys() and all(_value(x[k]) == _value(y[k]) for k in x) for x, y in zip(a, b))


def norm_tries(own: Sequence[dict], own_from: str, session: Sequence[dict]) -> List[dict]:
    """The input normalizations a model is scored through, each once, its own
    first: ``{"input_norm": steps, "from": where}``.

    ``own`` is what the model's start sets (its declaration, ``own_from``
    "declared"; or its kind's convention, "convention"), else what the model
    advice assumes for a model nothing is known of ("assumed"; [] when there
    is nothing to assume), ``session`` the session's input normalization
    ("session").
    The model's own is always tried: read through another's, or none, a model
    scores as noise (fly_organelles fed raw 0 to 255 matched 51%, its best
    channel nucleus for paint of mitochondria). The session's is tried too,
    as one the user set may fit this raw better. With neither, the raw as it
    is ("none").
    """
    tries = []
    # An assumption is tried after the session's, which someone chose, so it
    # loses a tie to it (best_try keeps the first among equals).
    order = ((own, own_from), (session, "session"))
    for steps, where in (order[::-1] if own_from == "assumed" else order):
        steps = [dict(s) for s in steps or []]
        if steps and not any(same_steps(steps, t["input_norm"]) for t in tries):
            tries.append({"input_norm": steps, "from": where})
    return tries or [{"input_norm": [], "from": "none"}]


def best_try(scores: Sequence[Optional[dict]]) -> Optional[int]:
    """The index of the score (``rank_score``'s, one per ``norm_tries``
    entry, None where it counted nothing) that matches best, the first among
    equals (the model's own); None when none counted."""
    best = None
    for i, s in enumerate(scores):
        if s is not None and (best is None or s["matches"] > scores[best]["matches"]):
            best = i
    return best


SIGMOID = "SigmoidPostprocessor"
THRESHOLD = "ThresholdPostprocessor"


def needed_postprocess(output_class: Optional[str], postprocess: Sequence[dict],
                       threshold: Optional[float] = None) -> Optional[List[dict]]:
    """The postprocessing a model's layer needs instead of ``postprocess``
    (the session's steps) to show a good prediction, or None when it shows
    one as it is: what a model's start sets on a chain with none (no
    ``threshold``), and what Use this model, Fit to Paint and Use these
    settings set beside the input normalization (the ranking row's
    ``threshold``). ``output_class`` is the model's output activation
    (portable/label_it.py, as its server's probe answers it); with None,
    not known, nothing changes.

    Two steps are this rule's, on a chain of nothing else:

    - The activation: a sigmoid for an output with none (UNBOUNDED, logits)
      when the chain has none, and is empty or is given a threshold. A
      threshold alone on logits shows them cut at a logit (0, the model's
      own boundary, as a rule); a sigmoid put ahead of it would cut the
      probabilities there instead, everything object at 0. An affinity
      model's logits need it as much as any model's: the chain into a
      segmentation (serving/probe.py suggest_affinity_chain) stays the
      model advice's optional offer.
    - ``threshold``, a probability as the ranking's row gives it (0.5 at
      the model's own boundary): a ThresholdPostprocessor at it, in place of
      one the chain has, in the units of the output after the activation:
      the probability itself after the sigmoid and on an output in [0, 1]
      (UNIT); 2p - 1 on one in [-1, 1] (SIGNED_UNIT, a tanh), whose
      probability label_it.probability reads as (x + 1) / 2.

    A chain with any other step (a watershed, DefaultPostprocessor, a
    ChannelSelection) is the user's and left as it is, but for one thing: a
    sigmoid on an output that has one already (UNIT, SIGNED_UNIT) is taken
    away from any chain, as a second sigmoid flattens the contrast (the
    model advice warns of it: serving/probe.py review_postprocess).
    """
    steps = [dict(s) for s in postprocess or []]
    needed = [s for s in steps if s.get("name") != SIGMOID] if output_class in (UNIT, SIGNED_UNIT) else steps
    if output_class not in (UNBOUNDED, UNIT, SIGNED_UNIT) or any(s.get("name") not in (SIGMOID, THRESHOLD) for s in needed):
        return None if needed == steps else needed
    if output_class == UNBOUNDED and (not needed or threshold is not None) and all(s.get("name") != SIGMOID for s in needed):
        needed = [{"name": SIGMOID}, *needed]
    if threshold is not None:
        value = round(2 * float(threshold) - 1, 3) if output_class == SIGNED_UNIT else float(threshold)
        needed = [s for s in needed if s.get("name") != THRESHOLD] + [{"name": THRESHOLD, "threshold": value}]
    return None if needed == steps else needed


def score_row(base: Optional[dict], stretched: Optional[dict], norm: Sequence[dict],
              raw: Optional[Sequence[float]], norm_from: str = "session",
              others: Sequence[dict] = ()) -> Optional[dict]:
    """A model's row of the ranking, as both modes answer it beside the
    model's name, the time taken and the notes; None when ``base`` is None
    (the paint gave the model nothing to score).

    ``base`` is the model's score (``rank_score``) read through ``norm``,
    the normalizer specs it was served with (the best of ``norm_tries``,
    ``norm_from`` its ``from``); ``stretched`` its score read through
    ``windowed(norm, *raw)``, ``raw`` the raw's 1st and 99th percentiles
    under the paint, or None when that was not tried (the raw is not narrow)
    or counted nothing. The stretched score is kept when it matches
    WINDOW_MARGIN better than ``base``. The row is the score kept, with:

    - ``window``: ``raw`` when the stretched score was kept, else None; and
      then ``matches_without_window``, ``base``'s match;
    - ``threshold``: the cut-off as a probability, as Seed from
      Prediction's threshold takes it, to three decimals: 0.5 at the
      model's own boundary (cut-off 0), else the probability the paint is
      fitted best above;
    - ``input_window``: the raw's range the model read as 0 to 1, as the
      Processing tab's MinMaxNormalizer takes it: ``window`` when kept, else
      ``norm``'s own range; None when ``norm`` has no MinMaxNormalizer;
    - ``input_norm``: the normalizer steps the score kept was read through
      (``norm``, stretched when the stretch was kept), which Use this model
      and Fit to Paint make the session's; and ``input_norm_from``, where
      they come from (``norm_from``);
    - ``input_norm_others``: what the other normalizations tried matched,
      ``others`` (``{from, matches}`` each), which says why this one.
    """
    if base is None:
        return None
    keep = stretched is not None and raw is not None and stretched["matches"] >= base["matches"] + WINDOW_MARGIN
    score = stretched if keep else base
    window = [float(raw[0]), float(raw[1])] if keep else None
    declared = _min_max(norm)
    if window is not None:
        input_window = window
    elif declared is not None:
        input_window = [declared[1], declared[2]]
    else:
        input_window = None
    return {
        **score,
        "window": window,
        **({"matches_without_window": base["matches"]} if keep else {}),
        "threshold": round(1 / (1 + math.exp(-score["cutoff"])), 3),
        "input_window": input_window,
        "input_norm": windowed(norm, *window) if keep else [dict(s) for s in norm],
        "input_norm_from": norm_from,
        "input_norm_others": [{"from": o["from"], "matches": o["matches"]} for o in others],
    }

