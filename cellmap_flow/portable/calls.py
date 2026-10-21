"""What the browser calls: the functions ``web/src/backend/routes/rank.ts``
runs through chunkmirage's ``Engine.call("cellmap_flow.portable.calls:<name>",
args, arrays)`` (``chunkmirage.calls`` says the shape: ``fn(args, arrays)``,
``args`` the JSON arguments, ``arrays`` the bytes sent beside them, in order;
a JSON value, bytes, or a pair of JSON and arrays back). Each decodes its
arrays, calls the maths in ``rank.py`` or ``carry.py`` and encodes the
answer; the maths is not here. The workers take turns, so what a call needs
from the one before (the rank's tallies) travels in its JSON.
"""

import numpy as np

from cellmap_flow.portable import rank
from cellmap_flow.portable.carry import Grid, carry as carry_core, parts

_BINS = ("object_bins", "background_bins")


def _tallies(encoded) -> list:
    return [{k: np.asarray(v, np.int64) if k in _BINS else v for k, v in t.items()} for t in encoded]


def _encoded(tallies) -> list:
    return [{k: v.tolist() if k in _BINS else v for k, v in t.items()} for t in tallies]


def rank_tally(args, arrays):
    """One painted chunk added to a model's tallies (``rank.tally``). ``args``:
    ``tallies`` (the ones so far, as this answered them; null for fresh ones),
    ``shape`` (z, y, x), ``channels``, ``labels_dtype`` and ``affinity``;
    ``arrays``: the chunk's labels (z, y, x), then its target, mask and the
    model's logits (each C, z, y, x float32, the logits NaN where not read).
    Answers the tallies."""
    shape, channels = tuple(args["shape"]), int(args["channels"])
    labels = np.frombuffer(arrays[0], np.dtype(args["labels_dtype"])).reshape(shape)
    target, mask, logits = (np.frombuffer(a, np.float32).reshape((channels, *shape)) for a in arrays[1:4])
    tallies = _tallies(args["tallies"]) if args.get("tallies") else rank.empty_tallies(channels)
    rank.tally(tallies, target, mask, logits, None if args["affinity"] else rank.paint_edges(labels))
    return _encoded(tallies)


def rank_score(args, arrays):
    """The score of ``tallies`` (``rank.rank_score``) for ``channels``, an ``affinity`` model's or not; null when nothing was counted."""
    return rank.rank_score(_tallies(args["tallies"]), args["channels"], bool(args["affinity"]))


def windowed(args, arrays):
    """``rank.windowed``: ``norm`` with its MinMaxNormalizer set to the raw's window [``lo``, ``hi``], or null."""
    return rank.windowed(args["norm"], float(args["lo"]), float(args["hi"]))


def carry_paint(args, arrays):
    """The old volume's paint carried onto the new grid (``carry.carry``).
    ``args``: ``old`` and ``new`` (each a ``Grid``'s fields), ``old_dtype``,
    ``new_dtype`` and ``indices``, the old painted chunks'; ``arrays``: those
    chunks' labels, whole, in that order. Answers the counts with ``indices``,
    the new chunks' in the order of the arrays beside them, each a whole chunk
    to write where it is not 0."""
    old, new = (Grid(**{k: tuple(v) for k, v in args[g].items()}) for g in ("old", "new"))
    old_dtype = np.dtype(args["old_dtype"])
    labels = {tuple(i): np.frombuffer(a, old_dtype).reshape(old.chunks) for i, a in zip(args["indices"], arrays)}

    def read(lo, hi):
        shape = [b - a for a, b in zip(lo, hi)]
        out = np.zeros(shape, old_dtype)
        for index, in_chunk, in_box in parts(old.chunks, lo, shape):
            if index in labels:
                out[in_box] = labels[index][in_chunk]
        return out

    chunks, counts = carry_core(old, new, sorted(labels.items()), read, np.dtype(args["new_dtype"]))
    indices = sorted(chunks)
    return {**counts, "indices": [list(i) for i in indices]}, [chunks[i] for i in indices]
