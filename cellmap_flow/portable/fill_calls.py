"""What the browser calls for Label it's box tools: the functions
``web/src/backend/routes/view-labels.ts``'s Seed from Prediction, All
Background, Split Objects and Undo run through
``Inference.python("cellmap_flow.portable.fill_calls:<name>", args, arrays)``,
in the shape ``chunkmirage.calls`` says: ``fn(args, arrays)``, ``args`` the
JSON arguments, ``arrays`` the bytes sent beside them, in order; a JSON
value, or a pair of JSON and arrays, back. Each decodes its arrays, calls
``fill.py`` and encodes the answer; the maths is not here, and the server's
routes (``dashboard/routes/finetune/view_labels.py``, through
``finetune/session/fill.py``) call the same functions around MinIO.

The route reads the box from the paint store and writes back what comes
out, and keeps the undo stack, as the AI annotation's accept does
(``ai_annotate_calls.accept``): nothing here reads or keeps a volume. A box
travels as the store holds it: ``shape`` (z, y, x) and ``dtype`` (the
volume's) in ``args``, its bytes in C order, always ``arrays[0]``. A write
answers the box to write as its one array, or no array when nothing
changes, so the route writes nothing and pushes no undo entry then.

A refusal the server answers 400 (new ids that do not fit in the volume's
dtype, a ValueError there) comes back as ``{"error": message}``, as
``label_it_calls`` answers its own, for the route to answer as the server
does. A ``ModuleNotFoundError`` (scipy, which Split Objects labels its
components with) is never caught: it is how chunkmirage's worker learns to
load scipy, before it calls again.
"""

import numpy as np

from cellmap_flow.portable import fill


def _box(args, data) -> np.ndarray:
    """The box's labels from ``data``, its bytes as the paint store holds them."""
    return np.frombuffer(data, dtype=np.dtype(args["dtype"])).reshape(tuple(args["shape"]))


def _written(after, value):
    """``value`` with the box to write beside it, or with no array when ``after`` is None."""
    return value, [] if after is None else [after]


def box(args, arrays):
    """The annotation voxels a box in world nm covers (``fill.box_voxels``).
    ``args``: ``volume`` (its ``output_voxel_size`` and
    ``dataset_offset_nm``), ``offset_nm`` and ``shape_nm`` (the box's lower
    corner and size, z, y, x) and ``volume_shape`` (the volume's voxels).
    Answers ``{"lo", "hi"}``, or null when no voxel of the volume is in the
    box."""
    voxels = fill.box_voxels(args["volume"], args["offset_nm"], args["shape_nm"], args["volume_shape"])
    if voxels is None:
        return None
    lo, hi = voxels
    return {"lo": [int(v) for v in lo], "hi": [int(v) for v in hi]}


def _paint_counts(counts) -> dict:
    n_foreground, n_background, n_overwritten = counts
    return {"n_foreground": n_foreground, "n_background": n_background, "n_overwritten": n_overwritten}


def seed(args, arrays):
    """A seed written into the box's unannotated voxels: ``fill.seed_labels``
    of the objects, written as ``fill.paint`` writes without overwrite.
    ``args``: the box's ``shape`` and ``dtype``, and ``instance_target``
    (the model trains on instances, so new ids count up on a volume of more
    than 8-bit ids: ``fill.counts_up``). ``arrays``: the box, then the
    objects over it (uint32, 0 background, as ``label_it_calls.seed``
    answers them). Answers the counts (``n_foreground``, ``n_background``,
    ``n_overwritten``) with the box to write; or the refusal, when the new
    ids do not fit."""
    existing = _box(args, arrays[0])
    objects = np.frombuffer(arrays[1], dtype=np.uint32).reshape(existing.shape)
    try:
        labels = fill.seed_labels(objects, existing, count_up=fill.counts_up(args["instance_target"], existing.dtype))
    except ValueError as e:
        return {"error": str(e)}
    after, counts = fill.paint(existing, labels)
    return _written(after, _paint_counts(counts))


def background(args, arrays):
    """Every unannotated voxel of the box background (1), written as
    ``fill.paint`` writes without overwrite: All Background. ``args``: the
    box's ``shape`` and ``dtype``; ``arrays``: the box. Answers as
    ``seed``."""
    existing = _box(args, arrays[0])
    after, counts = fill.paint(existing, np.ones_like(existing))
    return _written(after, _paint_counts(counts))


def split(args, arrays):
    """The box's foreground relabelled by connected component
    (``fill.relabel_objects``) and written over its foreground voxels
    (``fill.rewrite``): Split Objects. ``args``: the box's ``shape`` and
    ``dtype``, and ``connectivity`` and ``per_slice``, as the route has read
    them. ``arrays``: the box. Answers the counts (``objects``, ``split``,
    ``merged``) and ``changed``, the voxels whose label changed, with the box
    to write; or the refusal, when the split-off pieces' new ids do not
    fit."""
    existing = _box(args, arrays[0])
    try:
        labels, counts = fill.relabel_objects(
            existing, connectivity=int(args["connectivity"]), per_slice=bool(args["per_slice"])
        )
    except ValueError as e:
        return {"error": str(e)}
    after, changed = fill.rewrite(existing, labels)
    return _written(after, {**{k: int(v) for k, v in counts.items()}, "changed": changed})


def undo(args, arrays):
    """The box put back to before an action where it still holds what the
    action left (``fill.restore``): Undo. ``args``: the box's ``shape`` and
    ``dtype``; ``arrays``: the box as it is now, then the undo entry's
    ``before`` and ``after``, each shaped as the box. Answers ``restored``,
    the voxels put back, with the box to write."""
    current = _box(args, arrays[0])
    before, after = _box(args, arrays[1]), _box(args, arrays[2])
    restored, n = fill.restore(current, before, after)
    return _written(restored, {"restored": n})
