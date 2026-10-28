"""What the browser calls for Label it: the functions
``web/src/backend/routes/view-labels.ts`` (and the rank's
``web/src/backend/routes/rank.ts``, for the placement and the probability)
runs through ``Inference.python("cellmap_flow.portable.label_it_calls:<name>",
args, arrays)``, in the shape ``chunkmirage.calls`` says: ``fn(args, arrays)``,
``args`` the JSON arguments, ``arrays`` the bytes sent beside them; a JSON
value, or a pair of JSON and arrays, back. Each decodes its arrays, calls
``label_it`` and encodes the answer; the maths is not here.

A refusal the server answers 400 (a ValueError there) comes back as
``{"error": message}``, for the route to answer as the server does; any
other exception rejects the call, the route's 500. A ``ModuleNotFoundError``
is never caught: it is how chunkmirage's worker learns to load scipy, before
it calls again.
"""

import numpy as np

from cellmap_flow.portable import label_it


def _slice(channels):
    """The channels as ``label_it`` takes them, from ``[start, stop]`` or null."""
    return None if channels is None else slice(int(channels[0]), int(channels[1]))


def _placed(args):
    """``label_it.placement`` of ``args``' box (``lo``, ``hi``) of ``volume``
    (its ``output_voxel_size`` and ``dataset_offset_nm``) on ``level`` (the
    served level: ``origin``, its OME translation, ``voxel`` and ``shape``),
    as JSON: the box shrunk, the indices and the served box to read for
    them; null when the level covers none of it."""
    volume, level = args["volume"], args["level"]
    placed = label_it.placement(
        args["lo"], args["hi"], volume["output_voxel_size"], volume.get("dataset_offset_nm"),
        level["origin"], level["voxel"], level["shape"],
    )
    if placed is None:
        return None
    indices, lo, hi = placed
    read_lo, read_hi = label_it.read_box(indices)
    return {"lo": lo, "hi": hi, "indices": [i.tolist() for i in indices], "read_lo": read_lo, "read_hi": read_hi}


def seed_options(args, arrays):
    """The seed's options (``label_it.seed_options``) from ``args``, the
    request's body: ``{"options": ...}``, or the refusal."""
    try:
        return {"options": label_it.seed_options(args)}
    except ValueError as e:
        return {"error": str(e)}


def seed_plan(args, arrays):
    """What a seed reads of the model, and where. ``args``: the model's
    ``channel_names``, the request's ``select_channel`` (null when it gives
    none), and the box as ``place`` takes it. Answers ``channels`` (``[start,
    stop]``, or null for every one), ``offsets`` (an affinity model's, by
    ``label_it.affinity_offsets``, else null) and ``placement`` (``place``'s
    answer); or the refusal of the selected channel."""
    offsets = label_it.affinity_offsets(args["channel_names"])
    try:
        channels = label_it.seed_channels("affinities" if offsets else "binary", offsets, args.get("select_channel"))
    except ValueError as e:
        return {"error": str(e)}
    return {
        "channels": None if channels is None else [channels.start, channels.stop],
        "offsets": offsets,
        "placement": _placed(args),
    }


def field_of_view(args, arrays):
    """The size in nm (z, y, x) of what the viewer's panel shows
    (``label_it.field_of_view_nm``): ``args`` ``viewer_px`` (the page's
    viewer frame, width and height), ``state`` (the viewer's state, its
    JSON), ``pixel_nm``, ``depth_axis``, ``display_dimensions``,
    ``voxel_size`` (the volume's) and ``depth_nm`` (null: one voxel).
    Answers ``{"size_nm"}``, or the refusal."""
    try:
        return {"size_nm": label_it.field_of_view_nm(
            args.get("viewer_px"), args.get("state") or {}, args.get("pixel_nm"), int(args["depth_axis"]),
            args.get("display_dimensions"), args["voxel_size"], args.get("depth_nm"),
        )}
    except ValueError as e:
        return {"error": str(e)}


def place(args, arrays):
    """Which served voxel lies under each voxel of the box (``_placed``):
    ``args`` ``lo``, ``hi``, ``volume`` and ``level``."""
    return _placed(args)


def seed(args, arrays):
    """A seed's objects, or the mutex watershed's input. ``arrays``: the
    served region ``seed_plan``'s placement says to read (C, z, y, x
    float32, its ``shape`` in ``args``). ``args`` also: the ``indices`` of
    that placement, ``method`` and the options (``seed_options``'), the
    model's ``output_class`` (null: read off the values), and the plan's
    ``channels`` and ``offsets``.

    *components*: ``{"count"}`` and the objects, (z, y, x) uint32 over the
    box. *mutex_watershed*: ``{"shape"}`` and ``label_it.affinities``, float32,
    which the browser's mutex watershed takes from there. A refusal (a
    channel the model does not have) is ``{"error"}``.
    """
    region = np.frombuffer(arrays[0], np.float32).reshape(args["shape"])
    prediction = label_it.gather(region, args["indices"])
    output_class = args.get("output_class")
    try:
        if args["method"] == "components":
            objects = label_it.components(
                prediction, output_class, _slice(args.get("channels")), args["threshold"], args["min_size"],
                args["connectivity"], args["per_slice"],
            )
            return {"count": int(objects.max(initial=0))}, [objects.astype(np.uint32)]
        if args["method"] == "mutex_watershed":
            if not args.get("offsets"):
                raise ValueError("The model's affinity offsets are unknown, so it cannot seed by mutex_watershed")
            out = label_it.affinities(prediction, output_class, args["offsets"])
            return {"shape": list(out.shape)}, [np.ascontiguousarray(out, dtype=np.float32)]
    except ValueError as e:
        return {"error": str(e)}
    raise ValueError(f"the browser seeds by components or mutex_watershed, not {args['method']!r}")


def probability(args, arrays):
    """``label_it.probability`` of a block (``arrays``: C, z, y, x float32,
    its ``shape`` in ``args``) by the model's ``output_class`` (null: read
    off the values), every channel: the same shape, float32."""
    block = np.frombuffer(arrays[0], np.float32).reshape(args["shape"])
    return np.ascontiguousarray(label_it.probability(block, args.get("output_class")), dtype=np.float32)
