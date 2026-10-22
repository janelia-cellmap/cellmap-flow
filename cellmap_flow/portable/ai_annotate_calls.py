"""What the browser calls for an AI annotation: the functions
``web/src/backend/ai-annotate-job.ts`` and ``web/src/ai-annotate/gemini.ts``
run through chunkmirage's ``Engine.call("cellmap_flow.portable.ai_annotate_calls:<name>",
args, arrays)`` (``chunkmirage.calls`` says the shape: ``fn(args, arrays)``,
``args`` the JSON arguments, ``arrays`` the bytes sent beside them, in order;
a JSON value, bytes, or a pair of JSON and arrays back). Each decodes its
arrays, calls the maths in ``ai_annotate.py`` and ``fill.py`` and encodes the
answer; the maths is not here. Images travel as the canvas holds them, RGBA
bytes rows first; label boxes in the volume's dtype, C order (z, y, x).

A refusal meant for the user (a point outside the volume or the data, ids
running out) is answered as ``{"refused": message}`` rather than raised: the
worker would hand the page Python's traceback with it, where the page shows
the message alone, as the server's routes do. Nothing here logs or keeps
anything: an image's bytes go no further than the answer.
"""

import numpy as np

from cellmap_flow.portable import ai_annotate, fill
from cellmap_flow.portable.ai_annotate import PlanePlan, Refused


def _rgba(rgb) -> np.ndarray:
    """An RGB image (rows, cols, 3) as the canvas's RGBA, opaque."""
    out = np.full(rgb.shape[:2] + (4,), 255, dtype=np.uint8)
    out[:, :, :3] = rgb
    return out


def plan(args, arrays):
    """The plane through the point (``ai_annotate.plan_plane``) and the raw it
    reads (``raw_box``). ``args``: ``volume`` (its ``output_voxel_size``,
    ``input_voxel_size`` and ``dataset_offset_nm``), ``volume_shape``,
    ``point_nm``, ``depth_axis``, ``crop_size_px``, ``data_box_nm`` and
    ``display_dimensions`` (each null for none), and ``raw``, the grid of the
    raw level read (``voxel``, and ``corner``, its voxel 0's lower corner in
    nm). Answers the ``plan`` (``PlanePlan.to_dict``), its ``click_px`` and
    ``write_shape``, and the ``raw_box`` (``begin``, ``count``); or the
    refusal."""
    try:
        planned = ai_annotate.plan_plane(
            args["volume"], args["volume_shape"], args["point_nm"], int(args["depth_axis"]), int(args["crop_size_px"]),
            data_box_nm=args.get("data_box_nm"), display_dimensions=args.get("display_dimensions"),
        )
    except Refused as e:
        return {"refused": str(e)}
    begin, count = ai_annotate.raw_box(planned, args["raw"]["voxel"], args["raw"]["corner"])
    return {
        "plan": planned.to_dict(),
        "click_px": list(planned.click_px),
        "write_shape": list(planned.write_shape),
        "raw_box": {"begin": list(begin), "count": list(count)},
    }


def request_image(args, arrays):
    """The image sent for ``plan``'s plane: the raw block its ``raw_box``
    read (``arrays[0]``, ``count`` voxels z, y, x of ``dtype``, 0 past the
    data) laid out and resampled (``ai_annotate.plane_image``) and scaled to
    8-bit grey (``normalize_to_uint8``), as the server's read_plane and
    build_request make it. Answers its RGBA bytes, the plan's
    ``image_shape``, the grey in every channel."""
    planned = PlanePlan.from_dict(args["plan"])
    block = np.frombuffer(arrays[0], dtype=np.dtype(args["dtype"])).reshape(tuple(args["count"]))
    grey = ai_annotate.normalize_to_uint8(ai_annotate.plane_image(block, planned))
    return _rgba(np.repeat(grey[:, :, None], 3, axis=2))


def model_mask(args, arrays):
    """The mask in a model's reply, and the reply lined up with the plane
    (``ai_annotate.model_mask``). ``args``: ``shape`` (rows, cols) of the
    reply as decoded, ``sent`` and ``image`` (rows, cols of the image sent,
    padded square, and of the plane), ``target_rgb``; ``arrays``: the
    reply's RGBA bytes. Answers ``{}`` with the mask (a byte a pixel of the
    plane, 1 where painted) and the lined-up reply (RGBA)."""
    rows, cols = (int(s) for s in args["shape"])
    reply = np.frombuffer(arrays[0], dtype=np.uint8).reshape(rows, cols, 4)
    mask, aligned = ai_annotate.model_mask(reply, args["sent"], args["image"], args["target_rgb"])
    return {}, [mask.astype(np.uint8), _rgba(aligned)]


def prepare(args, arrays):
    """Nothing, once scipy.ndimage is imported, which ``stage`` labels the
    mask's objects with: the browser calls this while the model works, so
    scipy's first load (seconds, once a page) is over by the time the
    answer comes."""
    import scipy.ndimage  # noqa: F401

    return None


def stage(args, arrays):
    """What the review shows of a mask, as the server stages it: its objects
    on the write box (``ai_annotate.mask_to_write_shape``), the share of the
    box they mark, and the image sent with them tinted (``overlay``).
    ``args``: ``plan`` and ``target_rgb``; ``arrays``: the mask (a byte a
    pixel of the plan's image, nonzero where painted) and the image sent
    (RGBA). Answers ``mask_fraction`` with the objects (int32, the write
    box's in-plane shape, z, y, x order) and the overlay (RGBA)."""
    planned = PlanePlan.from_dict(args["plan"])
    rows, cols = planned.image_shape
    mask = np.frombuffer(arrays[0], dtype=np.uint8).reshape(rows, cols)
    image = np.frombuffer(arrays[1], dtype=np.uint8).reshape(rows, cols, 4)
    objects = ai_annotate.mask_to_write_shape(mask, planned)
    tinted = ai_annotate.overlay(image, planned.to_screen(objects > 0), args["target_rgb"])
    fraction = float((objects > 0).mean()) if objects.size else 0.0
    return {"mask_fraction": fraction}, [objects.astype(np.int32), _rgba(tinted)]


def accept(args, arrays):
    """What Accept writes into the write box: ``ai_annotate.accept_labels``,
    written over the box as ``fill.paint`` writes. ``args``: ``depth_axis``,
    ``shape`` (the write box's, z, y, x), ``dtype`` (the volume's),
    ``instance_target`` (the model is trained on instances), ``overwrite``,
    and ``beside``: whether the planes before and after the box along the
    depth axis are sent (false past the volume's edge). ``arrays``: the
    objects (int32, as ``stage`` answered them), the box as the volume holds
    it now, then the planes beside that are sent, each shaped as the box.
    Answers the counts (``n_foreground``, ``n_background``,
    ``n_overwritten``) with the box to write, or no array when nothing
    changes; or the refusal, when the ids run out."""
    shape, dtype, depth_axis = tuple(args["shape"]), np.dtype(args["dtype"]), int(args["depth_axis"])
    in_plane = tuple(n for a, n in enumerate(shape) if a != depth_axis)
    objects = np.frombuffer(arrays[0], dtype=np.int32).reshape(in_plane)
    existing = np.frombuffer(arrays[1], dtype=dtype).reshape(shape)
    sent = iter(arrays[2:])
    beside = [np.frombuffer(next(sent), dtype=dtype).reshape(shape) if present else None for present in args["beside"]]
    overwrite = bool(args["overwrite"])
    try:
        labels = ai_annotate.accept_labels(objects, existing, depth_axis, bool(args["instance_target"]), overwrite, beside)
    except ValueError as e:
        return {"refused": str(e)}
    after, (n_foreground, n_background, n_overwritten) = fill.paint(existing, labels, overwrite)
    counts = {"n_foreground": n_foreground, "n_background": n_background, "n_overwritten": n_overwritten}
    return counts, [] if after is None else [after]
