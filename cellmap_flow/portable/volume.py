"""Where a new annotation volume lies and what holds its labels: the planning
of ``finetune/session/volume.py`` (its module docstring says what a volume
is), on numbers alone, so that the server and the browser plan a volume the
same way. The server reads the raw level and the model's geometry
(``open_raw``, ``resolve_model_geometry``) and the browser its
``DatasetInfo`` and model manifest (``web/src/finetune/volume.ts``, through
``volume_calls.py``); both hand what they found to ``plan_volume`` here.

The grid. A model reads one level of the raw, the one ``select_level``'s
"floor" picks for its input voxel size (or, with Resample, the raw resampled
to that size), and both servers serve its prediction on that level's grid:
output voxel 0 starts at the level's corner, and an output voxel is the
level's voxel times the model's ratio of output to input voxel size. The
volume lies on that grid, one chunk per model output block, so that a chunk
of paint is a block of the prediction it is painted over, and it covers the
level's whole extent, a partial output voxel at the far end rounded up, then
padded to whole chunks.

A volume painted before any model (``plan_paint_volume``, a "paint-first"
volume) lies on the raw's own grid instead, at a voxel size the user picks
(``shown_level`` gives the default: the level the viewer shows), in chunks
PAINT_FIRST_CHUNK voxels a side. It is for no model (its ``model_name`` is
None), so it is not trained on (``training_refusal``): Which model to start
from ranks the models on its paint, and Use this model carries the paint onto
the chosen model's grid (``carry.py``), into a volume that is trained on.
"""

import math
from collections.abc import Mapping

import numpy as np

# The training targets whose annotation volumes hold an id per object
# rather than foreground and background alone.
INSTANCE_TARGETS = ("affinities", "flows")

# Within this many voxels of a whole number, a block size in nm divided by its
# voxel size is that number: the float noise of nm (10.48 nm voxels) is not a
# fraction of a voxel.
WHOLE_VOXEL_TOLERANCE = 1e-6

# A paint-first volume's chunk side, in its voxels: the volume has no model
# whose output block a chunk would be, and 64^3 uint8 voxels (256 KiB) is a
# chunk Neuroglancer writes at every stroke without a pause. An axis the
# volume is thinner along gets a chunk as thin as the volume.
PAINT_FIRST_CHUNK = 64

# Why a volume painted before any model is not trained on, said in both modes
# (training_refusal); the Finetune tab's Which model to start from is where
# the paint gets a model's grid.
PAINT_FIRST_TRAINING = (
    "This volume was painted before a model was chosen, so it is on no model's grid: "
    "under Which model to start from, rank the models on your paint and click Use this model "
    "on the one to finetune, which copies the paint onto its grid, then train."
)


def annotation_dtype(output_type) -> str:
    """The labels' dtype of a new volume for a model that trains on ``output_type``.

    uint8 (ids up to 255) for a model trained on foreground and background;
    uint16 (up to 65535) for one trained on instances (affinities, Cellpose's
    flows), whose seeds number each object: 282 of a Cellpose model's masks
    in one box did not fit a uint8 volume.
    """
    return "uint16" if output_type in INSTANCE_TARGETS else "uint8"


def volume_corner_nm(dataset_offset_nm, output_voxel_size) -> np.ndarray:
    """The world position of an annotation volume's voxel-0 lower corner, in nm.

    ``dataset_offset_nm`` (a root attr of every volume) is also written as the
    volume's OME-NGFF translation, and a translation is voxel 0's *centre*.
    Neuroglancer drew the volume that way while it was painted, so that is
    where the labels are: read as a corner, the value puts every label half
    an annotation voxel away from where it was drawn. None is an offset of 0.
    """
    offset = np.zeros(len(output_voxel_size)) if dataset_offset_nm is None else np.asarray(dataset_offset_nm, float)
    return offset - np.asarray(output_voxel_size, float) / 2


def whole_voxels(values, what: str) -> list:
    """``values`` (a block size in voxels, as floats) as whole numbers.

    A ValueError names ``what`` when one is not whole: truncating a block of
    177.6 voxels to 177 would make a volume whose chunks are not the model's
    blocks.
    """
    out = []
    for value in map(float, values):
        if not math.isfinite(value) or abs(value - round(value)) > WHOLE_VOXEL_TOLERANCE:
            raise ValueError(f"{what} ({[float(v) for v in values]} voxels) is not whole voxels")
        out.append(round(value))
    return out


def _field(model, name):
    return model.get(name) if isinstance(model, Mapping) else getattr(model, name, None)


def _numbers(values, what: str, positive: bool = False) -> list:
    """``values`` as a list of finite floats; a ValueError naming ``what`` otherwise."""
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be numbers, not {values!r}") from None
    if not all(math.isfinite(v) and (v > 0 or not positive) for v in out):
        raise ValueError(f"{what} must be {'positive' if positive else 'finite'} numbers, not {out}")
    return out


def block_shapes(model) -> tuple:
    """``(input_size, output_size)``, a model's blocks in voxels.

    ``model`` (a mapping, or an object with these attributes) has
    ``input_voxel_size`` and ``output_voxel_size``, and either
    ``input_shape``/``output_shape`` in voxels or ``read_shape``/
    ``write_shape`` in nm, which is what a server reports.
    """
    if _field(model, "input_shape") is not None:
        return (whole_voxels(_numbers(_field(model, "input_shape"), "input_shape", positive=True), "the model's input block"),
                whole_voxels(_numbers(_field(model, "output_shape"), "output_shape", positive=True), "the model's output block"))
    claimed_in = _numbers(_field(model, "input_voxel_size"), "input_voxel_size", positive=True)
    claimed_out = _numbers(_field(model, "output_voxel_size"), "output_voxel_size", positive=True)
    read = _numbers(_field(model, "read_shape"), "read_shape", positive=True)
    write = _numbers(_field(model, "write_shape"), "write_shape", positive=True)
    return (whole_voxels([r / v for r, v in zip(read, claimed_in)], "the model's input block"),
            whole_voxels([w / v for w, v in zip(write, claimed_out)], "the model's output block"))


def _level_numbers(level) -> tuple:
    """``(voxel, corner, shape)`` of a raw level as the plans take it (see
    ``plan_volume``), checked: positive voxel sizes, a finite corner, a
    whole shape, one of each per axis."""
    voxel = _numbers(_field(level, "voxel"), "the level's voxel size", positive=True)
    corner = _numbers(_field(level, "corner"), "the level's corner")
    shape = whole_voxels(_numbers(_field(level, "shape"), "the level's shape"), "the level's shape")
    if not len(voxel) == len(corner) == len(shape):
        raise ValueError(f"the level's voxel size, corner and shape ({len(voxel)}, {len(corner)} and {len(shape)} axes) "
                         "do not have the same axes")
    return voxel, corner, shape


def _extent_in(shape, voxel, output_voxel) -> list:
    """A level of ``shape`` voxels of size ``voxel`` in voxels of size
    ``output_voxel``, rounded up so a partial voxel at the far end is
    covered, after rounding off the float noise of nm."""
    return [math.ceil(round(n * v / w, 6)) for n, v, w in zip(shape, voxel, output_voxel)]


def plan_volume(level, model) -> dict:
    """Where a new volume lies for ``model`` over the raw ``level`` it reads.

    ``level`` is that level as the model reads it, z, y, x: ``voxel`` its
    voxel size in nm, ``corner`` its voxel-0 lower corner in nm (not the OME
    translation, which is the centre), ``shape`` its voxels. On the server
    it is the level ``select_level`` picks for the model's input voxel size,
    at its own voxel size, or the raw resampled to the model's input voxel
    size when Resample is on; in the browser, the level ``chooseRawLevel``
    picks. ``model`` is as ``block_shapes`` takes it.

    Answers the plan's fields of ``VolumeGeometry``, as JSON-ready lists:
    ``input_voxel_size`` the level's own, ``output_voxel_size`` the level's
    times the model's ratio of output to input voxel size (where both
    servers serve the prediction), the model's own as the claimed sizes,
    ``chunk_size`` the model's output block and ``input_size`` its input
    block, ``dataset_offset_nm`` voxel 0's centre (the level's corner plus
    half an output voxel) and ``dataset_shape_voxels`` the level's extent in
    output voxels, rounded up, then padded to whole chunks.

    The ratio is the model's: snapping the output voxel size to a raw level
    of its own could break it (on levels of 10, 20 and 40 nm, a model reading
    8 nm and writing 16 read the 10 nm level and wrote a 10 nm volume, while
    its server serves the prediction at 20), and the volume's chunks were
    then not the model's blocks.
    """
    voxel, corner, shape = _level_numbers(level)
    # The model's own, as plain numbers (ints kept ints): the root attrs are
    # written from them as they are.
    claimed_in = np.asarray(_field(model, "input_voxel_size")).tolist()
    claimed_out = np.asarray(_field(model, "output_voxel_size")).tolist()
    model_in = _numbers(claimed_in, "the model's input_voxel_size", positive=True)
    model_out = _numbers(claimed_out, "the model's output_voxel_size", positive=True)
    if not len(voxel) == len(model_in) == len(model_out):
        raise ValueError(f"the level ({len(voxel)} axes) and the model ({len(model_in)}) do not have the same axes")
    input_size, output_size = block_shapes(model)

    # In this order, as the browser's served level computes it
    # (infer/geometry.ts outputLevel), so the two agree to the last bit.
    output_voxel = [v * (o / i) for v, o, i in zip(voxel, model_out, model_in)]
    offset = [c + w / 2 for c, w in zip(corner, output_voxel)]
    covered = _extent_in(shape, voxel, output_voxel)
    padded = [-(-n // c) * c for n, c in zip(covered, output_size)]
    return {
        "output_voxel_size": output_voxel,
        "input_voxel_size": voxel,
        "claimed_output_voxel_size": claimed_out,
        "claimed_input_voxel_size": claimed_in,
        "chunk_size": output_size,
        "input_size": input_size,
        "dataset_offset_nm": offset,
        "dataset_shape_voxels": padded,
    }


# A level whose finest axis is at most this share coarser than a screen pixel
# is still one the viewer shows at that zoom (the slack viewer/raw.py's
# _view_contrast_range gives it, for the float noise of a zoom in nm).
SHOWN_LEVEL_SLACK = 1.01


def shown_level(voxel_sizes, pixel_nm=None) -> int:
    """The index in ``voxel_sizes`` (each a raw level's voxel size in nm,
    finest first) of the level the viewer shows when one screen pixel spans
    ``pixel_nm``: the coarsest whose finest axis is no coarser than a pixel,
    and the finest when zoomed in past it, as Neuroglancer picks the level
    to draw (viewer/raw.py's _view_contrast_range reads it so). The finest
    when the zoom is not known (None: no viewer, or none that has reported
    one yet). A paint-first volume's voxel size is this level's unless the
    user picks another. A ValueError when there is no level."""
    finest_axes = [min(_numbers(size, "a level's voxel size", positive=True)) for size in voxel_sizes]
    if not finest_axes:
        raise ValueError("the raw has no levels to choose a voxel size from")
    finest = min(range(len(finest_axes)), key=lambda i: (finest_axes[i], i))
    try:
        pixel = float(pixel_nm)
    except (TypeError, ValueError):
        return finest
    if not (math.isfinite(pixel) and pixel > 0):
        return finest
    shown = [i for i, axis in enumerate(finest_axes) if axis <= pixel * SHOWN_LEVEL_SLACK]
    return max(shown, key=lambda i: (finest_axes[i], -i)) if shown else finest


def plan_paint_volume(level, voxel_size) -> dict:
    """Where a new paint-first volume lies: over the raw ``level`` it is
    read from, on a grid of ``voxel_size`` nm (z, y, x), for no model.

    ``level`` is as ``plan_volume`` takes it: on the server the level
    ``raw_level`` opens for ``voxel_size`` (the level at that size, else the
    finest not coarser than it, at its own voxel size; with Resample on, the
    raw resampled to it), in the browser the one ``chooseRawLevel`` picks.
    The grid starts at the level's corner: at a level's own voxel size its
    voxels are the level's (a size within float noise of it is taken as
    it), and at another (one the user typed) the level's extent is covered
    in voxels of that size, a partial voxel at the far end rounded up. The
    chunks are PAINT_FIRST_CHUNK voxels a side, as thin as the volume along
    an axis it is thinner along, and the extent is padded to whole chunks.

    Answers ``plan_volume``'s fields: ``output_voxel_size`` the grid's,
    ``input_voxel_size`` the level's (what the raw under the paint is read
    at: Which model to start from's window of it), no claimed sizes (there
    is no model to claim any), ``chunk_size``, ``input_size`` a chunk in the
    level's voxels, rounded up, ``dataset_offset_nm`` voxel 0's centre and
    ``dataset_shape_voxels`` the padded extent. A ValueError says what is
    wrong with ``level`` or ``voxel_size``.
    """
    voxel, corner, shape = _level_numbers(level)
    asked = _numbers(voxel_size if voxel_size is not None else [], "the voxel size", positive=True)
    if len(asked) != len(voxel):
        raise ValueError(f"the voxel size ({len(asked)} axes) and the raw's level ({len(voxel)}) do not have the same axes")
    size = voxel if np.allclose(asked, voxel, rtol=WHOLE_VOXEL_TOLERANCE, atol=0) else asked
    covered = _extent_in(shape, voxel, size)
    chunk = [max(1, min(PAINT_FIRST_CHUNK, n)) for n in covered]
    return {
        "output_voxel_size": size,
        "input_voxel_size": voxel,
        "claimed_output_voxel_size": None,
        "claimed_input_voxel_size": None,
        "chunk_size": chunk,
        "input_size": _extent_in(chunk, size, voxel),
        "dataset_offset_nm": [c + w / 2 for c, w in zip(corner, size)],
        "dataset_shape_voxels": [-(-n // c) * c for n, c in zip(covered, chunk)],
    }


def training_refusal(volume):
    """Why ``volume`` (its record or root attrs: a mapping, or an object
    with ``model_name``) is not trained on, or None when it can be:
    PAINT_FIRST_TRAINING for a paint-first volume, which names no model,
    so no model's blocks are its chunks to train on."""
    return PAINT_FIRST_TRAINING if not _field(volume, "model_name") else None
