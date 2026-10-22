"""Good regions on numbers alone: the region a mark makes, its size, which
stored entries are regions, and where a region anchors the model on a
volume's grid, so that the server and the browser make, draw and place a
region the same way. ``dashboard/routes/finetune/good_regions.py`` says what
good regions are for; its routes and the server's trainer
(``finetune/data/sampler.py``) call these in their process, and the
browser's routes and trainer (``web/src/backend/routes/good-regions.ts``,
``web/src/finetune/run.ts``) call them through ``volume_calls.py``.

A region is ``{"id", "label", "offset_nm", "shape_nm"}``: a box in world nm,
z, y, x, its lower corner and its size, as ``good_regions.json`` keeps it.
"""

import math
import uuid
from collections.abc import Mapping
from numbers import Real
from typing import Optional

import numpy as np

# Used when neither the request nor the volume says how big a region
# should be. One model *output* patch is the natural unit: it is what you can
# actually see and judge on screen, and it is the region a training patch
# computes loss over. Sizing to the input field of view instead would claim
# the model is right across 2848nm when you only looked at the middle 896nm
# of it -- and would not match annotation crops, which are already write_shape.
DEFAULT_REGION_SIZE_NM = 896.0


def _is_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _triple(value, positive: bool) -> Optional[np.ndarray]:
    """``value`` as three finite floats (each positive with ``positive``), or None."""
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        return None
    if not all(_is_number(v) and math.isfinite(v) and (v > 0 or not positive) for v in value):
        return None
    return np.asarray(value, dtype=float)


def default_size_nm(output_size=None, output_voxel_size=None) -> list:
    """One model output patch of a volume, in nm (see DEFAULT_REGION_SIZE_NM):
    ``output_size`` voxels of ``output_voxel_size`` nm, or 896 nm a side when
    either is missing or not numbers."""
    if output_size and output_voxel_size:
        try:
            return (np.asarray(output_size, dtype=float) * np.asarray(output_voxel_size, dtype=float)).tolist()
        except (TypeError, ValueError):
            pass
    return [DEFAULT_REGION_SIZE_NM] * 3


def region_size_nm(size_nm, output_size=None, output_voxel_size=None) -> list:
    """The size a mark asks for, in nm, z, y, x.

    No size (None, 0, an empty list) is one output patch of the volume
    (``default_size_nm`` of ``output_size`` and ``output_voxel_size``);
    one number, alone or in a list, is the same along every axis; else three
    numbers. ValueError ("size_nm must be positive") for anything else: a
    size that is not a number, not finite or not positive, or of another
    length.
    """
    if not size_nm:
        return default_size_nm(output_size, output_voxel_size)
    values = list(size_nm) if isinstance(size_nm, (list, tuple)) else [size_nm]
    if len(values) == 1:
        values = values * 3
    size = _triple(values, positive=True)
    if size is None:
        raise ValueError("size_nm must be positive")
    return size.tolist()


def new_region(centre_nm, size_nm, count: int, label=None, region_id: Optional[str] = None) -> dict:
    """The region a mark makes: a box of ``size_nm`` centred on ``centre_nm``,
    labelled ``label`` (a non-empty string), or ``good-<count + 1>`` after
    ``count`` regions; its id eight hex digits unless ``region_id`` is given."""
    centre = np.asarray(centre_nm, dtype=float)
    size = np.asarray(size_nm, dtype=float)
    return {
        "id": region_id or uuid.uuid4().hex[:8],
        "label": label if isinstance(label, str) and label else f"good-{count + 1}",
        "offset_nm": (centre - size / 2).tolist(),
        "shape_nm": size.tolist(),
    }


def region_box_nm(region) -> Optional[tuple]:
    """``(offset_nm, shape_nm)`` of a stored region, as float arrays, or None
    when the entry is not a region: a mapping whose offset_nm and shape_nm
    are three finite numbers each, its shape positive. Its id and label are
    not needed to place it."""
    if not isinstance(region, Mapping):
        return None
    offset, shape = _triple(region.get("offset_nm"), positive=False), _triple(region.get("shape_nm"), positive=True)
    return None if offset is None or shape is None else (offset, shape)


def region_centre_voxels(region, corner_nm, voxel_nm, shape_voxels) -> tuple:
    """Where a region's centre lies on a volume's grid: ``(where, centre)``.

    ``corner_nm`` is the volume's voxel-0 lower corner (``volume_corner_nm``),
    ``voxel_nm`` its output voxel size and ``shape_voxels`` its shape. The
    centre is in voxels from the corner (voxel v spans [v, v + 1)). ``where``
    is "malformed" (not a region, ``region_box_nm``; the centre is None),
    "outside" (a region marked on another volume, which would anchor the
    model to padding) or "inside".
    """
    box = region_box_nm(region)
    if box is None:
        return "malformed", None
    offset, shape = box
    centre = (offset + shape / 2 - np.asarray(corner_nm, dtype=float)) / np.asarray(voxel_nm, dtype=float)
    if np.any(centre < 0) or np.any(centre >= np.asarray(shape_voxels)):
        return "outside", centre
    return "inside", centre


def anchor_boxes(regions, corner_nm, voxel_nm, shape_voxels, block) -> dict:
    """The box of the volume's voxels each region anchors, and which were skipped.

    The box is the rehearsal patch the server's trainer cuts for the region:
    one output ``block`` whose start is the region's centre snapped so that
    the block is whole voxels (``PatchReader.snap``: ``floor(c - block/2 +
    1/2)``), clipped to the volume. Answers ``{"boxes": [{"region", "lo",
    "hi"}], "malformed": [...], "outside": [...]}``, each region by its
    index in ``regions`` and each box [lo, hi) in voxels, z, y, x.
    """
    boxes, malformed, outside = [], [], []
    shape = np.asarray(shape_voxels, dtype=int)
    size = np.asarray(block, dtype=int)
    for index, region in enumerate(regions):
        where, centre = region_centre_voxels(region, corner_nm, voxel_nm, shape)
        if where != "inside":
            (malformed if where == "malformed" else outside).append(index)
            continue
        # A centre inside the volume leaves some of its block inside too.
        start = np.floor(centre - size / 2 + 0.5).astype(int)
        lo, hi = np.maximum(start, 0), np.minimum(start + size, shape)
        boxes.append({"region": index, "lo": lo.tolist(), "hi": hi.tolist()})
    return {"boxes": boxes, "malformed": malformed, "outside": outside}
