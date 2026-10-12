"""Which plane of the data an AI annotation reads, and which voxels it writes.

A click (or the view centre) picks a point in world nm and the plane the
user is looking at picks the depth axis. ``plan_plane`` turns those into a
``PlanePlan``: a square field of view around the point, snapped to the
annotation volume's voxels and clipped to the volume (the write box, one
annotation voxel thick along the depth axis), and the raw read that covers
the same physical area at the model's input voxel size (one input voxel
thick, at the point's depth). So whatever the model paints on the image
lands on the annotation voxels under it.

Annotation voxels are placed as ``finetune.session.volume`` places them:
the volume's ``dataset_offset_nm`` is voxel 0's *centre*, and a voxel is in
a box when its centre is (``fill.box_voxels``). The raw is read on its own
grid (``io.geometry``), from the voxel that holds the first pixel's centre.

Axes are z, y, x throughout, as everywhere in cellmap-flow. An image's rows
are the first in-plane axis in that order and its columns the second: y and
x for a z-depth (XY) plane, z and x for XZ, z and y for YZ.
"""

from dataclasses import asdict, dataclass

import numpy as np

from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.finetune.session.fill import box_voxels
from cellmap_flow.finetune.session.volume import volume_corner_nm
from cellmap_flow.io.geometry import Box
from cellmap_flow.io.metadata import snap_integral

AXIS_NAMES = ("z", "y", "x")

# The longest side of an image sent to a model. A hosted image model bills
# an image by the image, not by its pixels (gemini-3-pro-image: ~560 tokens
# whatever the size), and resizes what it gets to its own working size, so
# more pixels than this cost upload time and gain nothing.
MAX_IMAGE_SIDE = 1024

# The plane each depth axis leaves, named by the data axes it shows.
PLANE_NAMES = ("XY", "XZ", "YZ")

# Neuroglancer names its cross-section panels after display-dimension
# *slots*, not after the dimensions' names: the "xy" panel shows display
# dimensions 0 and 1 and looks along 2, "xz" (rotated about the first) looks
# along 1, "yz" (rotated about the second) along 0 (data_panel_layout.ts's
# AXES_RELATIVE_ORIENTATION). The display dimensions are the first three of
# the viewer's coordinate space unless set, and cellmap-flow's viewers name
# theirs z, y, x in that order, so the "xy" panel is a z-y slice looking
# along x, and the plane cellmap-flow calls XY is neuroglancer's "yz" panel.
_LAYOUT_DEPTH_SLOT = {"xy": 2, "xz": 1, "yz": 0}


def _layout_type(layout):
    """The panel name of a neuroglancer layout: a string, a dict or a state object."""
    if isinstance(layout, dict):
        layout = layout.get("type")
    elif layout is not None and not isinstance(layout, str):
        layout = getattr(layout, "type", None)
    if not isinstance(layout, str):
        return None
    name = layout.lower()
    return name[: -len("-3d")] if name.endswith("-3d") else name


def depth_axis_from_layout(layout, display_dimensions=None) -> int:
    """The data axis (0 z, 1 y, 2 x) the panel named by ``layout`` looks along.

    ``layout`` is neuroglancer's layout as a string ("xy", "xz-3d"...), a
    dict with a "type", or the state's layout object. ``display_dimensions``
    are the names of the viewer's display dimensions in order (z, y, x when
    None, as cellmap-flow's viewers have them); see ``_LAYOUT_DEPTH_SLOT``
    for why they matter. Only a single-plane layout says which plane the
    user is looking at: neuroglancer keeps one layout for the whole viewer,
    so in "4panel" or "3d" the panel a click landed in is not known. Those
    (and anything unrecognised) get the "xy" panel's axis, the panel
    neuroglancer puts first. A rotated cross-section is not followed.
    """
    names = tuple(display_dimensions) if display_dimensions else AXIS_NAMES
    slot = _LAYOUT_DEPTH_SLOT.get(_layout_type(layout), _LAYOUT_DEPTH_SLOT["xy"])
    if slot < len(names) and names[slot] in AXIS_NAMES:
        return AXIS_NAMES.index(names[slot])
    return 0


def depth_axis_for_view(viewer_state) -> int:
    """``depth_axis_from_layout`` for a neuroglancer ``ViewerState``.

    Takes the display dimensions the state sets, else the first three of
    its coordinate space, as neuroglancer does.
    """
    display = list(getattr(viewer_state, "display_dimensions", None) or [])
    if not display:
        dimensions = getattr(viewer_state, "dimensions", None)
        display = list(getattr(dimensions, "names", None) or [])[:3]
    return depth_axis_from_layout(getattr(viewer_state, "layout", None), display or None)


def _floats(values):
    return tuple(float(v) for v in values)


def _ints(values):
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class PlanePlan:
    """One plane to annotate: what is read, what is sent and what is written.

    All three-element tuples are z, y, x. ``raw_offset_nm`` and
    ``raw_shape_nm`` are the world box the raw is read over (lower corner
    and size): in plane, the write box's extent; along the depth axis, one
    input voxel centred on the point. ``image_shape`` is the (rows, cols)
    of the image sent, ``resolution_nm`` its pixel size in nm (square
    pixels, which a model needs to see shapes as they are). ``write_lo`` and
    ``write_hi`` are the annotation voxels written, ``[lo, hi)``, clipped to
    the volume and one voxel thick along the depth axis.
    """

    depth_axis: int
    point_nm: tuple
    raw_offset_nm: tuple
    raw_shape_nm: tuple
    image_shape: tuple
    input_voxel_size: tuple
    write_lo: tuple
    write_hi: tuple
    resolution_nm: float

    @property
    def plane_axes(self) -> tuple:
        """The two in-plane axes, z, y, x order: the image's rows, then its columns."""
        return tuple(a for a in range(3) if a != self.depth_axis)

    @property
    def plane_name(self) -> str:
        """"XY", "XZ" or "YZ": the data axes the plane shows."""
        return PLANE_NAMES[self.depth_axis]

    @property
    def write_shape(self) -> tuple:
        """The write box's in-plane shape in annotation voxels, (rows, cols) as the image's."""
        return tuple(int(self.write_hi[a] - self.write_lo[a]) for a in self.plane_axes)

    @property
    def click_px(self) -> tuple:
        """``(row, col)`` of the point in the image sent."""
        position = []
        for side, a in zip(self.image_shape, self.plane_axes):
            fraction = (self.point_nm[a] - self.raw_offset_nm[a]) / self.raw_shape_nm[a]
            position.append(int(np.clip(np.floor(fraction * side), 0, side - 1)))
        return tuple(position)

    def to_dict(self) -> dict:
        """The plan as JSON-safe lists, for a staging's meta.json."""
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "PlanePlan":
        """The plan ``to_dict`` wrote."""
        return cls(
            depth_axis=int(data["depth_axis"]),
            point_nm=_floats(data["point_nm"]),
            raw_offset_nm=_floats(data["raw_offset_nm"]),
            raw_shape_nm=_floats(data["raw_shape_nm"]),
            image_shape=_ints(data["image_shape"]),
            input_voxel_size=_floats(data["input_voxel_size"]),
            write_lo=_ints(data["write_lo"]),
            write_hi=_ints(data["write_hi"]),
            resolution_nm=float(data["resolution_nm"]),
        )


def plan_plane(volume: dict, volume_shape, point_nm, depth_axis: int, crop_size_px: int) -> PlanePlan:
    """The plane through ``point_nm`` (world nm, z, y, x) across ``depth_axis``.

    ``volume`` is the annotation volume's record (``output_voxel_size``,
    ``input_voxel_size``, ``dataset_offset_nm``), ``volume_shape`` its
    ``annotation/s0`` shape. The field of view is square in nm,
    ``crop_size_px`` annotation voxels along the finer in-plane axis (both,
    for isotropic voxels), centred on the point; its write box is the
    annotation voxels whose centres lie in it, clipped to the volume, in the
    annotation voxel that holds the point along the depth axis.

    The raw is read over the clipped box's extent, so the image and the box
    cover the same physical area. The image's pixels are the finer in-plane
    input voxel size, square; when that makes a side longer than
    ``MAX_IMAGE_SIDE`` the pixels are made larger to fit.

    Raises AIAnnotateError ("refused") when the point is outside the volume:
    a field clipped to an edge it does not reach is not what was aimed at.
    """
    if depth_axis not in (0, 1, 2):
        raise ValueError(f"depth_axis must be 0, 1 or 2 (z, y, x), got {depth_axis!r}")
    if int(crop_size_px) < 1:
        raise ValueError(f"crop_size_px must be at least 1, got {crop_size_px!r}")
    output_voxel_size = np.asarray(volume["output_voxel_size"], dtype=float)
    input_voxel_size = np.asarray(volume["input_voxel_size"], dtype=float)
    shape = np.asarray(volume_shape, dtype=int)
    corner = volume_corner_nm(volume.get("dataset_offset_nm"), output_voxel_size)
    point = np.asarray(point_nm, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"point_nm must be three finite numbers (z, y, x), got {point_nm!r}")

    voxel = np.floor(snap_integral((point - corner) / output_voxel_size)).astype(int)
    if np.any(voxel < 0) or np.any(voxel >= shape):
        extent = [f"{c:g}..{c + n * v:g}" for c, n, v in zip(corner, shape, output_voxel_size)]
        raise AIAnnotateError(
            "refused",
            f"The point ({', '.join(f'{p:g}' for p in point)}) nm is outside the annotation volume "
            f"(z, y, x {', '.join(extent)} nm): move the view onto the data and try again.",
            http_status=400,
        )

    plane = [a for a in range(3) if a != depth_axis]
    side_nm = int(crop_size_px) * float(output_voxel_size[plane].min())
    offset = point - side_nm / 2
    size = np.full(3, side_nm)
    # Along the depth axis, exactly the voxel that holds the point: a box
    # one voxel long whose only centre is that voxel's.
    offset[depth_axis] = corner[depth_axis] + voxel[depth_axis] * output_voxel_size[depth_axis]
    size[depth_axis] = output_voxel_size[depth_axis]
    box = box_voxels(volume, offset, size, shape)
    if box is None:  # cannot happen with the point inside, but never write an empty box
        raise AIAnnotateError("refused", "No annotation voxels lie around the point.", http_status=400)
    lo, hi = box

    raw_offset = corner + lo * output_voxel_size
    raw_shape = (hi - lo) * output_voxel_size
    raw_offset[depth_axis] = point[depth_axis] - input_voxel_size[depth_axis] / 2
    raw_shape[depth_axis] = input_voxel_size[depth_axis]

    extent = raw_shape[plane]
    resolution = max(float(input_voxel_size[plane].min()), float(extent.max()) / MAX_IMAGE_SIDE)
    image_shape = np.maximum(np.round(snap_integral(extent / resolution)), 1).astype(int)
    return PlanePlan(
        depth_axis=int(depth_axis),
        point_nm=_floats(point),
        raw_offset_nm=_floats(snap_integral(raw_offset)),
        raw_shape_nm=_floats(snap_integral(raw_shape)),
        image_shape=_ints(image_shape),
        input_voxel_size=_floats(input_voxel_size),
        write_lo=_ints(lo),
        write_hi=_ints(hi),
        resolution_nm=float(snap_integral(resolution)),
    )


def raw_box(plan: PlanePlan, voxel_size, corner) -> Box:
    """The raw voxels to read for ``plan``, on a raw grid of ``voxel_size`` from ``corner``.

    Each axis starts at the voxel that holds the first sample's centre (half
    a voxel in from the box's corner) and runs for as many voxels as fit in
    the box, at least one: so a box on the grid reads exactly its voxels,
    and the depth axis reads the voxel that holds the point.
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    corner = np.asarray(corner, dtype=float)
    first_centre = np.asarray(plan.raw_offset_nm, dtype=float) + voxel_size / 2
    begin = np.floor(snap_integral((first_centre - corner) / voxel_size)).astype(int)
    count = np.maximum(np.round(snap_integral(np.asarray(plan.raw_shape_nm) / voxel_size)), 1).astype(int)
    return Box(_ints(begin), _ints(count))
