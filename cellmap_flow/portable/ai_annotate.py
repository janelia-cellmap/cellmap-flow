"""AI-assisted annotation's maths, which both modes run: which plane of the
data an annotation reads and which voxels it writes, the image the model is
sent, the mask in the image it sends back, and the labels that mask becomes.

The server runs it in its process (``ai_annotate/geometry.py``,
``pipeline.py``, ``mask_extraction.py``, ``staging.py`` and the Vertex
backend call it, around their zarr reads and PNG files), the browser in
chunkmirage's Pyodide worker (``portable/ai_annotate_calls.py``, around the
engine's raw reads and the paint store). Nothing here reads data, calls a
model or sees a credential: what is sent and to whom stays with each mode's
provider.

The plane. A click (or the view centre) picks a point in world nm and the
plane the user is looking at picks the depth axis. ``plan_plane`` turns
those into a ``PlanePlan``: a square field of view around the point,
snapped to the annotation volume's voxels and clipped to the volume (the
write box, one annotation voxel thick along the depth axis), and the raw
read that covers the same physical area at the model's input voxel size (one
input voxel thick, at the point's depth). So whatever the model paints on
the image lands on the annotation voxels under it. Annotation voxels are
placed as ``finetune.session.volume`` places them: the volume's
``dataset_offset_nm`` is voxel 0's *centre*, and a voxel is in a box when
its centre is (``fill.box_voxels``). The raw is read on its own grid, from
the voxel that holds the first pixel's centre (``raw_box``).

Axes are z, y, x throughout, as everywhere in cellmap-flow, except in the
image sent: that is laid out as the viewer's panel shows the plane
(``to_screen``), so the model, and the user reviewing its answer, see what
was on screen. Which axis runs down it and which across, and whether across
runs backwards, depend on the viewer's display dimensions (``screen_axes``):
cellmap-flow's viewers show x, y, z, so the XY plane is drawn as in z, y, x
order, the XZ plane too, and the YZ plane transposed and mirrored.

The images are resampled by Pillow (``resize``, ``nearest``, and the
model's reply in ``model_mask``), in both modes: the server's installed
Pillow, and in the browser the one Pyodide ships, which chunkmirage's worker
loads the first time a call imports it (scipy, for the mask's objects,
likewise).
"""

from dataclasses import asdict, dataclass

import numpy as np

from cellmap_flow.portable.fill import box_voxels, counts_up, seed_labels, volume_corner_nm

AXIS_NAMES = ("z", "y", "x")

# The longest side of an image sent to a model. A hosted image model bills
# an image by the image, not by its pixels (gemini-3-pro-image: ~560 tokens
# whatever the size), and resizes what it gets to its own working size, so
# more pixels than this cost upload time and gain nothing.
MAX_IMAGE_SIDE = 1024

# The plane each depth axis leaves, named by the data axes it shows.
PLANE_NAMES = ("XY", "XZ", "YZ")

# How strongly the review's overlay tints the mask with the target colour.
OVERLAY_ALPHA = 0.5

# How neuroglancer's panels draw their plane, by the display-dimension slot
# each looks along: the slot drawn down, the slot drawn across, and whether
# across runs backwards. The "xy" panel (along slot 2) draws display
# dimension 0 across and 1 down; "xz" (along 1) turns it a quarter about the
# first, so 0 across and 2 down; "yz" (along 0) a quarter about the second,
# so 1 down and 2 across, increasing to the left (AXES_RELATIVE_ORIENTATION's
# quaternions in neuroglancer's data_panel_layout.ts). The image sent was z,
# y, x order before, so planes went to the model, and back to the user's
# review, transposed or mirrored from what was on screen.
_PANEL_SLOTS = {2: (1, 0, False), 1: (2, 0, False), 0: (1, 2, True)}

# How close (relative, and at least absolutely) a value must be to a whole
# number to be that number (io.metadata.snap_integral's tolerance).
_INTEGRAL_TOLERANCE = 1e-6


class Refused(Exception):
    """A plane the point cannot have, outside the volume or the data: the
    message says so, for the user (the server raises it as an
    ``AIAnnotateError("refused")``, the browser's calls answer it)."""


def snap_integral(values):
    """``values`` as floats, with near-whole numbers made exactly whole
    (``io.metadata.snap_integral``): voxel arithmetic in nm leaves
    511.99999999 where 512 is meant, and a floor or a round after it must
    not see the difference."""
    arr = np.asarray(values, dtype=float)
    rounded = np.round(arr)
    close = np.abs(arr - rounded) <= _INTEGRAL_TOLERANCE * np.maximum(1.0, np.abs(arr))
    return np.where(close, rounded, arr)


# ---------------------------------------------------------------------------
# The screen layout
# ---------------------------------------------------------------------------


def display_names(display_dimensions=None) -> tuple:
    """``display_dimensions`` when they are z, y and x in some order, else
    z, y, x: the display dimensions of a viewer that sets none over
    cellmap-flow's coordinate space, and of a plan saved before plans kept
    them."""
    names = tuple(display_dimensions or ())
    return names if sorted(names) == sorted(AXIS_NAMES) else AXIS_NAMES


def screen_axes(depth_axis, display_dimensions=None) -> tuple:
    """``(down, across, backwards)``: the data axes (0 z, 1 y, 2 x) the panel
    looking along ``depth_axis`` draws down and across, and whether across
    runs backwards, in a viewer whose display dimensions are
    ``display_dimensions``, as ``display_names`` gives them (a plan's), or
    None for z, y, x."""
    names = tuple(display_dimensions or AXIS_NAMES)
    down, across, backwards = _PANEL_SLOTS[names.index(AXIS_NAMES[int(depth_axis)])]
    return AXIS_NAMES.index(names[down]), AXIS_NAMES.index(names[across]), backwards


def _screen(depth_axis, display_dimensions=None):
    down, across, backwards = screen_axes(depth_axis, display_dimensions)
    return down > across, backwards


def screen_shape(shape, depth_axis, display_dimensions=None) -> tuple:
    """A plane's ``(rows, cols)`` in z, y, x order, as the panel draws it
    (``screen_axes``)."""
    transposed, _ = _screen(depth_axis, display_dimensions)
    rows, cols = (int(s) for s in shape)
    return (cols, rows) if transposed else (rows, cols)


def to_screen(plane, depth_axis, display_dimensions=None) -> np.ndarray:
    """``plane`` (rows and columns its two axes in z, y, x order) as the panel
    looking along ``depth_axis`` draws it, in a viewer showing
    ``display_dimensions`` (``screen_axes``)."""
    transposed, backwards = _screen(depth_axis, display_dimensions)
    plane = np.asarray(plane)
    if transposed:
        plane = np.swapaxes(plane, 0, 1)
    return plane[:, ::-1] if backwards else plane


def from_screen(image, depth_axis, display_dimensions=None) -> np.ndarray:
    """``to_screen`` undone: an image as the panel draws it, in z, y, x order."""
    transposed, backwards = _screen(depth_axis, display_dimensions)
    image = np.asarray(image)
    if backwards:
        image = image[:, ::-1]
    return np.swapaxes(image, 0, 1) if transposed else image


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


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
    of the image sent, laid out as on screen (``to_screen``),
    ``resolution_nm`` its pixel size in nm (square
    pixels, which a model needs to see shapes as they are). ``write_lo`` and
    ``write_hi`` are the annotation voxels written, ``[lo, hi)``, clipped to
    the volume and one voxel thick along the depth axis.
    ``display_dimensions`` are the viewer's (``display_names``), which say
    how its panel draws the plane (``screen_axes``); z, y, x in a plan saved
    before plans kept them.
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
    display_dimensions: tuple = AXIS_NAMES

    @property
    def plane_axes(self) -> tuple:
        """The two in-plane axes, in z, y, x order."""
        return tuple(a for a in range(3) if a != self.depth_axis)

    @property
    def plane_name(self) -> str:
        """"XY", "XZ" or "YZ": the data axes the plane shows."""
        return PLANE_NAMES[self.depth_axis]

    @property
    def write_shape(self) -> tuple:
        """The write box's in-plane shape in annotation voxels, its ``plane_axes`` in z, y, x order."""
        return tuple(int(self.write_hi[a] - self.write_lo[a]) for a in self.plane_axes)

    @property
    def click_px(self) -> tuple:
        """``(row, col)`` of the point in the image sent."""
        down, across, backwards = screen_axes(self.depth_axis, self.display_dimensions)
        position = []
        for side, a in zip(self.image_shape, (down, across)):
            fraction = (self.point_nm[a] - self.raw_offset_nm[a]) / self.raw_shape_nm[a]
            position.append(int(np.clip(np.floor(fraction * side), 0, side - 1)))
        if backwards:
            # The pixel counted from the near end, then mirrored, so a point
            # on a boundary takes the pixel it takes in z, y, x order.
            position[1] = self.image_shape[1] - 1 - position[1]
        return tuple(position)

    def to_screen(self, plane) -> np.ndarray:
        """``to_screen`` for this plan's plane."""
        return to_screen(plane, self.depth_axis, self.display_dimensions)

    def from_screen(self, image) -> np.ndarray:
        """``from_screen`` for this plan's plane."""
        return from_screen(image, self.depth_axis, self.display_dimensions)

    def to_dict(self) -> dict:
        """The plan as JSON-safe lists, for a staging's meta.json and the browser."""
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
            display_dimensions=display_names(data.get("display_dimensions")),
        )


def plan_plane(volume: dict, volume_shape, point_nm, depth_axis: int, crop_size_px: int,
               data_box_nm=None, display_dimensions=None) -> PlanePlan:
    """The plane through ``point_nm`` (world nm, z, y, x) across ``depth_axis``.

    ``volume`` is the annotation volume's record (``output_voxel_size``,
    ``input_voxel_size``, ``dataset_offset_nm``), ``volume_shape`` its
    ``annotation/s0`` shape. The field of view is square in nm,
    ``crop_size_px`` annotation voxels along the finer in-plane axis (both,
    for isotropic voxels), centred on the point; its write box is the
    annotation voxels whose centres lie in it, clipped to the volume, in the
    annotation voxel that holds the point along the depth axis.

    ``data_box_nm`` is the raw data's extent, ``(lo, hi)`` world nm z, y, x:
    the field is clipped to it too. The volume is the data rounded up to
    whole chunks of the model's output, which for Cellpose at 64 nm is up to
    33 um past the data's edge, and there the plane is only the read's
    zero padding: an image of it was labelled as background, and the cells
    beside it smeared into it.

    The raw is read over the clipped box's extent, so the image and the box
    cover the same physical area. The image's pixels are the finer in-plane
    input voxel size, square; when that makes a side longer than
    ``MAX_IMAGE_SIDE`` the pixels are made larger to fit. The image is laid
    out as the viewer's panel draws the plane, for the viewer's
    ``display_dimensions`` (z, y, x when None).

    Raises ``Refused`` when the point is outside the volume or the data: a
    field clipped to an edge it does not reach is not what was aimed at.
    ValueError for a depth axis, crop size or point that is not one.
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
        raise Refused(
            f"The point ({', '.join(f'{p:g}' for p in point)}) nm is outside the annotation volume "
            f"(z, y, x {', '.join(extent)} nm): move the view onto the data and try again."
        )

    plane = [a for a in range(3) if a != depth_axis]
    side_nm = int(crop_size_px) * float(output_voxel_size[plane].min())
    offset = point - side_nm / 2
    size = np.full(3, side_nm)
    if data_box_nm is not None:
        data_lo, data_hi = (np.asarray(v, dtype=float) for v in data_box_nm)
        if np.any(point < data_lo) or np.any(point >= data_hi):
            extent = [f"{lo:g}..{hi:g}" for lo, hi in zip(data_lo, data_hi)]
            raise Refused(
                f"The point ({', '.join(f'{p:g}' for p in point)}) nm is outside the data "
                f"(z, y, x {', '.join(extent)} nm): move the view onto the data and try again."
            )
        for a in plane:
            top = min(offset[a] + size[a], data_hi[a])
            offset[a] = max(offset[a], data_lo[a])
            size[a] = top - offset[a]
    # Along the depth axis, exactly the voxel that holds the point: a box
    # one voxel long whose only centre is that voxel's.
    offset[depth_axis] = corner[depth_axis] + voxel[depth_axis] * output_voxel_size[depth_axis]
    size[depth_axis] = output_voxel_size[depth_axis]
    box = box_voxels(volume, offset, size, shape)
    if box is None:  # cannot happen with the point inside, but never write an empty box
        raise Refused("No annotation voxels lie around the point.")
    lo, hi = box

    raw_offset = corner + lo * output_voxel_size
    raw_shape = (hi - lo) * output_voxel_size
    raw_offset[depth_axis] = point[depth_axis] - input_voxel_size[depth_axis] / 2
    raw_shape[depth_axis] = input_voxel_size[depth_axis]

    extent = raw_shape[plane]
    resolution = max(float(input_voxel_size[plane].min()), float(extent.max()) / MAX_IMAGE_SIDE)
    display = display_names(display_dimensions)
    image_shape = screen_shape(np.maximum(np.round(snap_integral(extent / resolution)), 1), depth_axis, display)
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
        display_dimensions=display,
    )


def raw_box(plan: PlanePlan, voxel_size, corner) -> tuple:
    """``(begin, count)``: the raw voxels to read for ``plan``, on a raw grid
    of ``voxel_size`` whose voxel 0's lower corner is ``corner`` (nm, z, y, x).

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
    return _ints(begin), _ints(count)


# ---------------------------------------------------------------------------
# Resampling, by Pillow
# ---------------------------------------------------------------------------
#
# Pillow is imported inside the functions that resample, not at the top: in
# the browser, chunkmirage's Pyodide worker loads a package the first time a
# call fails to import it (found by its module in Pyodide's lockfile), so
# Pillow is downloaded with the first image resized, not when a dataset
# opens, and never on a page that does not annotate with AI. Each answer is
# a writable array of its own, not a view of the image's bytes.


def resize(plane, shape) -> np.ndarray:
    """``plane`` resampled to ``shape`` (rows, cols), float32; itself when it already is that.

    Shrinking (neither side grows) averages each pixel's footprint (Pillow's
    box filter), so fine membranes fade rather than alias; otherwise (the
    coarse axis of an anisotropic plane) it interpolates linearly (Pillow's
    bilinear), on Pillow's 32-bit float ("F") image.
    """
    rows, cols = (int(s) for s in shape)
    plane = np.asarray(plane)
    if plane.shape == (rows, cols):
        return plane
    if rows < 1 or cols < 1:
        raise ValueError(f"cannot resize a plane to {rows}x{cols}")
    from PIL import Image

    shrinking = plane.shape[0] >= rows and plane.shape[1] >= cols
    kind = Image.Resampling.BOX if shrinking else Image.Resampling.BILINEAR
    image = Image.fromarray(np.ascontiguousarray(plane, dtype=np.float32))
    return np.array(image.resize((cols, rows), kind), dtype=np.float32)


def nearest(array, shape) -> np.ndarray:
    """A 2D ``array`` (uint8 or int32, as Pillow holds an image of each) at
    ``shape`` (rows, cols) by nearest neighbour, pixel centre to pixel centre
    (Pillow's NEAREST), in its own dtype; a copy when it already is that
    size."""
    rows, cols = (int(s) for s in shape)
    array = np.asarray(array)
    if array.shape == (rows, cols):
        return array.copy()
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(array))
    return np.array(image.resize((cols, rows), Image.Resampling.NEAREST), dtype=array.dtype)


# ---------------------------------------------------------------------------
# The image sent, and the mask in the image the model sends back
# ---------------------------------------------------------------------------


def plane_image(block, plan: PlanePlan) -> np.ndarray:
    """The plane sent, from the raw ``block`` read over ``raw_box``'s voxels (z, y, x).

    Its one slice along the depth axis, laid out as on screen
    (``PlanePlan.to_screen``) and resampled to the plan's image
    (``resize``). The raw's dtype is kept unless the plane had to be resized
    to the image's grid, when it is float32.
    """
    plane = np.take(np.asarray(block), 0, axis=plan.depth_axis)
    return resize(plan.to_screen(plane), plan.image_shape)


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """A 2D array as uint8 grey levels (from ask-to-mask's ``_normalize_to_uint8``).

    uint8 passes through unchanged. Anything else is clipped to its 1st-99th
    percentile over the plane and stretched to 0-255, as raw EM is not always
    8-bit and a few saturated pixels should not flatten the contrast.
    """
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return data
    p_lo, p_hi = np.percentile(data, (1, 99))
    if p_hi <= p_lo:
        p_hi = p_lo + 1
    clipped = np.clip(data.astype(np.float32), p_lo, p_hi)
    return ((clipped - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)


def extract_mask(rgb, target_rgb, threshold: float = 200.0) -> np.ndarray:
    """The pixels of the model's reply (rows, cols, 3 or 4, uint8) painted
    ``target_rgb``, as a bool (rows, cols) mask (ask-to-mask's
    ``extract_mask``, its colour-difference mode).

    Each pixel scores how strongly it shows the target colour: its weakest
    channel among those the target has at full strength, minus its strongest
    deviation elsewhere -- a channel the target has off (0) being on, or a
    channel the target has at part strength being away from that strength.
    For red (255, 0, 0) that is ``R - max(G, B)``: red that dominates. A
    pixel counts when its score reaches ``threshold``.

    The part-strength term is what lets orange (255, 128, 0) or purple
    (128, 0, 255) be found at all: counting their half channel as "on" would
    cap the score at 128, under the threshold, and counting it as "off"
    would reject the colour itself.
    """
    out = np.asarray(rgb)[:, :, :3].astype(np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)

    full = target >= 255
    off = target <= 0
    part = ~full & ~off

    strength = np.min(out[:, :, full], axis=-1) if full.any() else np.full(out.shape[:2], 255.0, np.float32)
    deviations = []
    if off.any():
        deviations.append(np.max(out[:, :, off], axis=-1))
    if part.any():
        deviations.append(np.max(np.abs(out[:, :, part] - target[part]), axis=-1))
    score = strength - np.max(deviations, axis=0) if deviations else strength
    return score >= threshold


def model_mask(rgb, sent_shape, image_shape, target_rgb):
    """The mask in a model's reply, and the reply lined up with the plane:
    ``(mask, aligned)``.

    ``rgb`` is the image the model sent back, decoded (rows, cols, 3 or 4,
    uint8), ``sent_shape`` the (rows, cols) of the image it was sent (the
    plane padded to a square), ``image_shape`` the plane's. A reply at
    another size (or shape: stretching it then is what the review shows) is
    resized back to the sent size by Pillow's bilinear filter on an RGB
    image (which averages when shrinking), then the padding is cut off the
    right and bottom, so what is left lines up with the plane; the mask is
    the pixels painted ``target_rgb`` in that (``extract_mask``).
    ``aligned`` is (rows, cols, 3) uint8, the plane's shape, for the review.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)[:, :, :3]
    sent_rows, sent_cols = (int(s) for s in sent_shape)
    if rgb.shape[:2] != (sent_rows, sent_cols):
        from PIL import Image

        reply = Image.fromarray(np.ascontiguousarray(rgb))
        rgb = np.array(reply.resize((sent_cols, sent_rows), Image.Resampling.BILINEAR))
    rows, cols = (int(s) for s in image_shape)
    if rows > rgb.shape[0] or cols > rgb.shape[1]:
        raise ValueError(f"The plane is {rows}x{cols}, larger than the {rgb.shape[0]}x{rgb.shape[1]} image sent")
    aligned = np.ascontiguousarray(rgb[:rows, :cols])
    return extract_mask(aligned, target_rgb), aligned


def overlay(rgb, mask, target_rgb, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    """``rgb`` (the image sent, rows, cols, 3 or 4) with the pixels ``mask``
    marks tinted ``target_rgb``: the review's third picture, what accepting
    writes, not the model's own image. ``mask`` is the write box's objects
    laid out as the image is (``to_screen``) and scaled to it by nearest
    neighbour; uint8 RGB back, rounded half to even."""
    out = np.asarray(rgb)[:, :, :3].astype(np.float32)
    on = nearest((np.asarray(mask) > 0).astype(np.uint8), out.shape[:2]) > 0
    tint = np.asarray(target_rgb, dtype=np.float32)
    out[on] = (1 - alpha) * out[on] + alpha * tint
    return np.round(out).astype(np.uint8)


# ---------------------------------------------------------------------------
# The mask on the write box, and its labels
# ---------------------------------------------------------------------------


def mask_to_write_shape(mask: np.ndarray, plan: PlanePlan) -> np.ndarray:
    """The objects of ``mask`` (the image's shape, as on screen) on the write
    box's in-plane voxels, in z, y, x order (``PlanePlan.write_shape``):
    int32, 0 for background and 1..n for the objects, numbered in the box's
    raster order as ``scipy.ndimage.label`` numbers a mask.

    Objects are the mask's 4-connected pieces (scipy's default, so pieces
    touching only at a corner stay apart, as the model was asked to leave
    them), found at the image's own resolution, where the model drew the
    lines between them: on voxels coarser than the pixels a line can fall
    between the pixels the voxels take, and the objects either side of it
    still keep ids of their own. Then nearest neighbour, pixel centre to
    voxel centre: each annotation voxel takes the object of the pixel at
    its centre, so the mask keeps hard edges and lands where it was drawn
    whatever the two sizes are. An object no voxel centre falls in is lost.
    """
    from scipy.ndimage import label

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"The mask must be 2D, got shape {mask.shape}")
    objects, _ = label(mask)
    rows, cols = screen_shape(plan.write_shape, plan.depth_axis, plan.display_dimensions)
    objects = nearest(objects.astype(np.int32), (rows, cols))
    objects = plan.from_screen(objects)
    # Renumbered by first voxel, so fresh ids go to the objects in the order
    # labelling the box itself would give them, whatever the screen layout.
    ids, first = np.unique(objects, return_index=True)
    order = ids[ids > 0][np.argsort(first[ids > 0])]
    number = np.zeros(int(objects.max(initial=0)) + 1, dtype=np.int32)
    number[order] = np.arange(1, order.size + 1)
    return number[objects]


def labels_for_box(objects: np.ndarray, existing: np.ndarray, depth_axis: int, count_up: bool,
                   beside=None) -> np.ndarray:
    """The labels for the write box: an id per object of ``objects``, 1 elsewhere.

    ``objects`` is the in-plane objects (``mask_to_write_shape``), 0 for
    background, ``existing`` the one-voxel-thick box as the volume holds
    it. The objects are labelled by ``fill.seed_labels``: an object the user
    has already painted part of keeps that id, the rest get fresh ones,
    counting up for an instance target's ``count_up``.

    ``beside`` is the planes on either side of the box along the depth axis,
    as the volume holds them (each shaped like ``existing``, or None at the
    volume's edge). Planes are annotated one at a time, so an object usually
    continues one in the plane next to it: where this plane is unpainted,
    the ids beside it count as painted here, and an object takes the id it
    overlaps most, rather than a fresh one that an instance target would
    read as a boundary between the two planes. Fresh ids skip those planes'
    ids too, so a new object does not take the id of a different object
    right next to it, which an instance target would read as one object.
    """
    objects = np.expand_dims(np.asarray(objects), depth_axis)
    if objects.shape != existing.shape:
        raise ValueError(f"The mask makes a box of {objects.shape}; the box holds {existing.shape}")
    planes = [np.asarray(p, dtype=existing.dtype) for p in (beside or ()) if p is not None]
    if any(p.shape != existing.shape for p in planes):
        raise ValueError(f"The planes beside the box must be shaped {existing.shape}")
    if not planes:
        return seed_labels(objects, existing, count_up=count_up)
    seen = existing.copy()
    for plane in planes:
        carry = (seen == 0) & (plane >= 2)
        seen[carry] = plane[carry]
    # The planes beside are stacked after the box with no objects in them:
    # seed_labels labels only the box's objects, but its fresh ids see every
    # id in the stack.
    stack_existing = np.concatenate([seen, *planes], axis=depth_axis)
    stack_objects = np.concatenate([objects, *[np.zeros_like(objects)] * len(planes)], axis=depth_axis)
    labels = seed_labels(stack_objects, stack_existing, count_up=count_up)
    return np.take(labels, [0], axis=depth_axis)


def accept_labels(objects, existing, depth_axis: int, instance_target: bool, overwrite: bool,
                  beside=None) -> np.ndarray:
    """What Accept writes into the write box: ``labels_for_box``'s labels, as
    a seed's ids (counting up for an instance target on a volume of more
    than 8-bit ids, ``fill.counts_up``; an object continuing one in a plane
    ``beside`` keeps its id), for ``fill.paint`` to write.

    With ``overwrite`` the model's objects replace what is under them, but
    its background only fills unlabelled voxels: an object already painted
    that the model missed is kept, not erased to background (those voxels
    get 0, "leave as it is").
    """
    labels = labels_for_box(objects, existing, depth_axis, count_up=counts_up(instance_target, existing.dtype),
                            beside=beside)
    if overwrite:
        labels[(labels == 1) & (existing != 0)] = 0
    return labels
