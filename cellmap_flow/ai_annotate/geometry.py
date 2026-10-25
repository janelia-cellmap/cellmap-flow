"""Which plane of the data an AI annotation reads, and which voxels it writes.

The maths is ``cellmap_flow.portable.ai_annotate``'s, which the browser runs
too, and is re-exported here under the names the server's callers use:
``plan_plane`` (a ``PlanePlan``: the square field around the point, snapped
to the annotation volume's voxels and clipped to the volume and the data,
and the raw read covering the same physical area), ``raw_box`` (that read on
the raw's grid, as voxel ``(begin, end)``) and the screen layout
(``screen_axes``, ``screen_shape``, ``to_screen``, ``from_screen``): the
image sent is laid out as the viewer's panel shows the plane. What is the
server's alone is reading which plane the viewer shows: its layout and its
display dimensions (``depth_axis_for_view``, ``display_dimensions_for_view``),
which the browser reads from its own viewer (web/src/viewer/state.ts).

Axes are z, y, x throughout, as everywhere in cellmap-flow, except in the
image sent (``portable/ai_annotate.py`` says how it is drawn).
"""

from cellmap_flow import viewer
from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.portable import ai_annotate as portable
from cellmap_flow.portable.ai_annotate import (  # noqa: F401  (re-exported for the server's callers)
    AXIS_NAMES,
    MAX_IMAGE_SIDE,
    PLANE_NAMES,
    PlanePlan,
    display_names,
    from_screen,
    screen_axes,
    screen_shape,
    to_screen,
)

# Neuroglancer names its cross-section panels after display-dimension
# *slots*, not after the dimensions' names: the "xy" panel shows display
# dimensions 0 and 1 and looks along 2, "xz" (rotated about the first) looks
# along 1, "yz" (rotated about the second) along 0 (data_panel_layout.ts's
# AXES_RELATIVE_ORIENTATION). The display dimensions are the first three of
# the viewer's coordinate space unless set. cellmap-flow's viewers order
# their dimensions x, y, z (viewer.VIEWER_AXES), so the "xy" panel is the XY
# plane, looking along z. A viewer in z, y, x order (a link made before)
# has a z-y slice looking along x there, and the XY plane is its "yz" panel.
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
    None, as a viewer that sets none over z, y, x shows them); see
    ``_LAYOUT_DEPTH_SLOT`` for why they matter. Only a single-plane layout says which plane the
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


def display_dimensions_for_view(viewer_state) -> tuple:
    """The display dimensions of a neuroglancer ``ViewerState``, as
    ``display_names`` gives them."""
    return display_names(viewer.display_dimensions_for_view(viewer_state))


def depth_axis_for_view(viewer_state) -> int:
    """``depth_axis_from_layout`` for a neuroglancer ``ViewerState``, over
    its display dimensions as they are: a panel looking along one that is
    not z, y or x (time) gets z."""
    display = viewer.display_dimensions_for_view(viewer_state)
    return depth_axis_from_layout(getattr(viewer_state, "layout", None), display or None)


def plan_plane(volume: dict, volume_shape, point_nm, depth_axis: int, crop_size_px: int,
               data_box_nm=None, display_dimensions=None) -> PlanePlan:
    """``portable.ai_annotate.plan_plane`` (its docstring says what the plane
    is), its refusal raised as the ``AIAnnotateError`` ("refused", 400) the
    routes answer: a point outside the volume or the data, which the user
    fixes by moving the view."""
    try:
        return portable.plan_plane(volume, volume_shape, point_nm, depth_axis, crop_size_px,
                                   data_box_nm=data_box_nm, display_dimensions=display_dimensions)
    except portable.Refused as e:
        raise AIAnnotateError("refused", str(e), http_status=400) from None


def raw_box(plan: PlanePlan, voxel_size, corner) -> tuple:
    """The raw voxels to read for ``plan``, ``(begin, end)`` on a raw grid of
    ``voxel_size`` from ``corner`` (its voxel 0's lower corner):
    ``portable.ai_annotate.raw_box``'s box."""
    begin, count = portable.raw_box(plan, voxel_size, corner)
    return tuple(int(b) for b in begin), tuple(int(b) + int(n) for b, n in zip(begin, count))
