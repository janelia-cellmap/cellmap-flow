"""The neuroglancer viewer: making one, and the layers it shows.

- ``bootstrap``: ``new_viewer()``, the one place a viewer is made.
- ``layers``: a model's prediction layer and the raw data's layer, built the
  same way by every path that shows one.
- ``DISPLAY_DIMENSIONS`` and ``display_dimensions_for_view``: which
  dimensions a viewer displays, for bootstrap and for the AI annotation's
  geometry. Defined here, so that importing them imports nothing else.

Nothing here reads the dashboard's state or starts the dashboard; callers
pass in what a layer needs.
"""

# The display dimensions cellmap-flow's viewers set (bootstrap.display_dimensions),
# over a coordinate space named z, y, x. Neuroglancer's "xy" panel shows the
# first two display dimensions, which are the first three of the coordinate
# space unless set: unset, that panel showed z across and y down, and a
# model that computes one z-slice at a time was asked for a block for every
# z position on screen. With x, y, z it is the XY plane, x across and y
# down, looking along z, and the positions stay z, y, x.
DISPLAY_DIMENSIONS = ("x", "y", "z")


def display_dimensions_for_view(viewer_state) -> tuple:
    """The display dimensions of a neuroglancer ``ViewerState``: those it
    sets, else the first three of its coordinate space, as neuroglancer
    takes them; empty when it has neither."""
    display = list(getattr(viewer_state, "display_dimensions", None) or [])
    if not display:
        dimensions = getattr(viewer_state, "dimensions", None)
        display = list(getattr(dimensions, "names", None) or [])[:3]
    return tuple(display)
