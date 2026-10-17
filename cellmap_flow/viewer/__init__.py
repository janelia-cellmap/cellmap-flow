"""The neuroglancer viewer: making one, and the layers it shows.

- ``bootstrap``: ``new_viewer()``, the one place a viewer is made.
- ``layers``: a model's prediction layer and the raw data's layer, built the
  same way by every path that shows one.
- ``DISPLAY_DIMENSIONS`` and ``display_dimensions_for_view``: which
  dimensions a viewer displays. Here, so that importing them imports
  nothing else.

Nothing here reads the dashboard's state or starts the dashboard; callers
pass in what a layer needs.
"""

# The display dimensions cellmap-flow's viewers set over their z, y, x
# coordinate space. Neuroglancer's "xy" panel shows the first two, which are
# the first three of the coordinate space unless set: unset, it showed z
# across and y down. With x, y, z it is the XY plane, looking along z.
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
