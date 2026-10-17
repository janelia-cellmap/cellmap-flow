"""The neuroglancer viewer: making one, the layers it shows and the keys bound in it.

- ``bootstrap``: ``new_viewer()``, the one place a viewer is made.
- ``layers``: a model's prediction layer and the raw data's layer, built the
  same way by every path that shows one.
- ``keys``: binding a key once per viewer, and its status-bar message.
- ``VIEWER_AXES`` and ``display_dimensions_for_view``: the order of a
  viewer's dimensions, and which it displays. Here, so that importing them
  imports nothing else.

Nothing here reads the dashboard's state or starts the dashboard; callers
pass in what a layer needs.
"""

# The order of cellmap-flow's viewers' dimensions, as Neuroglancer's own
# viewers have them. Its "xy" panel shows the first two display dimensions,
# which are the first three of the coordinate space: in this order the XY
# plane, x across and y down, looking along z. In z, y, x order it showed z
# across and y down. Setting the display dimensions to x, y, z over a z, y,
# x space does not do instead: the painting fork's brush reads a panel's
# normal, given in display order, in the space's order, and painted across
# the slice shown.
VIEWER_AXES = ("x", "y", "z")


def display_dimensions_for_view(viewer_state) -> tuple:
    """The display dimensions of a neuroglancer ``ViewerState``: those it
    sets, else the first three of its coordinate space, as neuroglancer
    takes them; empty when it has neither."""
    display = list(getattr(viewer_state, "display_dimensions", None) or [])
    if not display:
        dimensions = getattr(viewer_state, "dimensions", None)
        display = list(getattr(dimensions, "names", None) or [])[:3]
    return tuple(display)
