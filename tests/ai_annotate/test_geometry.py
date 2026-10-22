"""Which plane the viewer shows, and the server's refusal of a point outside the volume.

The plane's maths (``plan_plane``, ``raw_box``, the screen layout) is
``cellmap_flow.portable.ai_annotate``'s, tested in tests/portable/test_ai_annotate.py.
"""

import pytest

from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.ai_annotate.geometry import depth_axis_from_layout, plan_plane


@pytest.mark.parametrize(
    "layout, depth",
    [
        # cellmap-flow's viewers name their dimensions z, y, x: neuroglancer's
        # "xy" panel shows z and y, looking along x.
        ("xy", 2),
        ("xz", 1),
        ("yz", 0),
        ("yz-3d", 0),
        ("XZ", 1),
        ({"type": "xz"}, 1),
        # Several planes at once: the "xy" panel's, neuroglancer's first.
        ("4panel", 2),
        (None, 2),
    ],
)
def test_depth_axis_from_layout_with_zyx_dimensions(layout, depth):
    assert depth_axis_from_layout(layout) == depth


def test_a_point_outside_the_volume_is_refused_for_the_user():
    volume = {"output_voxel_size": [8.0] * 3, "input_voxel_size": [4.0] * 3, "dataset_offset_nm": [4.0] * 3}
    with pytest.raises(AIAnnotateError) as caught:
        plan_plane(volume, (100, 200, 300), (-1, 800, 1200), 0, crop_size_px=64)
    assert caught.value.category == "refused" and caught.value.http_status == 400
    assert "outside the annotation volume" in caught.value.user_message
