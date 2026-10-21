"""Planning the plane an AI annotation reads and the voxels it writes.

The volumes here have their voxel 0's corner at the world origin: an 8 nm
volume's ``dataset_offset_nm`` is 4 (voxel 0's centre), so annotation voxel
i covers [8i, 8i + 8) nm and the arithmetic in the expectations is plain.
"""

import numpy as np
import pytest

from cellmap_flow.ai_annotate.errors import AIAnnotateError
from cellmap_flow.ai_annotate.geometry import (
    depth_axis_from_layout,
    plan_plane,
    raw_box,
)


def _volume(output=(8, 8, 8), input=(4, 4, 4)):
    output = np.asarray(output, dtype=float)
    return {
        "output_voxel_size": list(output),
        "input_voxel_size": list(input),
        "dataset_offset_nm": list(output / 2),  # voxel 0's centre: its corner is at 0
    }


SHAPE = (100, 200, 300)


# --- which plane the user is looking at ---------------------------------------------


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


# How the image is laid out as the viewer's panel draws the plane (to_screen,
# from_screen) is web/test/fixtures/ai_annotate.json's "screen" cases, which
# tests/web/test_fixtures_fresh.py regenerates from this code and the vitest
# parity suite checks the TypeScript against.


# --- the plane --------------------------------------------------------------------


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_a_plane_is_centred_on_the_point_one_voxel_thick(depth_axis):
    # The point is in annotation voxel (50, 100, 150), off its centre.
    point = (50 * 8 + 1, 100 * 8 + 3, 150 * 8 + 6)
    plan = plan_plane(_volume(), SHAPE, point, depth_axis, crop_size_px=64)

    lo, hi = np.array(plan.write_lo), np.array(plan.write_hi)
    voxel = np.array([50, 100, 150])
    for axis in range(3):
        if axis == depth_axis:
            assert (lo[axis], hi[axis]) == (voxel[axis], voxel[axis] + 1)
        else:
            # 64 voxels whose centres lie within 256 nm of the point.
            assert hi[axis] - lo[axis] == 64
            assert np.all(np.abs((np.arange(lo[axis], hi[axis]) + 0.5) * 8 - point[axis]) <= 256)
    assert plan.write_shape == (64, 64)

    # The raw is the same area at 4 nm, one input voxel thick at the point.
    raw_lo, raw_size = np.array(plan.raw_offset_nm), np.array(plan.raw_shape_nm)
    for axis in plan.plane_axes:
        assert raw_lo[axis] == lo[axis] * 8
        assert raw_size[axis] == 64 * 8
    assert raw_size[depth_axis] == 4
    assert raw_lo[depth_axis] + 2 == point[depth_axis]
    assert plan.image_shape == (128, 128)
    assert plan.resolution_nm == 4
    assert plan.input_voxel_size == (4.0, 4.0, 4.0)
    assert plan.plane_name == ("XY", "XZ", "YZ")[depth_axis]
    # The point's pixel: 4 nm pixels from the read's corner, laid out as the
    # panel draws the plane (rows, cols): y, x with x leftwards; x, z; y, z.
    down, across = {0: (1, 2), 1: (2, 0), 2: (1, 0)}[depth_axis]
    row, col = (int((point[a] - raw_lo[a]) // 4) for a in (down, across))
    assert plan.click_px == (row, 128 - 1 - col if depth_axis == 0 else col)


def test_a_point_on_a_voxel_boundary_takes_the_voxel_above():
    plan = plan_plane(_volume(), SHAPE, (80, 800, 1200), 0, crop_size_px=8)
    assert (plan.write_lo[0], plan.write_hi[0]) == (10, 11)


def test_the_volume_offset_is_voxel_0s_centre():
    # dataset_offset_nm 100: voxel 0 covers [96, 104), so 103 nm is in voxel 0
    # and 104 nm in voxel 1, on every axis.
    volume = {**_volume(), "dataset_offset_nm": [100.0] * 3}
    plan = plan_plane(volume, SHAPE, (103, 104 + 8 * 20, 104 + 8 * 30), 0, crop_size_px=4)
    assert plan.write_lo == (0, 20 - 1, 30 - 1)
    assert plan.write_hi == (1, 20 + 3, 30 + 3)
    assert plan.raw_offset_nm[1:] == (96 + 19 * 8, 96 + 29 * 8)


def test_the_write_box_is_clipped_at_the_volume_edge():
    # Voxel (50, 3, 297): 3 voxels from y's start, 2 from x's end.
    point = (50 * 8 + 4, 3 * 8 + 4, 297 * 8 + 4)
    plan = plan_plane(_volume(), SHAPE, point, 0, crop_size_px=64)
    assert plan.write_lo == (50, 0, 297 - 32)
    assert plan.write_hi == (51, 3 + 32, 300)
    assert plan.write_shape == (35, 35)
    # The raw read covers exactly the clipped box, so the image does too.
    assert plan.raw_offset_nm[1:] == (0.0, (297 - 32) * 8)
    assert plan.raw_shape_nm[1:] == (35 * 8, 35 * 8)
    assert plan.image_shape == (70, 70)
    # x runs leftwards in the XY plane, as neuroglancer draws it.
    assert plan.click_px == ((3 * 8 + 4) // 4, 70 - 1 - (32 * 8 + 4) // 4)


@pytest.mark.parametrize(
    "point",
    [
        (-1, 800, 1200),  # before the volume along the depth axis
        (400, 800, 300 * 8),  # just past the far end in x
        (400, -0.5, 1200),
        (100 * 8 + 4, 800, 1200),
    ],
)
def test_a_point_outside_the_volume_is_refused(point):
    with pytest.raises(AIAnnotateError) as caught:
        plan_plane(_volume(), SHAPE, point, 0, crop_size_px=64)
    assert caught.value.category == "refused"
    assert "outside the annotation volume" in caught.value.user_message


def test_the_raw_box_reads_the_plane_on_the_raws_grid():
    plan = plan_plane(_volume(), SHAPE, (50 * 8 + 1, 800, 1200), 0, crop_size_px=64)
    # A raw at 4 nm from the origin: exactly the box's voxels, and the one
    # input voxel the point is in (401 nm is in [400, 404)).
    box = raw_box(plan, (4, 4, 4), (0, 0, 0))
    assert box.begin == (100, (800 - 256) // 4, (1200 - 256) // 4)
    assert box.shape == (1, 128, 128)
    # A raw whose corner is off the box's grid: each pixel reads the raw
    # voxel its centre is in (the first pixel's centre is 546 nm in y).
    for corner, begin in (((-1,) * 3, (100, 136, 236)), ((1,) * 3, (100, 136, 236)), ((3,) * 3, (99, 135, 235))):
        box = raw_box(plan, (4, 4, 4), corner)
        assert box.begin == begin, corner
        assert box.shape == (1, 128, 128)
    # A coarser raw: 8 nm voxels, 64 of them; along z the one holding 401 nm.
    box = raw_box(plan, (8, 8, 8), (0, 0, 0))
    assert box.begin == (50, 68, 118)
    assert box.shape == (1, 64, 64)
