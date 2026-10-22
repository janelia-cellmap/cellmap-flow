"""AI-assisted annotation's maths (cellmap_flow.portable.ai_annotate), which
both modes run: the plane planned around a point, the mask in a model's
reply, and the labels Accept writes.

The volumes here have their voxel 0's corner at the world origin: an 8 nm
volume's ``dataset_offset_nm`` is 4 (voxel 0's centre), so annotation voxel
i covers [8i, 8i + 8) nm and the arithmetic in the expectations is plain.
The round trips through a raw on disk are tests/ai_annotate/test_pipeline.py's.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.ai_annotate.organelles import ORGANELLES
from cellmap_flow.portable.ai_annotate import (
    Refused,
    accept_labels,
    extract_mask,
    from_screen,
    labels_for_box,
    mask_to_write_shape,
    overlay,
    plan_plane,
    raw_box,
    to_screen,
)


def _volume(output=(8, 8, 8), input=(4, 4, 4)):
    output = np.asarray(output, dtype=float)
    return {
        "output_voxel_size": list(output),
        "input_voxel_size": list(input),
        "dataset_offset_nm": list(output / 2),  # voxel 0's centre: its corner is at 0
    }


SHAPE = (100, 200, 300)


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
    with pytest.raises(Refused, match="outside the annotation volume"):
        plan_plane(_volume(), SHAPE, point, 0, crop_size_px=64)


def test_the_raw_box_reads_the_plane_on_the_raws_grid():
    plan = plan_plane(_volume(), SHAPE, (50 * 8 + 1, 800, 1200), 0, crop_size_px=64)
    # A raw at 4 nm from the origin: exactly the box's voxels, and the one
    # input voxel the point is in (401 nm is in [400, 404)).
    begin, count = raw_box(plan, (4, 4, 4), (0, 0, 0))
    assert begin == (100, (800 - 256) // 4, (1200 - 256) // 4)
    assert count == (1, 128, 128)
    # A raw whose corner is off the box's grid: each pixel reads the raw
    # voxel its centre is in (the first pixel's centre is 546 nm in y).
    for corner, first in (((-1,) * 3, (100, 136, 236)), ((1,) * 3, (100, 136, 236)), ((3,) * 3, (99, 135, 235))):
        assert raw_box(plan, (4, 4, 4), corner) == (first, (1, 128, 128)), corner
    # A coarser raw: 8 nm voxels, 64 of them; along z the one holding 401 nm.
    assert raw_box(plan, (8, 8, 8), (0, 0, 0)) == ((50, 68, 118), (1, 64, 64))


# --- the mask in the model's reply --------------------------------------------------


@pytest.mark.parametrize("key", list(ORGANELLES))
def test_every_catalog_colour_can_be_found_and_neighbours_are_not(key):
    target = ORGANELLES[key].rgb
    near = tuple(int(np.clip(c + (-20 if c > 128 else 15), 0, 255)) for c in target)
    others = [p.rgb for p in ORGANELLES.values() if p.rgb != target]
    mask = extract_mask(np.array([[target, near, (0, 0, 0), (255, 255, 255), *others]], dtype=np.uint8), target)
    assert mask[0, :2].all(), f"{key}: its own colour was not found"
    assert not mask[0, 2:].any(), f"{key}: another colour counted as {target}"


def test_the_overlay_tints_what_accept_writes_halfway_to_the_colour():
    # A 2x2 mask on a 4x4 image: each mask pixel covers 2x2 image pixels.
    # Halfway from grey 2 to 255 is 128.5, which rounds to the even 128.
    image = np.full((4, 4, 4), 2, dtype=np.uint8)
    tinted = overlay(image, np.array([[1, 0], [0, 0]]), (255, 0, 0))
    assert tinted.shape == (4, 4, 3)
    assert tinted[:2, :2].tolist() == [[[128, 1, 1]] * 2] * 2
    assert (tinted[2:] == 2).all() and (tinted[:2, 2:] == 2).all()


# --- the mask on the box ------------------------------------------------------------


def test_a_one_pixel_line_parts_two_objects_on_coarser_voxels():
    # 4 nm pixels on 8 nm voxels: each voxel takes every second pixel, and
    # the line the model left between the two objects is on one none takes.
    plan = plan_plane(_volume((8, 8, 8), (4, 4, 4)), (24, 32, 40), (81, 113, 177), 0, crop_size_px=16)
    assert plan.image_shape == (32, 32) and plan.write_shape == (16, 16)
    mask = np.ones(plan.image_shape, dtype=bool)
    mask[:, 16] = False
    labels = labels_for_box(mask_to_write_shape(mask, plan), np.zeros((1, 16, 16), dtype=np.uint8), 0, count_up=False)
    assert set(np.unique(labels)) == {2, 3}


def _objects(mask, depth_axis):
    """``mask`` (the box's plane, z, y, x order) cut into objects by
    ``mask_to_write_shape``, from an image of the box's own size."""
    plan = SimpleNamespace(write_shape=mask.shape, depth_axis=depth_axis, display_dimensions=None,
                           from_screen=lambda image: from_screen(image, depth_axis))
    return mask_to_write_shape(to_screen(mask, depth_axis), plan)


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_labels_for_box_gives_each_object_an_id(depth_axis):
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 4:7] = True
    mask[5, 7] = True  # touches the second only at a corner: an object of its own
    existing = np.zeros(np.insert(mask.shape, depth_axis, 1), dtype=np.uint8)
    labels = np.squeeze(labels_for_box(_objects(mask, depth_axis), existing, depth_axis, count_up=False), axis=depth_axis)
    assert labels.dtype == np.uint8
    assert np.all(labels[~mask] == 1)
    ids = {int(labels[0, 0]), int(labels[3, 4]), int(labels[5, 7])}
    assert ids == {2, 3, 4}
    assert len(np.unique(labels[0:2, 0:2])) == 1 and len(np.unique(labels[3:5, 4:7])) == 1


def test_labels_for_box_keeps_a_painted_objects_id():
    mask = np.zeros((1, 6, 6), dtype=bool)[0]
    mask[1:4, 1:4] = True
    existing = np.zeros((1, 6, 6), dtype=np.uint8)
    existing[0, 1, 1] = 7  # the user painted part of the object as 7
    existing[0, 5, 5] = 9  # and another object elsewhere
    labels = labels_for_box(mask, existing, 0, count_up=False)[0]
    assert np.all(labels[1:4, 1:4] == 7)


def test_labels_for_box_counts_up_on_an_instance_volume():
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, 0:2] = True
    mask[:, 4:6] = True
    existing = np.zeros((4, 1, 8), dtype=np.uint16)
    existing[0, 0, 7] = 300
    labels = labels_for_box(_objects(mask, 1), existing, 1, count_up=True)[:, 0, :]
    assert labels.dtype == np.uint16
    assert {int(labels[0, 0]), int(labels[0, 4])} == {301, 302}


@pytest.mark.parametrize("dtype, instance_target, ids", [
    (np.uint8, True, {2, 3}),  # a uint8 volume reuses free ids, whatever the model
    (np.uint16, False, {2, 3}),  # a binary target only sees foreground
    (np.uint16, True, {301, 302}),  # an instance target on wider ids counts up from the box's largest
    (np.uint32, True, {301, 302}),
])
def test_accept_counts_ids_up_only_for_an_instance_target_on_a_wide_volume(dtype, instance_target, ids):
    # A one-voxel-thick box along z, 1 x 8 in plane, with an id painted at its end.
    objects = np.zeros((1, 8), dtype=np.int32)
    objects[0, :2], objects[0, 4:6] = 1, 2
    existing = np.zeros((1, 1, 8), dtype=dtype)
    existing[0, 0, 7] = 300 if np.dtype(dtype).itemsize > 1 else 30
    labels = accept_labels(objects, existing, 0, instance_target, overwrite=False)[0, 0]
    assert {int(labels[0]), int(labels[4])} == ids


def test_overwrite_lets_objects_replace_paint_but_keeps_objects_the_model_missed():
    objects = np.array([[1, 1, 0, 0]], dtype=np.int32)
    existing = np.array([[[5, 0, 6, 0]]], dtype=np.uint8)
    labels = accept_labels(objects, existing, 0, False, overwrite=True)[0, 0]
    # The model's object keeps the id painted on it; its background fills
    # only the unpainted voxel (0 leaves a voxel as it is), so object 6,
    # which the model missed, stays.
    assert labels.tolist() == [5, 5, 0, 1]
