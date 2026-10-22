"""Reading the planned plane, mapping a mask onto the box and labelling it.

The round trips are the point: a raw with a bright block at a known world
position, a plane planned through it, read, masked by brightness, mapped to
the write box and labelled must label exactly the annotation voxels that
cover the block, on every plane and whatever the voxel sizes are.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.ai_annotate.geometry import from_screen, plan_plane, to_screen
from cellmap_flow.ai_annotate.pipeline import (
    labels_for_box,
    mask_to_write_shape,
    raw_extent_nm,
    read_plane,
)

DIM, BRIGHT = 20, 220

# The bright block, world nm, z, y, x: [lo, hi) per axis.
BLOCK_NM = ((64, 96), (96, 128), (160, 192))


def _raw_with_block(raw_zarr, voxel_size, extent_nm, block_nm=BLOCK_NM, name="raw"):
    """A raw of ``extent_nm`` at ``voxel_size`` from the origin, DIM but for the block."""
    voxel_size = np.asarray(voxel_size)
    shape = tuple(int(e // v) for e, v in zip(extent_nm, voxel_size))
    data = np.full(shape, DIM, dtype=np.uint8)
    data[tuple(slice(int(lo // v), int(hi // v)) for (lo, hi), v in zip(block_nm, voxel_size))] = BRIGHT
    return raw_zarr(data, voxel_size=tuple(int(v) for v in voxel_size), offset=(0, 0, 0), name=name)


def _volume(raw_path, output, input):
    output = np.asarray(output, dtype=float)
    return {
        "dataset_path": raw_path,
        "output_voxel_size": list(output),
        "input_voxel_size": list(input),
        "dataset_offset_nm": list(output / 2),  # corner at the origin, as the raw's
        "resample": False,
    }


def _labelled_voxels(volume, volume_shape, point, depth_axis, crop_size_px):
    """The annotation voxels (global indices) the plane through ``point`` labels as foreground."""
    plan = plan_plane(volume, volume_shape, point, depth_axis, crop_size_px)
    plane = read_plane(volume, plan)
    assert plane.shape == tuple(plan.image_shape)
    mask = mask_to_write_shape(plane > (DIM + BRIGHT) / 2, plan)
    existing = np.zeros(np.subtract(plan.write_hi, plan.write_lo), dtype=np.uint8)
    labels = labels_for_box(mask, existing, depth_axis, count_up=False)
    # One object, id 2; the rest of the plane background.
    assert set(np.unique(labels)) == {1, 2}
    return plan, {tuple(int(i) for i in v) for v in np.argwhere(labels >= 2) + np.asarray(plan.write_lo)}


def _expected(block_nm, output, depth_axis, depth_voxel):
    """The annotation voxels covering the block, in the plane's slice."""
    ranges = [range(int(lo // v), int(hi // v)) for (lo, hi), v in zip(block_nm, output)]
    ranges[depth_axis] = [depth_voxel]
    return {(z, y, x) for z in ranges[0] for y in ranges[1] for x in ranges[2]}


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_a_bright_block_lands_on_its_annotation_voxels(raw_zarr, depth_axis):
    # 4 nm raw, 8 nm annotation, the point inside the block.
    raw = _raw_with_block(raw_zarr, (4, 4, 4), (192, 256, 320))
    volume = _volume(raw, (8, 8, 8), (4, 4, 4))
    point = (81, 113, 177)
    plan, labelled = _labelled_voxels(volume, (24, 32, 40), point, depth_axis, crop_size_px=16)
    assert plan.image_shape == (32, 32)
    assert labelled == _expected(BLOCK_NM, (8, 8, 8), depth_axis, int(point[depth_axis] // 8))


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_anisotropic_voxels_land_too(raw_zarr, depth_axis):
    # 8 nm z in the raw and 16 nm in the annotation; the coarse axis of an
    # XZ or YZ plane is drawn at 4 nm, like the others.
    raw = _raw_with_block(raw_zarr, (8, 4, 4), (192, 256, 320))
    volume = _volume(raw, (16, 8, 8), (8, 4, 4))
    point = (81, 113, 177)
    plan, labelled = _labelled_voxels(volume, (12, 32, 40), point, depth_axis, crop_size_px=16)
    assert plan.resolution_nm == 4
    assert labelled == _expected(BLOCK_NM, (16, 8, 8), depth_axis, int(point[depth_axis] // (16, 8, 8)[depth_axis]))


def test_a_downsampled_plane_lands_too(raw_zarr):
    # 1 nm in-plane raw, 4 nm annotation, a field of 1100 x 1200 nm: 1100 x
    # 1200 px at 1 nm, so sent at 1200/1024 nm per pixel.
    block = ((0, 8), (400, 600), (500, 720))
    raw = _raw_with_block(raw_zarr, (4, 1, 1), (8, 1300, 1300), block)
    volume = _volume(raw, (4, 4, 4), (4, 1, 1))
    plan, labelled = _labelled_voxels(volume, (2, 325, 325), (6, 500, 600), 0, crop_size_px=300)
    assert plan.write_shape == (275, 300)
    assert max(plan.image_shape) == 1024
    assert plan.resolution_nm == pytest.approx(1200 / 1024)
    assert labelled == _expected(block, (4, 4, 4), 0, 1)


def test_a_resampled_raw_is_read_at_the_input_voxel_size(raw_zarr):
    # The raw is at 8 nm; the volume was made with Resample at 4 nm input.
    raw = _raw_with_block(raw_zarr, (8, 8, 8), (192, 256, 320))
    volume = {**_volume(raw, (8, 8, 8), (4, 4, 4)), "resample": True}
    plan, labelled = _labelled_voxels(volume, (24, 32, 40), (81, 113, 177), 0, crop_size_px=16)
    assert plan.image_shape == (32, 32)
    assert labelled == _expected(BLOCK_NM, (8, 8, 8), 0, 10)


def test_a_field_at_the_datas_edge_stops_there(raw_zarr):
    # A volume larger than its raw (padded to whole chunks), with the data's
    # extent given: the field stops at the raw's y end, 256 nm, rather than
    # reading and labelling the padding past it.
    raw = _raw_with_block(raw_zarr, (4, 4, 4), (192, 256, 320))
    volume = _volume(raw, (8, 8, 8), (4, 4, 4))
    data_box = raw_extent_nm(volume)
    assert [list(v) for v in data_box] == [[0, 0, 0], [192, 256, 320]]
    plan = plan_plane(volume, (24, 40, 40), (81, 250, 177), 0, crop_size_px=16, data_box_nm=data_box)
    assert plan.write_hi[1] == 256 // 8
    assert plan.raw_offset_nm[1] + plan.raw_shape_nm[1] == 256
    assert np.all(read_plane(volume, plan) == DIM)


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_the_click_marks_the_points_voxel(raw_zarr, depth_axis):
    # A mask of the click's pixel alone, as a model answering about the
    # point would paint it, lands on the annotation voxel the point is in
    # (a pixel a voxel, so no sampling can skip it).
    raw = _raw_with_block(raw_zarr, (8, 8, 8), (192, 256, 320))
    volume = _volume(raw, (8, 8, 8), (8, 8, 8))
    point = (81, 113, 177)
    plan = plan_plane(volume, (24, 32, 40), point, depth_axis, crop_size_px=16)
    image = np.zeros(plan.image_shape, dtype=bool)
    image[plan.click_px] = True
    marked = np.argwhere(mask_to_write_shape(image, plan))
    voxel = [int(point[a] // 8) - plan.write_lo[a] for a in plan.plane_axes]
    assert marked.tolist() == [voxel]


@pytest.mark.parametrize("depth_axis, down, across", [(0, 1, 2), (1, 2, 0), (2, 1, 0)])
def test_the_image_is_the_plane_as_the_panel_draws_it(raw_zarr, depth_axis, down, across):
    # Neuroglancer's panels, with cellmap-flow's z, y, x display dimensions:
    # the XY plane is y down and x across, increasing to the left; XZ is x
    # down and z across; YZ is y down and z across.
    point, extent = (96, 128, 160), (192, 256, 320)

    def bright_below_the_point_along(axis):
        block = [(0, e) for e in extent]
        block[axis] = (0, point[axis])
        raw = _raw_with_block(raw_zarr, (4, 4, 4), extent, block, name=f"raw_{axis}")
        volume = _volume(raw, (8, 8, 8), (4, 4, 4))
        plan = plan_plane(volume, (24, 32, 40), point, depth_axis, crop_size_px=16)
        image = read_plane(volume, plan)
        assert image.shape == tuple(plan.image_shape) == (32, 32)
        return image == BRIGHT

    top = bright_below_the_point_along(down)
    assert top[:16].all() and not top[16:].any()
    side = bright_below_the_point_along(across)
    near, far = (side[:, 16:], side[:, :16]) if depth_axis == 0 else (side[:, :16], side[:, 16:])
    assert near.all() and not far.any()


# --- the mask on the box ----------------------------------------------------------


def test_a_one_pixel_line_parts_two_objects_on_coarser_voxels():
    # 4 nm pixels on 8 nm voxels: each voxel takes every second pixel, and
    # the line the model left between the two objects is on one none takes.
    plan = plan_plane(_volume(None, (8, 8, 8), (4, 4, 4)), (24, 32, 40), (81, 113, 177), 0, crop_size_px=16)
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
