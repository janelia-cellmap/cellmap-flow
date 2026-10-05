"""Reading the planned plane, asking for a mask and labelling the box.

The round trips are the point: a raw with a bright block at a known world
position, a plane planned through it, read, masked by brightness, mapped to
the write box and labelled must label exactly the annotation voxels that
cover the block, on every plane and whatever the voxel sizes are.
"""

import numpy as np
import pytest

from cellmap_flow.ai_annotate.backends.base import SegmentRequest
from cellmap_flow.ai_annotate.backends.fake import FakeBackend
from cellmap_flow.ai_annotate.geometry import plan_plane
from cellmap_flow.ai_annotate.organelles import ORGANELLES
from cellmap_flow.ai_annotate.pipeline import (
    build_request,
    labels_for_box,
    mask_to_write_shape,
    read_plane,
)

DIM, BRIGHT = 20, 220

# The bright block, world nm, z, y, x: [lo, hi) per axis.
BLOCK_NM = ((64, 96), (96, 128), (160, 192))


def _raw_with_block(raw_zarr, voxel_size, extent_nm, block_nm=BLOCK_NM):
    """A raw of ``extent_nm`` at ``voxel_size`` from the origin, DIM but for the block."""
    voxel_size = np.asarray(voxel_size)
    shape = tuple(int(e // v) for e, v in zip(extent_nm, voxel_size))
    data = np.full(shape, DIM, dtype=np.uint8)
    data[tuple(slice(int(lo // v), int(hi // v)) for (lo, hi), v in zip(block_nm, voxel_size))] = BRIGHT
    return raw_zarr(data, voxel_size=tuple(int(v) for v in voxel_size), offset=(0, 0, 0))


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


def test_a_plane_clipped_at_the_volume_edge_lands_too(raw_zarr):
    # A block touching the volume's y start, the point 2 voxels from it.
    block = ((64, 96), (0, 24), (160, 192))
    raw = _raw_with_block(raw_zarr, (4, 4, 4), (192, 256, 320), block)
    volume = _volume(raw, (8, 8, 8), (4, 4, 4))
    plan, labelled = _labelled_voxels(volume, (24, 32, 40), (81, 17, 177), 0, crop_size_px=16)
    assert plan.write_lo[1] == 0 and plan.write_shape == (2 + 8, 16)
    assert labelled == _expected(block, (8, 8, 8), 0, 10)


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


def test_the_read_is_padded_outside_the_raw(raw_zarr):
    # A volume larger than its raw (padded to whole chunks): the far part reads 0.
    raw = _raw_with_block(raw_zarr, (4, 4, 4), (192, 256, 320))
    volume = _volume(raw, (8, 8, 8), (4, 4, 4))
    plan = plan_plane(volume, (24, 40, 40), (81, 250, 177), 0, crop_size_px=16)
    plane = read_plane(volume, plan)
    rows_in_raw = (256 - plan.raw_offset_nm[1]) // 4
    assert np.all(plane[: int(rows_in_raw)] == DIM)
    assert np.all(plane[int(rows_in_raw):] == 0)


# --- the request ------------------------------------------------------------------


def test_build_request_states_the_resolution_sent_and_the_click(raw_zarr):
    raw = _raw_with_block(raw_zarr, (4, 4, 4), (192, 256, 320))
    volume = _volume(raw, (8, 8, 8), (4, 4, 4))
    plan = plan_plane(volume, (24, 32, 40), (81, 113, 177), 0, crop_size_px=16)
    request = build_request(read_plane(volume, plan), plan, ORGANELLES["mito"])
    assert isinstance(request, SegmentRequest)
    assert request.image.mode == "RGB"
    assert request.image.size == (plan.image_shape[1], plan.image_shape[0])
    assert request.target_rgb == tuple(ORGANELLES["mito"].rgb)
    assert request.click_px == plan.click_px == ((113 - 48) // 4, (177 - 112) // 4)
    assert "4 nm/px" in request.prompt

    edited = build_request(read_plane(volume, plan), plan, ORGANELLES["mito"], "Paint the bright block red.")
    assert "Paint the bright block red." in edited.prompt
    assert "4 nm/px" in edited.prompt

    # A backend's mask comes back the image's size and maps onto the box.
    result = FakeBackend().segment(request, "fake-threshold")
    assert mask_to_write_shape(result.mask, plan).shape == plan.write_shape


def test_build_request_refuses_a_plane_off_the_plan(raw_zarr):
    raw = _raw_with_block(raw_zarr, (4, 4, 4), (192, 256, 320))
    volume = _volume(raw, (8, 8, 8), (4, 4, 4))
    plan = plan_plane(volume, (24, 32, 40), (81, 113, 177), 0, crop_size_px=16)
    with pytest.raises(ValueError):
        build_request(np.zeros((5, 5)), plan, ORGANELLES["mito"])


# --- the mask on the box ----------------------------------------------------------


def test_mask_to_write_shape_samples_each_voxel_centre():
    from types import SimpleNamespace

    # A 4 px image over 2 voxels: voxel 0 takes pixel 1 (its centre is at
    # 1.0, the pixel [1, 2)), voxel 1 pixel 3.
    plan = SimpleNamespace(write_shape=(1, 2))
    assert mask_to_write_shape(np.array([[True, False, False, True]]), plan).tolist() == [[False, True]]
    assert mask_to_write_shape(np.array([[False, True, False, False]]), plan).tolist() == [[True, False]]
    with pytest.raises(ValueError):
        mask_to_write_shape(np.zeros((2, 2, 2)), plan)


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_labels_for_box_gives_each_object_an_id(depth_axis):
    mask = np.zeros((6, 8), dtype=bool)
    mask[0:2, 0:2] = True
    mask[3:5, 4:7] = True
    mask[5, 7] = True  # touches the second only at a corner: an object of its own
    existing = np.zeros(np.insert(mask.shape, depth_axis, 1), dtype=np.uint8)
    labels = np.squeeze(labels_for_box(mask, existing, depth_axis, count_up=False), axis=depth_axis)
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
    labels = labels_for_box(mask, existing, 1, count_up=True)[:, 0, :]
    assert labels.dtype == np.uint16
    assert {int(labels[0, 0]), int(labels[0, 4])} == {301, 302}


@pytest.mark.parametrize("depth_axis", [0, 1, 2])
def test_an_object_continuing_one_in_the_plane_beside_keeps_its_id(depth_axis):
    # The plane before holds two objects, 2 on the left and 3 on the right.
    # This plane's mask continues only the right one, and adds a new one
    # where the plane before has nothing.
    before = np.ones((6, 12), dtype=np.uint16)
    before[1:4, 0:3] = 2
    before[1:4, 5:8] = 3
    mask = np.zeros((6, 12), dtype=bool)
    mask[1:4, 5:8] = True
    mask[1:4, 9:12] = True

    def box(plane):
        return np.expand_dims(plane, depth_axis)

    existing = box(np.zeros((6, 12), dtype=np.uint16))
    labels = np.squeeze(
        labels_for_box(mask, existing, depth_axis, count_up=True, beside=[box(before), None]), axis=depth_axis
    )
    assert np.all(labels[1:4, 5:8] == 3)  # the same object, not a fresh id
    # A new object, not 2: that id is a different object right beside it.
    assert np.all(labels[1:4, 9:12] == 4)
    assert np.all(labels[~mask] == 1)  # the ids beside are only consulted, never written


def test_a_new_object_on_a_uint8_volume_skips_the_ids_beside():
    after = np.zeros((1, 4, 8), dtype=np.uint8)
    after[0, :, 0:2] = 2
    mask = np.zeros((4, 8), dtype=bool)
    mask[:, 5:7] = True
    labels = labels_for_box(mask, np.zeros((1, 4, 8), dtype=np.uint8), 0, count_up=False, beside=[None, after])[0]
    assert np.all(labels[:, 5:7] == 3)


def test_the_planes_beside_must_match_the_box():
    with pytest.raises(ValueError):
        labels_for_box(np.ones((4, 4), dtype=bool), np.zeros((1, 4, 4), dtype=np.uint8), 0, count_up=False,
                       beside=[np.zeros((1, 4, 5), dtype=np.uint8), None])


def test_labels_for_box_refuses_a_box_of_another_shape():
    with pytest.raises(ValueError):
        labels_for_box(np.ones((4, 4), dtype=bool), np.zeros((1, 4, 5), dtype=np.uint8), 0, count_up=False)
