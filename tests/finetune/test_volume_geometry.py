"""An annotation volume trains where Neuroglancer drew it.

A volume's ``dataset_offset_nm`` is also its OME translation, which is voxel
0's centre, and Neuroglancer drew the volume that way while it was painted.
The trainer read the value as a corner, and the raw's translation as a
corner too, so over a Janelia pyramid every label was paired with raw
fractions of a voxel away from what it had been painted on.
"""

import numpy as np
import pytest
import zarr

from cellmap_flow.dashboard.finetune_utils import create_annotation_volume_zarr
from cellmap_flow.finetune.virtual_dataset import (
    VirtualPatchDataset,
    new_volume_geometry,
    volume_corner_nm,
)

# Janelia pyramids: translation = scale/2 - 4, so every level's corner is -4 nm.
LEVELS = [("s0", 8.0, 0.0), ("s1", 16.0, 4.0)]


def _multiscales(levels):
    return [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": name,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [vs] * 3},
                        {"type": "translation", "translation": [t] * 3},
                    ],
                }
                for name, vs, t in levels
            ],
        }
    ]


@pytest.fixture
def raw(tmp_path):
    """A 32^3 s0 whose voxels hold z index + 1, with a matching s1."""
    group = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w").create_group("em")
    for i, (name, _, _) in enumerate(LEVELS):
        n = 32 >> i
        data = np.arange(1, n + 1, dtype=np.uint8)[:, None, None]
        group.create_dataset(name, data=np.broadcast_to(data, (n, n, n)).copy(), chunks=(8, 8, 8))
    group.attrs["multiscales"] = _multiscales(LEVELS)
    return str(tmp_path / "raw.zarr" / "em")


def _volume(tmp_path, raw, output_size):
    offset, shape = new_volume_geometry(raw, (16.0,) * 3, output_size)
    path = str(tmp_path / "vol.zarr")
    ok, info = create_annotation_volume_zarr(
        zarr_path=path,
        dataset_shape_voxels=shape,
        output_voxel_size=np.array([16.0] * 3),
        dataset_offset_nm=offset,
        chunk_size=np.array(output_size),
        dataset_path=raw,
        model_name="m",
        input_size=np.array(output_size) * 2,
        input_voxel_size=np.array([8.0] * 3),
    )
    assert ok, info
    return path


def test_a_new_volume_sits_on_the_raw_grid(tmp_path, raw):
    offset, shape = new_volume_geometry(raw, (16.0,) * 3, (4, 4, 4))
    # s1's corner is -4, so voxel 0's centre is 4: the same value volumes
    # were given before, when s1's translation was read as its corner.
    assert offset.tolist() == [4.0] * 3
    assert volume_corner_nm(offset, (16.0,) * 3).tolist() == [-4.0] * 3
    assert shape.tolist() == [16] * 3

    path = _volume(tmp_path, raw, (4, 4, 4))
    transforms = zarr.open_group(path, mode="r")["annotation"].attrs["multiscales"][0][
        "datasets"
    ][0]["coordinateTransformations"]
    assert transforms[1]["translation"] == [4.0] * 3


@pytest.mark.parametrize("output_size", [2, 3])
def test_a_label_is_paired_with_the_raw_it_covers(tmp_path, raw, output_size):
    """Annotation voxel v spans raw voxels 2v and 2v + 1, exactly."""
    size = (output_size,) * 3
    path = _volume(tmp_path, raw, size)
    zarr.open_group(path, mode="r+")["annotation/s0"][5, 8, 8] = 2

    ds = VirtualPatchDataset(
        volume_zarr_path=path,
        raw_dataset_path=raw,
        input_size_voxels=(2 * output_size,) * 3,
        output_size_voxels=size,
        input_voxel_size_nm=(8.0,) * 3,
        output_voxel_size_nm=(16.0,) * 3,
        patches_per_epoch=1,
        jitter_voxels=(0, 0, 0),
    )
    raw_patch, ann_patch = ds[0]

    # Which annotation z the patch starts at, from where the label landed.
    z0 = 5 - int(np.argwhere(ann_patch[0].numpy() == 2)[0][0])
    # Raw voxel i holds i + 1, and annotation voxel z covers raw 2z, 2z + 1.
    expected = np.arange(2 * z0, 2 * z0 + 2 * output_size) + 1
    assert raw_patch[0, :, 0, 0].numpy().tolist() == expected.tolist()


def test_a_good_region_patch_is_paired_too(tmp_path, raw):
    """A region's centre falls anywhere, and the patch around it must still
    be whole annotation voxels with the raw read around that same patch."""
    path = _volume(tmp_path, raw, (2, 2, 2))
    # Training needs something painted; this is well away from the region.
    zarr.open_group(path, mode="r+")["annotation/s0"][14, 14, 14] = 2
    ds = VirtualPatchDataset(
        volume_zarr_path=path,
        raw_dataset_path=raw,
        input_size_voxels=(4, 4, 4),
        output_size_voxels=(2, 2, 2),
        input_voxel_size_nm=(8.0,) * 3,
        output_voxel_size_nm=(16.0,) * 3,
        patches_per_epoch=1,
        # Centred 3 nm off the voxel grid.
        good_regions=[{"id": "r", "offset_nm": [63.0] * 3, "shape_nm": [32.0] * 3}],
        rehearsal_fraction=1.0,
    )
    raw_patch, _, anchor = ds[0]
    assert float(anchor.sum()) == anchor.numel()
    # The region spans [63, 95) nm, from annotation voxel 4.19 (corner -4):
    # the nearest whole patch starts at voxel 4, which is raw 8.
    assert raw_patch[0, :, 0, 0].numpy().tolist() == [9, 10, 11, 12]
