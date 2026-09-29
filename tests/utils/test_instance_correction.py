"""Instance-correction volumes: where they are seeded, and what they hold."""

import numpy as np
import pytest
import zarr

from cellmap_flow.dashboard import finetune_utils as fu

CENTRE = [8.0, 168.0, 8.0]  # voxel 0's centre: corner (0, 160, 0) at 16 nm
OME = {
    "multiscales": [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [16.0] * 3},
                        {"type": "translation", "translation": CENTRE},
                    ],
                }
            ],
        }
    ]
}


def make_instances(path, group_attrs=None, s0_attrs=None):
    group = zarr.open_group(str(path), mode="w")
    group.attrs.update(group_attrs or {})
    ids = np.zeros((8, 8, 8), np.uint32)
    ids[2:4, 2:4, 2:4] = 7
    s0 = group.create_dataset("s0", data=ids, chunks=(4, 4, 4))
    s0.attrs.update(s0_attrs or {})
    return str(path)


@pytest.mark.parametrize(
    "group_attrs, s0_attrs",
    [
        (None, {"resolution": [16] * 3, "offset": [0, 160, 0]}),  # offset is the corner
        (OME, None),  # OME only: the translation is already the centre
    ],
    ids=["resolution-offset", "ome"],
)
def test_a_seeded_volume_lies_on_its_segmentation(tmp_path, group_attrs, s0_attrs):
    ok, path = fu.create_instance_annotation_volume_from_seg(
        str(tmp_path / "roi_annotation.zarr"),
        make_instances(tmp_path / "instances.zarr", group_attrs, s0_attrs),
        "/raw.zarr",
        "model",
        input_size=[12] * 3,
        input_voxel_size=[16] * 3,
        dilation_radius_voxels=1,
    )
    assert ok, path

    volume = zarr.open_group(path, mode="r")
    transforms = volume["annotation"].attrs["multiscales"][0]["datasets"][0]
    assert transforms["coordinateTransformations"][1]["translation"] == CENTRE
    assert volume.attrs["dataset_offset_nm"] == CENTRE
    assert volume.attrs["type"] == "annotation_volume"

    labels = volume["annotation/s0"][:]
    assert labels.dtype == np.uint16
    assert labels[2, 2, 2] == 8  # instance 7 is label 8
    assert labels[1, 2, 2] == 1  # the background shell, one voxel out
    assert labels[6, 6, 6] == 0  # unannotated
