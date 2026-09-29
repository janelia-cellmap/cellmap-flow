"""The raw layer is drawn where its data is.

get_raw_layer handed LocalVolume the offset in nm as ``voxel_offset``, which
counts whole voxels: a dataset at 80 nm on 8 nm voxels was drawn at 640 nm,
and an OME corner such as -4 nm could not be expressed at all. The position
now goes in the source transform, in voxels of the layer's own dimensions.
"""

import numpy as np
import zarr

from cellmap_flow.utils.scale_pyramid import get_raw_layer


def _source(layer):
    source = layer.to_json()["source"]
    return source[0] if isinstance(source, list) else source


def _translation(layer):
    matrix = np.array(_source(layer)["transform"]["matrix"])
    assert np.array_equal(matrix[:, :-1], np.eye(3))
    return matrix[:, -1].tolist()


def test_a_janelia_pyramid_is_drawn_from_its_corner(tmp_path):
    group = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w").create_group("em")
    levels = [("s0", 8.0, 0.0), ("s1", 16.0, 4.0)]
    for i, (name, _, _) in enumerate(levels):
        group.create_dataset(name, data=np.zeros((16 >> i,) * 3, np.uint8), chunks=(8, 8, 8))
    group.attrs["multiscales"] = [
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

    layer = get_raw_layer(str(tmp_path / "raw.zarr" / "em"), normalize=False)
    # Every level's corner is -4 nm: half an 8 nm voxel below the origin.
    assert _source(layer)["transform"]["outputDimensions"]["z"] == [8e-9, "m"]
    assert _translation(layer) == [-0.5] * 3


def test_an_offset_array_is_drawn_at_its_offset_not_at_offset_voxels(tmp_path):
    arr = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w").create_dataset(
        "raw", data=np.zeros((4, 4, 4), np.uint8)
    )
    arr.attrs["resolution"] = [8, 8, 8]
    arr.attrs["offset"] = [80, 40, 40]

    layer = get_raw_layer(str(tmp_path / "raw.zarr" / "raw"), normalize=False)
    assert _translation(layer) == [10.0, 5.0, 5.0]
