"""Dataset metadata and reads come out spatial, C-ordered and in nanometers."""

import json
import os

import numpy as np
import tensorstore as ts
import zarr
from funlib.geometry import Roi

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.utils import ds
from cellmap_flow.utils.ds import get_ds_info


def _ome(axes, datasets):
    return [{"version": "0.4", "axes": axes, "datasets": datasets}]


def _space(units="nanometer", names="zyx"):
    return [{"name": n, "type": "space", "unit": units} for n in names]


def _level(path, scale, translation=None):
    transforms = [{"type": "scale", "scale": scale}]
    if translation is not None:
        transforms.append({"type": "translation", "translation": translation})
    return {"path": path, "coordinateTransformations": transforms}


def test_n5_is_read_in_the_same_axis_order_as_its_metadata(tmp_path):
    from zarr.n5 import N5FSStore

    path = str(tmp_path / "a.n5")
    data = np.arange(10 * 20 * 30, dtype=np.uint16).reshape(10, 20, 30)  # z, y, x
    arr = zarr.open(N5FSStore(path), mode="w").create_dataset(
        "raw", data=data, chunks=(5, 10, 15)
    )
    arr.attrs["resolution"] = [3, 2, 1]  # N5 attributes are x, y, z
    arr.attrs["offset"] = [0, 0, 0]

    voxel_size, _, shape, _, axes, _ = get_ds_info(path + "/raw")
    assert tuple(voxel_size) == (1, 2, 3) and tuple(shape) == (10, 20, 30)
    assert axes == ["z", "y", "x"]

    idi = ImageDataInterface(path + "/raw")
    assert idi.ts.shape == (10, 20, 30)
    got = idi.to_ndarray_ts(Roi((0, 0, 0), (2, 4, 6)))
    np.testing.assert_array_equal(got, data[:2, :2, :2])


def test_precomputed_keeps_all_three_spatial_axes(tmp_path):
    path = str(tmp_path / "pc")
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
            "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
            "scale_metadata": {
                "size": [20, 10, 2],
                "resolution": [4, 8, 16],
                "encoding": "raw",
                "chunk_size": [20, 10, 2],
            },
        },
        create=True,
    ).result()
    xyz = np.arange(20 * 10 * 2, dtype=np.uint8).reshape(20, 10, 2)
    store[..., 0] = xyz

    voxel_size, _, shape, _, axes, _ = get_ds_info("precomputed://" + path)
    assert tuple(voxel_size) == (16, 8, 4)
    assert tuple(shape) == (2, 10, 20)
    assert axes == ["z", "y", "x"]

    idi = ImageDataInterface("precomputed://" + path)
    got = idi.to_ndarray_ts(Roi((0, 0, 0), (32, 80, 80)))
    np.testing.assert_array_equal(got, xyz.transpose(2, 1, 0))


def test_border_reads_of_an_array_with_a_fill_value(tmp_path):
    root = zarr.open(str(tmp_path / "f.zarr"), mode="w")
    arr = root.create_dataset("raw", data=np.ones((4, 4, 4), np.uint8), fill_value=7)
    arr.attrs["resolution"] = [1, 1, 1]
    arr.attrs["offset"] = [0, 0, 0]
    idi = ImageDataInterface(str(tmp_path / "f.zarr" / "raw"))
    got = idi.to_ndarray_ts(Roi((-1, 0, 0), (2, 1, 1)))
    assert got.ravel().tolist() == [0, 1]


def test_s3_goes_through_the_generic_remote_reader(tmp_path, monkeypatch):
    local = str(tmp_path / "data.zarr")
    group = zarr.open(local, mode="w").create_group("raw")
    group.create_dataset("s0", data=np.zeros((4, 4, 4), np.uint8))
    group.create_dataset("s1", data=np.zeros((2, 2, 2), np.uint8))
    group.attrs["multiscales"] = _ome(
        _space(),
        [_level("s0", [4, 4, 4], [8, 8, 8]), _level("s1", [8, 8, 8], [10, 10, 10])],
    )
    monkeypatch.setattr(
        ds,
        "_open_zarr",
        lambda p, mode="r": zarr.open(p.replace("s3://bucket", str(tmp_path)), mode=mode),
    )

    info = get_ds_info("s3://bucket/data.zarr/raw/s1")
    assert len(info) == 6
    voxel_size, chunk_shape, shape, roi, axes, filetype = info
    assert tuple(voxel_size) == (8, 8, 8)
    assert tuple(roi.offset) == (10, 10, 10)
    assert tuple(shape) == (2, 2, 2) and axes == ["z", "y", "x"]


def test_multichannel_ome_zarr_keeps_its_voxel_size(tmp_path):
    path = str(tmp_path / "mc.zarr")
    root = zarr.open(path, mode="w")
    root.create_dataset("s0", data=np.zeros((2, 4, 4, 4), np.uint8), chunks=(1, 2, 2, 2))
    root.attrs["multiscales"] = _ome(
        [{"name": "c", "type": "channel"}] + _space(),
        [_level("s0", [1, 8, 4, 4], [0, 80, 40, 40])],
    )
    voxel_size, chunk_shape, shape, roi, _, _ = get_ds_info(path + "/s0")
    assert tuple(voxel_size) == (8, 4, 4)
    assert tuple(roi.offset) == (80, 40, 40)
    assert tuple(shape) == (4, 4, 4) and tuple(chunk_shape) == (2, 2, 2)


def test_micrometer_units_are_converted_to_nanometers(tmp_path):
    path = str(tmp_path / "um.zarr")
    root = zarr.open(path, mode="w")
    root.create_dataset("s0", data=np.zeros((4, 4, 4), np.uint8))
    root.create_dataset("s1", data=np.zeros((2, 2, 2), np.uint8))
    root.attrs["multiscales"] = _ome(
        _space("micrometer"),
        [
            _level("s0", [0.008, 0.004, 0.004], [0.08, 0.04, 0.04]),
            _level("s1", [0.016, 0.008, 0.008]),
        ],
    )
    voxel_size, _, _, roi, _, _ = get_ds_info(path + "/s0")
    assert tuple(voxel_size) == (8, 4, 4)
    assert tuple(roi.offset) == (80, 40, 40)
    # Scale selection compares nanometers too.
    assert ImageDataInterface(path, voxel_size=(16, 8, 8)).path.endswith("s1")


def test_array_at_the_container_root_reads_its_own_attrs(tmp_path):
    path = str(tmp_path / "root.zarr")
    arr = zarr.open(path, mode="w", shape=(4, 4, 4), dtype=np.uint8)
    arr.attrs["resolution"] = [8, 4, 4]
    arr.attrs["offset"] = [80, 40, 40]
    voxel_size, _, _, roi, _, _ = get_ds_info(path)
    assert tuple(voxel_size) == (8, 4, 4)
    assert tuple(roi.offset) == (80, 40, 40)


def test_fortran_memory_order_is_not_an_axis_order(tmp_path):
    path = str(tmp_path / "forder.zarr")
    arr = zarr.open(path, mode="w").create_dataset(
        "raw", data=np.zeros((4, 4, 4), np.uint8), order="F"
    )
    arr.attrs["resolution"] = [8, 4, 2]
    arr.attrs["offset"] = [0, 0, 0]
    assert tuple(get_ds_info(path + "/raw")[0]) == (8, 4, 2)


def _two_level_group(path):
    root = zarr.open(path, mode="w")
    root.create_dataset("s0", data=np.zeros((4, 4, 4), np.uint8))
    root.create_dataset("s1", data=np.zeros((2, 2, 2), np.uint8))
    root.attrs["multiscales"] = _ome(
        _space(), [_level("s0", [8, 8, 8]), _level("s1", [16, 16, 16])]
    )
    return path


def test_a_v2_group_without_a_voxel_size_opens_its_finest_scale(tmp_path):
    idi = ImageDataInterface(_two_level_group(str(tmp_path / "ms.zarr")))
    assert idi.path.endswith("s0")
    assert tuple(idi.voxel_size) == (8, 8, 8)


def _v3_multichannel(tmp_path):
    path = str(tmp_path / "v3.zarr")
    for level, (shape, chunks) in enumerate(
        [((2, 4, 4, 4), (1, 2, 2, 2)), ((2, 2, 2, 2), (1, 1, 1, 1))]
    ):
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": f"{path}/s{level}"},
                "metadata": {
                    "shape": list(shape),
                    "data_type": "uint8",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": list(chunks)},
                    },
                },
            },
            create=True,
        ).result()
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "multiscales": _ome(
                            [{"name": "c", "type": "channel"}] + _space(),
                            [_level("s0", [1, 8, 8, 8]), _level("s1", [1, 16, 16, 16])],
                        )
                    }
                },
            },
            f,
        )
    return path


def test_v3_chunk_shape_is_spatial_like_the_shape(tmp_path):
    path = _v3_multichannel(tmp_path)
    _, chunk_shape, shape, _, _, _ = get_ds_info(path + "/s0")
    assert tuple(shape) == (4, 4, 4) and tuple(chunk_shape) == (2, 2, 2)


def test_closest_raw_scale_from_a_per_scale_path(tmp_path):
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

    v3 = _v3_multichannel(tmp_path)
    assert tuple(get_raw_closest_scale(v3 + "/s0", (16, 16, 16))) == (16, 16, 16)
    v2 = _two_level_group(str(tmp_path / "ms.zarr"))
    assert tuple(get_raw_closest_scale(v2 + "/s0", (16, 16, 16))) == (16, 16, 16)
