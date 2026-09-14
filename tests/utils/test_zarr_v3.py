"""Tests for Zarr v3 (``zarr.json``) read support in ``cellmap_flow.utils.zarr_v3``."""

import json
import os
import tempfile
import unittest

import numpy as np

pytest_importorskip = None
try:
    import pytest

    pytest_importorskip = pytest.importorskip
except ImportError:  # pragma: no cover - pytest is a test-only dependency
    pass

if pytest_importorskip is not None:
    pytest_importorskip("tensorstore")

from cellmap_flow.utils import zarr_v3


def _create_v3_array(path, data, chunk_shape=None, attributes=None):
    import tensorstore as ts

    chunk_shape = chunk_shape or list(data.shape)
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": path},
        "metadata": {
            "shape": list(data.shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": chunk_shape},
            },
            "data_type": str(data.dtype),
        },
        "create": True,
    }
    if attributes:
        spec["metadata"]["attributes"] = attributes
    store = ts.open(spec).result()
    store[:] = data
    return path


def _write_group_zarr_json(group_path, multiscales=None):
    os.makedirs(group_path, exist_ok=True)
    attributes = {}
    if multiscales is not None:
        attributes["ome"] = {"multiscales": multiscales}
    meta = {"zarr_format": 3, "node_type": "group", "attributes": attributes}
    with open(os.path.join(group_path, "zarr.json"), "w") as f:
        json.dump(meta, f)


def _multiscales(datasets):
    axes = [{"name": ax, "type": "space", "unit": "nanometer"} for ax in "zyx"]
    return [{"axes": axes, "datasets": datasets, "version": "0.5"}]


def _dataset_entry(path, scale, translation=(0.0, 0.0, 0.0)):
    return {
        "path": path,
        "coordinateTransformations": [
            {"type": "scale", "scale": list(scale)},
            {"type": "translation", "translation": list(translation)},
        ],
    }


class ZarrV3Tests(unittest.TestCase):
    def test_is_v3_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr_path = os.path.join(tmp, "arr")
            _create_v3_array(arr_path, np.arange(8, dtype=np.uint8).reshape(2, 2, 2))
            self.assertTrue(zarr_v3.is_v3_container(arr_path))
            self.assertFalse(zarr_v3.is_v3_container(tmp))

    def test_plain_array_with_transform_attrs(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr_path = os.path.join(tmp, "arr")
            data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
            _create_v3_array(
                arr_path,
                data,
                chunk_shape=[2, 2, 2],
                attributes={
                    "transform": {
                        "scale": [8.0, 8.0, 8.0],
                        "translate": [100.0, 200.0, 300.0],
                    }
                },
            )

            voxel_size, chunk_shape, shape, roi, axes_names, filetype = (
                zarr_v3.get_ds_info_v3(arr_path)
            )
            self.assertEqual(tuple(voxel_size), (8, 8, 8))
            self.assertEqual(tuple(roi.offset), (100, 200, 300))
            self.assertEqual(tuple(shape), (4, 4, 4))
            self.assertEqual(filetype, "zarr")

            arr = zarr_v3.open_array_v3(arr_path)
            self.assertEqual(arr.shape, (4, 4, 4))
            self.assertTrue(np.array_equal(arr[:], data))
            self.assertTrue(np.array_equal(arr[0:2], data[0:2]))

    def test_plain_array_no_attrs_defaults_to_unit_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            arr_path = os.path.join(tmp, "arr")
            data = np.zeros((2, 2, 2), dtype=np.uint8)
            _create_v3_array(arr_path, data)

            voxel_size, chunk_shape, shape, roi, axes_names, filetype = (
                zarr_v3.get_ds_info_v3(arr_path)
            )
            self.assertEqual(tuple(voxel_size), (1, 1, 1))
            self.assertEqual(tuple(roi.offset), (0, 0, 0))

    def test_multiscale_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_path = os.path.join(tmp, "group")
            _write_group_zarr_json(
                group_path,
                multiscales=_multiscales(
                    [
                        _dataset_entry("s0", (4.0, 4.0, 4.0)),
                        _dataset_entry("s1", (8.0, 8.0, 8.0)),
                    ]
                ),
            )
            data_s0 = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
            data_s1 = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            _create_v3_array(os.path.join(group_path, "s0"), data_s0, chunk_shape=[2, 2, 2])
            _create_v3_array(os.path.join(group_path, "s1"), data_s1, chunk_shape=[2, 2, 2])

            offsets, resolutions, shapes = zarr_v3.get_scale_info_v3(group_path)
            self.assertEqual(resolutions["s0"], [4.0, 4.0, 4.0])
            self.assertEqual(shapes["s1"], (2, 2, 2))

            scale, offset, shape = zarr_v3.find_closest_scale_v3(
                group_path, [8.0, 8.0, 8.0]
            )
            self.assertEqual(scale, "s1")

            voxel_size, chunk_shape, shape, roi, axes_names, filetype = (
                zarr_v3.get_ds_info_v3(group_path)
            )
            self.assertEqual(tuple(voxel_size), (4, 4, 4))
            self.assertEqual(axes_names, ["z", "y", "x"])

            arr = zarr_v3.open_array_v3(os.path.join(group_path, "s1"))
            self.assertTrue(np.array_equal(arr[:], data_s1))

    def test_get_ds_info_on_scale_array_path_uses_group_transform(self):
        """Regression test: calling get_ds_info_v3 directly on a per-scale
        array path (e.g. .../group/s1), as ImageDataInterface does once it
        has resolved a scale, must use the ancestor group's multiscales
        transform for that scale rather than defaulting to unit scale (each
        v3 array has its own zarr.json, so naively it looks like its own
        "container")."""
        with tempfile.TemporaryDirectory() as tmp:
            group_path = os.path.join(tmp, "group")
            _write_group_zarr_json(
                group_path,
                multiscales=_multiscales(
                    [
                        _dataset_entry("s0", (4.0, 4.0, 4.0), translation=(1.0, 2.0, 3.0)),
                        _dataset_entry("s1", (8.0, 8.0, 8.0), translation=(1.0, 2.0, 3.0)),
                    ]
                ),
            )
            data_s1 = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            _create_v3_array(
                os.path.join(group_path, "s0"),
                np.arange(64, dtype=np.uint8).reshape(4, 4, 4),
                chunk_shape=[2, 2, 2],
            )
            _create_v3_array(os.path.join(group_path, "s1"), data_s1, chunk_shape=[2, 2, 2])

            voxel_size, chunk_shape, shape, roi, axes_names, filetype = (
                zarr_v3.get_ds_info_v3(os.path.join(group_path, "s1"))
            )
            self.assertEqual(tuple(voxel_size), (8, 8, 8))
            self.assertEqual(tuple(roi.offset), (1, 2, 3))
            self.assertEqual(tuple(shape), (2, 2, 2))

    def test_find_closest_scale_v3_none_target_defaults_to_first_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_path = os.path.join(tmp, "group")
            _write_group_zarr_json(
                group_path,
                multiscales=_multiscales(
                    [
                        _dataset_entry("s0", (4.0, 4.0, 4.0)),
                        _dataset_entry("s1", (8.0, 8.0, 8.0)),
                    ]
                ),
            )
            _create_v3_array(
                os.path.join(group_path, "s0"),
                np.zeros((4, 4, 4), dtype=np.uint8),
                chunk_shape=[2, 2, 2],
            )
            _create_v3_array(
                os.path.join(group_path, "s1"),
                np.zeros((2, 2, 2), dtype=np.uint8),
                chunk_shape=[2, 2, 2],
            )

            scale, offset, shape = zarr_v3.find_closest_scale_v3(group_path, None)
            self.assertEqual(scale, "s0")

    def test_group_with_s0_only_defaults_to_unit_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_path = os.path.join(tmp, "group")
            _write_group_zarr_json(group_path)
            data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            _create_v3_array(os.path.join(group_path, "s0"), data)

            voxel_size, chunk_shape, shape, roi, axes_names, filetype = (
                zarr_v3.get_ds_info_v3(group_path)
            )
            self.assertEqual(tuple(voxel_size), (1, 1, 1))
            self.assertEqual(tuple(roi.offset), (0, 0, 0))

    def test_group_with_no_children_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            group_path = os.path.join(tmp, "group")
            _write_group_zarr_json(group_path)

            with self.assertRaises(RuntimeError):
                zarr_v3.get_ds_info_v3(group_path)


if __name__ == "__main__":
    unittest.main()
