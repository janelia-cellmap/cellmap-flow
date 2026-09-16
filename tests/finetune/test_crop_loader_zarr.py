"""Tests proving crop_loader.py reads Zarr v2 and v3 crop sources identically."""

import json
import os
import tempfile
import unittest

import numpy as np
import zarr

from cellmap_flow.finetune.crop_loader import _open_array, _read_voxel_size_and_offset

try:
    import pytest

    _importorskip = pytest.importorskip
except ImportError:  # pragma: no cover - pytest is a test-only dependency
    _importorskip = None

if _importorskip is not None:
    _importorskip("tensorstore")


def _axes():
    return [{"name": ax, "type": "space", "unit": "nanometer"} for ax in "zyx"]


def _v2_multiscale_group(tmp, data, voxel_size, translation):
    path = os.path.join(tmp, "crop_v2.zarr")
    grp = zarr.open_group(path, mode="w")
    grp.attrs["multiscales"] = [
        {
            "axes": _axes(),
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": list(voxel_size)},
                        {"type": "translation", "translation": list(translation)},
                    ],
                }
            ],
            "version": "0.4",
        }
    ]
    grp.create_dataset("s0", data=data, chunks=data.shape)
    return path


def _v3_multiscale_group(tmp, data, voxel_size, translation):
    import tensorstore as ts

    path = os.path.join(tmp, "crop_v3.zarr")
    os.makedirs(path, exist_ok=True)
    meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "multiscales": [
                    {
                        "axes": _axes(),
                        "datasets": [
                            {
                                "path": "s0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": list(voxel_size)},
                                    {
                                        "type": "translation",
                                        "translation": list(translation),
                                    },
                                ],
                            }
                        ],
                        "version": "0.5",
                    }
                ]
            }
        },
    }
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(meta, f)

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": os.path.join(path, "s0")},
        "metadata": {
            "shape": list(data.shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(data.shape)},
            },
            "data_type": str(data.dtype),
        },
        "create": True,
    }
    store = ts.open(spec).result()
    store[:] = data
    return path


class CropLoaderZarrTests(unittest.TestCase):
    def test_v2_multiscale_group_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
            path = _v2_multiscale_group(
                tmp, data, voxel_size=(4.0, 4.0, 4.0), translation=(10.0, 20.0, 30.0)
            )

            sub, scale, translation = _read_voxel_size_and_offset(path)
            self.assertEqual(sub, ("s0",))
            self.assertTrue(np.array_equal(scale, [4.0, 4.0, 4.0]))
            self.assertTrue(np.array_equal(translation, [10.0, 20.0, 30.0]))

            arr = _open_array(path, sub)
            self.assertTrue(np.array_equal(arr[:], data))

    def test_v3_multiscale_group_matches_v2_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
            path = _v3_multiscale_group(
                tmp, data, voxel_size=(4.0, 4.0, 4.0), translation=(10.0, 20.0, 30.0)
            )

            sub, scale, translation = _read_voxel_size_and_offset(path)
            self.assertEqual(sub, ("s0",))
            self.assertTrue(np.array_equal(scale, [4.0, 4.0, 4.0]))
            self.assertTrue(np.array_equal(translation, [10.0, 20.0, 30.0]))

            arr = _open_array(path, sub)
            self.assertTrue(np.array_equal(arr[:], data))

    def test_v3_plain_array_with_transform_attrs(self):
        import tensorstore as ts

        with tempfile.TemporaryDirectory() as tmp:
            arr_path = os.path.join(tmp, "arr")
            data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            spec = {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": arr_path},
                "metadata": {
                    "shape": list(data.shape),
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": list(data.shape)},
                    },
                    "data_type": str(data.dtype),
                    "attributes": {
                        "transform": {
                            "scale": [2.0, 2.0, 2.0],
                            "translate": [1.0, 1.0, 1.0],
                        }
                    },
                },
                "create": True,
            }
            store = ts.open(spec).result()
            store[:] = data

            sub, scale, translation = _read_voxel_size_and_offset(arr_path)
            self.assertEqual(sub, ())
            self.assertTrue(np.array_equal(scale, [2.0, 2.0, 2.0]))
            self.assertTrue(np.array_equal(translation, [1.0, 1.0, 1.0]))

            arr = _open_array(arr_path, sub)
            self.assertTrue(np.array_equal(arr[:], data))

    def test_v3_group_with_s0_only_defaults_to_unit_scale(self):
        import tensorstore as ts

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "crop_v3.zarr")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "zarr.json"), "w") as f:
                json.dump(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}}, f
                )
            data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
            spec = {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": os.path.join(path, "s0")},
                "metadata": {
                    "shape": list(data.shape),
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": list(data.shape)},
                    },
                    "data_type": str(data.dtype),
                },
                "create": True,
            }
            store = ts.open(spec).result()
            store[:] = data

            sub, scale, translation = _read_voxel_size_and_offset(path)
            self.assertEqual(sub, ("s0",))
            self.assertTrue(np.array_equal(scale, [1.0, 1.0, 1.0]))
            self.assertTrue(np.array_equal(translation, [0.0, 0.0, 0.0]))

    def test_v3_group_with_neither_multiscales_nor_s0_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "crop_v3.zarr")
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "zarr.json"), "w") as f:
                json.dump(
                    {"zarr_format": 3, "node_type": "group", "attributes": {}}, f
                )

            with self.assertRaises(ValueError):
                _read_voxel_size_and_offset(path)


if __name__ == "__main__":
    unittest.main()


class MissingCropPathTests(unittest.TestCase):
    """A path that does not exist must name itself.

    zarr.open reports the path *inside* the store, which for a missing
    directory is the empty string -- "nothing found at path ''" identified
    neither the crop nor the typo. A real manifest listing
    .../rc_amphiuma-means-liver-1/... instead of .../jrc_amphiuma-.../ hit
    exactly that, twice, so the diagnostics are worth pinning down.
    """

    def test_names_the_missing_path_and_where_it_breaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = os.path.join(tmp, "annotations")
            os.makedirs(existing)
            missing = os.path.join(existing, "typoed-dataset", "crop.zarr", "s0")

            with self.assertRaises(FileNotFoundError) as caught:
                _read_voxel_size_and_offset(missing)

            message = str(caught.exception)
            self.assertIn(missing, message)
            # The deepest surviving ancestor is what points at the bad
            # component -- here, that "typoed-dataset" is the first thing
            # that does not exist.
            self.assertIn(existing, message)

    def test_reports_when_nothing_in_the_path_exists(self):
        with self.assertRaises(FileNotFoundError) as caught:
            _read_voxel_size_and_offset("/nonexistent-root-for-tests/a/b.zarr")
        self.assertIn("/nonexistent-root-for-tests/a/b.zarr", str(caught.exception))

    def test_existing_crop_still_reads_normally(self):
        """The guard must not shadow the normal path."""
        with tempfile.TemporaryDirectory() as tmp:
            data = np.zeros((4, 4, 4), dtype=np.uint8)
            path = _v2_multiscale_group(tmp, data, (8, 8, 8), (16, 32, 48))
            sub, voxel_size, offset = _read_voxel_size_and_offset(path)
            self.assertEqual(tuple(voxel_size), (8, 8, 8))
            self.assertEqual(tuple(offset), (16, 32, 48))
            self.assertTrue(sub)
