"""Regression test: _write_crop_into_volume must resample a crop whose
native voxel size differs from the annotation volume's output_voxel_size,
rather than writing raw array indices as-is (which silently doubles/halves
the written data's physical extent -- see the jrc_axolotl-heart-1 mito005
voxel-size-mismatch bug report)."""

import os
import tempfile
import unittest

import numpy as np
import zarr

from cellmap_flow.dashboard.finetune_utils import create_annotation_volume_zarr
from cellmap_flow.dashboard.routes.finetune.yaml_crops import (
    _majority_vote_downsample,
    _write_crop_into_volume,
)
from cellmap_flow.finetune.crop_loader import CropEntry


def _make_crop(tmp, data, translation=(160.0, 160.0, 160.0), voxel_size=8.0):
    path = os.path.join(tmp, "crop.zarr")
    grp = zarr.open_group(path, mode="w")
    grp.attrs["multiscales"] = [
        {
            "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in "zyx"],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [voxel_size] * 3},
                        {"type": "translation", "translation": list(translation)},
                    ],
                }
            ],
            "version": "0.4",
        }
    ]
    grp.create_dataset("s0", data=data, chunks=data.shape)
    return path


class WriteCropIntoVolumeResamplingTests(unittest.TestCase):
    def test_mismatched_voxel_size_is_resampled_not_written_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = np.ones((8, 8, 8), dtype=np.uint8)
            crop_path = _make_crop(tmp, data, translation=(160.0, 160.0, 160.0))

            zarr_path = os.path.join(tmp, "volume.zarr")
            output_voxel_size = (16.0, 16.0, 16.0)
            success, info = create_annotation_volume_zarr(
                zarr_path=zarr_path,
                dataset_shape_voxels=(64, 64, 64),
                output_voxel_size=output_voxel_size,
                dataset_offset_nm=(0.0, 0.0, 0.0),
                chunk_size=(64, 64, 64),
                dataset_path="unused",
                model_name="test_model",
                input_size=(64, 64, 64),
                input_voxel_size=output_voxel_size,
            )
            self.assertTrue(success, info)

            volume_meta = {
                "zarr_path": zarr_path,
                "output_voxel_size": list(output_voxel_size),
                "dataset_offset_nm": [0.0, 0.0, 0.0],
            }
            entry = CropEntry(path=crop_path, fg_ids=[1])

            n_fg = _write_crop_into_volume(volume_meta, entry)

            # Crop is 8x8x8 voxels at 8nm = 64nm per side physically. At the
            # volume's 16nm voxel size that must occupy 4x4x4 voxels, not 8x8x8
            # (which would claim double the true physical extent).
            self.assertEqual(n_fg, 4 * 4 * 4)

            vol = zarr.open(zarr_path, mode="r")
            arr = vol["annotation/s0"]
            written = np.asarray(arr[:])
            fg_positions = np.argwhere(written >= 2)
            self.assertEqual(fg_positions.shape[0], 4 * 4 * 4)
            lo = fg_positions.min(axis=0)
            hi = fg_positions.max(axis=0) + 1
            self.assertTrue(np.array_equal(hi - lo, [4, 4, 4]))
            # translation (160nm) / 16nm voxel size = voxel offset 10
            self.assertTrue(np.array_equal(lo, [10, 10, 10]))

    def test_majority_vote_beats_single_corner_sample(self):
        """_majority_vote_downsample must represent each output voxel by
        the value most common across its whole block -- not by picking one
        fixed corner sample, which is what a naive nearest-neighbor zoom
        does (e.g. scipy.ndimage.zoom(..., grid_mode=True) deterministically
        picks each block's *last* voxel on every axis, confirmed directly:
        zoom(np.arange(20), 0.5, order=0, grid_mode=True) picks index 1, not
        0, for block [0,1], and so on for every block).

        Build a factor-3 block where the corner a corner-sampler would pick
        is background, but 2 of the 3 voxels (a real majority) are
        foreground -- majority vote must return foreground, unlike
        corner-sampling which would miss it entirely."""
        # z axis: 6 voxels -> 2 blocks of 3. Each block's last voxel (the
        # single-corner-sample choice for a fine/coarse=1/3 ratio) is 0, but
        # 2 of the 3 voxels per block are foreground (5).
        labels = np.zeros((6, 2, 2), dtype=np.uint8)
        labels[[0, 1, 3, 4]] = 5  # positions 0,1 (block 0) and 3,4 (block 1) fg
        down = _majority_vote_downsample(labels, factors=(3, 1, 1))
        self.assertEqual(down.shape, (2, 2, 2))
        self.assertTrue(np.all(down == 5))

    def _first_written_voxel(self, tmp, crop_path):
        zarr_path = os.path.join(tmp, "volume.zarr")
        output_voxel_size = (16.0, 16.0, 16.0)
        success, info = create_annotation_volume_zarr(
            zarr_path=zarr_path,
            dataset_shape_voxels=(32, 32, 32),
            output_voxel_size=output_voxel_size,
            dataset_offset_nm=(0.0, 0.0, 0.0),
            chunk_size=(32, 32, 32),
            dataset_path="unused",
            model_name="test_model",
            input_size=(32, 32, 32),
            input_voxel_size=output_voxel_size,
        )
        self.assertTrue(success, info)

        volume_meta = {
            "zarr_path": zarr_path,
            "output_voxel_size": list(output_voxel_size),
            "dataset_offset_nm": [0.0, 0.0, 0.0],
        }
        _write_crop_into_volume(volume_meta, CropEntry(path=crop_path, fg_ids=[1]))

        written = np.asarray(zarr.open(zarr_path, mode="r")["annotation/s0"][:])
        return np.argwhere(written >= 2).min(axis=0)

    def test_a_downsampled_crop_is_placed_corner_to_corner(self):
        """Both translations are voxel-0 centres, so the corners are 4 nm
        below the crop's 70 and 8 nm below the volume's 0. The crop's corner,
        66, is 74 nm = 4.6 volume voxels up, so it lands in voxel 5. Reading
        the translation itself as the position gave round(70/16) = 4."""
        with tempfile.TemporaryDirectory() as tmp:
            data = np.ones((8, 8, 8), dtype=np.uint8)
            crop_path = _make_crop(tmp, data, translation=(70.0, 70.0, 70.0))
            self.assertTrue(np.array_equal(self._first_written_voxel(tmp, crop_path), [5, 5, 5]))

    def test_a_4x_downsampled_crop_is_placed_corner_to_corner(self):
        """The fixed +fine/2 that stood in for the corners was only right at
        2x. A 4 nm crop centred at 4 has its corner at 2; its first 16 nm
        block spans [2, 18), mostly volume voxel 1 ([8, 24)), where +fine/2
        rounded (4 + 2) / 16 down to voxel 0."""
        with tempfile.TemporaryDirectory() as tmp:
            data = np.ones((16, 16, 16), dtype=np.uint8)
            crop_path = _make_crop(tmp, data, translation=(4.0, 4.0, 4.0), voxel_size=4.0)
            self.assertTrue(np.array_equal(self._first_written_voxel(tmp, crop_path), [1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
