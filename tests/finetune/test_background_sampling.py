"""Painted background corrections are sampled; an all-background session trains.

Patches were only centred on foreground voxels (>= 2). A background-only
correction -- painting 1 where the model hallucinates -- further than about
half a patch from any foreground was never in a patch, so the false-positive
fix did nothing, and a session that painted only background raised "no
foreground voxels" (on a restart, outside the CLI's try, killing the job).
"""

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset


def _volume(tmp_path, arr, imported_crops=()):
    path = tmp_path / "vol.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.create_group("annotation").create_dataset(
        "s0", shape=arr.shape, chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    root["annotation"]["s0"][:] = arr
    root.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    root.attrs["imported_crops"] = list(imported_crops)
    return str(path)


def _dataset(path):
    return VirtualPatchDataset(
        volume_zarr_path=path, raw_dataset_path="/unused", input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4), input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16), seed=0,
    )


def _as_set(index):
    return {tuple(v) for v in index.tolist()}


def test_a_painted_background_correction_far_from_foreground_is_sampled(tmp_path):
    arr = np.zeros((64, 64, 64), dtype=np.uint8)
    arr[2:4, 2:4, 2:4] = 2          # a foreground scribble in one corner
    arr[50:52, 50:52, 50:52] = 1    # a false-positive fix far away from it
    ds = _dataset(_volume(tmp_path, arr))

    assert (50, 50, 50) in _as_set(ds._fg_index_sparse)
    assert ds.patches_per_epoch == 2, "both chunks with annotations count"


def test_a_session_that_painted_only_background_can_train(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[20:24, 20:24, 20:24] = 1
    ds = _dataset(_volume(tmp_path, arr))
    assert ds._fg_index_sparse.shape[0] == 64
    assert len(ds) == 1


def test_imported_crops_stay_centred_on_their_foreground(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[0:16, 0:16, 0:16] = 1
    arr[4:8, 4:8, 4:8] = 2
    crop = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}
    ds = _dataset(_volume(tmp_path, arr, [crop]))
    assert ds._fg_index_dense.shape[0] == 64, "the crop's background is not a centre"
    assert ds._fg_index_sparse.shape[0] == 0


def test_all_background_crops_are_still_trainable(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[0:16, 0:16, 0:16] = 1
    crop = {"annotation_offset_voxels": [0, 0, 0], "annotation_shape_voxels": [16, 16, 16]}
    ds = _dataset(_volume(tmp_path, arr, [crop]))
    assert ds._fg_index_dense.shape[0] == 16 ** 3


def test_a_volume_with_no_annotations_is_still_an_error(tmp_path):
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    path = _volume(tmp_path, arr)
    zarr.open(f"{path}/annotation/s0", mode="r+")[0:1, 0:1, 0:1] = 0  # a fill chunk on disk
    with pytest.raises(ValueError):
        _dataset(path)
