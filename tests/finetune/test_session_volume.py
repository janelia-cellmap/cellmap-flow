"""finetune.session.volume, through plan_volume and read_volume."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.session.volume import (
    VolumeGeometry,
    create_volume_zarr,
    plan_volume,
    read_volume,
)


def test_a_level_whose_extent_is_not_whole_nm_gets_no_extra_chunk(tmp_path):
    """21 voxels of 10.48 nm, voxel 0 centred at 5.24: the whole-nm box around
    them is 221 nm, 21.09 voxels, which rounded up would add a voxel and so a
    chunk of 3."""
    group = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w")
    group.create_dataset("s0", data=np.zeros((21, 20, 20), np.uint8), chunks=(8, 8, 8))
    group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [10.48, 8, 8]},
            {"type": "translation", "translation": [5.24, 4, 4]}]}],
    }]
    model = SimpleNamespace(input_shape=[9, 12, 12], output_shape=[3, 4, 4],
                            input_voxel_size=[10.48, 8, 8], output_voxel_size=[10.48, 8, 8])
    geometry = plan_volume(str(tmp_path / "raw.zarr"), model)
    assert geometry.dataset_shape_voxels == (21, 20, 20)
    assert geometry.dataset_offset_nm == pytest.approx((5.24, 4, 4))  # voxel 0's centre
    assert (geometry.chunk_size, geometry.input_size) == ((3, 4, 4), (9, 12, 12))


def test_a_volume_without_its_geometry_is_not_given_one(tmp_path):
    geometry = VolumeGeometry(
        output_voxel_size=(16.0,) * 3, input_voxel_size=(8.0,) * 3,
        claimed_output_voxel_size=None, claimed_input_voxel_size=None, chunk_size=(4,) * 3,
        input_size=(12,) * 3, dataset_offset_nm=(4.0,) * 3, dataset_shape_voxels=(8,) * 3,
    )
    path = create_volume_zarr(str(tmp_path / "v.zarr"), geometry, dataset_path="/raw", model_name="m")
    root = zarr.open_group(path, mode="r+")
    root.attrs.put({k: v for k, v in root.attrs.asdict().items() if k != "chunk_size"})

    with pytest.raises(ValueError, match="chunk_size"):
        read_volume(path)
    # Serving and syncing need no geometry: the record says what is missing.
    record = read_volume(path, require_geometry=False)
    assert record["output_size"] is None and record["input_size"] == [12, 12, 12]


def test_no_two_slabs_of_a_crop_write_the_same_chunk(tmp_path, monkeypatch):
    """A crop is written in parallel z slabs. A slab edge inside a chunk has two
    threads rewrite that chunk whole, and one's half is lost."""
    from cellmap_flow.finetune.crop_loader import CropEntry
    from cellmap_flow.finetune.session import sync
    from cellmap_flow.finetune.session.volume import write_crop_into_volume

    crop = zarr.open_group(str(tmp_path / "crop.zarr"), mode="w")
    crop.create_dataset("s0", data=np.ones((20, 4, 4), np.uint8), chunks=(20, 4, 4))
    crop.attrs["multiscales"] = [{"version": "0.4", "axes": [
        {"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"], "datasets": [
        {"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [16.0] * 3}, {"type": "translation", "translation": [40.0, 8.0, 8.0]}]}]}]
    geometry = VolumeGeometry(
        output_voxel_size=(16.0,) * 3, input_voxel_size=(16.0,) * 3, claimed_output_voxel_size=None,
        claimed_input_voxel_size=None, chunk_size=(4,) * 3, input_size=(4,) * 3,
        dataset_offset_nm=(8.0,) * 3, dataset_shape_voxels=(32, 4, 4),
    )
    path = create_volume_zarr(str(tmp_path / "v.zarr"), geometry, dataset_path="/raw", model_name="m")

    written = []
    store_set = zarr.storage.DirectoryStore.__setitem__
    monkeypatch.setattr(zarr.storage.DirectoryStore, "__setitem__",
                        lambda self, key, value: written.append(key) or store_set(self, key, value))
    monkeypatch.setattr(sync, "worker_count", lambda: 4)
    record = write_crop_into_volume({"zarr_path": path, "output_voxel_size": [16.0] * 3,
                                     "dataset_offset_nm": [8.0] * 3}, CropEntry(path=str(tmp_path / "crop.zarr")))

    # Voxel 0's corner is 32 nm: rows 2..21, chunks 0 to 5, each written once.
    assert record["annotation_offset_voxels"] == [2, 0, 0]
    chunks = [k for k in written if k.startswith("annotation/s0/") and not k.endswith((".zarray", ".zattrs"))]
    assert sorted(chunks) == [f"annotation/s0/{z}.0.0" for z in range(6)]
