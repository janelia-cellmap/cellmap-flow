"""finetune.session.volume: planning a volume over a raw dataset, reading one
back, and writing a crop into it (read with crop_loader, from zarr v2 or v3).
What each volume creator writes is pinned by test_volume_snapshot."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.crop_loader import CropEntry
from cellmap_flow.finetune.session import sync
from cellmap_flow.finetune.session.volume import (
    VolumeGeometry,
    create_volume_zarr,
    plan_volume,
    read_volume,
    write_crop_into_volume,
)


def _volume(tmp_path, shape, chunk=None, offset=0.0):
    """An empty 16 nm volume whose voxel 0 is centred at ``offset``, and its record."""
    geometry = VolumeGeometry(
        output_voxel_size=(16.0,) * 3, input_voxel_size=(16.0,) * 3, claimed_output_voxel_size=None,
        claimed_input_voxel_size=None, chunk_size=chunk or shape, input_size=chunk or shape,
        dataset_offset_nm=(offset,) * 3, dataset_shape_voxels=shape,
    )
    path = create_volume_zarr(str(tmp_path / "volume.zarr"), geometry, dataset_path="/raw", model_name="m")
    return {"zarr_path": path, "output_voxel_size": [16.0] * 3, "dataset_offset_nm": [offset] * 3}


MAJORITY = np.ones((2, 2, 2), np.uint8)
MAJORITY[1, 1, 1] = MAJORITY[1, 1, 0] = MAJORITY[1, 0, 1] = 0  # 5 of 8 foreground; the last corner is not


@pytest.mark.parametrize("data, voxel_size, translation, zarr_format, first, last", [
    # 8 voxels of 8 nm are 4 of 16 nm: written as 8 they claimed twice the extent.
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 160.0, 2, 10, 13, id="resampled"),
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 160.0, 3, 10, 13, id="resampled, from a zarr v3 group"),
    # Both translations are voxel-0 centres: the crop's corner, 66 nm, is 4.6
    # volume voxels above the volume's, -8; reading 70 as the position gave 4.
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 70.0, 2, 5, 8, id="placed corner to corner"),
    # A 4 nm crop centred at 4 starts at 2: its first 16 nm block is mostly voxel 1.
    pytest.param(np.ones((16, 16, 16), np.uint8), 4.0, 4.0, 2, 1, 4, id="placed corner to corner at 4x"),
    # Each voxel takes its block's most common label; a nearest-neighbour zoom takes the last corner.
    pytest.param(MAJORITY, 8.0, 28.0, 2, 2, 2, id="by majority vote"),
    # A plain v3 array whose (legacy) transform gives its corner, 156 nm.
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 156.0, "v3 array", 10, 13, id="from a plain zarr v3 array"),
    # A plain v2 array whose funlib resolution/offset give its corner, 156 nm.
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 156.0, "v2 array", 10, 13, id="from a plain zarr v2 array"),
])
def test_a_crop_is_written_at_its_place_and_the_volumes_resolution(tmp_path, ome_zarr, data, voxel_size,
                                                                  translation, zarr_format, first, last):
    if zarr_format == "v3 array":
        crop = ome_zarr.v3_array(tmp_path / "crop", data, attributes={
            "transform": {"scale": [voxel_size] * 3, "translate": [translation] * 3}})
    elif zarr_format == "v2 array":
        crop = str(tmp_path / "crop.zarr")
        zarr.open(crop, mode="w", shape=data.shape, dtype=data.dtype)[...] = data
        zarr.open(crop, mode="r+").attrs.update(resolution=[voxel_size] * 3, offset=[translation] * 3)
    else:
        crop = ome_zarr(tmp_path / "crop.zarr", ("s0", data, voxel_size, translation), zarr_format=zarr_format)
    volume = _volume(tmp_path, (64, 64, 64))
    n_fg = (last - first + 1) ** 3
    assert write_crop_into_volume(volume, CropEntry(path=crop, fg_ids=[1]))["n_fg_voxels"] == n_fg
    written = np.argwhere(zarr.open(volume["zarr_path"], mode="r")["annotation/s0"][:] >= 2)
    assert len(written) == n_fg
    assert (written.min(axis=0).tolist(), written.max(axis=0).tolist()) == ([first] * 3, [last] * 3)


@pytest.mark.parametrize("crop, error, deepest", [
    pytest.param("annotations/typoed-dataset/crop.zarr/s0", FileNotFoundError, "annotations",
                 id="a typo, and where the path breaks"),
    pytest.param("/nonexistent-root-for-tests/a/b.zarr", FileNotFoundError, None, id="nothing of it exists"),
    pytest.param("annotations/empty.zarr", ValueError, None, id="a group with neither multiscales nor s0"),
])
def test_a_crop_that_cannot_be_read_names_itself(tmp_path, crop, error, deepest):
    """zarr names the path inside the store, "" for a missing directory: a
    manifest's typo (rc_amphiuma for jrc_amphiuma) named neither crop nor typo.
    The deepest part of the path that exists shows where it breaks."""
    (tmp_path / "annotations" / "empty.zarr").mkdir(parents=True)
    (tmp_path / "annotations" / "empty.zarr" / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
    path = crop if crop.startswith("/") else str(tmp_path / crop)
    with pytest.raises(error) as caught:
        write_crop_into_volume({"zarr_path": "/unused"}, CropEntry(path=path))
    assert path in str(caught.value) and (deepest is None or str(tmp_path / deepest) in str(caught.value))


def test_no_two_slabs_of_a_crop_write_the_same_chunk(tmp_path, ome_zarr, monkeypatch):
    """A crop is written in parallel z slabs; a slab edge inside a chunk has two
    threads rewrite that chunk whole, and one's half is lost."""
    crop = ome_zarr(tmp_path / "crop.zarr", ("s0", np.ones((20, 4, 4), np.uint8), 16.0, [40.0, 8.0, 8.0]))
    volume = _volume(tmp_path, (32, 4, 4), chunk=(4, 4, 4), offset=8.0)
    written = []
    store_set = zarr.storage.DirectoryStore.__setitem__
    monkeypatch.setattr(zarr.storage.DirectoryStore, "__setitem__",
                        lambda self, key, value: written.append(key) or store_set(self, key, value))
    monkeypatch.setattr(sync, "worker_count", lambda: 4)

    # Voxel 0's corner is 32 nm: rows 2..21, chunks 0 to 5, each written once.
    assert write_crop_into_volume(volume, CropEntry(path=crop))["annotation_offset_voxels"] == [2, 0, 0]
    chunks = [k for k in written if k.startswith("annotation/s0/") and not k.endswith((".zarray", ".zattrs"))]
    assert sorted(chunks) == [f"annotation/s0/{z}.0.0" for z in range(6)]


def test_a_level_whose_extent_is_not_whole_nm_gets_no_extra_chunk(tmp_path, ome_zarr):
    """21 voxels of 10.48 nm, voxel 0 centred at 5.24: the whole-nm box around
    them is 221 nm, 21.09 voxels, which rounded up would add a voxel and a chunk."""
    raw = ome_zarr(tmp_path / "raw.zarr", ("s0", np.zeros((21, 20, 20), np.uint8), [10.48, 8, 8], [5.24, 4, 4]))
    model = SimpleNamespace(input_shape=[9, 12, 12], output_shape=[3, 4, 4],
                            input_voxel_size=[10.48, 8, 8], output_voxel_size=[10.48, 8, 8])
    geometry = plan_volume(raw, model)
    assert geometry.dataset_shape_voxels == (21, 20, 20)
    assert geometry.dataset_offset_nm == pytest.approx((5.24, 4, 4))  # voxel 0's centre
    assert (geometry.chunk_size, geometry.input_size) == ((3, 4, 4), (9, 12, 12))


def test_a_volume_without_its_geometry_is_not_given_one(tmp_path):
    """The record used to invent 56^3 / 178^3 / 16 nm; the manifest backfill now refuses."""
    path = _volume(tmp_path, (8, 8, 8), chunk=(4, 4, 4))["zarr_path"]
    root = zarr.open_group(path, mode="r+")
    root.attrs.put({k: v for k, v in root.attrs.asdict().items() if k != "chunk_size"})

    with pytest.raises(ValueError, match="chunk_size"):
        read_volume(path)
    # Serving and syncing need no geometry: the record says what is missing.
    record = read_volume(path, require_geometry=False)
    assert record["output_size"] is None and record["input_size"] == [4, 4, 4]
