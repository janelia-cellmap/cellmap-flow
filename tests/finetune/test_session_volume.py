"""finetune.session.volume: writing a crop into a volume (read with
crop_loader, from zarr v2 or v3), and a volume made to resample, read back
and trained on. Making and resuming volumes through the routes is
test_volume_snapshot's."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.crop_loader import CropEntry
from cellmap_flow.finetune.session.volume import (
    VolumeGeometry,
    build_manifest,
    create_volume_zarr,
    plan_volume,
    read_volume,
    write_crop_into_volume,
)


def _volume(tmp_path, shape):
    """An empty 16 nm volume whose voxel 0 is centred at 0, and its record."""
    geometry = VolumeGeometry(
        output_voxel_size=(16.0,) * 3, input_voxel_size=(16.0,) * 3, claimed_output_voxel_size=None,
        claimed_input_voxel_size=None, chunk_size=shape, input_size=shape,
        dataset_offset_nm=(0.0,) * 3, dataset_shape_voxels=shape,
    )
    path = create_volume_zarr(str(tmp_path / "volume.zarr"), geometry, dataset_path="/raw", model_name="m")
    return {"zarr_path": path, "output_voxel_size": [16.0] * 3, "dataset_offset_nm": [0.0] * 3}


MAJORITY = np.ones((2, 2, 2), np.uint8)
MAJORITY[1, 1, 1] = MAJORITY[1, 1, 0] = MAJORITY[1, 0, 1] = 0  # 5 of 8 foreground; the last corner is not


@pytest.mark.parametrize("data, voxel_size, translation, zarr_format, first, last", [
    # 8 voxels of 8 nm are 4 of 16 nm: written as 8 they claimed twice the extent.
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 160.0, 2, 10, 13, id="resampled"),
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 160.0, 3, 10, 13, id="resampled, from a zarr v3 group"),
    # Both translations are voxel-0 centres: the crop's corner, 66 nm, is 4.6
    # volume voxels above the volume's, -8; reading 70 as the position gave 4.
    pytest.param(np.ones((8, 8, 8), np.uint8), 8.0, 70.0, 2, 5, 8, id="placed corner to corner"),
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


@pytest.mark.parametrize("resample, voxel_size, shape", [
    # Snapped to the raw level, which the trainer then reads as it is stored.
    pytest.param(False, [16, 4, 4], [16, 16, 16], id="snapped-to-the-level"),
    # At the model's own 8 nm, over the data's whole extent, and read resampled.
    pytest.param(True, [8, 8, 8], [32, 8, 8], id="resampled"),
])
def test_a_volume_made_to_resample_is_trained_on_the_raw_resampled(tmp_path, ome_pyramid, resample, voxel_size,
                                                                   shape):
    """The Resample setting travels with the volume (its attrs, then the
    manifest), so the trainer reads the raw at the model's voxel size as a
    --resample server does, whatever the dashboard is set to later."""
    from cellmap_flow.finetune.data.reader import PatchReader
    from cellmap_flow.io.raw import RawInput, open_raw
    from funlib.geometry import Roi

    raw = ome_pyramid((((16, 4, 4), 0),))  # 16^3 voxels from (-8, -2, -2) nm; no 8 nm level
    model = SimpleNamespace(input_voxel_size=(8, 8, 8), output_voxel_size=(8, 8, 8),
                            input_shape=(4, 4, 4), output_shape=(4, 4, 4))
    geometry = plan_volume(raw, model, resample=resample)
    assert (list(geometry.output_voxel_size), list(geometry.dataset_shape_voxels)) == (voxel_size, shape)

    path = create_volume_zarr(str(tmp_path / "volume.zarr"), geometry, dataset_path=raw, model_name="m")
    record = {**read_volume(path), "zarr_path": path}
    manifest = build_manifest(record, input_norm=None, postprocess=None)
    assert (record["resample"], manifest["resample"]) == (resample, resample)
    if not resample:
        return

    corner = np.array([-8.0, -2.0, -2.0])
    reader = PatchReader(path, raw, (4, 4, 4), (4, 4, 4), (8, 8, 8), (8, 8, 8),
                         corner_nm=corner, shape_voxels=np.array(shape), resample=manifest["resample"])
    # Centred 6, 2, 2 output voxels in: the input box starts 32, 0, 0 nm from the corner.
    expected = RawInput(open_raw(raw, (8, 8, 8), resample=True), normalize=False).to_ndarray_ts(
        Roi((24, -2, -2), (32, 32, 32)))
    assert np.array_equal(reader.raw(np.array([6, 2, 2])), expected)
