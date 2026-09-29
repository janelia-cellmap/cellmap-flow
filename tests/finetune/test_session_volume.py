"""finetune.session.volume: where a new volume lies, through plan_volume."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from cellmap_flow.finetune.session.volume import plan_volume


def _level(path, voxel_size, translation, shape):
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("s0", data=np.zeros(shape, np.uint8), chunks=(8, 8, 8))
    group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": list(voxel_size)},
            {"type": "translation", "translation": list(translation)}]}],
    }]
    return str(path)


@pytest.mark.parametrize(
    "voxel_size, translation, shape, chunk, offset, expected",
    [
        # Janelia: corner -4, 21 voxels padded to whole 4-voxel chunks.
        ((16, 16, 16), (4, 4, 4), (21, 21, 21), (4, 4, 4), (4, 4, 4), (24, 24, 24)),
        # Levels whose extent is not whole nm: counting from the whole-nm box
        # around the data instead of the data would add a voxel, here a chunk.
        ((10.48, 8, 8), (5.24, 4, 4), (21, 20, 20), (3, 4, 4), (5.24, 4, 4), (21, 20, 20)),
        ((16, 16, 16), (8.5, 8, 8), (20, 20, 20), (4, 4, 4), (8.5, 8, 8), (20, 20, 20)),
    ],
    ids=["janelia", "fractional-extent", "fractional-corner"],
)
def test_a_volume_covers_the_raw_level_in_whole_chunks(
    tmp_path, voxel_size, translation, shape, chunk, offset, expected
):
    raw = _level(tmp_path / "raw.zarr", voxel_size, translation, shape)
    model = SimpleNamespace(input_shape=[3 * c for c in chunk], output_shape=list(chunk),
                            input_voxel_size=list(voxel_size), output_voxel_size=list(voxel_size))
    geometry = plan_volume(raw, model)
    assert geometry.dataset_shape_voxels == expected
    assert geometry.dataset_offset_nm == pytest.approx(offset)  # voxel 0's centre
    assert geometry.chunk_size == chunk and geometry.input_size == tuple(3 * c for c in chunk)


@pytest.mark.parametrize("drop", [None, "chunk_size", "input_voxel_size"])
def test_a_volume_reads_back_its_geometry_and_none_is_invented(tmp_path, drop):
    from cellmap_flow.finetune.session.sync import volume_record
    from cellmap_flow.finetune.session.volume import (
        VolumeGeometry, build_manifest, create_volume_zarr, read_volume,
    )

    geometry = VolumeGeometry(
        output_voxel_size=(16.0,) * 3, input_voxel_size=(8.0,) * 3,
        claimed_output_voxel_size=None, claimed_input_voxel_size=None, chunk_size=(4,) * 3,
        input_size=(12,) * 3, dataset_offset_nm=(4.0,) * 3, dataset_shape_voxels=(8,) * 3,
    )
    path = create_volume_zarr(str(tmp_path / "v.zarr"), geometry, dataset_path="/raw", model_name="m")
    if drop:
        root = zarr.open_group(path, mode="r+")
        root.attrs.put({k: v for k, v in root.attrs.asdict().items() if k != drop})
        with pytest.raises(ValueError, match=drop):
            read_volume(path)

    # The sync reads it anyway, without the geometry it lacks; a manifest refuses it.
    record = volume_record("v", path, volumes={})
    assert record["output_size"] == (None if drop == "chunk_size" else [4, 4, 4])
    assert record["input_voxel_size"] == (None if drop == "input_voxel_size" else [8.0, 8.0, 8.0])
    assert record["dataset_offset_nm"] == [4.0, 4.0, 4.0] and record["dataset_path"] == "/raw"
    if drop:
        with pytest.raises(ValueError, match="cannot be written"):
            build_manifest(record, input_norm=None, postprocess=None)
    else:
        assert read_volume(path) == record
        assert build_manifest(record, input_norm=None, postprocess=None)["output_size_voxels"] == [4, 4, 4]
