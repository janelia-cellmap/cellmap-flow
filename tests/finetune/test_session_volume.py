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
