"""OME-NGFF translation is the centre of voxel 0; cellmap-flow works in corners.

Janelia pyramids store ``translation = scale/2 - 4`` per level, so every level
shares a lower corner at -4 nm. Reading translation as a corner put each
level half its own voxel off (s1 8 nm, s2 16 nm), which misaligned every
prediction made from s1/s2 and made the levels disagree with each other.
"""

import os

import numpy as np
import pytest
import zarr
from funlib.geometry import Roi

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.utils.ds import read_ds_meta

LEVELS = [("s0", 8.0, 0.0), ("s1", 16.0, 4.0), ("s2", 32.0, 12.0)]


def _multiscales(levels=LEVELS, axes="zyx"):
    return [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in axes],
            "datasets": [
                {
                    "path": name,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [vs] * 3},
                        {"type": "translation", "translation": [t] * 3},
                    ],
                }
                for name, vs, t in levels
            ],
        }
    ]


def _level_data(n):
    # Each voxel holds its own z index, so a read shows which voxel it hit.
    return np.broadcast_to(np.arange(n, dtype=np.uint16)[:, None, None], (n, n, n)).copy()


@pytest.fixture
def v2_pyramid(tmp_path):
    root = zarr.open_group(str(tmp_path / "janelia.zarr"), mode="w")
    group = root.create_group("em")
    for i, (name, _, _) in enumerate(LEVELS):
        n = 32 >> i
        group.create_dataset(name, data=_level_data(n), chunks=(8, 8, 8))
    group.attrs["multiscales"] = _multiscales()
    return str(tmp_path / "janelia.zarr" / "em")


@pytest.fixture
def v3_pyramid(tmp_path):
    from tests.utils.test_zarr_v3 import _create_v3_array, _dataset_entry, _write_group_zarr_json
    from tests.utils.test_zarr_v3 import _multiscales as v3_multiscales

    group = str(tmp_path / "janelia_v3.zarr")
    _write_group_zarr_json(
        group, v3_multiscales([_dataset_entry(name, (vs,) * 3, (t,) * 3) for name, vs, t in LEVELS])
    )
    for i, (name, _, _) in enumerate(LEVELS):
        _create_v3_array(os.path.join(group, name), _level_data(32 >> i), chunk_shape=[8, 8, 8])
    return group


@pytest.mark.parametrize("level", range(len(LEVELS)))
def test_every_janelia_level_reports_the_shared_corner(v2_pyramid, level):
    name, vs, _ = LEVELS[level]
    voxel_size, offset, *_ = read_ds_meta(os.path.join(v2_pyramid, name))
    assert tuple(voxel_size) == (vs,) * 3
    assert tuple(offset) == (-4.0,) * 3


@pytest.mark.parametrize("level", range(len(LEVELS)))
def test_zarr_v3_levels_report_the_shared_corner_too(v3_pyramid, level):
    name, vs, _ = LEVELS[level]
    voxel_size, offset, *_ = read_ds_meta(os.path.join(v3_pyramid, name))
    assert tuple(voxel_size) == (vs,) * 3
    assert tuple(offset) == (-4.0,) * 3


def test_the_interface_reads_voxel_0_at_the_corner(v2_pyramid):
    idi = ImageDataInterface(os.path.join(v2_pyramid, "s2"), normalize=False)
    assert tuple(idi.roi.offset) == (-4, -4, -4)
    # World [-4, 28) is exactly voxel 0 of s2; read as a corner it was voxel -1/0.
    block = idi.to_ndarray_ts(Roi((-4, -4, -4), (32, 32, 32)))
    assert block.shape == (1, 1, 1) and int(block.ravel()[0]) == 0
    # The memory-note example: a read starting at 57408 nm at s2 is voxel
    # floor((57408 + 4) / 32) = 1794, not 1793 -- here scaled down to the
    # small fixture: world 60 nm -> voxel floor(64 / 32) = 2.
    block = idi.to_ndarray_ts(Roi((60, -4, -4), (32, 32, 32)))
    assert int(block.ravel()[0]) == 2


def test_a_missing_translation_means_voxel_0_is_centred_on_the_origin(tmp_path):
    root = zarr.open_group(str(tmp_path / "plain.zarr"), mode="w")
    root.create_dataset("s0", data=np.zeros((4, 4, 4), np.uint8))
    ms = _multiscales([("s0", 8.0, 0.0)])
    ms[0]["datasets"][0]["coordinateTransformations"] = [{"type": "scale", "scale": [8.0] * 3}]
    root.attrs["multiscales"] = ms
    _, offset, *_ = read_ds_meta(str(tmp_path / "plain.zarr" / "s0"))
    assert tuple(offset) == (-4.0,) * 3


def test_legacy_offset_attributes_are_still_corners(tmp_path):
    root = zarr.open_group(str(tmp_path / "legacy.zarr"), mode="w")
    arr = root.create_dataset("raw", data=np.zeros((4, 4, 4), np.uint8))
    arr.attrs["resolution"] = [8, 8, 8]
    arr.attrs["offset"] = [80, 40, 40]
    voxel_size, offset, *_ = read_ds_meta(str(tmp_path / "legacy.zarr" / "raw"))
    assert tuple(voxel_size) == (8.0,) * 3
    assert tuple(offset) == (80.0, 40.0, 40.0)


def test_a_model_on_s1_is_served_on_s1s_own_grid(tmp_path):
    """The served grid starts at the corner of the level the model reads.

    A 16 nm model reads s1, whose translation 4 puts its corner at -4 nm.
    Anchored at 0 (and with s1's corner read as 4), chunk 0 was computed
    from a box a quarter voxel into s1 and came back shifted.
    """
    from cellmap_flow.models.models_config import ScriptModelConfig
    from cellmap_flow.server import CellMapFlowServer
    from tests.utils.serving_helpers import decode_chunk, get_json, write_script

    root = zarr.open_group(str(tmp_path / "janelia.zarr"), mode="w")
    group = root.create_group("em")
    for i, (name, _, _) in enumerate(LEVELS[:2]):
        n = 16 >> i
        # z index + 1, so padding (0) cannot pass for data.
        group.create_dataset(name, data=(_level_data(n) + 1).astype(np.uint8), chunks=(4, 4, 4))
    group.attrs["multiscales"] = _multiscales(LEVELS[:2])

    model = """
input_voxel_size = Coordinate(16, 16, 16)
output_voxel_size = Coordinate(16, 16, 16)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
block_shape = np.array((4, 4, 4, 1))


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


model = Identity()
"""
    server = CellMapFlowServer(
        str(tmp_path / "janelia.zarr" / "em"),
        ScriptModelConfig(script_path=write_script(tmp_path, model)),
    )
    client = server.app.test_client()

    meta = get_json(client, "/plain/s0/.zarray")
    chunk = decode_chunk(server, client.get("/plain/s0/0.0.0.0").data, meta["dtype"], meta["chunks"])
    assert chunk[:, 0, 0, 0].tolist() == [1, 2, 3, 4]

    attrs = get_json(client, "/plain/.zattrs")
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    translation = next(t["translation"] for t in transforms if t["type"] == "translation")
    assert translation[:3] == [4.0, 4.0, 4.0]
