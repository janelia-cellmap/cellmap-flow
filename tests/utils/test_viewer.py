"""The viewer's raw layer (viewer.raw.get_raw_layer).

get_raw_layer handed LocalVolume the offset in nm as ``voxel_offset``, which
counts whole voxels: a dataset at 80 nm on 8 nm voxels was drawn at 640 nm,
and an OME corner such as -4 nm could not be expressed at all. The position
now goes in the source transform, in voxels of the layer's own dimensions.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.globals import g
from cellmap_flow.viewer.raw import get_raw_layer


def _source(layer):
    source = layer.to_json()["source"]
    return source[0] if isinstance(source, list) else source


def _placement(layer):
    """(voxel size per axis, translation in voxels) of a layer's source."""
    transform = _source(layer)["transform"]
    matrix = np.array(transform["matrix"])
    assert np.array_equal(matrix[:, :-1], np.eye(3))
    return [v[0] for v in transform["outputDimensions"].values()], matrix[:, -1].tolist()


@pytest.mark.parametrize(
    "write, scales, translation",
    [
        # Every Janelia level's corner is -4 nm: half an 8 nm voxel below the origin.
        pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))), [8e-9] * 3, [-0.5] * 3, id="janelia-pyramid"),
        pytest.param(lambda f: f.raw_zarr(np.zeros((4, 4, 4), np.uint8), offset=(80, 40, 40)), [8e-9] * 3,
                     [10.0, 5.0, 5.0], id="funlib-offset"),
        # Voxel 0's centre at (10, 20, 30): its corner is (8, 18, 28).
        pytest.param(lambda f: f.ome_pyramid(((4, (10, 20, 30)), (8, (12, 22, 32))), zarr_format=3), [4e-9] * 3,
                     [2.0, 4.5, 7.0], id="zarr-v3"),
        # x, y, z voxels (3, 2, 1).
        pytest.param(lambda f: f.write_array("precomputed", np.zeros((2, 10, 20), np.uint8), {
            "resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]}),
            [16e-9, 8e-9, 4e-9], [1.0, 2.0, 3.0], id="precomputed"),
    ],
)
def test_the_raw_layer_is_drawn_where_its_data_is(ome_pyramid, raw_zarr, write_array, write, scales, translation):
    path = write(SimpleNamespace(ome_pyramid=ome_pyramid, raw_zarr=raw_zarr, write_array=write_array))
    got_scales, got_translation = _placement(get_raw_layer(path, normalize=False))
    assert got_scales == pytest.approx(scales) and got_translation == translation


def _named(pyramid, names):
    """``pyramid`` with its levels renamed to ``names`` on disk and in its multiscales."""
    import os

    import zarr

    group = zarr.open_group(pyramid, mode="r+")
    multiscales = group.attrs["multiscales"]
    for dataset, name in zip(multiscales[0]["datasets"], names):
        os.rename(os.path.join(pyramid, dataset["path"]), os.path.join(pyramid, name))
        dataset["path"] = name
    group.attrs["multiscales"] = multiscales
    return pyramid


def _unlisted_s2(pyramid):
    """``pyramid`` with its last level left out of its multiscales, though still on disk."""
    import zarr

    group = zarr.open_group(pyramid, mode="r+")
    multiscales = group.attrs["multiscales"]
    multiscales[0]["datasets"] = multiscales[0]["datasets"][:-1]
    group.attrs["multiscales"] = multiscales
    return pyramid


def _funlib_pyramid(f):
    """s0 and s1 of 8 and 16 nm, with funlib attributes and no multiscales."""
    f.raw_zarr(np.zeros((8, 8, 8), np.uint8), voxel_size=(16, 16, 16), name="pyramid/s1")
    return f.raw_zarr(np.zeros((16, 16, 16), np.uint8), name="pyramid/s0")


@pytest.mark.parametrize("write", [
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))), id="named-sN"),
    pytest.param(lambda f: _named(f.ome_pyramid(((8, 0), (16, 4))), ["0", "1"]), id="named-by-number"),
    pytest.param(lambda f: _named(f.ome_pyramid(((8, 0), (16, 4))), ["0", "1"]) + "/1", id="one-numbered-level"),
    pytest.param(lambda f: _unlisted_s2(f.ome_pyramid(((8, 0), (16, 4), (32, 12)))), id="a-level-not-listed"),
    # Without OME multiscales, an sN level's siblings are the pyramid.
    pytest.param(_funlib_pyramid, id="funlib-sN-without-multiscales"),
])
def test_a_pyramids_levels_are_the_ones_its_multiscales_list(ome_pyramid, raw_zarr, write):
    layer = get_raw_layer(write(SimpleNamespace(ome_pyramid=ome_pyramid, raw_zarr=raw_zarr)), normalize=False)
    assert sorted(layer.source[0].url.volume_layers) == [(1, 1, 1), (2, 2, 2)]
    assert _placement(layer)[0] == pytest.approx([8e-9] * 3)


def test_a_label_volume_is_a_segmentation_layer_in_the_same_place(raw_zarr):
    from cellmap_flow.norm.input_normalize import MinMaxNormalizer

    ids = np.arange(64, dtype=np.uint64).reshape(4, 4, 4)
    path = raw_zarr(ids, offset=(80, 40, 40), name="ids")
    g.input_norms = [MinMaxNormalizer(0, 63)]
    layer = get_raw_layer(path, segmentation=True, disable_meshes=True)
    assert layer.to_json()["type"] == "segmentation" and _placement(layer)[1] == [10.0, 5.0, 5.0]
    assert _source(layer)["subsources"] == {"meshes": False}
    np.testing.assert_array_equal(np.asarray(layer.source[0].url.data[...]), ids, "ids as stored, never normalized")
    assert "subsources" not in _source(get_raw_layer(path, segmentation=True))
