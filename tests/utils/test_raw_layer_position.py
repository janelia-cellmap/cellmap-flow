"""The raw layer is drawn where its data is.

get_raw_layer handed LocalVolume the offset in nm as ``voxel_offset``, which
counts whole voxels: a dataset at 80 nm on 8 nm voxels was drawn at 640 nm,
and an OME corner such as -4 nm could not be expressed at all. The position
now goes in the source transform, in voxels of the layer's own dimensions.
"""

import contextlib
import types

import numpy as np
import pytest
import zarr

from cellmap_flow.utils.scale_pyramid import get_raw_layer


def _source(layer):
    source = layer.to_json()["source"]
    return source[0] if isinstance(source, list) else source


def _translation(layer):
    matrix = np.array(_source(layer)["transform"]["matrix"])
    assert np.array_equal(matrix[:, :-1], np.eye(3))
    return matrix[:, -1].tolist()


def _janelia_pyramid(tmp_path):
    """raw.zarr/em: s0 at 8 nm and s1 at 16 nm, both with their corner at -4 nm."""
    group = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w").create_group("em")
    levels = [("s0", 8.0, 0.0), ("s1", 16.0, 4.0)]
    for i, (name, _, _) in enumerate(levels):
        group.create_dataset(name, data=np.zeros((16 >> i,) * 3, np.uint8), chunks=(8, 8, 8))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
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
    return str(tmp_path / "raw.zarr" / "em")


def test_a_janelia_pyramid_is_drawn_from_its_corner(tmp_path):
    layer = get_raw_layer(_janelia_pyramid(tmp_path), normalize=False)
    # Every level's corner is -4 nm: half an 8 nm voxel below the origin.
    assert _source(layer)["transform"]["outputDimensions"]["z"] == [8e-9, "m"]
    assert _translation(layer) == [-0.5] * 3


def test_an_offset_array_is_drawn_at_its_offset_not_at_offset_voxels(tmp_path):
    arr = zarr.open_group(str(tmp_path / "raw.zarr"), mode="w").create_dataset(
        "raw", data=np.zeros((4, 4, 4), np.uint8)
    )
    arr.attrs["resolution"] = [8, 8, 8]
    arr.attrs["offset"] = [80, 40, 40]

    layer = get_raw_layer(str(tmp_path / "raw.zarr" / "raw"), normalize=False)
    assert _translation(layer) == [10.0, 5.0, 5.0]


def test_a_label_volume_is_a_segmentation_layer_in_the_same_place(tmp_path):
    from cellmap_flow.globals import g
    from cellmap_flow.norm.input_normalize import MinMaxNormalizer

    ids = np.arange(64, dtype=np.uint64).reshape(4, 4, 4)
    arr = zarr.open_group(str(tmp_path / "labels.zarr"), mode="w").create_dataset("ids", data=ids)
    arr.attrs["resolution"] = [8, 8, 8]
    arr.attrs["offset"] = [80, 40, 40]
    path = str(tmp_path / "labels.zarr" / "ids")
    g.input_norms = [MinMaxNormalizer(0, 63)]

    layer = get_raw_layer(path, segmentation=True, disable_meshes=True)
    assert layer.to_json()["type"] == "segmentation"
    assert _translation(layer) == [10.0, 5.0, 5.0]
    assert _source(layer)["subsources"] == {"meshes": False}
    np.testing.assert_array_equal(np.asarray(layer.source[0].url.data[...]), ids)
    assert "subsources" not in _source(get_raw_layer(path, segmentation=True))


@pytest.mark.parametrize("level", ["", "/s1"])
def test_the_viewer_takes_its_dimensions_from_the_finest_raw_level(tmp_path, monkeypatch, level):
    from cellmap_flow.utils import neuroglancer_utils

    state = types.SimpleNamespace(layers={}, dimensions=None)

    class FakeViewer:
        def txn(self):
            return contextlib.nullcontext(state)

    monkeypatch.setattr(neuroglancer_utils.neuroglancer, "Viewer", FakeViewer)
    monkeypatch.setattr(neuroglancer_utils, "create_and_run_app", lambda **k: "url")

    neuroglancer_utils.generate_neuroglancer_url(_janelia_pyramid(tmp_path) + level)

    assert state.dimensions.to_json() == {axis: [8e-9, "m"] for axis in "zyx"}


@pytest.mark.parametrize(
    "fmt, scales, translation",
    [("zarr3", [4e-9] * 3, [2.0, 4.5, 7.0]), ("precomputed", [16e-9, 8e-9, 4e-9], [1.0, 2.0, 3.0])],
)
def test_zarr_v3_and_precomputed_layers_are_placed_by_their_metadata(fmt, scales, translation, ome_pyramid,
                                                                      write_array):
    if fmt == "zarr3":  # voxel 0's centre at (10, 20, 30): its corner is (8, 18, 28)
        path = ome_pyramid(((4, (10, 20, 30)), (8, (12, 22, 32))), zarr_format=3)
    else:  # x, y, z voxels (3, 2, 1)
        path = write_array("precomputed", np.zeros((2, 10, 20), np.uint8),
                           {"resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]})
    layer = get_raw_layer(path, normalize=False)
    assert [v[0] for v in _source(layer)["transform"]["outputDimensions"].values()] == pytest.approx(scales)
    assert _translation(layer) == translation
