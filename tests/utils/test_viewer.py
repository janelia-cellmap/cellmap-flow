"""The viewer's raw and prediction layers (scale_pyramid, neuroglancer_utils).

get_raw_layer handed LocalVolume the offset in nm as ``voxel_offset``, which
counts whole voxels: a dataset at 80 nm on 8 nm voxels was drawn at 640 nm,
and an OME corner such as -4 nm could not be expressed at all. The position
now goes in the source transform, in voxels of the layer's own dimensions.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from neuroglancer.viewer_base import ViewerBase

from cellmap_flow.globals import g
from cellmap_flow.utils import neuroglancer_utils
from cellmap_flow.utils.scale_pyramid import get_raw_layer


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


@pytest.fixture
def new_viewer(monkeypatch):
    """generate_neuroglancer_url, with a viewer that has no web server and no dashboard after it."""
    monkeypatch.setattr(neuroglancer_utils.neuroglancer, "Viewer", ViewerBase)
    monkeypatch.setattr(neuroglancer_utils, "create_and_run_app", lambda **k: None)
    monkeypatch.setattr(neuroglancer_utils, "fetch_model_info", lambda host: {})
    g.jobs, g.models_config, g.input_norms, g.postprocess = [], [], [], []
    return neuroglancer_utils.generate_neuroglancer_url


@pytest.mark.parametrize("dataset", [
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))), id="pyramid"),
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))) + "/s1", id="its-coarser-level"),
    pytest.param(lambda f: f.raw_zarr(np.zeros((4, 4, 4), np.uint8)), id="plain-array"),  # flat: no contrast range
])
def test_the_viewer_takes_its_dimensions_from_the_finest_raw_level(new_viewer, ome_pyramid, raw_zarr, dataset):
    new_viewer(dataset(SimpleNamespace(ome_pyramid=ome_pyramid, raw_zarr=raw_zarr)))
    assert g.viewer.state.dimensions.to_json() == {axis: [8e-9, "m"] for axis in "zyx"}


def test_a_job_without_a_host_gets_no_layer(new_viewer, ome_pyramid):
    """zarr://None/... is never going to load, and nothing replaces it later."""
    g.jobs = [type("Job", (), {"model_name": name, "host": host})() for name, host in
              [("ghost", None), ("real", "http://node:3")]]
    new_viewer(ome_pyramid(((8, 0),)))
    assert [layer.name for layer in g.viewer.state.layers] == ["data", "real"]
