"""The viewer's raw layer (viewer.raw.get_raw_layer).

get_raw_layer handed LocalVolume the offset in nm as ``voxel_offset``, which
counts whole voxels: a dataset at 80 nm on 8 nm voxels was drawn at 640 nm,
and an OME corner such as -4 nm could not be expressed at all. The position
now goes in the source transform, in voxels of the layer's own dimensions.
"""

import re
from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.norm.input_normalize import LambdaNormalizer, MinMaxNormalizer
from cellmap_flow.process_chain import process_chain
from cellmap_flow.viewer.raw import ScalePyramid, get_raw_layer
from tests.utils.test_io_metadata import at_url  # noqa: F401  (a fixture: a directory served at URLs)


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


def _funlib_pyramid(f):
    """s0 and s1 of 8 and 16 nm, with funlib attributes and no multiscales."""
    f.raw_zarr(np.zeros((8, 8, 8), np.uint8), voxel_size=(16, 16, 16), name="pyramid/s1")
    return f.raw_zarr(np.zeros((16, 16, 16), np.uint8), name="pyramid/s0")


def _precomputed_scales(f, count=2):
    """A precomputed volume of ``count`` scales: 8 nm, then twice the one before."""
    return f.write_array("precomputed", np.zeros((8,) * 3, np.uint8), {"resolution": [8] * 3}, scales=count)


TWO_LEVELS = [(1, 1, 1), (2, 2, 2)]


@pytest.mark.parametrize("write, levels", [
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))), TWO_LEVELS, id="named-sN"),
    # Without OME multiscales, an sN level's siblings are the pyramid.
    pytest.param(_funlib_pyramid, TWO_LEVELS, id="funlib-sN-without-multiscales"),
    # A precomputed volume's levels are its scales.
    pytest.param(_precomputed_scales, TWO_LEVELS, id="precomputed-scales"),
    # A level 1.5x or 3x the finest is not shown; the finest always is.
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (12, 2), (16, 4))), TWO_LEVELS, id="a-1.5x-level-dropped"),
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (24, 8))), [(1, 1, 1)], id="a-3x-level-dropped"),
])
def test_a_pyramids_levels_are_the_ones_its_multiscales_list(ome_pyramid, raw_zarr, write_array, write, levels):
    path = write(SimpleNamespace(ome_pyramid=ome_pyramid, raw_zarr=raw_zarr, write_array=write_array))
    layer = get_raw_layer(path, normalize=False)
    url = layer.source[0].url
    assert (sorted(url.volume_layers) if isinstance(url, ScalePyramid) else None) == levels
    assert _placement(layer)[0] == pytest.approx([8e-9] * 3)
    # Each level is served as the finest one (8 nm) downsampled by its key.
    for key, level in getattr(url, "volume_layers", {}).items():
        assert level.dimensions.scales == pytest.approx(np.multiply(key, 8e-9)), key


def _contrast(layer):
    """The contrast range of a raw layer's shader."""
    lo, hi = re.search(r"range=\[([^,]+), ([^\]]+)\]", layer.shader).groups()
    return float(lo), float(hi)


FLAT = np.zeros((16, 16, 16), np.uint8)


@pytest.mark.parametrize("data, input_norms, contrast", [
    # The 1st and 99th percentiles of what is shown...
    pytest.param(np.repeat(np.arange(256, dtype=np.uint8), 16).reshape(16, 16, 16), [], (2, 253), id="sampled"),
    # ...or, with nothing to sample (flat, like padding, or too big to read whole),
    # the range of its dtype...
    pytest.param(FLAT, [], (0, 255), id="uint8"),
    pytest.param(FLAT.astype(np.uint16), [], (0, 65535), id="uint16"),
    # ...put through the input chain: [0, 1] after a MinMaxNormalizer...
    pytest.param(FLAT, [MinMaxNormalizer(0, 255)], (0, 1), id="through-a-float-chain"),
    pytest.param(FLAT, [MinMaxNormalizer(0, 255), LambdaNormalizer("x*2-1")], (-1, 1), id="to-minus-one-to-one"),
])
def test_a_single_arrays_contrast_is_sampled_or_its_dtypes(raw_zarr, data, input_norms, contrast):
    process_chain().input_norms = input_norms
    assert _contrast(get_raw_layer(raw_zarr(data))) == contrast


def _bounds(shader):
    """An invlerp control's range and window, as written: [lo, hi, wlo, whi]."""
    return list(re.search(r"range=\[([^,]+), ([^\]]+)\], window=\[([^,]+), ([^\]]+)\]", shader).groups())


def test_a_uint8_layers_bounds_are_uint8_values(raw_zarr):
    """neuroglancer refuses the whole shader when a bound is no value of the
    data's type: over uint8, a window padded below 0 ("Invalid uint8 value:
    "-75"") or a fractional percentile. Through a float chain they stay as
    sampled, padding and all."""
    data = (np.arange(16**3) % 151).astype(np.uint8).reshape((16,) * 3)
    process_chain().input_norms = []
    bounds = _bounds(get_raw_layer(raw_zarr(data)).shader)
    assert all(b.isdigit() for b in bounds) and int(bounds[2]) == 0 and int(bounds[3]) <= 255, bounds
    process_chain().input_norms = [MinMaxNormalizer(0, 255)]
    assert float(_bounds(get_raw_layer(raw_zarr(data, name="again")).shader)[2]) < 0


def test_a_label_volume_is_a_segmentation_layer_in_the_same_place(raw_zarr):
    ids = np.arange(64, dtype=np.uint64).reshape(4, 4, 4)
    path = raw_zarr(ids, offset=(80, 40, 40), name="ids")
    process_chain().input_norms = [MinMaxNormalizer(0, 63)]
    layer = get_raw_layer(path, segmentation=True, disable_meshes=True)
    assert layer.to_json()["type"] == "segmentation" and _placement(layer)[1] == [10.0, 5.0, 5.0]
    assert _source(layer)["subsources"] == {"meshes": False}
    np.testing.assert_array_equal(np.asarray(layer.source[0].url.data[...]), ids, "ids as stored, never normalized")
    assert "subsources" not in _source(get_raw_layer(path, segmentation=True))


@pytest.mark.parametrize("scheme", ["http", "s3", "gs"])
def test_a_pyramid_at_a_url_is_shown_as_it_is_from_disk(ome_pyramid, at_url, scheme):  # noqa: F811
    path = ome_pyramid(((8, 0), (16, 4)))

    def shown(path):
        layer = get_raw_layer(path)
        levels = layer.source[0].url.volume_layers
        return _placement(layer), {key: np.asarray(level.data[...]).tolist() for key, level in levels.items()}

    assert shown(at_url(scheme, path)) == shown(path)

