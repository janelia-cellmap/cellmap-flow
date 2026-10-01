"""The viewer's raw layer (viewer.raw.get_raw_layer).

get_raw_layer handed LocalVolume the offset in nm as ``voxel_offset``, which
counts whole voxels: a dataset at 80 nm on 8 nm voxels was drawn at 640 nm,
and an OME corner such as -4 nm could not be expressed at all. The position
now goes in the source transform, in voxels of the layer's own dimensions.
"""

import logging
import re
from types import SimpleNamespace

import neuroglancer
import numpy as np
import pytest

from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.process_chain import process_chain
from cellmap_flow.viewer.raw import ScalePyramid, get_raw_layer, neuroglancer_source
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


def _precomputed_scales(f, count=2):
    """A precomputed volume of ``count`` scales: 8 nm, then twice the one before."""
    return f.write_array("precomputed", np.zeros((8,) * 3, np.uint8), {"resolution": [8] * 3}, scales=count)


TWO_LEVELS = [(1, 1, 1), (2, 2, 2)]


@pytest.mark.parametrize("write, levels", [
    pytest.param(lambda f: f.ome_pyramid(((8, 0), (16, 4))), TWO_LEVELS, id="named-sN"),
    pytest.param(lambda f: _named(f.ome_pyramid(((8, 0), (16, 4))), ["0", "1"]), TWO_LEVELS, id="named-by-number"),
    pytest.param(lambda f: _named(f.ome_pyramid(((8, 0), (16, 4))), ["0", "1"]) + "/1", TWO_LEVELS,
                 id="one-numbered-level"),
    pytest.param(lambda f: _unlisted_s2(f.ome_pyramid(((8, 0), (16, 4), (32, 12)))), TWO_LEVELS,
                 id="a-level-not-listed"),
    # Without OME multiscales, an sN level's siblings are the pyramid.
    pytest.param(_funlib_pyramid, TWO_LEVELS, id="funlib-sN-without-multiscales"),
    # A precomputed volume's levels are its scales, found from any one of them...
    pytest.param(_precomputed_scales, TWO_LEVELS, id="precomputed-scales"),
    pytest.param(lambda f: _precomputed_scales(f) + "/s1", TWO_LEVELS, id="one-precomputed-scale"),
    # ...but one scale stays one array, which neuroglancer downsamples on the fly.
    pytest.param(lambda f: _precomputed_scales(f, count=1), None, id="precomputed-of-one-scale"),
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


def _level(voxel_size):
    return neuroglancer.LocalVolume(
        np.zeros((8, 8, 8), np.uint8),
        dimensions=neuroglancer.CoordinateSpace(names=list("zyx"), units="nm", scales=voxel_size),
    )


@pytest.mark.parametrize("voxel_sizes, kept, max_downsampling", [
    # neuroglancer may ask as far past the coarsest level as its LocalVolume
    # downsamples (4x4x4, 64), so a one-level pyramid zooms out like one array.
    pytest.param([(8,) * 3, (16,) * 3, (32,) * 3],
                 {(1, 1, 1): (8,) * 3, (2, 2, 2): (16,) * 3, (4, 4, 4): (32,) * 3}, 64 * 64, id="powers-of-two"),
    # neuroglancer asks only for power-of-two downsamplings, which these never answer.
    pytest.param([(8,) * 3, (12,) * 3, (16,) * 3, (24,) * 3],
                 {(1, 1, 1): (8,) * 3, (2, 2, 2): (16,) * 3}, 8 * 64, id="1.5x-and-3x-dropped"),
    pytest.param([(8,) * 3, (24,) * 3], {(1, 1, 1): (8,) * 3}, 64, id="only-the-finest-left"),
    pytest.param([(8, 8, 8), (8, 16, 16), (8, 24, 24)],
                 {(1, 1, 1): (8, 8, 8), (1, 2, 2): (8, 16, 16)}, 4 * 64, id="anisotropic"),
    pytest.param([(4,) * 3, (7.999999999,) * 3],
                 {(1, 1, 1): (4,) * 3, (2, 2, 2): (7.999999999,) * 3}, 8 * 64, id="float-noise"),
])
def test_a_scale_pyramids_levels_and_how_far_neuroglancer_may_zoom_out(
    caplog, voxel_sizes, kept, max_downsampling
):
    with caplog.at_level(logging.WARNING, logger="cellmap_flow.viewer.raw"):
        pyramid = ScalePyramid([_level(voxel_size) for voxel_size in voxel_sizes])
    got = {key: tuple(np.round(level.dimensions.scales * 1e9, 9)) for key, level in pyramid.volume_layers.items()}
    assert got == kept
    # One warning when a level is dropped, naming them all; none when none is.
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == (len(kept) < len(voxel_sizes))
    assert pyramid.info()["maxDownsampling"] == max_downsampling
    # ...and the furthest request that allows is served.
    furthest = np.multiply(max(pyramid.volume_layers, key=np.prod), 4)
    pyramid.get_encoded_subvolume("npz", np.zeros(3, int), np.ones(3, int), ",".join(map(str, furthest)))


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
    # ...as the input chain returns it: floats have none, so [-1, 1].
    pytest.param(FLAT, [MinMaxNormalizer(0, 255)], (-1, 1), id="through-a-float-chain"),
])
def test_a_single_arrays_contrast_is_sampled_or_its_dtypes(raw_zarr, data, input_norms, contrast):
    process_chain().input_norms = input_norms
    assert _contrast(get_raw_layer(raw_zarr(data))) == contrast


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


@pytest.mark.parametrize("path, source", [
    pytest.param("s3://b/x.zarr/em", "zarr://s3://b/x.zarr/em", id="s3-zarr"),
    pytest.param("gs://b/x.zarr/em/s0", "zarr://gs://b/x.zarr/em/s0", id="gs-zarr"),
    pytest.param("https://h/x.n5/em", "n5://https://h/x.n5/em", id="https-n5"),
    # A zarr at a URL with no .zarr in it was named a precomputed volume.
    pytest.param("https://h/era5/t2m", "zarr://https://h/era5/t2m", id="https-zarr-without-a-suffix"),
    pytest.param("gs://b/vol", "precomputed://gs://b/vol", id="bare-gs-is-precomputed"),
    # The volume, never one of its scales: neuroglancer finds no info under .../s2.
    pytest.param("precomputed://gs://b/vol/s2", "precomputed://gs://b/vol", id="precomputed-scale"),
    pytest.param("precomputed://s3://b/vol", "precomputed://s3://b/vol", id="precomputed-s3"),
])
def test_the_source_neuroglancer_reads_unwrapped_raw_data_from(path, source):
    """With wrap_raw=False the browser reads the data itself, at this URL."""
    assert neuroglancer_source(path) == source
