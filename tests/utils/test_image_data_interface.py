"""ImageDataInterface: which level it opens, where it puts it, and what a read returns.

A read that lands a voxel off, or through the wrong chain, gives a model the
wrong input with nothing to show for it. The datasets hold each voxel's z
index + 1 (see conftest's ``ome_pyramid``), so a read's z column says which
voxels it hit.
"""

import json
import logging
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from cellmap_flow.globals import g
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.norm.input_normalize import ChannelSelector, LambdaNormalizer, MinMaxNormalizer

Z = np.arange(1, 11, dtype=np.uint8)[:, None, None]  # z index + 1


def _at_the_origin(f):
    """16^3 voxels of 8 nm, voxel 0's corner at the origin."""
    return f.ome_pyramid(((8, 4),)) + "/s0"


def _two_channels(f):
    """(c, z, y, x): 2 channels of 16^3 voxels of 8 nm; channel c holds z index + 1 + 100 c."""
    z = np.broadcast_to(np.arange(1, 17, dtype=np.uint8)[:, None, None], (16, 16, 16))
    return f.write_array("zarr2", np.stack([z, z + 100]), {"resolution": [8] * 3, "offset": [0] * 3})


def _precomputed_pyramid(f):
    """Scales 0 and 1 of a precomputed volume: 4, 10, 20 z, y, x voxels of 16, 8,
    4 nm, each holding its z index + 1, then every other one of them at twice the
    size; both with their corner at (32, 16, 8) nm."""
    return f.write_array("precomputed", np.broadcast_to(Z[:4], (4, 10, 20)), {
        "resolution": [4, 8, 16], "voxel_offset": [2] * 3}, scales=2)


def _extra_compressor_field(f):
    """Newer numcodecs write compressor fields, such as zstd's checksum, that
    tensorstore's zarr driver rejects as extra members."""
    import numcodecs

    path = f.write_array("zarr2", np.broadcast_to(Z[:4], (4, 4, 4)), compressor=numcodecs.Zstd(level=1))
    with open(os.path.join(path, ".zarray")) as file:
        meta = json.load(file)
    meta["compressor"]["checksum"] = False
    with open(os.path.join(path, ".zarray"), "w") as file:
        json.dump(meta, file)
    return path


# layout: (how it is written, ImageDataInterface arguments, where it starts, read ROI,
#          (shape read, its dtype, its z column))
READS = {
    # N5 attributes are x, y, z; the data is z, y, x like its metadata says.
    "n5": (
        lambda f: f.write_array("n5", np.broadcast_to(Z, (10, 20, 30)), {"resolution": [3, 2, 1], "offset": [0] * 3}),
        {}, (0, 0, 0), Roi((0, 0, 0), (2, 4, 6)), ((2, 2, 2), "uint8", [1, 2]),
    ),
    # Stored x, y, z: read back z, y, x, where voxel_offset puts it.
    "precomputed": (
        lambda f: f.write_array("precomputed", np.broadcast_to(Z[:2], (2, 10, 20)), {
            "resolution": [4, 8, 16], "chunk_size": [20, 10, 2], "voxel_offset": [3, 2, 1]}),
        {}, (16, 16, 12), Roi((16, 16, 12), (32, 80, 80)), ((2, 10, 20), "uint8", [1, 2]),
    ),
    # A trailing /s1 opens scale 1, z, y, x as well: s0's voxels 0 and 2.
    "precomputed-level": (
        lambda f: _precomputed_pyramid(f) + "/s1", {}, (32, 16, 8), Roi((32, 16, 8), (64, 80, 80)),
        ((2, 5, 10), "uint8", [1, 3]),
    ),
    "whole-array": (_at_the_origin, {}, (0, 0, 0), None, ((16, 16, 16), "uint8", list(range(1, 17)))),
    # A start inside a voxel reads from that voxel.
    "off-grid-start": (_at_the_origin, {}, (0, 0, 0), Roi((4, 0, 0), (16, 8, 8)), ((2, 1, 1), "uint8", [1, 2])),
    "negative-start": (_at_the_origin, {}, (0, 0, 0), Roi((-16, 0, 0), (32, 8, 8)), ((4, 1, 1), "uint8", [0, 0, 1, 2])),
    # Half a voxel before the array is in voxel -1, as on the other side of voxel 0.
    "negative-off-grid-start": (
        _at_the_origin, {}, (0, 0, 0), Roi((-4, 0, 0), (16, 8, 8)), ((2, 1, 1), "uint8", [0, 1]),
    ),
    "past-the-end": (_at_the_origin, {}, (0, 0, 0), Roi((112, 0, 0), (32, 8, 8)), ((4, 1, 1), "uint8", [15, 16, 0, 0])),
    # Outside the array a read is padded with 0, not the array's fill value...
    "border": (
        lambda f: f.write_array("zarr2", np.ones((4, 4, 4), np.uint8), {"resolution": [1] * 3, "offset": [0] * 3},
                                fill_value=7),
        {}, (0, 0, 0), Roi((-1, 0, 0), (2, 1, 1)), ((2, 1, 1), "uint8", [0, 1]),
    ),
    # ...or with custom_fill_value, or the border voxels repeated...
    "custom-fill-value": (
        _at_the_origin, {"custom_fill_value": 9}, (0, 0, 0), Roi((-8, 0, 0), (16, 8, 8)), ((2, 1, 1), "uint8", [9, 1]),
    ),
    "edge-fill": (
        _at_the_origin, {"custom_fill_value": "edge"}, (0, 0, 0), Roi((-8, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [1, 1]),
    ),
    # ...and the padding is added after the chain: it is never normalized.
    "chain-then-padding": (
        _at_the_origin, {"input_norms": [LambdaNormalizer("x * 2")]}, (0, 0, 0), Roi((-8, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "float32", [0.0, 2.0]),
    ),
    # A ROI wholly outside the array is all padding, in the chain's dtype...
    "wholly-past-the-end": (
        _at_the_origin, {"input_norms": [LambdaNormalizer("x * 2")]}, (0, 0, 0), Roi((200, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "float32", [0.0, 0.0]),
    ),
    # ...and "edge" repeats the array's voxels nearest it, on either side.
    "wholly-before-the-start-edge": (
        _at_the_origin, {"custom_fill_value": "edge"}, (0, 0, 0), Roi((-32, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [1, 1]),
    ),
    "wholly-past-the-end-edge": (
        _at_the_origin, {"custom_fill_value": "edge"}, (0, 0, 0), Roi((200, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [16, 16]),
    ),
    "not-normalized": (
        _at_the_origin, {"input_norms": [LambdaNormalizer("x * 2")], "normalize": False}, (0, 0, 0),
        Roi((0, 0, 0), (16, 8, 8)), ((2, 1, 1), "uint8", [1, 2]),
    ),
    "channel-selected": (
        _two_channels, {"input_norms": [ChannelSelector(1)]}, (0, 0, 0), Roi((0, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [101, 102]),
    ),
    # output_voxel_size resamples by the z factor: finer repeats each voxel, read
    # widened to whole voxels (here [0, 24) nm) and cropped back to the ROI...
    "upsampled": (
        _at_the_origin, {"output_voxel_size": (4, 4, 4)}, (0, 0, 0), Roi((4, 0, 0), (16, 8, 8)),
        ((4, 2, 2), "uint8", [1, 2, 2, 3]),
    ),
    # ...coarser takes each block's median...
    "downsampled": (
        _at_the_origin, {"output_voxel_size": (16, 16, 16)}, (0, 0, 0), Roi((0, 0, 0), (32, 16, 16)),
        ((2, 1, 1), "float64", [1.5, 3.5]),
    ),
    # ...and the same z voxel size resamples nothing, even when y and x differ.
    "resampled-z-unchanged": (
        _at_the_origin, {"output_voxel_size": (8, 4, 4)}, (0, 0, 0), Roi((0, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [1, 2]),
    ),
    "extra-compressor-field": (
        _extra_compressor_field, {}, (0, 0, 0), Roi((0, 0, 0), (4, 4, 4)), ((4, 4, 4), "uint8", [1, 2, 3, 4]),
    ),
    # z = 524 nm is voxel 100 at 5.24 nm (it was voxel 104 at 5 nm).
    "fractional-voxel-size": (
        lambda f: f.ome_pyramid((((5.24, 4, 4), (2.62, 2, 2)),), shape=(200, 4, 4)) + "/s0",
        {}, (0, 0, 0), Roi((524, 0, 0), (52, 4, 4)), ((9, 1, 1), "uint8", list(range(101, 110))),
    ),
    # Every Janelia level's corner is -4 nm: world [-4, 28) is exactly s2's voxel 0.
    "janelia-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4), (32, 12)), shape=(32, 32, 32)) + "/s2",
        {}, (-4, -4, -4), Roi((60, -4, -4), (64, 32, 32)), ((2, 1, 1), "uint8", [3, 4]),
    ),
    "ome-v3-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4)), zarr_format=3),
        {"voxel_size": (16, 16, 16)}, (-4, -4, -4), Roi((-4, -4, -4), (32, 16, 16)), ((2, 1, 1), "uint8", [1, 2]),
    ),
    # Relabelled voxel for voxel: s1's first voxel, at 120 nm on 12 nm voxels, is at 160 on 16.
    "relabelled-level": (
        lambda f: f.ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4)),
        {"voxel_size": (16, 16, 16)}, (160, 160, 160), Roi((160, 160, 160), (48, 16, 16)),
        ((3, 1, 1), "uint8", [1, 2, 3]),
    ),
    "exact-level": (
        lambda f: f.ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4)),
        {"voxel_size": (12, 12, 12)}, (120, 120, 120), Roi((120, 120, 120), (12, 12, 12)), ((1, 1, 1), "uint8", [1]),
    ),
}


def _open(write, kwargs, ome_pyramid, write_array):
    path = write(SimpleNamespace(ome_pyramid=ome_pyramid, write_array=write_array))
    return ImageDataInterface(path, **{"input_norms": [], **kwargs})


# The resampling rows pass the deprecated arguments, which work until they go.
@pytest.mark.filterwarnings("ignore:ImageDataInterface's .* is deprecated:DeprecationWarning")
@pytest.mark.parametrize("layout", READS)
def test_a_read_lands_where_the_metadata_says(layout, ome_pyramid, write_array):
    write, kwargs, corner, roi, (shape, dtype, column) = READS[layout]
    idi = _open(write, kwargs, ome_pyramid, write_array)
    assert tuple(idi.roi.offset) == corner
    got = idi.to_ndarray_ts(roi)
    assert (got.shape, str(got.dtype), got[:, 0, 0].tolist()) == (shape, dtype, column)


# layout: (what ImageDataInterface reports: voxel size, offset, roi, shape, chunk shape, axes, file type)
OPENED = {
    "n5": ((1, 2, 3), (0, 0, 0), Roi((0, 0, 0), (10, 40, 90)), (10, 20, 30), (5, 10, 15), ["z", "y", "x"], "n5"),
    "precomputed": (
        (16, 8, 4), (16, 16, 12), Roi((16, 16, 12), (32, 80, 80)), (2, 10, 20), (2, 10, 20), ["z", "y", "x"],
        "precomputed",
    ),
    "precomputed-level": (
        (32, 16, 8), (32, 16, 8), Roi((32, 16, 8), (64, 80, 80)), (2, 5, 10), (2, 5, 10), ["z", "y", "x"],
        "precomputed",
    ),
    # A local zarr v2 array reports its last three axes, whatever is before them.
    "channel-selected": (
        (8, 8, 8), (0, 0, 0), Roi((0, 0, 0), (128, 128, 128)), (16, 16, 16), (8, 8, 8), ["z", "y", "x"], "zarr",
    ),
    "fractional-voxel-size": (
        (5.24, 4.0, 4.0), (0, 0, 0), Roi((0, 0, 0), (1048, 16, 16)), (200, 4, 4), (100, 2, 2), ["z", "y", "x"], "zarr",
    ),
    "ome-v3-level": ((16, 16, 16), (-4, -4, -4), Roi((-4, -4, -4), (128, 128, 128)), (8, 8, 8), (4, 4, 4),
                     ["z", "y", "x"], "zarr"),
    "relabelled-level": ((16, 16, 16), (160, 160, 160), Roi((160, 160, 160), (64, 32, 32)), (4, 2, 2), (2, 1, 1),
                         ["z", "y", "x"], "zarr"),
}


@pytest.mark.parametrize("layout", OPENED)
def test_what_a_dataset_opens_as(layout, ome_pyramid, write_array):
    write, kwargs = READS[layout][:2]
    idi = _open(write, kwargs, ome_pyramid, write_array)
    assert (tuple(idi.voxel_size), tuple(idi.offset), idi.roi, tuple(idi.shape), tuple(idi.chunk_shape), idi.axes_names,
            idi.filetype) == OPENED[layout]
    assert idi.output_voxel_size == idi.voxel_size


# layout: (what ts is, its dtype, domain origin and shape, ts[:2, 0, 0])
TS_VIEWS = {
    "n5": ("LazyNormalization", np.uint8, ((0, 0, 0), (10, 20, 30)), [1, 2]),
    # C order, indexed from 0 though tensorstore starts the volume at voxel_offset.
    "precomputed": ("LazyNormalization", np.uint8, ((0, 0, 0), (2, 10, 20)), [1, 2]),
    "chain-then-padding": ("LazyNormalization", np.float32, ((0, 0, 0), (16, 16, 16)), [2.0, 4.0]),
    "not-normalized": ("TensorStore", np.uint8, ((0, 0, 0), (16, 16, 16)), [1, 2]),
    "channel-selected": ("LazyNormalization", np.uint8, ((0, 0, 0), (16, 16, 16)), [101, 102]),
}


@pytest.mark.parametrize("layout", TS_VIEWS)
def test_ts_is_one_channel_of_the_dataset_through_the_chain(layout, ome_pyramid, write_array):
    """What neuroglancer indexes, and what process_chunk scripts may read directly."""
    write, kwargs = READS[layout][:2]
    ts = _open(write, kwargs, ome_pyramid, write_array).ts
    dtype = np.dtype(getattr(ts.dtype, "numpy_dtype", ts.dtype))
    assert (type(ts).__name__, dtype, (tuple(ts.domain.inclusive_min), tuple(ts.shape)),
            np.asarray(ts[:2, 0, 0]).tolist()) == TS_VIEWS[layout]


@pytest.mark.parametrize("kwargs, deprecated", [
    pytest.param({"output_voxel_size": (4, 4, 4)}, "output_voxel_size", id="output_voxel_size"),
    pytest.param({"custom_fill_value": 9}, "custom_fill_value", id="custom_fill_value"),
    # The inference server's parallel reads: in use, and kept.
    pytest.param({"concurrency_limit": 3}, None, id="concurrency_limit-is-not"),
])
def test_the_resampling_arguments_are_deprecated(raw_zarr, kwargs, deprecated):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ImageDataInterface(raw_zarr(), **kwargs)
    ours = [str(w.message) for w in caught if w.category is DeprecationWarning and "ImageDataInterface" in str(w.message)]
    assert [deprecated in message for message in ours] == ([True] if deprecated else [])


def test_without_a_voxel_size_the_finest_level_is_opened(ome_pyramid):
    assert ImageDataInterface(ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4))).path.endswith("s0")


def test_a_level_at_another_voxel_size_is_relabelled_with_a_warning_or_refused(ome_pyramid, caplog):
    pyramid = ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4))
    with caplog.at_level(logging.WARNING):
        idi = ImageDataInterface(pyramid, voxel_size=(16, 16, 16))
    assert [r.name for r in caplog.records if r.levelno >= logging.WARNING] == ["cellmap_flow.image_data_interface"]
    assert (idi.path[-2:], idi.voxel_size, idi.actual_voxel_size, idi.requested_voxel_size) == (
        "s1", Coordinate(16, 16, 16), Coordinate(12, 12, 12), Coordinate(16, 16, 16),
    )
    assert tuple(idi.roi.shape) == (64, 32, 32)
    with pytest.raises(ValueError, match="requested"):
        ImageDataInterface(pyramid, voxel_size=(16, 16, 16), on_voxel_size_mismatch="error")


@pytest.mark.parametrize("levels, unit, requested, level, voxel_size, roi_shape", [
    # Levels are compared in nanometers.
    pytest.param((((0.008, 0.004, 0.004), None), ((0.016, 0.008, 0.008), None)), "micrometer", (16, 8, 8),
                 "s1", Coordinate(16, 8, 8), None, id="micrometers-picked-in-nm"),
    # A whole number despite the float noise of converting (70 A is 7.000000000000001 nm)...
    pytest.param((((70, 40, 40), None),), "angstrom", None, "s0", Coordinate(7, 4, 4), None, id="angstroms-whole-nm"),
    # ...or kept as floats: Coordinate would truncate 5.24 to 5, and 200 voxels span 1048 nm, not 1000.
    pytest.param((((5.24, 4, 4), (2.62, 2, 2)),), "nanometer", None, "s0", (5.24, 4.0, 4.0), (1048, 16, 16),
                 id="fractional-nm-kept"),
])
def test_the_voxel_size_is_in_nanometers_whole_or_fractional(ome_pyramid, levels, unit, requested, level, voxel_size,
                                                             roi_shape):
    idi = ImageDataInterface(ome_pyramid(levels, shape=(200, 4, 4), unit=unit), voxel_size=requested)
    assert (idi.path[-2:], idi.voxel_size) == (level, voxel_size)
    assert roi_shape is None or tuple(idi.roi.shape) == roi_shape


def test_each_read_goes_through_its_own_chain(raw_zarr):
    roi = Roi((0, 0, 0), (32, 32, 32))
    idi = ImageDataInterface(raw_zarr(np.full((4, 4, 4), 7, np.uint8)))
    g.input_norms = [LambdaNormalizer("x * 2")]
    # With no chain of its own it follows g, on region and whole-array reads.
    assert np.all(idi.to_ndarray_ts(roi) == 14) and np.all(idi.to_ndarray_ts() == 14)
    # An explicit chain wins, and leaves the original alone; both share one store.
    tripled = idi.with_input_norms([LambdaNormalizer("x * 3")])
    assert np.all(tripled.to_ndarray_ts(roi) == 21) and np.all(idi.to_ndarray_ts(roi) == 14)
    assert tripled.source.ts is idi.source.ts
    assert np.all(ImageDataInterface(idi.path, normalize=False).to_ndarray_ts(roi) == 7)


def test_the_channel_is_chosen_on_every_read_and_keeps_the_last_declared_dtype(raw_zarr):
    """A server that has read a chunk must follow a new ChannelSelector."""
    roi = Roi((0, 0, 0), (32, 32, 32))
    channels = ImageDataInterface(raw_zarr(np.stack([np.full((4, 4, 4), c, np.uint8) for c in (1, 2)])))
    g.input_norms = []
    assert np.all(channels.to_ndarray_ts(roi) == 1)
    g.input_norms = [ChannelSelector(1)]
    assert np.all(channels.to_ndarray_ts(roi) == 2) and np.all(np.asarray(channels.ts[...]) == 2)
    # ChannelSelector declares no dtype: MinMax's float32 still reaches the viewer.
    view = channels.with_input_norms([MinMaxNormalizer(), ChannelSelector(0)])
    assert view.ts.dtype == np.float32 and view.to_ndarray_ts(roi).dtype == np.float32
    assert channels.with_input_norms([]).ts.dtype == np.uint8


CACHE, CONCURRENCY = "CELLMAP_FLOW_RAW_CACHE_BYTES", "CELLMAP_FLOW_RAW_READ_CONCURRENCY"


@pytest.mark.parametrize(
    "env, cache_pool, limit",
    [pytest.param({}, {"total_bytes_limit": 1 << 30}, None, id="default-cache"),
     pytest.param({CACHE: "0", CONCURRENCY: "3"}, {}, 3, id="set-by-the-environment")],
)
def test_the_inference_server_reads_in_parallel_through_a_cache(raw_zarr, model_script, monkeypatch, env,
                                                                 cache_pool, limit):
    from cellmap_flow.models.models_config import ScriptModelConfig
    from cellmap_flow.server import CellMapFlowServer

    for name in (CACHE, CONCURRENCY):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    raw = raw_zarr(np.arange(512).reshape(8, 8, 8).astype(np.uint8))
    served = CellMapFlowServer(raw, ScriptModelConfig(script_path=model_script())).idi_raw
    plain = ImageDataInterface(raw, voxel_size=(8, 8, 8))

    def context(idi):
        return idi.source.ts.spec(retain_context=True).to_json()["context"]

    assert (context(served)["cache_pool"], context(served)["data_copy_concurrency"].get("limit")) == (cache_pool, limit)
    # Every other reader keeps one thread and no cache.
    assert (context(plain)["cache_pool"], context(plain)["data_copy_concurrency"]) == ({}, {"limit": 1})
    for _ in range(2):  # the second read comes from the cache
        assert np.array_equal(served.to_ndarray_ts(served.roi), plain.to_ndarray_ts(plain.roi))
