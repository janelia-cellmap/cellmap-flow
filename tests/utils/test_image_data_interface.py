"""ImageDataInterface: which level it opens, where it puts it, and what a read returns.

A read that lands a voxel off, or through the wrong chain, gives a model the
wrong input with nothing to show for it. The datasets hold each voxel's z
index + 1 (see conftest's ``ome_pyramid``), so a read's z column says which
voxels it hit.
"""

import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest
from funlib.geometry import Coordinate, Roi

from cellmap_flow.globals import g
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.norm.input_normalize import ChannelSelector, LambdaNormalizer, MinMaxNormalizer

Z = np.arange(1, 11, dtype=np.uint8)[:, None, None]  # z index + 1


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


# layout: (how it is written, ImageDataInterface arguments, where it starts, read ROI, (shape read, its z column))
READS = {
    # N5 attributes are x, y, z; the data is z, y, x like its metadata says.
    "n5": (
        lambda f: f.write_array("n5", np.broadcast_to(Z, (10, 20, 30)), {"resolution": [3, 2, 1], "offset": [0] * 3}),
        {}, (0, 0, 0), Roi((0, 0, 0), (2, 4, 6)), ((2, 2, 2), [1, 2]),
    ),
    # Stored x, y, z: read back z, y, x, where voxel_offset puts it.
    "precomputed": (
        lambda f: f.write_array("precomputed", np.broadcast_to(Z[:2], (2, 10, 20)), {
            "resolution": [4, 8, 16], "chunk_size": [20, 10, 2], "voxel_offset": [3, 2, 1]}),
        {}, (16, 16, 12), Roi((16, 16, 12), (32, 80, 80)), ((2, 10, 20), [1, 2]),
    ),
    # Outside the array a read is padded with 0, not the array's fill value.
    "border": (
        lambda f: f.write_array("zarr2", np.ones((4, 4, 4), np.uint8), {"resolution": [1] * 3, "offset": [0] * 3},
                                fill_value=7),
        {}, (0, 0, 0), Roi((-1, 0, 0), (2, 1, 1)), ((2, 1, 1), [0, 1]),
    ),
    "extra-compressor-field": (_extra_compressor_field, {}, (0, 0, 0), Roi((0, 0, 0), (4, 4, 4)), ((4, 4, 4), [1, 2, 3, 4])),
    # z = 524 nm is voxel 100 at 5.24 nm (it was voxel 104 at 5 nm).
    "fractional-voxel-size": (
        lambda f: f.ome_pyramid((((5.24, 4, 4), (2.62, 2, 2)),), shape=(200, 4, 4)) + "/s0",
        {}, (0, 0, 0), Roi((524, 0, 0), (52, 4, 4)), ((9, 1, 1), list(range(101, 110))),
    ),
    # Every Janelia level's corner is -4 nm: world [-4, 28) is exactly s2's voxel 0.
    "janelia-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4), (32, 12)), shape=(32, 32, 32)) + "/s2",
        {}, (-4, -4, -4), Roi((60, -4, -4), (64, 32, 32)), ((2, 1, 1), [3, 4]),
    ),
    "ome-v3-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4)), zarr_format=3),
        {"voxel_size": (16, 16, 16)}, (-4, -4, -4), Roi((-4, -4, -4), (32, 16, 16)), ((2, 1, 1), [1, 2]),
    ),
    # Relabelled voxel for voxel: s1's first voxel, at 120 nm on 12 nm voxels, is at 160 on 16.
    "relabelled-level": (
        lambda f: f.ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4)),
        {"voxel_size": (16, 16, 16)}, (160, 160, 160), Roi((160, 160, 160), (48, 16, 16)), ((3, 1, 1), [1, 2, 3]),
    ),
    "exact-level": (
        lambda f: f.ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4)),
        {"voxel_size": (12, 12, 12)}, (120, 120, 120), Roi((120, 120, 120), (12, 12, 12)), ((1, 1, 1), [1]),
    ),
}


@pytest.mark.parametrize("layout", READS)
def test_a_read_lands_where_the_metadata_says(layout, ome_pyramid, write_array):
    write, kwargs, corner, roi, (shape, column) = READS[layout]
    idi = ImageDataInterface(write(SimpleNamespace(ome_pyramid=ome_pyramid, write_array=write_array)),
                             input_norms=[], **kwargs)
    assert tuple(idi.roi.offset) == corner
    got = idi.to_ndarray_ts(roi)
    assert (got.shape, got[:, 0, 0].tolist()) == (shape, column)


def test_the_level_opened_and_the_voxel_size_reported(ome_pyramid, caplog):
    pyramid = ome_pyramid(((6, 123), (12, 126)), shape=(8, 4, 4))
    # No voxel size: the finest level.
    assert ImageDataInterface(pyramid).path.endswith("s0")

    # A level at another voxel size is relabelled, with a warning, or refused.
    with caplog.at_level(logging.WARNING):
        idi = ImageDataInterface(pyramid, voxel_size=(16, 16, 16))
    assert [r.name for r in caplog.records if r.levelno >= logging.WARNING] == ["cellmap_flow.image_data_interface"]
    assert (idi.path[-2:], idi.voxel_size, idi.actual_voxel_size, idi.requested_voxel_size) == (
        "s1", Coordinate(16, 16, 16), Coordinate(12, 12, 12), Coordinate(16, 16, 16),
    )
    assert tuple(idi.roi.shape) == (64, 32, 32)
    with pytest.raises(ValueError, match="requested"):
        ImageDataInterface(pyramid, voxel_size=(16, 16, 16), on_voxel_size_mismatch="error")

    # Levels are compared in nanometers, and a voxel size is a whole number
    # of them despite the float noise of converting units (70 A is
    # 7.000000000000001 nm), or else kept as floats (5.24 nm).
    microns = ome_pyramid((((0.008, 0.004, 0.004), None), ((0.016, 0.008, 0.008), None)), name="um.zarr",
                          unit="micrometer")
    assert ImageDataInterface(microns, voxel_size=(16, 8, 8)).path.endswith("s1")
    angstroms = ome_pyramid((((70, 40, 40), None),), name="a.zarr", unit="angstrom")
    assert ImageDataInterface(angstroms).voxel_size == Coordinate(7, 4, 4)
    fractional = ome_pyramid((((5.24, 4, 4), (2.62, 2, 2)),), shape=(200, 4, 4), name="f.zarr")
    idi = ImageDataInterface(fractional)
    assert idi.voxel_size == (5.24, 4.0, 4.0) and tuple(idi.roi.shape) == (1048, 16, 16)


def test_each_read_goes_through_its_own_chain(raw_zarr):
    roi = Roi((0, 0, 0), (32, 32, 32))
    idi = ImageDataInterface(raw_zarr(np.full((4, 4, 4), 7, np.uint8)))
    g.input_norms = [LambdaNormalizer("x * 2")]
    # With no chain of its own it follows g, on region and whole-array reads.
    assert np.all(idi.to_ndarray_ts(roi) == 14) and np.all(idi.to_ndarray_ts() == 14)
    # An explicit chain wins, and leaves the original alone; both share one store.
    tripled = idi.with_input_norms([LambdaNormalizer("x * 3")])
    assert np.all(tripled.to_ndarray_ts(roi) == 21) and np.all(idi.to_ndarray_ts(roi) == 14)
    assert tripled._raw_ts() is idi._raw_ts()
    unnormalized = ImageDataInterface(idi.path, normalize=False)
    assert np.all(unnormalized.to_ndarray_ts(roi) == 7)

    # The channel is chosen on every read, not when the store is opened: a
    # server that has read a chunk must follow a new ChannelSelector.
    channels = ImageDataInterface(raw_zarr(np.stack([np.full((4, 4, 4), c, np.uint8) for c in (1, 2)]), name="c"))
    g.input_norms = []
    assert np.all(channels.to_ndarray_ts(roi) == 1)
    g.input_norms = [ChannelSelector(1)]
    assert np.all(channels.to_ndarray_ts(roi) == 2) and np.all(np.asarray(channels.ts[...]) == 2)
    # A step without a dtype (ChannelSelector) keeps the last declared one.
    view = channels.with_input_norms([MinMaxNormalizer(), ChannelSelector(0)])
    assert view.ts.dtype == np.float32 and view.to_ndarray_ts(roi).dtype == np.float32
    assert channels.with_input_norms([]).ts.dtype == np.uint8


CACHE, CONCURRENCY = "CELLMAP_FLOW_RAW_CACHE_BYTES", "CELLMAP_FLOW_RAW_READ_CONCURRENCY"


@pytest.mark.parametrize(
    "env, cache_pool, limit",
    [({}, {"total_bytes_limit": 1 << 30}, None), ({CACHE: "0", CONCURRENCY: "3"}, {}, 3)],
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
        return idi._raw_ts().spec(retain_context=True).to_json()["context"]

    assert (context(served)["cache_pool"], context(served)["data_copy_concurrency"].get("limit")) == (cache_pool, limit)
    # Every other reader keeps one thread and no cache.
    assert (context(plain)["cache_pool"], context(plain)["data_copy_concurrency"]) == ({}, {"limit": 1})
    for _ in range(2):  # the second read comes from the cache
        assert np.array_equal(served.to_ndarray_ts(served.roi), plain.to_ndarray_ts(plain.roi))
