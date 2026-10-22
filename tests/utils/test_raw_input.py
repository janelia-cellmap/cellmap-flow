"""open_raw and RawInput: which level opens, where worldarray puts it, and what a read returns.

A read that lands a voxel off, or through the wrong chain, gives a model the
wrong input with nothing to show for it. The datasets hold each voxel's z
index + 1 (see conftest's ``ome_pyramid``), so a read's z column says which
voxels it hit. A world box is read by worldarray's rule: a voxel is read when
its centre lies in the half-open box.
"""

import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import worldarray as wa
from funlib.geometry import Roi

from cellmap_flow.io.raw import RawInput, open_raw, spatial_shape
from cellmap_flow.norm.input_normalize import ChannelSelector, LambdaNormalizer
from cellmap_flow.process_chain import process_chain

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


def _openorganelle_level(f):
    """OpenOrganelle's s1: 8^3 voxels of 10.48, 8, 8 nm. s0 (5.24, 4, 4 nm) is
    centred on 0, so every level's corner is at (-2.62, -2, -2) nm, which is
    not a whole nanometer in z."""
    return f.ome_pyramid((((5.24, 4, 4), 0), ((10.48, 8, 8), (2.62, 2, 2)))) + "/s1"


# Read the array's own box (``raw.array.box``).
OWN_BOX = "raw.array.box"

# What RawInput takes; a row's other arguments are open_raw's.
INPUT_ARGUMENTS = ("input_norms", "normalize")

# layout: (how it is written, open_raw and RawInput arguments, where it starts
#          (the corner), read box, (shape read, its dtype, its z column))
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
    # A box that begins at voxel -1's centre reads voxel -1: the box is half-open.
    "negative-off-grid-start": (
        _at_the_origin, {}, (0, 0, 0), Roi((-4, 0, 0), (16, 8, 8)), ((2, 1, 1), "uint8", [0, 1]),
    ),
    "past-the-end": (_at_the_origin, {}, (0, 0, 0), Roi((112, 0, 0), (32, 8, 8)), ((4, 1, 1), "uint8", [15, 16, 0, 0])),
    # Outside the array a read is padded with 0 (padding=0), not the array's fill value...
    "border": (
        lambda f: f.write_array("zarr2", np.ones((4, 4, 4), np.uint8), {"resolution": [1] * 3, "offset": [0] * 3},
                                fill_value=7),
        {}, (0, 0, 0), Roi((-1, 0, 0), (2, 1, 1)), ((2, 1, 1), "uint8", [0, 1]),
    ),
    # ...or with the border voxels repeated (padding="edge", open_raw's default
    # and what a model's input is read with)...
    "edge-fill": (
        _at_the_origin, {"padding": "edge"}, (0, 0, 0), Roi((-8, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [1, 1]),
    ),
    # ...which is added after the chain: the edge repeated is the normalized one.
    "chain-then-edge": (
        _at_the_origin, {"input_norms": [LambdaNormalizer("x * 2")], "padding": "edge"}, (0, 0, 0),
        Roi((-8, 0, 0), (16, 8, 8)), ((2, 1, 1), "float32", [2.0, 2.0]),
    ),
    "channel-selected": (
        _two_channels, {"input_norms": [ChannelSelector(1)]}, (0, 0, 0), Roi((0, 0, 0), (16, 8, 8)),
        ((2, 1, 1), "uint8", [101, 102]),
    ),
    "extra-compressor-field": (
        _extra_compressor_field, {}, (0, 0, 0), Roi((0, 0, 0), (4, 4, 4)), ((4, 4, 4), "uint8", [1, 2, 3, 4]),
    ),
    # z = 524 nm is voxel 100 at 5.24 nm (it was voxel 104 at 5 nm). 47 nm is
    # just short of 9 voxels (47.16 nm) and holds the centres of voxels
    # 100..108, as 9 whole voxels would; 52 nm (9.92 voxels) holds voxel 109's too.
    "fractional-voxel-size": (
        lambda f: f.ome_pyramid((((5.24, 4, 4), (2.62, 2, 2)),), shape=(200, 4, 4)) + "/s0",
        {}, (0, 0, 0), Roi((524, 0, 0), (47, 4, 4)), ((9, 1, 1), "uint8", list(range(101, 110))),
    ),
    # Every Janelia level's corner is -4 nm: world [-4, 28) is exactly s2's voxel 0.
    "janelia-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4), (32, 12)), shape=(32, 32, 32)) + "/s2",
        {}, (-4, -4, -4), Roi((60, -4, -4), (64, 32, 32)), ((2, 1, 1), "uint8", [3, 4]),
    ),
    # The corner is not a whole nanometer in z, and the array's box keeps it:
    # reading the box gives the array's own voxels, not a voxel of padding
    # and voxels 0..6.
    "openorganelle-own-box": (
        _openorganelle_level, {}, (-2.62, -2, -2), OWN_BOX, ((8, 8, 8), "uint8", list(range(1, 9))),
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
}


def _open(write, kwargs, ome_pyramid, write_array):
    """The row's dataset, written and opened. Its reads go through no chain and
    are padded with 0 unless the row says otherwise, so padding (open_raw's
    default repeats the edge) cannot pass for data."""
    path = write(SimpleNamespace(ome_pyramid=ome_pyramid, write_array=write_array))
    input_kwargs = {"input_norms": [], **{k: v for k, v in kwargs.items() if k in INPUT_ARGUMENTS}}
    open_kwargs = {"padding": 0, **{k: v for k, v in kwargs.items() if k not in INPUT_ARGUMENTS}}
    return RawInput(open_raw(path, **open_kwargs), **input_kwargs)


@pytest.mark.parametrize("layout", READS)
def test_a_read_lands_where_the_metadata_says(layout, ome_pyramid, write_array):
    write, kwargs, corner, box, (shape, dtype, column) = READS[layout]
    raw = _open(write, kwargs, ome_pyramid, write_array)
    assert raw.array.corner == pytest.approx(corner)
    got = raw.to_ndarray_ts(raw.array.box if box == OWN_BOX else box)
    assert (got.shape, str(got.dtype), got[:, 0, 0].tolist()) == (shape, dtype, column)


# --- resampling (open_raw's resample=True) ------------------------------------------


def _scipy_resampled(data, source_vs, target_vs):
    """What a resampled read should give, computed directly: a block mean on the
    axes whose factor is a whole number of 2 or more (the last block, cut short
    by the end of the data, over the voxels it has), then ``map_coordinates``
    (linear) at each target voxel's centre, clamped to the data's first and
    last voxel centres, on the others. Label data takes the voxel each target
    voxel's centre is in."""
    from scipy.ndimage import map_coordinates

    factors = [t / s for s, t in zip(source_vs, target_vs)]
    targets = [int(np.ceil(n / f - 1e-9)) for n, f in zip(data.shape, factors)]
    if data.dtype.kind == "u" and data.dtype.itemsize >= 4:
        index = [np.minimum(np.floor((np.arange(m) + 0.5) * f).astype(int), n - 1)
                 for n, m, f in zip(data.shape, targets, factors)]
        return data[np.ix_(*index)]
    values, coordinates = data.astype(np.float64), []
    for axis, (f, m) in enumerate(zip(factors, targets)):
        if f >= 2 and float(f).is_integer():
            blocks = np.array_split(values, range(int(f), values.shape[axis], int(f)), axis=axis)
            values = np.concatenate([block.mean(axis=axis, keepdims=True) for block in blocks], axis=axis)
            coordinates.append(np.arange(m, dtype=float))
        else:
            coordinates.append(np.clip((np.arange(m) + 0.5) * f - 0.5, 0, data.shape[axis] - 1))
    resampled = map_coordinates(values, np.meshgrid(*coordinates, indexing="ij"), order=1)
    return np.rint(resampled).astype(data.dtype) if data.dtype.kind in "iu" else resampled.astype(data.dtype)


# layout: (source voxel size, shape, dtype, voxel size asked for)
RESAMPLED_VALUES = {
    "block-mean-rounded-in-uint8": ((8, 8, 8), (8, 8, 8), np.uint8, (16, 16, 16)),
    "block-mean-of-a-last-block-cut-short": ((8, 8, 8), (5, 8, 7), np.float32, (24, 16, 24)),
    "upsampled-linear": ((8, 8, 8), (4, 4, 4), np.float32, (4, 4, 4)),
    "non-integer-factor-linear": ((12, 12, 12), (8, 8, 8), np.float32, (16, 16, 16)),
    # z linear (40 -> 16), y and x block means (8 -> 16).
    "16nm-model-on-8x8x40-data": ((40, 8, 8), (4, 8, 8), np.float32, (16, 16, 16)),
    # Labels take the nearest voxel, whatever the factor.
    "labels-nearest": ((12, 8, 8), (8, 8, 8), np.uint64, (16, 16, 16)),
}


def _random_volume(write_array, layout):
    source_vs, shape, dtype, requested = RESAMPLED_VALUES[layout]
    data = (np.random.default_rng(0).random(shape) * 250).astype(dtype)
    path = write_array("zarr2", data, {"resolution": list(source_vs), "offset": [0, 0, 0]})
    raw = RawInput(open_raw(path, voxel_size=requested, resample=True, padding=0), input_norms=[])
    return raw, data


@pytest.mark.parametrize("layout", RESAMPLED_VALUES)
def test_a_resampled_read_is_the_direct_computation(write_array, layout):
    """Every target voxel, read by index: ``raw.array.box`` is the data's own
    extent, which can end before the last target voxel's centre (5 x voxels
    of 8 nm are 40 nm, two 24 nm voxels centred at 12 and 36 nm, the second
    reaching past the data; at 7 they are 56 nm and the third, at 60 nm, is
    outside it)."""
    raw, data = _random_volume(write_array, layout)
    source_vs, _, dtype, requested = RESAMPLED_VALUES[layout]
    shape = spatial_shape(raw.array)
    got, expected = raw.read_index((0, 0, 0), shape), _scipy_resampled(data, source_vs, requested)
    assert got.dtype == dtype and got.shape == expected.shape == shape
    if np.dtype(dtype).kind == "f":
        np.testing.assert_allclose(got, expected, rtol=1e-6)
    else:
        np.testing.assert_array_equal(got, expected)
    # Through the chain after resampling, as a stored level would be.
    doubled = raw.with_input_norms([LambdaNormalizer("x * 2")]).read_index((0, 0, 0), shape)
    np.testing.assert_allclose(doubled, expected.astype(np.float32) * 2, rtol=1e-6)


@pytest.mark.parametrize("layout", RESAMPLED_VALUES)
def test_neighbouring_resampled_chunks_meet_exactly(write_array, layout):
    """The server reads every chunk on its own: two that meet must give, at
    their shared border, the voxels one read across both gives, including
    where they run past the data into padding."""
    raw, _ = _random_volume(write_array, layout)
    voxel = np.array(raw.array.voxel_size)
    # Every target voxel and two more on each side, cut below on target voxel corners.
    shape = spatial_shape(raw.array)
    whole = wa.Box.from_index((-2,) * 3, tuple(n + 2 for n in shape), voxel, raw.array.translation)
    expected = raw.to_ndarray_ts(whole)
    for axis in range(3):
        for split in range(1, expected.shape[axis]):
            at = whole.begin[axis] + split * voxel[axis]
            first = wa.Box(whole.begin, tuple(at if i == axis else e for i, e in enumerate(whole.end)))
            rest = wa.Box(tuple(at if i == axis else b for i, b in enumerate(whole.begin)), whole.end)
            pieces = np.concatenate([raw.to_ndarray_ts(first), raw.to_ndarray_ts(rest)], axis=axis)
            assert np.array_equal(pieces, expected), (axis, split)


def test_each_read_goes_through_its_own_chain(raw_zarr):
    roi = Roi((0, 0, 0), (32, 32, 32))
    raw = RawInput(open_raw(raw_zarr(np.full((4, 4, 4), 7, np.uint8))))
    process_chain().input_norms = [LambdaNormalizer("x * 2")]
    # With no chain of its own it follows the process's, on a region and on the array's own box.
    assert np.all(raw.to_ndarray_ts(roi) == 14) and np.all(raw.to_ndarray_ts(raw.array.box) == 14)
    # An explicit chain wins, and leaves the original alone; both share one open array.
    tripled = raw.with_input_norms([LambdaNormalizer("x * 3")])
    assert np.all(tripled.to_ndarray_ts(roi) == 21) and np.all(raw.to_ndarray_ts(roi) == 14)
    assert tripled.array is raw.array
    assert np.all(RawInput(raw.array, normalize=False).to_ndarray_ts(roi) == 7)
    # Pickled, it still has no chain of its own: it reads the chain of the
    # process that unpickles it, which in a spawned loader worker is empty.
    unpickled = pickle.loads(pickle.dumps(raw))
    process_chain().input_norms = []
    assert np.all(unpickled.to_ndarray_ts(roi) == 7)
