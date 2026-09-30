"""What cellmap-flow reads from each layout it supports, through io/.

Every reader (ImageDataInterface, the viewer, blockwise, the finetune crop
loader) takes voxel sizes, offsets and levels from here, so one row per
layout: OME-Zarr v2 and v3 at every placement of their transforms, funlib
and N5 attributes, precomputed, http. Offsets are voxel 0's corner, in nm;
an OME translation is voxel 0's centre.
"""

import functools
import http.server
import json
import logging
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from cellmap_flow.io import paths
from cellmap_flow.io.metadata import ArrayMeta, list_levels, read_array_meta
from cellmap_flow.io.multiscale import closest_raw_scale, select_dataset, select_level
from cellmap_flow.io.ome import multiscales_attrs, singlescale_attrs

ZYX = ("z", "y", "x")
DATA = np.zeros((4, 4, 4), np.uint8)


def _v3_group(path):
    """A zarr v3 group with no attributes; its path."""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)
    return str(path)


def _v3_group_with_s0(f):
    f.write_array("zarr3", DATA, name="g.zarr/s0")
    return _v3_group(f.tmp / "g.zarr")


@pytest.fixture
def http_root(tmp_path):
    """tmp_path served over http; its URL."""
    quiet = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(quiet, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


# layout: (how it is written, (format, axes, voxel size, corner, shape, chunk shape))
LAYOUTS = {
    # Janelia's levels share a corner at -4 nm: translation = scale / 2 - 4.
    "ome-v2-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4))) + "/s1",
        ("zarr2", ZYX, (16.0,) * 3, (-4.0,) * 3, (8, 8, 8), (4, 4, 4)),
    ),
    "ome-v2-no-translation": (
        lambda f: f.ome_pyramid(((8, None),)) + "/s0",
        ("zarr2", ZYX, (8.0,) * 3, (-4.0,) * 3, (16, 16, 16), (8, 8, 8)),
    ),
    "ome-v2-channels": (
        lambda f: f.ome_pyramid(((8, 4),), shape=(8, 8, 8), channels=2) + "/s0",
        ("zarr2", ("c",) + ZYX, (1.0, 8.0, 8.0, 8.0), (0.0,) * 4, (2, 8, 8, 8), (1, 4, 4, 4)),
    ),
    # Each v3 level has its own zarr.json; the transform is the group's.
    "ome-v3-level": (
        lambda f: f.ome_pyramid(((4, (1, 2, 3)), (8, (1, 2, 3))), zarr_format=3) + "/s1",
        ("zarr3", ZYX, (8.0,) * 3, (-3.0, -2.0, -1.0), (8, 8, 8), (4, 4, 4)),
    ),
    "ome-v3-channels": (
        lambda f: f.ome_pyramid(((8, 4),), shape=(8, 8, 8), channels=2, zarr_format=3) + "/s0",
        ("zarr3", ("c",) + ZYX, (1.0, 8.0, 8.0, 8.0), (0.0,) * 4, (2, 8, 8, 8), (1, 4, 4, 4)),
    ),
    # A group's own path reads its first level.
    "ome-v3-group": (
        lambda f: f.ome_pyramid(((4, 2), (8, 4)), zarr_format=3),
        ("zarr3", ZYX, (4.0,) * 3, (0.0,) * 3, (16, 16, 16), (8, 8, 8)),
    ),
    "ome-float-voxel-size": (
        lambda f: f.ome_pyramid((((5.24, 4, 4), (2.62, 2, 2)),), shape=(200, 4, 4)) + "/s0",
        ("zarr2", ZYX, (5.24, 4.0, 4.0), (0.0,) * 3, (200, 4, 4), (100, 2, 2)),
    ),
    "ome-micrometer": (
        lambda f: f.ome_pyramid((((0.008, 0.004, 0.004), (0.08, 0.04, 0.04)),), unit="micrometer") + "/s0",
        ("zarr2", ZYX, (8.0, 4.0, 4.0), (76.0, 38.0, 38.0), (16, 16, 16), (8, 8, 8)),
    ),
    "http-ome-group": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4))),
        ("zarr2", ZYX, (8.0,) * 3, (-4.0,) * 3, (16, 16, 16), (8, 8, 8)),
    ),
    # The multiscales are on em, the group above the level.
    "http-ome-nested-level": (
        lambda f: f.ome_pyramid(((8, 0), (16, 4)), name="n.zarr/em") + "/s1",
        ("zarr2", ZYX, (16.0,) * 3, (-4.0,) * 3, (8, 8, 8), (4, 4, 4)),
    ),
    "http-funlib-attrs": (
        lambda f: f.raw_zarr(DATA, offset=(4, 4, 4)),
        ("zarr2", ZYX, (8.0,) * 3, (8.0,) * 3, (4, 4, 4), (4, 4, 4)),
    ),
    "funlib-attrs": (
        lambda f: f.raw_zarr(DATA, voxel_size=(8, 4, 4), offset=(80, 40, 40)),
        ("zarr2", ZYX, (8.0, 4.0, 4.0), (80.0, 40.0, 40.0), (4, 4, 4), (4, 4, 4)),
    ),
    # A v2 offset off the voxel grid is rounded onto it (funlib's rule).
    "funlib-offset-off-grid": (
        lambda f: f.raw_zarr(DATA, offset=(4, 4, 4)),
        ("zarr2", ZYX, (8.0,) * 3, (8.0,) * 3, (4, 4, 4), (4, 4, 4)),
    ),
    "funlib-no-attrs": (
        lambda f: f.write_array("zarr2", DATA),
        ("zarr2", ZYX, (1.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "funlib-root-array": (
        lambda f: f.write_array("zarr2", DATA, {"resolution": [8, 4, 4], "offset": [80, 40, 40]}, "root.zarr"),
        ("zarr2", ZYX, (8.0, 4.0, 4.0), (80.0, 40.0, 40.0), (4, 4, 4), (2, 2, 2)),
    ),
    # Fortran memory order is not an axis order.
    "funlib-f-order": (
        lambda f: f.write_array("zarr2", np.zeros((4, 6, 8), np.uint8), {"resolution": [8, 4, 2]}, order="F"),
        ("zarr2", ZYX, (8.0, 4.0, 2.0), (0.0,) * 3, (4, 6, 8), (2, 3, 4)),
    ),
    # Attributes covering the last axes: the others are channels.
    "funlib-channels": (
        lambda f: f.write_array("zarr2", np.zeros((3, 4, 4, 4), np.uint8), {"resolution": [8, 4, 4], "offset": [0] * 3}),
        ("zarr2", ("c^",) + ZYX, (1.0, 8.0, 4.0, 4.0), (0.0,) * 4, (3, 4, 4, 4), (1, 2, 2, 2)),
    ),
    "v3-transform-attrs": (
        lambda f: f.write_array("zarr3", DATA, {"transform": {"scale": [8] * 3, "translate": [100, 200, 300]}}),
        ("zarr3", ZYX, (8.0,) * 3, (100.0, 200.0, 300.0), (4, 4, 4), (2, 2, 2)),
    ),
    # A v3 array's own offset is taken as written.
    "v3-funlib-attrs": (
        lambda f: f.write_array("zarr3", DATA, {"resolution": [8] * 3, "offset": [4] * 3}),
        ("zarr3", ZYX, (8.0,) * 3, (4.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "v3-no-attrs": (
        lambda f: f.write_array("zarr3", DATA),
        ("zarr3", ZYX, (1.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "v3-group-without-multiscales": (
        _v3_group_with_s0,
        ("zarr3", ZYX, (1.0,) * 3, (0.0,) * 3, (4, 4, 4), (2, 2, 2)),
    ),
    "v3-empty-group": (lambda f: _v3_group(f.tmp / "g.zarr"), RuntimeError),
    # N5's attributes are x, y, z.
    "n5-resolution": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"resolution": [3, 2, 1], "offset": [0] * 3}),
        ("n5", ZYX, (1.0, 2.0, 3.0), (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # COSEM's N5: a C-order transform.
    "n5-transform": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"transform": {
            "ordering": "C", "scale": [8, 4, 2], "translate": [80, 40, 20], "units": ["nm"] * 3}}),
        ("n5", ZYX, (8.0, 4.0, 2.0), (80.0, 40.0, 20.0), (10, 20, 30), (5, 10, 15)),
    ),
    # One unit for every axis, not a string to reverse with them ("um" read as "mu").
    "n5-units-string": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {
            "resolution": [3, 2, 1], "offset": [0] * 3, "units": "um"}),
        ("n5", ZYX, (1000.0, 2000.0, 3000.0), (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # Without an offset the whole lookup falls through, voxel size and all.
    "n5-resolution-without-offset": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {"resolution": [3, 2, 1]}),
        ("n5", ZYX, (1.0,) * 3, (0.0,) * 3, (10, 20, 30), (5, 10, 15)),
    ),
    # BigDataViewer's and Paintera's N5.
    "n5-pixel-resolution": (
        lambda f: f.write_array("n5", np.zeros((10, 20, 30), np.uint8), {
            "pixelResolution": {"dimensions": [2, 4, 8], "unit": "nm"},
            "downsamplingFactors": [2, 2, 1], "offset": [30, 20, 10]}),
        ("n5", ZYX, (8.0, 8.0, 4.0), (8.0, 24.0, 32.0), (10, 20, 30), (5, 10, 15)),
    ),
    # x, y, z voxels (3, 2, 1) at (4, 8, 16) nm: the corner is (16, 16, 12) nm.
    "precomputed": (
        lambda f: f.write_array("precomputed", np.zeros((2, 10, 20), np.uint8), {
            "resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]}),
        ("precomputed", ("channel",) + ZYX, (1.0, 16.0, 8.0, 4.0), (0.0, 16.0, 16.0, 12.0), (1, 2, 10, 20), (1, 2, 5, 10)),
    ),
    # The scale is named by the last /s<N> only: not by /scans, as on /groups/scicompsoft.
    "precomputed-scale-under-an-s-directory": (
        lambda f: f.write_array("precomputed", np.zeros((2, 10, 20), np.uint8), {
            "resolution": [4, 8, 16], "chunk_size": [10, 5, 2], "voxel_offset": [3, 2, 1]}, "scans/pc") + "/s0",
        ("precomputed", ("channel",) + ZYX, (1.0, 16.0, 8.0, 4.0), (0.0, 16.0, 16.0, 12.0), (1, 2, 10, 20), (1, 2, 5, 10)),
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_what_each_layout_reads_as(layout, tmp_path, ome_pyramid, raw_zarr, write_array, request):
    write, expected = LAYOUTS[layout]
    path = write(SimpleNamespace(tmp=tmp_path, ome_pyramid=ome_pyramid, raw_zarr=raw_zarr, write_array=write_array))
    if layout.startswith("http"):
        path = request.getfixturevalue("http_root") + path[len(str(tmp_path)):]
    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            read_array_meta(path)
        return
    meta = read_array_meta(path)
    assert (meta.format, meta.axes, meta.voxel_size, meta.translation, meta.shape, meta.chunk_shape) == expected
    assert meta.spatial().axes == ZYX and meta.spatial().chunk_shape == expected[-1][-3:]


def test_the_full_record_of_a_channel_first_array(ome_pyramid):
    path = ome_pyramid(((8, 4),), shape=(8, 8, 8), channels=2) + "/s0"
    meta = read_array_meta(path)
    assert meta == ArrayMeta(
        path=path, format="zarr2", shape=(2, 8, 8, 8), dtype=np.dtype("u1"), chunk_shape=(1, 4, 4, 4),
        axes=("c",) + ZYX, units=(None,) + ("nanometer",) * 3, voxel_size=(1.0, 8.0, 8.0, 8.0),
        translation=(0.0,) * 4, fill_value=0,
    )
    assert meta.channel_axis == 0 and meta.spatial().channel_axis is None


# --- levels --------------------------------------------------------------------


def _level(path, voxel_size):
    return path, ArrayMeta(path, "zarr2", (4, 4, 4), np.dtype("u1"), (4, 4, 4), ZYX, ("nanometer",) * 3,
                           tuple(float(v) for v in voxel_size), (0.0,) * 3)


LEVELS = [_level("s0", (8, 4, 4)), _level("s1", (16, 8, 8)), _level("s2", (32, 16, 16))]
FLOAT_LEVELS = [_level("s0", (5.24, 4, 4)), _level("s1", (10.48, 8, 8))]


@pytest.mark.parametrize(
    "levels, voxel_size, mode, expected",
    [
        pytest.param(LEVELS, None, "floor", "s0", id="none-is-the-finest"),
        pytest.param(LEVELS, (16, 8, 8), "floor", "s1", id="floor-exact-match"),
        pytest.param(LEVELS, (20, 10, 10), "floor", "s1", id="floor-finest-not-too-coarse"),
        pytest.param(LEVELS, (4, 2, 2), "floor", "s0", id="floor-even-s0-too-coarse"),
        pytest.param(LEVELS, (16, 8, 4), "floor", "s0", id="floor-s1-coarser-in-x"),
        pytest.param(LEVELS, (64, 32, 32), "floor", "s2", id="floor-coarsest"),
        pytest.param(FLOAT_LEVELS, (10.48, 8, 8), "floor", "s1", id="floor-fractional-match"),
        pytest.param(FLOAT_LEVELS, (10, 8, 8), "floor", "s0", id="floor-10-is-not-10.48"),
        pytest.param(LEVELS, (16, 8, 8), "exact", "s1", id="exact"),
        pytest.param(FLOAT_LEVELS, (10, 8, 8), "exact", ValueError, id="exact-none-there"),
        pytest.param(LEVELS, (12, 6, 6), "nearest", "s1", id="nearest-on-a-log-scale"),
    ],
)
def test_select_level(levels, voxel_size, mode, expected):
    if expected is ValueError:
        with pytest.raises(ValueError, match="no level"):
            select_level(levels, voxel_size, mode)
    else:
        assert select_level(levels, voxel_size, mode)[0] == expected


@pytest.fixture(params=[pytest.param(2, id="zarr2"), pytest.param(3, id="zarr3")])
def janelia(ome_pyramid, request):
    """A Janelia pyramid, 8, 16 and 32 nm, each level's corner at -4 nm."""
    return ome_pyramid(((8, 0), (16, 4), (32, 12)), shape=(32, 32, 32), zarr_format=request.param)


def test_a_janelia_pyramids_levels_share_their_corner(janelia):
    assert [(path, meta.voxel_size, meta.translation) for path, meta in list_levels(janelia)] == [
        ("s0", (8.0,) * 3, (-4.0,) * 3), ("s1", (16.0,) * 3, (-4.0,) * 3), ("s2", (32.0,) * 3, (-4.0,) * 3),
    ]


def test_a_precomputed_volumes_levels_are_its_scales(write_array):
    """As s<N>, the path that opens scale N; the path of one scale is not the volume."""
    volume = write_array("precomputed", np.zeros((8,) * 3, np.uint8), {"resolution": [8] * 3}, scales=2)
    assert [(path, meta.path, meta.spatial().voxel_size) for path, meta in list_levels(volume)] == [
        ("s0", volume + "/s0", (8.0,) * 3), ("s1", volume + "/s1", (16.0,) * 3),
    ]
    with pytest.raises(ValueError, match="scale 1"):
        list_levels(volume + "/s1")


def test_a_group_resolves_to_a_level_and_an_array_is_read_as_it_is(janelia):
    assert select_dataset(janelia, (16, 16, 16)) == (os.path.join(janelia, "s1"), "s1")
    assert select_dataset(janelia + "/s0", (16, 16, 16)) == (janelia + "/s0", None)


def test_the_closest_raw_scale_from_the_group_or_one_of_its_levels(janelia, tmp_path):
    assert closest_raw_scale(janelia, (16, 16, 16)) == closest_raw_scale(janelia + "/s0", (20,) * 3) == (16.0,) * 3
    assert closest_raw_scale(str(tmp_path / "missing.zarr"), (8, 8, 8)) is None, "undetermined"


@pytest.mark.parametrize("path", [pytest.param("precomputed:///d/pc", id="local"), pytest.param("gs://b/pc", id="gs")])
def test_a_precomputed_path_is_read_at_the_scale_it_names(path, caplog):
    """Nothing looks for zarr levels there: a gs:// URL had fsspec ask for gcsfs, in a
    warning on every ImageDataInterface opened on it."""
    with caplog.at_level(logging.WARNING):
        assert select_dataset(path, (16, 16, 16)) == (path, None)
        assert closest_raw_scale(path, (16, 16, 16)) is None
    assert caplog.records == []


@pytest.mark.parametrize("zarr_format", [pytest.param(2, id="zarr2"), pytest.param(3, id="zarr3")])
def test_a_missing_level_has_no_closest_scale(ome_pyramid, zarr_format):
    """Not the scale of the group above it, whose zarr.json a v3 lookup found."""
    missing = ome_pyramid(((8, 0), (16, 4)), zarr_format=zarr_format) + "/missing"
    assert closest_raw_scale(missing, (8, 8, 8)) is None


@pytest.mark.parametrize("zarr_format, error", [pytest.param(2, KeyError, id="zarr2"),
                                                pytest.param(3, ValueError, id="zarr3")])
def test_a_group_without_multiscales_has_no_levels(tmp_path, zarr_format, error):
    import zarr

    path = str(tmp_path / "plain.zarr")
    zarr.open_group(path, mode="w") if zarr_format == 2 else _v3_group(path)
    with pytest.raises(error):
        list_levels(path)


# --- what is written ---------------------------------------------------------------


@pytest.mark.parametrize(
    "args, written",
    [
        pytest.param(
            ("s0", (16, 8, 8), [-4, 0, 4], ["nanometer"] * 3, ["z", "y", "x"]),
            '{"multiscales": [{"axes": [{"name": "z", "type": "space", "unit": "nanometer"}, '
            '{"name": "y", "type": "space", "unit": "nanometer"}, {"name": "x", "type": '
            '"space", "unit": "nanometer"}], "coordinateTransformations": [{"scale": [1.0, 1.0, '
            '1.0], "type": "scale"}], "datasets": [{"coordinateTransformations": [{"scale": '
            '[16, 8, 8], "type": "scale"}, {"translation": [4.0, 4.0, 8.0], "type": '
            '"translation"}], "path": "s0"}], "name": "", "version": "0.4"}]}',
            id="zyx",
        ),
        pytest.param(
            ("s0", (1, 5.24, 4.0, 4.0), [0, 0.0, -2, 2], ["", "nanometer", "nanometer", "nanometer"],
             ["c", "z", "y", "x"]),
            '{"multiscales": [{"axes": [{"name": "c", "type": "channel"}, {"name": "z", "type": '
            '"space", "unit": "nanometer"}, {"name": "y", "type": "space", "unit": "nanometer"}, '
            '{"name": "x", "type": "space", "unit": "nanometer"}], "coordinateTransformations": '
            '[{"scale": [1.0, 1.0, 1.0, 1.0], "type": "scale"}], "datasets": '
            '[{"coordinateTransformations": [{"scale": [1, 5.24, 4.0, 4.0], "type": "scale"}, '
            '{"translation": [0.0, 2.62, 0.0, 4.0], "type": "translation"}], "path": "s0"}], '
            '"name": "", "version": "0.4"}]}',
            id="channel-and-fractional-voxels",
        ),
    ],
)
def test_single_scale_attributes_are_written_byte_for_byte(args, written):
    """Blockwise writes its outputs' .zattrs with these; corners in, centres out."""
    assert json.dumps(singlescale_attrs(*args)) == written


def test_written_corners_read_back(tmp_path):
    import zarr

    group = zarr.open_group(str(tmp_path / "w.zarr"), mode="w")
    group.create_dataset("s0", shape=(4, 4, 4), dtype="u1")
    group.create_dataset("s1", shape=(2, 2, 2), dtype="u1")
    group.attrs.update(multiscales_attrs(["z", "y", "x"], ["nanometer"] * 3,
                                         [("s0", [8.0] * 3, [-4.0] * 3), ("s1", [16.0] * 3, [-4.0] * 3)]))
    # Written as voxel 0's centre (4 at 16 nm), read back as the corner.
    assert group.attrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][1]["translation"] == [4.0] * 3
    assert [meta.translation for _, meta in list_levels(str(tmp_path / "w.zarr"))] == [(-4.0,) * 3] * 2


@pytest.mark.parametrize("path, expected", [
    pytest.param("/d/x.zarr/em/s0", ("/d/x.zarr", "em/s0"), id="at-the-suffix"),
    pytest.param("/d/a.n5/b.zarr/raw", ("/d/a.n5/b.zarr", "raw"), id="at-the-innermost-suffix"),
    pytest.param("{tmp}/plain/em/s0", ("{tmp}/plain/em", "s0"), id="without-a-suffix-the-nearest-zgroup"),
    pytest.param("https://host/no/suffix", RuntimeError, id="remote-without-a-suffix"),
])
def test_splitting_a_path_into_its_container_and_dataset(tmp_path, path, expected):
    import zarr

    zarr.open_group(f"{tmp_path}/plain", mode="w").create_group("em").create_dataset("s0", shape=(2,), dtype="u1")
    path = path.format(tmp=tmp_path)
    if expected is RuntimeError:
        with pytest.raises(RuntimeError):
            paths.split_container(path)
    else:
        assert paths.split_container(path) == tuple(p.format(tmp=tmp_path) for p in expected)


@pytest.mark.parametrize("path, fmt", [
    pytest.param("{tmp}/v3.zarr/s0", "zarr3", id="under-a-zarr-json"),
    pytest.param("{tmp}/v2.zarr/s0", "zarr2", id="zarr-suffix"),
    pytest.param("/d/a.n5/raw", "n5", id="n5-suffix"),
    pytest.param("https://h/a.n5", "n5", id="remote-n5"),
    pytest.param("gs://b/pc", "precomputed", id="gs"),
])
def test_a_paths_format(tmp_path, path, fmt):
    _v3_group(tmp_path / "v3.zarr")
    os.makedirs(tmp_path / "v3.zarr" / "s0")
    assert paths.detect_format(path.format(tmp=tmp_path)) == fmt


def test_v3_is_read_from_local_disk_only_and_paths_join_and_unescape(tmp_path):
    _v3_group(tmp_path / "v3.zarr")
    assert paths.is_v3_container(str(tmp_path / "v3.zarr")) and not paths.is_v3_container(str(tmp_path))
    assert paths.find_v3_container("https://host/v3.zarr") is None
    assert paths.join("https://host/x.zarr/", "em", "s0") == "https://host/x.zarr/em/s0"
    # A shell-escaped space is a space, on disk only.
    assert paths.normalize_path("/d/my\\ data.zarr") == "/d/my data.zarr"
    assert paths.normalize_path("https://h/my\\ data.zarr") == "https://h/my\\ data.zarr"
