"""The io/ package through its public interface.

What the old readers returned on real layouts is pinned in test_io_matrix;
this file covers what is new: ArrayMeta with every axis, list_levels,
select_level's modes, the path helpers and the OME writers.
"""

import json
import os

import numpy as np
import pytest
import tensorstore as ts
import zarr

from cellmap_flow.io import paths
from cellmap_flow.io.metadata import ArrayMeta, list_levels, read_array_meta
from cellmap_flow.io.multiscale import closest_raw_scale, select_dataset, select_level
from cellmap_flow.io.ome import multiscales_attrs, singlescale_attrs


def _v3_node(path, node_type="group", attributes=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": node_type, "attributes": attributes or {}}, f)


def test_paths(tmp_path):
    from cellmap_flow.utils.ds import split_dataset_path

    t = str(tmp_path)
    _v3_node(f"{t}/v3.zarr")
    _v3_node(f"{t}/v3.zarr/s0", "array")
    zarr.open_group(f"{t}/plain", mode="w").create_group("em").create_dataset("s0", shape=(2,), dtype="u1")
    assert paths.split_container("/d/x.zarr/em/s0") == ("/d/x.zarr", "em/s0")
    assert paths.split_container("/d/a.n5/b.zarr/raw") == ("/d/a.n5/b.zarr", "raw")
    # Without a suffix: the nearest .zgroup going up (em's own).
    assert paths.split_container(f"{t}/plain/em/s0") == (f"{t}/plain/em", "s0")
    with pytest.raises(RuntimeError):
        paths.split_container("https://host/no/suffix")
    assert split_dataset_path("/d/x.zarr/em", scale=1) == ("/d/x.zarr", "em/s1")
    assert [
        paths.detect_format(p)
        for p in (f"{t}/v3.zarr/s0", f"{t}/v2.zarr/s0", "/d/a.n5/raw", "gs://b/pc", "https://h/a.n5")
    ] == ["zarr3", "zarr2", "n5", "precomputed", "n5"]
    # zarr v3 is only read from the local filesystem.
    assert paths.find_v3_container("https://host/v3.zarr") is None
    assert paths.join("https://host/x.zarr/", "em", "s0") == "https://host/x.zarr/em/s0"
    assert paths.normalize_path("/d/my\\ data.zarr") == "/d/my data.zarr"
    assert paths.normalize_path("https://h/my\\ data.zarr") == "https://h/my\\ data.zarr"


def _ome_axes(channel=False):
    axes = [{"name": n, "type": "space", "unit": "nanometer"} for n in "zyx"]
    return ([{"name": "c", "type": "channel"}] if channel else []) + axes


def _datasets(levels):
    return [
        {
            "path": path,
            "coordinateTransformations": [
                {"type": "scale", "scale": list(scale)},
                {"type": "translation", "translation": list(translation)},
            ],
        }
        for path, scale, translation in levels
    ]


@pytest.fixture
def czyx(tmp_path):
    group = zarr.open_group(str(tmp_path / "c.zarr"), mode="w")
    group.create_dataset("s0", shape=(2, 8, 8, 8), chunks=(1, 4, 4, 4), dtype="u2", fill_value=3)
    group.create_dataset("s1", shape=(2, 4, 4, 4), chunks=(1, 2, 2, 2), dtype="u2")
    levels = [("s0", (1, 8, 8, 8), (0, 4, 4, 4)), ("s1", (1, 16, 16, 16), (0, 8, 8, 8))]
    group.attrs["multiscales"] = [
        {"version": "0.4", "axes": _ome_axes(channel=True), "datasets": _datasets(levels)}
    ]
    return str(tmp_path / "c.zarr")


def test_array_meta_keeps_every_axis(czyx):
    meta = read_array_meta(czyx + "/s0")
    assert meta == ArrayMeta(
        path=czyx + "/s0",
        format="zarr2",
        shape=(2, 8, 8, 8),
        dtype=np.dtype("u2"),
        chunk_shape=(1, 4, 4, 4),
        axes=("c", "z", "y", "x"),
        units=(None, "nanometer", "nanometer", "nanometer"),
        voxel_size=(1.0, 8.0, 8.0, 8.0),
        # Translation (4, 4, 4) is voxel 0's centre: the corner is at 0.
        translation=(0.0, 0.0, 0.0, 0.0),
        fill_value=3,
    )
    assert meta.channel_axis == 0
    assert meta.spatial().axes == ("z", "y", "x") and meta.spatial().channel_axis is None
    assert meta.spatial().chunk_shape == (4, 4, 4)


def _precomputed(path):
    ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
            "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
            "scale_metadata": {"size": [20, 10, 2], "resolution": [4, 8, 16], "encoding": "raw"},
        },
        create=True,
    ).result()
    return "precomputed://" + path


def _legacy_channels(path):
    arr = zarr.open_group(path, mode="w").create_dataset("raw", shape=(3, 4, 4, 4), dtype="u1")
    arr.attrs.update({"resolution": [8, 4, 4], "offset": [0, 0, 0]})
    return path + "/raw"


def _janelia_v3(path):
    multiscale = {"version": "0.5", "axes": _ome_axes(), "datasets": _datasets([("s1", (16,) * 3, (4,) * 3)])}
    _v3_node(path, attributes={"ome": {"multiscales": [multiscale]}})
    ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": path + "/s1"},
            "metadata": {
                "shape": [4, 4, 4],
                "data_type": "uint8",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2, 2]}},
            },
        },
        create=True,
    ).result()
    return path + "/s1"


@pytest.mark.parametrize(
    "make, fmt, axes, voxel_size, translation",
    [
        (_precomputed, "precomputed", ("channel", "z", "y", "x"), (1.0, 16.0, 8.0, 4.0), (0.0,) * 4),
        # Legacy attributes covering the last axes: the others are channels.
        (_legacy_channels, "zarr2", ("c^", "z", "y", "x"), (1.0, 8.0, 4.0, 4.0), (0.0,) * 4),
        # -4 nm at 16 nm: exact, not rounded onto the grid.
        (_janelia_v3, "zarr3", ("z", "y", "x"), (16.0,) * 3, (-4.0,) * 3),
    ],
)
def test_read_array_meta(tmp_path, make, fmt, axes, voxel_size, translation):
    meta = read_array_meta(make(str(tmp_path / "data")))
    assert (meta.format, meta.axes, meta.voxel_size, meta.translation) == (
        fmt,
        axes,
        voxel_size,
        translation,
    )
    assert meta.spatial().axes == ("z", "y", "x")


def test_list_levels(czyx, tmp_path):
    levels = list_levels(czyx)
    assert [(path, meta.path, meta.spatial().voxel_size) for path, meta in levels] == [
        ("s0", os.path.join(czyx, "s0"), (8.0, 8.0, 8.0)),
        ("s1", os.path.join(czyx, "s1"), (16.0, 16.0, 16.0)),
    ]
    zarr.open_group(str(tmp_path / "plain.zarr"), mode="w")
    with pytest.raises(KeyError):
        list_levels(str(tmp_path / "plain.zarr"))
    _v3_node(str(tmp_path / "plain_v3.zarr"))
    with pytest.raises(ValueError):
        list_levels(str(tmp_path / "plain_v3.zarr"))


def _level(path, voxel_size):
    return path, ArrayMeta(
        path=path,
        format="zarr2",
        shape=(4, 4, 4),
        dtype=np.dtype("u1"),
        chunk_shape=(4, 4, 4),
        axes=("z", "y", "x"),
        units=("nanometer",) * 3,
        voxel_size=tuple(float(v) for v in voxel_size),
        translation=(0.0,) * 3,
    )


LEVELS = [_level("s0", (8, 4, 4)), _level("s1", (16, 8, 8)), _level("s2", (32, 16, 16))]


@pytest.mark.parametrize(
    "voxel_size, mode, expected",
    [
        (None, "floor", "s0"),
        ((16, 8, 8), "floor", "s1"),
        ((20, 10, 10), "floor", "s1"),  # the finest level that is not too coarse
        ((4, 2, 2), "floor", "s0"),  # even s0 is too coarse
        ((16, 8, 4), "floor", "s0"),  # s1 is coarser in x
        ((64, 32, 32), "floor", "s2"),
        ((16, 8, 8), "exact", "s1"),
        ((10.48, 8, 8), "exact", ValueError),
        ((12, 6, 6), "nearest", "s1"),  # nearer 16 than 8 on a log scale
    ],
)
def test_select_level(voxel_size, mode, expected):
    if expected is ValueError:
        with pytest.raises(ValueError, match="no level"):
            select_level(LEVELS, voxel_size, mode)
    else:
        assert select_level(LEVELS, voxel_size, mode)[0] == expected


def test_select_dataset_and_closest_raw_scale(czyx):
    assert select_dataset(czyx, (16, 16, 16)) == (os.path.join(czyx, "s1"), "s1")
    # An array is read as it is.
    assert select_dataset(czyx + "/s0", (16, 16, 16)) == (czyx + "/s0", None)
    # From the group or one of its levels alike.
    assert closest_raw_scale(czyx + "/s0", (16, 16, 16)) == (16.0, 16.0, 16.0)
    assert closest_raw_scale(czyx + "/missing", (8, 8, 8)) is None


@pytest.mark.parametrize(
    "args, written",
    [
        # As ds.generate_singlescale_metadata wrote them before it moved.
        (
            ("s0", "Coordinate(16, 8, 8)", [-4, 0, 4], ["nanometer"] * 3, ["z", "y", "x"]),
            '{"multiscales": [{"axes": [{"name": "z", "type": "space", "unit": "nanometer"}, '
            '{"name": "y", "type": "space", "unit": "nanometer"}, {"name": "x", "type": '
            '"space", "unit": "nanometer"}], "coordinateTransformations": [{"scale": [1.0, 1.0, '
            '1.0], "type": "scale"}], "datasets": [{"coordinateTransformations": [{"scale": '
            '[16, 8, 8], "type": "scale"}, {"translation": [4.0, 4.0, 8.0], "type": '
            '"translation"}], "path": "s0"}], "name": "", "version": "0.4"}]}',
        ),
        (
            (
                "s0",
                (1, 5.24, 4.0, 4.0),
                [0, 0.0, -2, 2],
                ["", "nanometer", "nanometer", "nanometer"],
                ["c", "z", "y", "x"],
            ),
            '{"multiscales": [{"axes": [{"name": "c", "type": "channel"}, {"name": "z", "type": '
            '"space", "unit": "nanometer"}, {"name": "y", "type": "space", "unit": "nanometer"}, '
            '{"name": "x", "type": "space", "unit": "nanometer"}], "coordinateTransformations": '
            '[{"scale": [1.0, 1.0, 1.0, 1.0], "type": "scale"}], "datasets": '
            '[{"coordinateTransformations": [{"scale": [1, 5.24, 4.0, 4.0], "type": "scale"}, '
            '{"translation": [0.0, 2.62, 0.0, 4.0], "type": "translation"}], "path": "s0"}], '
            '"name": "", "version": "0.4"}]}',
        ),
    ],
)
def test_singlescale_attrs_are_written_byte_for_byte_as_before(args, written):
    from funlib.geometry import Coordinate

    from cellmap_flow.utils.ds import generate_singlescale_metadata

    args = tuple(Coordinate(16, 8, 8) if a == "Coordinate(16, 8, 8)" else a for a in args)
    assert json.dumps(generate_singlescale_metadata(*args)) == written
    assert json.dumps(singlescale_attrs(*args)) == written


def test_written_corners_read_back(tmp_path):
    group = zarr.open_group(str(tmp_path / "w.zarr"), mode="w")
    group.create_dataset("s0", shape=(4, 4, 4), dtype="u1")
    group.create_dataset("s1", shape=(2, 2, 2), dtype="u1")
    levels = [("s0", [8.0] * 3, [-4.0] * 3), ("s1", [16.0] * 3, [-4.0] * 3)]
    group.attrs.update(multiscales_attrs(["z", "y", "x"], ["nanometer"] * 3, levels, name="raw"))
    # Written as voxel 0's centre (4 at 16 nm), read back as the corner.
    assert group.attrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][1] == {
        "translation": [4.0, 4.0, 4.0],
        "type": "translation",
    }
    assert [meta.translation for _, meta in list_levels(str(tmp_path / "w.zarr"))] == [(-4.0,) * 3] * 2
