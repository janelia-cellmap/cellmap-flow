"""The io/ package: paths, metadata, multiscale levels and OME attributes.

What the old readers returned on real layouts is pinned in test_io_matrix;
this file covers the new interfaces themselves.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import tensorstore as ts
import zarr

from cellmap_flow.io import metadata, paths
from cellmap_flow.io.metadata import ArrayMeta, list_levels, read_array_meta

# ---------------------------------------------------------------------------
# io/ stays importable without the application around it
# ---------------------------------------------------------------------------

_HEAVY = ("cellmap_flow.globals", "flask", "neuroglancer", "torch", "huggingface_hub", "peft")


@pytest.mark.parametrize(
    "module",
    ["cellmap_flow.io", "cellmap_flow.io.paths", "cellmap_flow.io.metadata", "cellmap_flow.io.ome"],
)
def test_io_modules_import_nothing_heavy(module, tmp_path):
    # globals configures logging and reads ~/.cellmap_flow on import; the
    # others are slow or optional. A fresh interpreter, so that what this
    # test process has already imported does not hide anything.
    code = (
        f"import sys, {module}; "
        f"loaded = [m for m in {_HEAVY!r} if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    env = {**os.environ, "HOME": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=os.getcwd()
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_split_container_at_the_last_suffix():
    assert paths.split_container("/d/x.zarr/em/s0") == ("/d/x.zarr", "em/s0")
    assert paths.split_container("/d/x.zarr") == ("/d/x.zarr", "")
    assert paths.split_container("/d/a.n5/b.zarr/raw") == ("/d/a.n5/b.zarr", "raw")
    assert paths.split_container("/d/a.zarr/b.n5/raw") == ("/d/a.zarr/b.n5", "raw")
    assert paths.split_container("s3://bucket/x.zarr/raw") == ("s3://bucket/x.zarr", "raw")


def test_split_container_without_a_suffix_finds_the_group(tmp_path):
    root = zarr.open_group(str(tmp_path / "plain"), mode="w")
    root.create_group("em").create_dataset("s0", shape=(2, 2, 2), dtype="u1")
    # The nearest .zgroup going up is em's own.
    assert paths.split_container(str(tmp_path / "plain" / "em" / "s0")) == (
        str(tmp_path / "plain" / "em"),
        "s0",
    )
    with pytest.raises(RuntimeError):
        paths.split_container(str(tmp_path / "nothing" / "here"))
    with pytest.raises(RuntimeError):
        paths.split_container("https://host/no/suffix")


def test_split_dataset_path_still_appends_a_scale(tmp_path):
    from cellmap_flow.utils.ds import split_dataset_path

    assert split_dataset_path("/d/x.zarr/em", scale=1) == ("/d/x.zarr", "em/s1")
    assert split_dataset_path("/d/x.zarr", scale=0) == ("/d/x.zarr", "/s0")
    zarr.open_group(str(tmp_path / "plain"), mode="w")
    assert split_dataset_path(str(tmp_path / "plain"), scale=2) == (str(tmp_path / "plain"), "s2")


def test_join_and_normalize():
    assert paths.join("https://host/x.zarr/", "em", "s0") == "https://host/x.zarr/em/s0"
    assert paths.join("/d/x.zarr", "em", "s0") == os.path.join("/d/x.zarr", "em", "s0")
    assert paths.normalize_path("/d/my\\ data.zarr") == "/d/my data.zarr"
    # Shell escapes are a filesystem thing; a URL is left alone.
    assert paths.normalize_path("https://host/my\\ data.zarr") == "https://host/my\\ data.zarr"
    assert paths.is_remote("s3://b/x") and paths.is_remote("http://h/x")
    assert not paths.is_remote("gs://b/x") and not paths.is_remote("/d/x")


def _v3_node(path, node_type):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": node_type}, f)


def test_detect_format(tmp_path):
    _v3_node(str(tmp_path / "v3.zarr"), "group")
    _v3_node(str(tmp_path / "v3.zarr" / "s0"), "array")
    assert paths.detect_format(str(tmp_path / "v3.zarr" / "s0")) == "zarr3"
    assert paths.detect_format(str(tmp_path / "v3.zarr")) == "zarr3"
    assert paths.detect_format(str(tmp_path / "v2.zarr" / "s0")) == "zarr2"
    assert paths.detect_format(str(tmp_path / "a.n5" / "raw")) == "n5"
    assert paths.detect_format("precomputed:///d/pc") == "precomputed"
    assert paths.detect_format("gs://bucket/pc") == "precomputed"
    assert paths.detect_format("https://host/a.n5/raw") == "n5"
    assert paths.detect_format("https://host/x.zarr/raw") == "zarr2"
    # zarr v3 is only read from the local filesystem.
    assert paths.find_v3_container("https://host/v3.zarr") is None
    assert paths.find_v3_container(str(tmp_path / "v3.zarr" / "s0" / "c")) == str(
        tmp_path / "v3.zarr" / "s0"
    )


def test_zarr_container_markers(tmp_path):
    zarr.open_group(str(tmp_path / "g"), mode="w")
    _v3_node(str(tmp_path / "v3"), "group")
    (tmp_path / "empty").mkdir()
    assert paths.is_zarr_container(str(tmp_path / "g"))
    assert paths.is_zarr_container(str(tmp_path / "v3"))
    assert not paths.is_zarr_container(str(tmp_path / "empty"))
    assert not paths.is_zarr_container("https://host/x.zarr")


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def _ome_axes(channel=False, unit="nanometer"):
    axes = [{"name": n, "type": "space", "unit": unit} for n in "zyx"]
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


JANELIA = [("s0", (8,) * 3, (0,) * 3), ("s1", (16,) * 3, (4,) * 3), ("s2", (32,) * 3, (12,) * 3)]


@pytest.fixture
def czyx(tmp_path):
    group = zarr.open_group(str(tmp_path / "c.zarr"), mode="w")
    group.create_dataset("s0", shape=(2, 8, 8, 8), chunks=(1, 4, 4, 4), dtype="u2", fill_value=3)
    group.create_dataset("s1", shape=(2, 4, 4, 4), chunks=(1, 2, 2, 2), dtype="u2")
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": _ome_axes(channel=True),
            "datasets": _datasets(
                [("s0", (1, 8, 8, 8), (0, 4, 4, 4)), ("s1", (1, 16, 16, 16), (0, 8, 8, 8))]
            ),
        }
    ]
    return str(tmp_path / "c.zarr")


def test_every_axis_is_kept(czyx):
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
    spatial = meta.spatial()
    assert spatial.axes == ("z", "y", "x") and spatial.shape == (8, 8, 8)
    assert spatial.chunk_shape == (4, 4, 4) and spatial.channel_axis is None


def test_levels_of_a_group(czyx, tmp_path):
    levels = list_levels(czyx)
    assert [path for path, _ in levels] == ["s0", "s1"]
    assert levels[1][1].spatial().voxel_size == (16.0, 16.0, 16.0)
    assert levels[1][1].path == os.path.join(czyx, "s1")

    zarr.open_group(str(tmp_path / "plain.zarr"), mode="w")
    with pytest.raises(KeyError):
        list_levels(str(tmp_path / "plain.zarr"))
    _v3_node(str(tmp_path / "plain_v3.zarr"), "group")
    with pytest.raises(ValueError):
        list_levels(str(tmp_path / "plain_v3.zarr"))


def _v3_group(path, levels, unit="nanometer"):
    for name, _, _ in levels:
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": os.path.join(path, name)},
                "metadata": {
                    "shape": [4, 4, 4],
                    "data_type": "uint8",
                    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2, 2]}},
                },
            },
            create=True,
        ).result()
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {
                        "multiscales": [
                            {"version": "0.5", "axes": _ome_axes(unit=unit), "datasets": _datasets(levels)}
                        ]
                    }
                },
            },
            f,
        )
    return path


@pytest.mark.parametrize("fmt", ["zarr2", "zarr3"])
def test_the_ome_centre_becomes_an_exact_corner(tmp_path, fmt):
    if fmt == "zarr3":
        group = _v3_group(str(tmp_path / "j.zarr"), JANELIA)
    else:
        group = str(tmp_path / "j.zarr")
        root = zarr.open_group(group, mode="w")
        for name, _, _ in JANELIA:
            root.create_dataset(name, shape=(4, 4, 4), dtype="u1")
        root.attrs["multiscales"] = [
            {"version": "0.4", "axes": _ome_axes(), "datasets": _datasets(JANELIA)}
        ]
    for name, meta in list_levels(group):
        # -4 nm at 8, 16 and 32 nm voxels: not on any level's grid, and not
        # rounded onto it.
        assert meta.translation == (-4.0, -4.0, -4.0), name
        assert read_array_meta(os.path.join(group, name)).translation == (-4.0,) * 3
        assert meta.format == fmt


def test_units_are_converted_before_the_centre_is(tmp_path):
    group = _v3_group(
        str(tmp_path / "um.zarr"),
        [("s0", (0.008, 0.004, 0.004), (0.08, 0.04, 0.04))],
        unit="micrometer",
    )
    meta = read_array_meta(group + "/s0")
    assert meta.voxel_size == (8.0, 4.0, 4.0)
    assert meta.translation == (76.0, 38.0, 38.0)
    assert meta.units == ("nanometer",) * 3


def test_legacy_offsets_are_rounded_onto_the_grid_but_v3_array_attrs_are_not(tmp_path):
    root = zarr.open_group(str(tmp_path / "l.zarr"), mode="w")
    arr = root.create_dataset("raw", shape=(4, 4, 4), dtype="u1")
    arr.attrs.update({"resolution": [8, 8, 8], "offset": [4, 4, 4]})
    assert read_array_meta(str(tmp_path / "l.zarr" / "raw")).translation == (8.0, 8.0, 8.0)

    ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(tmp_path / "plain")},
            "metadata": {
                "shape": [4, 4, 4],
                "data_type": "uint8",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4, 4, 4]}},
                "attributes": {"resolution": [8, 8, 8], "offset": [4, 4, 4]},
            },
        },
        create=True,
    ).result()
    meta = read_array_meta(str(tmp_path / "plain"))
    assert meta.format == "zarr3" and meta.translation == (4.0, 4.0, 4.0)


def test_a_legacy_array_with_a_leading_channel_axis(tmp_path):
    root = zarr.open_group(str(tmp_path / "l.zarr"), mode="w")
    arr = root.create_dataset("raw", shape=(3, 4, 4, 4), dtype="u1")
    arr.attrs.update({"resolution": [8, 4, 4], "offset": [0, 0, 0]})
    meta = read_array_meta(str(tmp_path / "l.zarr" / "raw"))
    assert meta.axes == ("c^", "z", "y", "x") and meta.channel_axis == 0
    assert meta.voxel_size == (1.0, 8.0, 4.0, 4.0)
    assert meta.spatial().shape == (4, 4, 4)


def test_n5_transform_in_c_order(tmp_path):
    from zarr.n5 import N5FSStore

    root = zarr.open(N5FSStore(str(tmp_path / "a.n5")), mode="w")
    arr = root.create_dataset("raw", shape=(10, 20, 30), dtype="u1")
    arr.attrs["transform"] = {
        "axes": ["z", "y", "x"],
        "ordering": "C",
        "scale": [8, 4, 2],
        "translate": [80, 40, 20],
        "units": ["nm", "nm", "nm"],
    }
    meta = read_array_meta(str(tmp_path / "a.n5" / "raw"))
    assert meta.format == "n5"
    assert meta.shape == (10, 20, 30)
    assert meta.voxel_size == (8.0, 4.0, 2.0) and meta.translation == (80.0, 40.0, 20.0)


def test_precomputed_is_read_in_c_order(tmp_path):
    path = str(tmp_path / "pc")
    ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
            "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
            "scale_metadata": {
                "size": [20, 10, 2],
                "resolution": [4, 8, 16],
                "encoding": "raw",
                "chunk_size": [10, 5, 2],
            },
        },
        create=True,
    ).result()
    meta = read_array_meta("precomputed://" + path)
    assert meta.format == "precomputed"
    assert meta.axes == ("channel", "z", "y", "x")
    assert meta.shape == (1, 2, 10, 20)
    assert meta.voxel_size == (1.0, 16.0, 8.0, 4.0)
    assert meta.spatial().chunk_shape == (2, 5, 10)


def test_float_voxel_sizes_are_kept(tmp_path):
    root = zarr.open_group(str(tmp_path / "f.zarr"), mode="w")
    root.create_dataset("s0", shape=(4, 4, 4), dtype="u1")
    root.attrs["multiscales"] = [
        {"version": "0.4", "axes": _ome_axes(), "datasets": _datasets([("s0", (5.24, 4, 4), (2.62, 2, 2))])}
    ]
    meta = read_array_meta(str(tmp_path / "f.zarr" / "s0"))
    assert meta.voxel_size == (5.24, 4.0, 4.0) and meta.translation == (0.0, 0.0, 0.0)


def test_to_nm_and_unknown_units(caplog):
    assert metadata.to_nm([0.008, 4], ["micrometer", "nm"]) == (8.0, 4.0)
    assert metadata.to_nm([8], None) == (8.0,)
    with caplog.at_level("WARNING"):
        assert metadata.nm_per_unit("furlong-for-this-test") == 1.0
        assert metadata.nm_per_unit("furlong-for-this-test") == 1.0
    assert sum("furlong" in r.getMessage() for r in caplog.records) == 1


def test_the_zarr_object_attribute_lookups_are_kept(tmp_path):
    from cellmap_flow.utils import ds

    root = zarr.open_group(str(tmp_path / "l.zarr"), mode="w")
    group = root.create_group("g")
    group.attrs["units"] = ["nm", "nm", "nm"]
    arr = group.create_dataset("raw", shape=(4, 4, 4), dtype="u1")
    arr.attrs["transform"] = {"ordering": "C", "scale": [8, 4, 2], "translate": [1, 2, 3], "units": ["nm"] * 3}
    assert ds.check_for_voxel_size(arr, "F") == [2, 4, 8]
    assert ds.check_for_offset(arr, "C") == [1, 2, 3]
    # The array's own transform comes before its parent's units.
    assert ds.check_for_units(arr, "C") == ["nm"] * 3
    assert ds.check_for_units(root.create_dataset("bare", shape=(2,), dtype="u1"), "C") == "pixels"
    multiscales, found_in = ds.check_for_multiscale(group)
    assert multiscales is None and found_in.path == ""
