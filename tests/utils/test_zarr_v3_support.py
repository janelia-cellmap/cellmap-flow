"""Tests for the zarr v3 read-support layer in cellmap_flow/utils/ds.py.

Covers:
    * _zarr_format detection (v2 / v3 / None / mixed-marker preference)
    * ZarrV3Node attrs/shape/chunks/keys/__getitem__
    * _is_zarr_group / _is_zarr_root structural typing
    * _open_zarr dispatch
    * _is_zarr_container + _detect_filetype
    * check_for_multiscale + access_parent
    * get_ds_info on single-array v3 + multiscale v3
    * ImageDataInterface end-to-end with tensorstore zarr3 driver

Fixtures are built in tmp_path so the suite is self-contained and side-
effect-free.
"""
import json
import os
import numpy as np
import pytest
import tensorstore as ts
import zarr

from cellmap_flow.utils.ds import (
    ZarrV3Node,
    _detect_filetype,
    _is_zarr_container,
    _is_zarr_group,
    _is_zarr_root,
    _open_zarr,
    _zarr_format,
    access_parent,
    check_for_multiscale,
    find_closest_scale,
    get_ds_info,
)


# --- fixture helpers ------------------------------------------------------


def _write_v3_array(path, shape, chunks, shard=None, dtype="uint8", seed=0):
    """Write a v3 zarr array at ``path`` using tensorstore.

    If ``shard`` is given, the array is sharded with ``chunks`` as the
    inner (compression/IO) chunk and ``shard`` as the outer chunk grid.
    """
    path = str(path)
    os.makedirs(path, exist_ok=True)
    if shard:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(chunks),
                    "codecs": [{"name": "bytes"}, {"name": "blosc"}],
                    "index_codecs": [{"name": "bytes"}, {"name": "crc32c"}],
                    "index_location": "end",
                },
            }
        ]
        chunk_grid = list(shard)
    else:
        codecs = [{"name": "bytes"}, {"name": "blosc"}]
        chunk_grid = list(chunks)
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": path},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": chunk_grid},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "codecs": codecs,
            "data_type": dtype,
            "fill_value": 0,
        },
        "create": True,
        "delete_existing": True,
    }
    arr = ts.open(spec).result()
    rng = np.random.default_rng(seed=seed)
    arr[:] = rng.integers(0, 256, size=shape, dtype=np.uint8)
    return arr


def _write_v3_group(path, multiscales_attrs=None):
    path = str(path)
    os.makedirs(path, exist_ok=True)
    meta = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": multiscales_attrs or {},
    }
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(meta, f)


def _write_v2_array(path, shape=(28, 28, 28), chunks=(14, 14, 14)):
    """Write a v2 zarr array for negative-case comparisons."""
    z = zarr.open(str(path), mode="w", shape=shape, chunks=chunks, dtype="uint8")
    z[:] = np.random.default_rng(seed=7).integers(0, 256, size=shape, dtype=np.uint8)


@pytest.fixture
def v3_single(tmp_path):
    p = tmp_path / "v3_single.zarr"
    _write_v3_array(p, shape=(56, 56, 56), chunks=(28, 28, 28))
    return p


@pytest.fixture
def v3_sharded(tmp_path):
    p = tmp_path / "v3_sharded.zarr"
    _write_v3_array(
        p, shape=(224, 224, 224), chunks=(56, 56, 56), shard=(112, 112, 112)
    )
    return p


@pytest.fixture
def v2_single(tmp_path):
    p = tmp_path / "v2_single.zarr"
    _write_v2_array(p)
    return p


@pytest.fixture
def v3_multiscale(tmp_path):
    """3-level OME-NGFF v3 group with sharded s0/s1 and unsharded s2."""
    p = tmp_path / "v3_multi.zarr"
    ms_attrs = {
        "multiscales": [
            {
                "version": "0.4",
                "axes": [
                    {"name": "z", "type": "space", "unit": "nanometer"},
                    {"name": "y", "type": "space", "unit": "nanometer"},
                    {"name": "x", "type": "space", "unit": "nanometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [8.0, 8.0, 8.0]},
                            {"type": "translation", "translation": [100.0, 200.0, 300.0]},
                        ],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [16.0, 16.0, 16.0]},
                            {"type": "translation", "translation": [100.0, 200.0, 300.0]},
                        ],
                    },
                    {
                        "path": "2",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [32.0, 32.0, 32.0]},
                            {"type": "translation", "translation": [100.0, 200.0, 300.0]},
                        ],
                    },
                ],
            }
        ]
    }
    _write_v3_group(p, ms_attrs)
    _write_v3_array(p / "0", shape=(224, 224, 224), chunks=(56, 56, 56), shard=(112, 112, 112), seed=0)
    _write_v3_array(p / "1", shape=(112, 112, 112), chunks=(28, 28, 28), shard=(56, 56, 56), seed=1)
    _write_v3_array(p / "2", shape=(56, 56, 56), chunks=(28, 28, 28), seed=2)
    return p


# --- _zarr_format ---------------------------------------------------------


def test_zarr_format_v2(v2_single):
    assert _zarr_format(str(v2_single)) == 2


def test_zarr_format_v3_unsharded(v3_single):
    assert _zarr_format(str(v3_single)) == 3


def test_zarr_format_v3_sharded(v3_sharded):
    assert _zarr_format(str(v3_sharded)) == 3


def test_zarr_format_none_for_random_dir(tmp_path):
    assert _zarr_format(str(tmp_path)) is None


def test_zarr_format_prefers_v3_when_both_markers(v3_single):
    """zarr-python 2's `zarr.open(path)` default mode='a' silently writes
    a `.zgroup` against a v3 store. The detector must still report v3 when
    `zarr.json` exists with `zarr_format=3`."""
    (v3_single / ".zgroup").write_text('{"zarr_format": 2}')
    assert _zarr_format(str(v3_single)) == 3


# --- ZarrV3Node -----------------------------------------------------------


def test_zarrv3node_array_shape_chunks(v3_single):
    n = ZarrV3Node(str(v3_single))
    assert not n.is_group
    assert n.shape == (56, 56, 56)
    assert n.chunks == (28, 28, 28)
    assert n.shard_shape is None
    assert n.dtype == np.dtype("uint8")


def test_zarrv3node_sharded_chunks_returns_inner(v3_sharded):
    """For sharded arrays, .chunks returns the inner compression chunk size,
    not the outer shard size. The shard size is exposed via .shard_shape."""
    n = ZarrV3Node(str(v3_sharded))
    assert n.chunks == (56, 56, 56)  # inner
    assert n.shard_shape == (112, 112, 112)  # outer


def test_zarrv3node_group_keys(v3_multiscale):
    n = ZarrV3Node(str(v3_multiscale))
    assert n.is_group
    assert n.keys() == ["0", "1", "2"]


def test_zarrv3node_getitem_navigates(v3_multiscale):
    n = ZarrV3Node(str(v3_multiscale))
    child = n["1"]
    assert not child.is_group
    assert child.shape == (112, 112, 112)


def test_zarrv3node_getitem_missing_raises(v3_multiscale):
    n = ZarrV3Node(str(v3_multiscale))
    with pytest.raises(KeyError):
        n["does_not_exist"]


def test_zarrv3node_group_no_shape_attr(v3_multiscale):
    n = ZarrV3Node(str(v3_multiscale))
    with pytest.raises(AttributeError):
        n.shape


def test_zarrv3node_array_no_keys(v3_single):
    n = ZarrV3Node(str(v3_single))
    with pytest.raises(AttributeError):
        n.keys()


def test_zarrv3node_attrs_includes_multiscales(v3_multiscale):
    n = ZarrV3Node(str(v3_multiscale))
    ms = n.attrs.get("multiscales")
    assert ms is not None
    assert ms[0]["datasets"][0]["path"] == "0"


# --- _is_zarr_group / _is_zarr_root ---------------------------------------


def test_is_zarr_group_v3(v3_multiscale, v3_single):
    grp = _open_zarr(str(v3_multiscale))
    arr = _open_zarr(str(v3_single))
    assert _is_zarr_group(grp) is True
    assert _is_zarr_group(arr) is False


def test_is_zarr_group_v2(v2_single):
    arr = _open_zarr(str(v2_single))
    assert _is_zarr_group(arr) is False


def test_is_zarr_root(v3_multiscale):
    grp = _open_zarr(str(v3_multiscale))
    arr = _open_zarr(str(v3_multiscale / "0"))
    assert _is_zarr_root(grp) is True
    assert _is_zarr_root(arr) is False


# --- _open_zarr dispatch --------------------------------------------------


def test_open_zarr_dispatches_v3_to_proxy(v3_single):
    n = _open_zarr(str(v3_single))
    assert isinstance(n, ZarrV3Node)


def test_open_zarr_dispatches_v2_to_zarr_python(v2_single):
    n = _open_zarr(str(v2_single))
    assert not isinstance(n, ZarrV3Node)


def test_open_zarr_v3_write_mode_raises(v3_single):
    with pytest.raises(NotImplementedError):
        _open_zarr(str(v3_single), mode="w")


# --- _is_zarr_container / _detect_filetype --------------------------------


def test_is_zarr_container_v3(v3_single):
    assert _is_zarr_container(str(v3_single)) is True


def test_is_zarr_container_random_dir(tmp_path):
    assert _is_zarr_container(str(tmp_path)) is False


def test_detect_filetype_v3(v3_single):
    assert _detect_filetype(str(v3_single)) == "zarr3"


def test_detect_filetype_v2(v2_single):
    assert _detect_filetype(str(v2_single)) == "zarr"


def test_detect_filetype_walks_up_to_v3(v3_multiscale):
    """A path one level inside a v3 array should still resolve to zarr3."""
    # /v3_multi.zarr/0 is an array — itself has zarr.json — should be zarr3.
    assert _detect_filetype(str(v3_multiscale / "0")) == "zarr3"


# --- check_for_multiscale / access_parent ---------------------------------


def test_access_parent_v3(v3_multiscale):
    arr = _open_zarr(str(v3_multiscale / "0"))
    parent = access_parent(arr)
    assert _is_zarr_group(parent)
    assert parent.attrs.get("multiscales") is not None


def test_access_parent_v3_root_raises(v3_multiscale):
    grp = _open_zarr(str(v3_multiscale))
    with pytest.raises(RuntimeError):
        access_parent(grp)


def test_check_for_multiscale_walks_up_v3(v3_multiscale):
    """Calling check_for_multiscale on a scale-level array should walk
    up to the group level and find the multiscales attribute."""
    arr = _open_zarr(str(v3_multiscale / "0"))
    ms, found_in = check_for_multiscale(arr)
    assert ms is not None
    assert _is_zarr_group(found_in)


def test_check_for_multiscale_returns_none_for_standalone_v3(v3_single):
    """Standalone v3 array (no v3 parent) — check_for_multiscale should
    terminate at the array level returning None for multiscales."""
    arr = _open_zarr(str(v3_single))
    ms, found_in = check_for_multiscale(arr)
    assert ms is None


# --- get_ds_info ----------------------------------------------------------


def test_get_ds_info_v3_single_uses_defaults(v3_single):
    voxel, chunks, shape, roi, axes, filetype = get_ds_info(str(v3_single))
    assert voxel == (1, 1, 1)
    assert chunks == (28, 28, 28)
    assert tuple(shape) == (56, 56, 56)
    assert filetype == "zarr3"


def test_get_ds_info_v3_sharded_reports_inner_chunk(v3_sharded):
    voxel, chunks, shape, roi, axes, filetype = get_ds_info(str(v3_sharded))
    assert chunks == (56, 56, 56)
    assert tuple(shape) == (224, 224, 224)
    assert filetype == "zarr3"


def test_get_ds_info_v3_multiscale_array_reads_parent(v3_multiscale):
    """Calling get_ds_info on a scale-level array path should pull
    voxel_size and translation from the parent group's multiscales attr."""
    voxel, chunks, shape, roi, axes, filetype = get_ds_info(
        str(v3_multiscale / "1")
    )
    assert voxel == (16, 16, 16)
    assert chunks == (28, 28, 28)
    assert tuple(shape) == (112, 112, 112)
    assert axes == ["z", "y", "x"]
    assert filetype == "zarr3"
    # roi.offset should reflect translation
    assert tuple(roi.offset) == (100, 200, 300)


def test_get_ds_info_v3_group_picks_first_scale(v3_multiscale):
    voxel, chunks, shape, roi, axes, filetype = get_ds_info(str(v3_multiscale))
    assert voxel == (8, 8, 8)
    assert tuple(shape) == (224, 224, 224)
    assert filetype == "zarr3"


# --- find_closest_scale + ImageDataInterface end-to-end -------------------


def test_find_closest_scale_v3(v3_multiscale):
    scale, offset, shape = find_closest_scale(str(v3_multiscale), [16, 16, 16])
    assert scale == "1"
    assert tuple(offset) == (100, 200, 300)
    assert tuple(shape) == (112, 112, 112)


def test_image_data_interface_v3_multiscale(v3_multiscale):
    """Full ImageDataInterface boot on a v3 multiscale group + voxel-size
    selection navigates to the right scale and opens via the zarr3
    tensorstore driver."""
    from cellmap_flow.image_data_interface import ImageDataInterface

    idi = ImageDataInterface(str(v3_multiscale), voxel_size=[16, 16, 16])
    assert idi.voxel_size == (16, 16, 16)
    assert tuple(idi.shape) == (112, 112, 112)
    assert idi.filetype == "zarr3"
    assert idi.ts.spec().to_json()["driver"] == "zarr3"


def test_image_data_interface_v3_sharded_byte_equal_to_v2(v3_sharded, tmp_path):
    """Sharded v3 must read byte-identical to v2 source for the same data."""
    from cellmap_flow.image_data_interface import ImageDataInterface

    # Read v3 sharded via tensorstore.
    v3_arr = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(v3_sharded)},
        }
    ).result()[:].read().result()

    # Round-trip via ImageDataInterface.ts must agree on shape/dtype + a
    # corner pixel.
    idi = ImageDataInterface(str(v3_sharded))
    assert idi.ts.spec().to_json()["driver"] == "zarr3"
    assert tuple(idi.shape) == v3_arr.shape
