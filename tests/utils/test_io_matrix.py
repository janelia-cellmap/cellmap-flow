"""What every dataset reader reports, recorded as literals.

The metadata readers (zarr v2 local and over http, zarr v3, N5,
neuroglancer precomputed), scale selection, ImageDataInterface, the crop
loader and the raw display layer each had their own copy of the parsing.
This matrix pins what they return on one set of fixtures so that
consolidating them changes nothing that is not written down here.

Each case is computed by one function and rendered by ``describe``, which
keeps the distinctions that matter downstream: list vs tuple, Coordinate vs
floats, int vs float, the numpy dtype. Where a value looks wrong it is still
recorded as it is; the comment next to it says why it is that way.
"""

import functools
import http.server
import json
import os
import threading

import numpy as np
import pytest
import tensorstore as ts
import zarr
from funlib.geometry import Coordinate, Roi
from zarr.n5 import N5FSStore


def describe(value):
    """A literal for ``value`` that is exact about types."""
    if isinstance(value, Roi):
        return f"Roi({describe(value.offset)}, {describe(value.shape)})"
    if isinstance(value, Coordinate):
        return "Coordinate(" + ", ".join(describe(v) for v in value) + ")"
    if isinstance(value, np.ndarray):
        return f"ndarray[{value.dtype}]({describe(value.tolist())})"
    if isinstance(value, np.dtype):
        return f"dtype({str(value)!r})"
    if isinstance(value, np.generic):
        return f"{type(value).__name__}({value.item()!r})"
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, tuple):
        inner = ", ".join(describe(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(describe(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{describe(k)}: {describe(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"no literal for {type(value)}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _space(unit="nanometer", names="zyx"):
    return [{"name": n, "type": "space", "unit": unit} for n in names]


def _ome(axes, levels, version="0.4"):
    """``levels``: (path, scale, translation or None)."""
    datasets = []
    for path, scale, translation in levels:
        transforms = [{"type": "scale", "scale": list(scale)}]
        if translation is not None:
            transforms.append({"type": "translation", "translation": list(translation)})
        datasets.append({"path": path, "coordinateTransformations": transforms})
    return [{"version": version, "axes": axes, "datasets": datasets}]


def _z_index(shape, dtype=np.uint16):
    """Each voxel holds its own z index (the first spatial axis)."""
    data = np.zeros(shape, dtype=dtype)
    z = shape[-3]
    data += np.arange(z, dtype=dtype).reshape((z, 1, 1))
    return data


def _v2_group(path, axes, levels, shapes, chunks=None):
    # Groups on the way down from the container are real groups, as a
    # writer makes them.
    container, _, inner = path.partition(".zarr")
    group = zarr.open_group(container + ".zarr", mode="a")
    if inner.strip("/"):
        group = group.require_group(inner.strip("/"))
    for i, (name, _, _) in enumerate(levels):
        group.create_dataset(
            name, data=_z_index(shapes[i]), chunks=(chunks[i] if chunks else shapes[i])
        )
    group.attrs["multiscales"] = _ome(axes, levels)


def _v3_array(path, data, chunk_shape=None, attributes=None):
    metadata = {
        "shape": list(data.shape),
        "data_type": str(data.dtype),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunk_shape or data.shape)},
        },
    }
    if attributes:
        metadata["attributes"] = attributes
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}, "metadata": metadata},
        create=True,
    ).result()
    store[...] = data


def _v3_group(path, axes, levels, shapes, chunks=None):
    for i, (name, _, _) in enumerate(levels):
        _v3_array(
            os.path.join(path, name),
            _z_index(shapes[i]),
            chunk_shape=(chunks[i] if chunks else None),
        )
    with open(os.path.join(path, "zarr.json"), "w") as f:
        json.dump(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"multiscales": _ome(axes, levels, "0.5")}},
            },
            f,
        )


def _v2_array(root_path, name, shape, attrs, order="C", chunks=None):
    root = zarr.open_group(root_path, mode="a")
    arr = root.create_dataset(
        name, data=_z_index(shape), chunks=chunks or shape, order=order
    )
    arr.attrs.update(attrs)


def _n5_array(path, name, shape, attrs, chunks=None):
    root = zarr.open(N5FSStore(path), mode="a")
    arr = root.create_dataset(name, data=_z_index(shape), chunks=chunks or shape)
    arr.attrs.update(attrs)


def _precomputed(path):
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
            "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
            "scale_metadata": {
                "size": [20, 10, 2],
                "resolution": [4, 8, 16],
                "encoding": "raw",
                "chunk_size": [10, 5, 2],
                # Ignored today: every reader reports offset 0.
                "voxel_offset": [3, 2, 1],
            },
        },
        create=True,
    ).result()
    store[...] = np.arange(20 * 10 * 2, dtype=np.uint8).reshape(20, 10, 2, 1)


V2 = [("s0", (8, 8, 8), (100, 204, 308)), ("s1", (16, 16, 16), (104, 208, 312))]
CZYX = [("s0", (1, 8, 8, 8), (0, 4, 4, 4)), ("s1", (1, 16, 16, 16), (0, 8, 8, 8))]
V3 = [("s0", (4, 4, 4), (10, 20, 30)), ("s1", (8, 8, 8), (12, 22, 32))]
FLOAT = [("s0", (5.24, 4, 4), (2.62, 2, 2)), ("s1", (10.48, 8, 8), (5.24, 4, 4))]
MICRON = [
    ("s0", (0.008, 0.004, 0.004), (0.08, 0.04, 0.04)),
    ("s1", (0.016, 0.008, 0.008), None),
]
JANELIA = [("s0", (8,) * 3, (0,) * 3), ("s1", (16,) * 3, (4,) * 3), ("s2", (32,) * 3, (12,) * 3)]
CHANNEL = [{"name": "c", "type": "channel"}]


def build(root):
    """Every fixture, under ``root``."""
    r = functools.partial(os.path.join, root)
    _v2_group(r("v2.zarr", "em"), _space(), V2, [(8, 8, 8), (4, 4, 4)], [(4, 4, 4), (2, 2, 2)])
    _v2_group(
        r("czyx.zarr", "raw"),
        CHANNEL + _space(),
        CZYX,
        [(2, 8, 8, 8), (2, 4, 4, 4)],
        [(1, 4, 4, 4), (1, 2, 2, 2)],
    )
    _v3_group(r("v3.zarr"), _space(), V3, [(8, 8, 8), (4, 4, 4)], [[4, 4, 4], [2, 2, 2]])
    _v3_group(r("v3_czyx.zarr"), CHANNEL + _space(), CZYX, [(2, 8, 8, 8), (2, 4, 4, 4)])
    _v3_array(
        r("v3_plain_tx"),
        _z_index((4, 4, 4)),
        attributes={"transform": {"scale": [8.0, 8.0, 8.0], "translate": [100.0, 200.0, 300.0]}},
    )
    # A v3 array's own resolution/offset are not rounded onto the voxel grid
    # (the v2 reader's are).
    _v3_array(
        r("v3_plain_res"),
        _z_index((4, 4, 4)),
        attributes={"resolution": [8, 8, 8], "offset": [4, 4, 4]},
    )
    _v3_array(r("v3_plain_bare"), _z_index((2, 2, 2)))
    # COSEM-style N5: transform in C order, like Davis writes it.
    _n5_array(
        r("a.n5"),
        "tx",
        (10, 20, 30),
        {
            "transform": {
                "axes": ["z", "y", "x"],
                "ordering": "C",
                "scale": [8, 4, 2],
                "translate": [80, 40, 20],
                "units": ["nm", "nm", "nm"],
            }
        },
        chunks=(5, 10, 15),
    )
    # BigDataViewer/Paintera-style N5: x, y, z. Without an offset attribute
    # the whole lookup falls through to voxel size 1 (recorded as it is).
    pixres = {
        "pixelResolution": {"dimensions": [2, 4, 8], "unit": "nm"},
        "downsamplingFactors": [2, 2, 1],
    }
    _n5_array(r("a.n5"), "pixres", (10, 20, 30), pixres)
    _n5_array(r("a.n5"), "pixres_off", (10, 20, 30), {**pixres, "offset": [30, 20, 10]})
    _precomputed(r("pc"))
    _v2_group(r("float.zarr"), _space(), FLOAT, [(20, 4, 4), (10, 2, 2)])
    _v2_group(r("um.zarr"), _space("micrometer"), MICRON, [(4, 4, 4), (2, 2, 2)])
    _v3_group(r("um_v3.zarr"), _space("micrometer"), MICRON, [(4, 4, 4), (2, 2, 2)])
    zarr.open(r("root.zarr"), mode="w", shape=(4, 4, 4), dtype=np.uint8).attrs.update(
        {"resolution": [8, 4, 4], "offset": [80, 40, 40]}
    )
    # An offset that is not a multiple of the voxel size is rounded onto it.
    _v2_array(r("legacy.zarr"), "unaligned", (4, 4, 4), {"resolution": [8, 8, 8], "offset": [4, 4, 4]})
    _v2_array(
        r("legacy.zarr"), "forder", (4, 6, 8), {"resolution": [8, 4, 2], "offset": [0, 0, 0]}, order="F"
    )
    _v2_array(r("legacy.zarr"), "bare", (4, 4, 4), {})
    _v2_group(r("janelia.zarr", "em"), _space(), JANELIA, [(32,) * 3, (16,) * 3, (8,) * 3], [(8,) * 3] * 3)
    _v3_group(r("janelia_v3.zarr"), _space(), JANELIA, [(32,) * 3, (16,) * 3, (8,) * 3], [[8] * 3] * 3)
    # Crop-loader layouts: an OME group with its first scale, an array with
    # transform attrs, one with resolution/offset, and a group with only s0.
    _v2_group(r("crop_v2.zarr"), _space(), [("s0", (4, 4, 4), (10, 20, 30))], [(4, 4, 4)])
    _v3_group(r("crop_v3.zarr"), _space(), [("s0", (4, 4, 4), (10, 20, 30))], [(4, 4, 4)])
    _v2_array(
        r("crop_plain.zarr"), "tx", (4, 4, 4), {"transform": {"scale": [2, 2, 2], "translate": [1, 1, 1]}}
    )
    _v2_array(r("crop_plain.zarr"), "res", (4, 4, 4), {"resolution": [4, 4, 4], "offset": [2, 2, 2]})
    s0_only = zarr.open_group(r("crop_s0.zarr"), mode="w")
    s0_only.create_dataset("s0", data=_z_index((2, 2, 2)))
    os.makedirs(r("crop_s0_v3.zarr"))
    with open(r("crop_s0_v3.zarr", "zarr.json"), "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)
    _v3_array(r("crop_s0_v3.zarr", "s0"), _z_index((2, 2, 2)))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(root):
    """An http server over ``root``; returns (server, base URL)."""
    handler = functools.partial(_QuietHandler, directory=root)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """(root directory, base URL of an http server over it)."""
    root = str(tmp_path_factory.mktemp("io_matrix"))
    build(root)
    server, url = serve(root)
    yield root, url
    server.shutdown()
    server.server_close()


def _locate(world, where):
    """``where`` is a fixture-relative path, "http:" or "precomputed:" prefixed."""
    root, url = world
    if where.startswith("http:"):
        return f"{url}/{where[len('http:'):]}"
    if where.startswith("precomputed:"):
        return "precomputed://" + os.path.join(root, where[len("precomputed:"):])
    return os.path.join(root, where)


def _relative(world, path):
    root, url = world
    return path.replace(url, "http:").replace(root, "<root>")


# ---------------------------------------------------------------------------
# What is recorded for each kind of case
# ---------------------------------------------------------------------------


def _meta(world, where):
    from cellmap_flow.utils.ds import get_ds_info, read_ds_meta

    path = _locate(world, where)
    return {"read_ds_meta": describe(read_ds_meta(path)), "get_ds_info": describe(get_ds_info(path))}


def _meta_v3(world, where):
    from cellmap_flow.utils import zarr_v3

    path = _locate(world, where)
    return {
        "read_ds_meta_v3": describe(zarr_v3.read_ds_meta_v3(path)),
        "get_ds_info_v3": describe(zarr_v3.get_ds_info_v3(path)),
    }


def _idi(world, where, voxel_size=None):
    from cellmap_flow.image_data_interface import ImageDataInterface

    idi = ImageDataInterface(_locate(world, where), voxel_size=voxel_size)
    return {
        name: describe(getattr(idi, name))
        for name in (
            "voxel_size",
            "offset",
            "roi",
            "shape",
            "chunk_shape",
            "axes_names",
            "filetype",
            "actual_voxel_size",
            "requested_voxel_size",
            "output_voxel_size",
            "_voxel_size_f",
            "_offset_f",
        )
    } | {"path": _relative(world, idi.path)}


def _scales_v2(world, where, targets):
    from cellmap_flow.utils.ds import _open_zarr, find_closest_scale, get_scale_info

    path = _locate(world, where)
    out = {"get_scale_info": describe(get_scale_info(_open_zarr(path, mode="r")))}
    for target in targets:
        out[f"find_closest_scale{target}"] = describe(find_closest_scale(path, target))
    return out


def _scales_v3(world, where, targets):
    from cellmap_flow.utils import zarr_v3

    path = _locate(world, where)
    out = {"get_scale_info_v3": describe(zarr_v3.get_scale_info_v3(path))}
    for target in targets:
        out[f"find_closest_scale_v3{target}"] = describe(zarr_v3.find_closest_scale_v3(path, target))
    return out


def _closest_raw(world, where, targets):
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

    path = _locate(world, where)
    return {f"get_raw_closest_scale{t}": describe(get_raw_closest_scale(path, t)) for t in targets}


def _crop(world, where):
    from cellmap_flow.finetune import crop_loader

    return {"_read_voxel_size_and_offset": describe(crop_loader._read_voxel_size_and_offset(_locate(world, where)))}


def _crop_v3(world, where):
    from cellmap_flow.finetune import crop_loader

    return {
        "_read_voxel_size_and_offset_v3": describe(
            crop_loader._read_voxel_size_and_offset_v3(_locate(world, where))
        )
    }


def _raw_layer(world, where):
    from cellmap_flow.utils.scale_pyramid import get_raw_layer

    source = get_raw_layer(_locate(world, where), normalize=False).to_json()["source"]
    source = source[0] if isinstance(source, list) else source
    transform = source["transform"]
    return {
        "outputDimensions": describe(transform["outputDimensions"]),
        "matrix": describe(transform.get("matrix")),
    }


T3 = (None, (8, 8, 8), (12, 12, 12), (16, 16, 16), (32, 32, 32), (4, 4, 4))

CASES = {
    # read_ds_meta / get_ds_info
    "meta v2 group": (_meta, "v2.zarr/em"),
    "meta v2 s1": (_meta, "v2.zarr/em/s1"),
    "meta czyx s0": (_meta, "czyx.zarr/raw/s0"),
    "meta v3 group": (_meta, "v3.zarr"),
    "meta v3 s1": (_meta, "v3.zarr/s1"),
    "meta v3 czyx s1": (_meta, "v3_czyx.zarr/s1"),
    "meta v3 plain tx": (_meta, "v3_plain_tx"),
    "meta v3 plain res": (_meta, "v3_plain_res"),
    "meta v3 plain bare": (_meta, "v3_plain_bare"),
    "meta n5 tx": (_meta, "a.n5/tx"),
    "meta n5 pixres": (_meta, "a.n5/pixres"),
    "meta n5 pixres offset": (_meta, "a.n5/pixres_off"),
    "meta precomputed": (_meta, "precomputed:pc"),
    "meta float s0": (_meta, "float.zarr/s0"),
    "meta float s1": (_meta, "float.zarr/s1"),
    "meta micrometer s0": (_meta, "um.zarr/s0"),
    "meta micrometer s1": (_meta, "um.zarr/s1"),
    "meta micrometer v3 s0": (_meta, "um_v3.zarr/s0"),
    "meta root array": (_meta, "root.zarr"),
    "meta unaligned offset": (_meta, "legacy.zarr/unaligned"),
    "meta F order": (_meta, "legacy.zarr/forder"),
    "meta no attrs": (_meta, "legacy.zarr/bare"),
    "meta janelia s0": (_meta, "janelia.zarr/em/s0"),
    "meta janelia s1": (_meta, "janelia.zarr/em/s1"),
    "meta janelia s2": (_meta, "janelia.zarr/em/s2"),
    "meta janelia v3 s2": (_meta, "janelia_v3.zarr/s2"),
    "meta http group": (_meta, "http:v2.zarr/em"),
    "meta http s1": (_meta, "http:v2.zarr/em/s1"),
    "meta http czyx s1": (_meta, "http:czyx.zarr/raw/s1"),
    "meta http legacy": (_meta, "http:legacy.zarr/unaligned"),
    "meta http janelia s2": (_meta, "http:janelia.zarr/em/s2"),
    "meta v3 direct group": (_meta_v3, "v3.zarr"),
    "meta v3 direct s1": (_meta_v3, "v3.zarr/s1"),
    "meta v3 direct plain": (_meta_v3, "v3_plain_res"),
    # Scale selection
    "scales v2": (_scales_v2, "v2.zarr/em", T3),
    "scales czyx": (_scales_v2, "czyx.zarr/raw", T3),
    "scales float": (_scales_v2, "float.zarr", (None, (10.48, 8, 8), (10, 8, 8), (5.24, 4, 4))),
    "scales micrometer": (_scales_v2, "um.zarr", (None, (8, 4, 4), (16, 8, 8))),
    "scales janelia": (_scales_v2, "janelia.zarr/em", T3),
    "scales http": (_scales_v2, "http:v2.zarr/em", T3),
    "scales v3": (_scales_v3, "v3.zarr", T3),
    "scales v3 czyx": (_scales_v3, "v3_czyx.zarr", T3),
    "scales v3 micrometer": (_scales_v3, "um_v3.zarr", (None, (8, 4, 4), (16, 8, 8))),
    "scales janelia v3": (_scales_v3, "janelia_v3.zarr", T3),
    "closest raw v2 group": (_closest_raw, "v2.zarr/em", T3),
    "closest raw v2 s0": (_closest_raw, "v2.zarr/em/s0", T3),
    "closest raw v3 group": (_closest_raw, "v3.zarr", T3),
    "closest raw v3 s1": (_closest_raw, "v3.zarr/s1", T3),
    "closest raw czyx s0": (_closest_raw, "czyx.zarr/raw/s0", ((16, 16, 16),)),
    "closest raw micrometer": (_closest_raw, "um.zarr", ((16, 8, 8),)),
    "closest raw http": (_closest_raw, "http:v2.zarr/em", ((16, 16, 16),)),
    "closest raw plain array": (_closest_raw, "root.zarr", ((8, 4, 4),)),
    # ImageDataInterface
    "idi v2 group": (_idi, "v2.zarr/em"),
    "idi v2 group 16": (_idi, "v2.zarr/em", (16, 16, 16)),
    "idi v2 group 12": (_idi, "v2.zarr/em", (12, 12, 12)),
    "idi v2 group 32": (_idi, "v2.zarr/em", (32, 32, 32)),
    "idi v2 group 4": (_idi, "v2.zarr/em", (4, 4, 4)),
    "idi v2 s1 16": (_idi, "v2.zarr/em/s1", (16, 16, 16)),
    "idi czyx group": (_idi, "czyx.zarr/raw"),
    "idi czyx group 16": (_idi, "czyx.zarr/raw", (16, 16, 16)),
    "idi v3 group": (_idi, "v3.zarr"),
    "idi v3 group 8": (_idi, "v3.zarr", (8, 8, 8)),
    "idi v3 s1": (_idi, "v3.zarr/s1"),
    "idi v3 czyx 16": (_idi, "v3_czyx.zarr", (16, 16, 16)),
    "idi v3 plain res": (_idi, "v3_plain_res"),
    "idi n5 tx": (_idi, "a.n5/tx"),
    "idi n5 pixres": (_idi, "a.n5/pixres"),
    "idi n5 pixres offset": (_idi, "a.n5/pixres_off"),
    "idi precomputed": (_idi, "precomputed:pc"),
    "idi float 10.48": (_idi, "float.zarr", (10.48, 8, 8)),
    "idi float 10": (_idi, "float.zarr", (10, 8, 8)),
    "idi micrometer 16": (_idi, "um.zarr", (16, 8, 8)),
    "idi root array": (_idi, "root.zarr"),
    "idi unaligned offset": (_idi, "legacy.zarr/unaligned"),
    "idi F order": (_idi, "legacy.zarr/forder"),
    "idi janelia s2": (_idi, "janelia.zarr/em/s2"),
    "idi janelia 16": (_idi, "janelia.zarr/em", (16, 16, 16)),
    "idi janelia v3 32": (_idi, "janelia_v3.zarr", (32, 32, 32)),
    "idi http group 16": (_idi, "http:v2.zarr/em", (16, 16, 16)),
    "idi http s1": (_idi, "http:v2.zarr/em/s1"),
    # Crop loader
    "crop v2 group": (_crop, "crop_v2.zarr"),
    "crop v3 group": (_crop, "crop_v3.zarr"),
    "crop v3 group direct": (_crop_v3, "crop_v3.zarr"),
    "crop v2 transform": (_crop, "crop_plain.zarr/tx"),
    "crop v2 resolution": (_crop, "crop_plain.zarr/res"),
    "crop v3 transform": (_crop, "v3_plain_tx"),
    "crop v3 resolution": (_crop_v3, "v3_plain_res"),
    "crop v2 s0 only": (_crop, "crop_s0.zarr"),
    "crop v3 s0 only": (_crop, "crop_s0_v3.zarr"),
    # The crop loader reads the first scale as written: no unit conversion,
    # every axis.
    "crop micrometer": (_crop, "um.zarr"),
    "crop czyx": (_crop, "czyx.zarr/raw"),
    "crop janelia v3": (_crop_v3, "janelia_v3.zarr"),
    # Raw display layer
    "raw layer v2 group": (_raw_layer, "v2.zarr/em"),
    "raw layer v2 s1": (_raw_layer, "v2.zarr/em/s1"),
    "raw layer janelia": (_raw_layer, "janelia.zarr/em"),
    "raw layer v3 group": (_raw_layer, "v3.zarr"),
    "raw layer root array": (_raw_layer, "root.zarr"),
    "raw layer float": (_raw_layer, "float.zarr/s0"),
    "raw layer precomputed": (_raw_layer, "precomputed:pc"),
}


def compute(world, case):
    function, *args = CASES[case]
    try:
        return function(world, *args)
    except Exception as e:
        # Only the type: messages carry temporary paths.
        return {"raises": type(e).__name__}


@pytest.mark.parametrize("case", list(CASES))
def test_matrix(world, case):
    assert compute(world, case) == EXPECTED[case]


EXPECTED = {
    'meta v2 group': {
        'raises': 'AttributeError',
    },
    'meta v2 s1': {
        'read_ds_meta': "([16.0, 16.0, 16.0], [96.0, 200.0, 304.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(16, 16, 16), (2, 2, 2), Coordinate(4, 4, 4), Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta czyx s0': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [0.0, 0.0, 0.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(8, 8, 8), Roi(Coordinate(0, 0, 0), Coordinate(64, 64, 64)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 group': {
        'read_ds_meta': "([4.0, 4.0, 4.0], [8.0, 18.0, 28.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(4, 4, 4), (4, 4, 4), Coordinate(8, 8, 8), Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 s1': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [8.0, 18.0, 28.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (2, 2, 2), Coordinate(4, 4, 4), Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 czyx s1': {
        'read_ds_meta': "([16.0, 16.0, 16.0], [0.0, 0.0, 0.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(16, 16, 16), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(0, 0, 0), Coordinate(64, 64, 64)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 plain tx': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [100.0, 200.0, 300.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(100, 200, 300), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 plain res': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [4.0, 4.0, 4.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(4, 4, 4), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 plain bare': {
        'read_ds_meta': "([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (2, 2, 2), (2, 2, 2), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(1, 1, 1), (2, 2, 2), Coordinate(2, 2, 2), Roi(Coordinate(0, 0, 0), Coordinate(2, 2, 2)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta n5 tx': {
        'read_ds_meta': "([8.0, 4.0, 2.0], [80.0, 40.0, 20.0], (5, 10, 15), (10, 20, 30), ['z', 'y', 'x'], 'n5')",
        'get_ds_info': "(Coordinate(8, 4, 2), (5, 10, 15), Coordinate(10, 20, 30), Roi(Coordinate(80, 40, 20), Coordinate(80, 80, 60)), ['z', 'y', 'x'], 'n5')",
    },
    'meta n5 pixres': {
        'read_ds_meta': "([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (10, 20, 30), (10, 20, 30), ['z', 'y', 'x'], 'n5')",
        'get_ds_info': "(Coordinate(1, 1, 1), (10, 20, 30), Coordinate(10, 20, 30), Roi(Coordinate(0, 0, 0), Coordinate(10, 20, 30)), ['z', 'y', 'x'], 'n5')",
    },
    'meta n5 pixres offset': {
        'read_ds_meta': "([8.0, 8.0, 4.0], [8.0, 24.0, 32.0], (10, 20, 30), (10, 20, 30), ['z', 'y', 'x'], 'n5')",
        'get_ds_info': "(Coordinate(8, 8, 4), (10, 20, 30), Coordinate(10, 20, 30), Roi(Coordinate(8, 24, 32), Coordinate(80, 160, 120)), ['z', 'y', 'x'], 'n5')",
    },
    'meta precomputed': {
        'read_ds_meta': "([16.0, 8.0, 4.0], [0.0, 0.0, 0.0], (2, 5, 10), (2, 10, 20), ['z', 'y', 'x'], 'precomputed')",
        'get_ds_info': "(Coordinate(16, 8, 4), (2, 5, 10), Coordinate(2, 10, 20), Roi(Coordinate(0, 0, 0), Coordinate(32, 80, 80)), ['z', 'y', 'x'], 'precomputed')",
    },
    'meta float s0': {
        'read_ds_meta': "([5.24, 4.0, 4.0], [0.0, 0.0, 0.0], (20, 4, 4), (20, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "((5.24, 4.0, 4.0), (20, 4, 4), Coordinate(20, 4, 4), Roi(Coordinate(0, 0, 0), Coordinate(105, 16, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta float s1': {
        'read_ds_meta': "([10.48, 8.0, 8.0], [0.0, 0.0, 0.0], (10, 2, 2), (10, 2, 2), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "((10.48, 8.0, 8.0), (10, 2, 2), Coordinate(10, 2, 2), Roi(Coordinate(0, 0, 0), Coordinate(105, 16, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta micrometer s0': {
        'read_ds_meta': "([8.0, 4.0, 4.0], [76.0, 38.0, 38.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 4, 4), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(76, 38, 38), Coordinate(32, 16, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta micrometer s1': {
        'read_ds_meta': "([16.0, 8.0, 8.0], [-8.0, -4.0, -4.0], (2, 2, 2), (2, 2, 2), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(16, 8, 8), (2, 2, 2), Coordinate(2, 2, 2), Roi(Coordinate(-8, -4, -4), Coordinate(32, 16, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta micrometer v3 s0': {
        'read_ds_meta': "([8.0, 4.0, 4.0], [76.0, 38.0, 38.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 4, 4), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(76, 38, 38), Coordinate(32, 16, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta root array': {
        'read_ds_meta': "([8.0, 4.0, 4.0], [80.0, 40.0, 40.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 4, 4), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(80, 40, 40), Coordinate(32, 16, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta unaligned offset': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [8.0, 8.0, 8.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(8, 8, 8), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta F order': {
        'read_ds_meta': "([8.0, 4.0, 2.0], [0.0, 0.0, 0.0], (4, 6, 8), (4, 6, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 4, 2), (4, 6, 8), Coordinate(4, 6, 8), Roi(Coordinate(0, 0, 0), Coordinate(32, 24, 16)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta no attrs': {
        'read_ds_meta': "([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(1, 1, 1), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(0, 0, 0), Coordinate(4, 4, 4)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta janelia s0': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [-4.0, -4.0, -4.0], (8, 8, 8), (32, 32, 32), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (8, 8, 8), Coordinate(32, 32, 32), Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta janelia s1': {
        'read_ds_meta': "([16.0, 16.0, 16.0], [-4.0, -4.0, -4.0], (8, 8, 8), (16, 16, 16), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(16, 16, 16), (8, 8, 8), Coordinate(16, 16, 16), Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta janelia s2': {
        'read_ds_meta': "([32.0, 32.0, 32.0], [-4.0, -4.0, -4.0], (8, 8, 8), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(32, 32, 32), (8, 8, 8), Coordinate(8, 8, 8), Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta janelia v3 s2': {
        'read_ds_meta': "([32.0, 32.0, 32.0], [-4.0, -4.0, -4.0], (8, 8, 8), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(32, 32, 32), (8, 8, 8), Coordinate(8, 8, 8), Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta http group': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [96.0, 200.0, 304.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(8, 8, 8), Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta http s1': {
        'read_ds_meta': "([16.0, 16.0, 16.0], [96.0, 200.0, 304.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(16, 16, 16), (2, 2, 2), Coordinate(4, 4, 4), Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta http czyx s1': {
        'read_ds_meta': "([16.0, 16.0, 16.0], [0.0, 0.0, 0.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(16, 16, 16), (2, 2, 2), Coordinate(4, 4, 4), Roi(Coordinate(0, 0, 0), Coordinate(64, 64, 64)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta http legacy': {
        'read_ds_meta': "([8.0, 8.0, 8.0], [8.0, 8.0, 8.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(8, 8, 8), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta http janelia s2': {
        'read_ds_meta': "([32.0, 32.0, 32.0], [-4.0, -4.0, -4.0], (8, 8, 8), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info': "(Coordinate(32, 32, 32), (8, 8, 8), Coordinate(8, 8, 8), Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 direct group': {
        'read_ds_meta_v3': "([4.0, 4.0, 4.0], [8.0, 18.0, 28.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info_v3': "(Coordinate(4, 4, 4), (4, 4, 4), Coordinate(8, 8, 8), Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 direct s1': {
        'read_ds_meta_v3': "([8.0, 8.0, 8.0], [8.0, 18.0, 28.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info_v3': "(Coordinate(8, 8, 8), (2, 2, 2), Coordinate(4, 4, 4), Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'meta v3 direct plain': {
        'read_ds_meta_v3': "([8.0, 8.0, 8.0], [4.0, 4.0, 4.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')",
        'get_ds_info_v3': "(Coordinate(8, 8, 8), (4, 4, 4), Coordinate(4, 4, 4), Roi(Coordinate(4, 4, 4), Coordinate(32, 32, 32)), ['z', 'y', 'x'], 'zarr')",
    },
    'scales v2': {
        'get_scale_info': "({'s0': [96.0, 200.0, 304.0], 's1': [96.0, 200.0, 304.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, 16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)})",
        'find_closest_scaleNone': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
        'find_closest_scale(8, 8, 8)': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
        'find_closest_scale(12, 12, 12)': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
        'find_closest_scale(16, 16, 16)': "('s1', [96.0, 200.0, 304.0], (4, 4, 4))",
        'find_closest_scale(32, 32, 32)': "('s1', [96.0, 200.0, 304.0], (4, 4, 4))",
        'find_closest_scale(4, 4, 4)': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
    },
    'scales czyx': {
        'get_scale_info': "({'s0': [0.0, 0.0, 0.0], 's1': [0.0, 0.0, 0.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, 16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)})",
        'find_closest_scaleNone': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
        'find_closest_scale(8, 8, 8)': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
        'find_closest_scale(12, 12, 12)': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
        'find_closest_scale(16, 16, 16)': "('s1', [0.0, 0.0, 0.0], (4, 4, 4))",
        'find_closest_scale(32, 32, 32)': "('s1', [0.0, 0.0, 0.0], (4, 4, 4))",
        'find_closest_scale(4, 4, 4)': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
    },
    'scales float': {
        'get_scale_info': "({'s0': [0.0, 0.0, 0.0], 's1': [0.0, 0.0, 0.0]}, {'s0': [5.24, 4.0, 4.0], 's1': [10.48, 8.0, 8.0]}, {'s0': (20, 4, 4), 's1': (10, 2, 2)})",
        'find_closest_scaleNone': "('s0', [0.0, 0.0, 0.0], (20, 4, 4))",
        'find_closest_scale(10.48, 8, 8)': "('s1', [0.0, 0.0, 0.0], (10, 2, 2))",
        'find_closest_scale(10, 8, 8)': "('s0', [0.0, 0.0, 0.0], (20, 4, 4))",
        'find_closest_scale(5.24, 4, 4)': "('s0', [0.0, 0.0, 0.0], (20, 4, 4))",
    },
    'scales micrometer': {
        'get_scale_info': "({'s0': [76.0, 38.0, 38.0], 's1': [-8.0, -4.0, -4.0]}, {'s0': [8.0, 4.0, 4.0], 's1': [16.0, 8.0, 8.0]}, {'s0': (4, 4, 4), 's1': (2, 2, 2)})",
        'find_closest_scaleNone': "('s0', [76.0, 38.0, 38.0], (4, 4, 4))",
        'find_closest_scale(8, 4, 4)': "('s0', [76.0, 38.0, 38.0], (4, 4, 4))",
        'find_closest_scale(16, 8, 8)': "('s1', [-8.0, -4.0, -4.0], (2, 2, 2))",
    },
    'scales janelia': {
        'get_scale_info': "({'s0': [-4.0, -4.0, -4.0], 's1': [-4.0, -4.0, -4.0], 's2': [-4.0, -4.0, -4.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, 16.0, 16.0], 's2': [32.0, 32.0, 32.0]}, {'s0': (32, 32, 32), 's1': (16, 16, 16), 's2': (8, 8, 8)})",
        'find_closest_scaleNone': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
        'find_closest_scale(8, 8, 8)': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
        'find_closest_scale(12, 12, 12)': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
        'find_closest_scale(16, 16, 16)': "('s1', [-4.0, -4.0, -4.0], (16, 16, 16))",
        'find_closest_scale(32, 32, 32)': "('s2', [-4.0, -4.0, -4.0], (8, 8, 8))",
        'find_closest_scale(4, 4, 4)': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
    },
    'scales http': {
        'get_scale_info': "({'s0': [96.0, 200.0, 304.0], 's1': [96.0, 200.0, 304.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, 16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)})",
        'find_closest_scaleNone': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
        'find_closest_scale(8, 8, 8)': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
        'find_closest_scale(12, 12, 12)': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
        'find_closest_scale(16, 16, 16)': "('s1', [96.0, 200.0, 304.0], (4, 4, 4))",
        'find_closest_scale(32, 32, 32)': "('s1', [96.0, 200.0, 304.0], (4, 4, 4))",
        'find_closest_scale(4, 4, 4)': "('s0', [96.0, 200.0, 304.0], (8, 8, 8))",
    },
    'scales v3': {
        'get_scale_info_v3': "({'s0': [8.0, 18.0, 28.0], 's1': [8.0, 18.0, 28.0]}, {'s0': [4.0, 4.0, 4.0], 's1': [8.0, 8.0, 8.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)})",
        'find_closest_scale_v3None': "('s0', [8.0, 18.0, 28.0], (8, 8, 8))",
        'find_closest_scale_v3(8, 8, 8)': "('s1', [8.0, 18.0, 28.0], (4, 4, 4))",
        'find_closest_scale_v3(12, 12, 12)': "('s1', [8.0, 18.0, 28.0], (4, 4, 4))",
        'find_closest_scale_v3(16, 16, 16)': "('s1', [8.0, 18.0, 28.0], (4, 4, 4))",
        'find_closest_scale_v3(32, 32, 32)': "('s1', [8.0, 18.0, 28.0], (4, 4, 4))",
        'find_closest_scale_v3(4, 4, 4)': "('s0', [8.0, 18.0, 28.0], (8, 8, 8))",
    },
    'scales v3 czyx': {
        'get_scale_info_v3': "({'s0': [0.0, 0.0, 0.0], 's1': [0.0, 0.0, 0.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, 16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)})",
        'find_closest_scale_v3None': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
        'find_closest_scale_v3(8, 8, 8)': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
        'find_closest_scale_v3(12, 12, 12)': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
        'find_closest_scale_v3(16, 16, 16)': "('s1', [0.0, 0.0, 0.0], (4, 4, 4))",
        'find_closest_scale_v3(32, 32, 32)': "('s1', [0.0, 0.0, 0.0], (4, 4, 4))",
        'find_closest_scale_v3(4, 4, 4)': "('s0', [0.0, 0.0, 0.0], (8, 8, 8))",
    },
    'scales v3 micrometer': {
        'get_scale_info_v3': "({'s0': [76.0, 38.0, 38.0], 's1': [-8.0, -4.0, -4.0]}, {'s0': [8.0, 4.0, 4.0], 's1': [16.0, 8.0, 8.0]}, {'s0': (4, 4, 4), 's1': (2, 2, 2)})",
        'find_closest_scale_v3None': "('s0', [76.0, 38.0, 38.0], (4, 4, 4))",
        'find_closest_scale_v3(8, 4, 4)': "('s0', [76.0, 38.0, 38.0], (4, 4, 4))",
        'find_closest_scale_v3(16, 8, 8)': "('s1', [-8.0, -4.0, -4.0], (2, 2, 2))",
    },
    'scales janelia v3': {
        'get_scale_info_v3': "({'s0': [-4.0, -4.0, -4.0], 's1': [-4.0, -4.0, -4.0], 's2': [-4.0, -4.0, -4.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, 16.0, 16.0], 's2': [32.0, 32.0, 32.0]}, {'s0': (32, 32, 32), 's1': (16, 16, 16), 's2': (8, 8, 8)})",
        'find_closest_scale_v3None': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
        'find_closest_scale_v3(8, 8, 8)': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
        'find_closest_scale_v3(12, 12, 12)': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
        'find_closest_scale_v3(16, 16, 16)': "('s1', [-4.0, -4.0, -4.0], (16, 16, 16))",
        'find_closest_scale_v3(32, 32, 32)': "('s2', [-4.0, -4.0, -4.0], (8, 8, 8))",
        'find_closest_scale_v3(4, 4, 4)': "('s0', [-4.0, -4.0, -4.0], (32, 32, 32))",
    },
    'closest raw v2 group': {
        'get_raw_closest_scaleNone': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(8, 8, 8)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(12, 12, 12)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(16, 16, 16)': '(16.0, 16.0, 16.0)',
        'get_raw_closest_scale(32, 32, 32)': '(16.0, 16.0, 16.0)',
        'get_raw_closest_scale(4, 4, 4)': '(8.0, 8.0, 8.0)',
    },
    'closest raw v2 s0': {
        'get_raw_closest_scaleNone': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(8, 8, 8)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(12, 12, 12)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(16, 16, 16)': '(16.0, 16.0, 16.0)',
        'get_raw_closest_scale(32, 32, 32)': '(16.0, 16.0, 16.0)',
        'get_raw_closest_scale(4, 4, 4)': '(8.0, 8.0, 8.0)',
    },
    'closest raw v3 group': {
        'get_raw_closest_scaleNone': '(4.0, 4.0, 4.0)',
        'get_raw_closest_scale(8, 8, 8)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(12, 12, 12)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(16, 16, 16)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(32, 32, 32)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(4, 4, 4)': '(4.0, 4.0, 4.0)',
    },
    'closest raw v3 s1': {
        'get_raw_closest_scaleNone': '(4.0, 4.0, 4.0)',
        'get_raw_closest_scale(8, 8, 8)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(12, 12, 12)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(16, 16, 16)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(32, 32, 32)': '(8.0, 8.0, 8.0)',
        'get_raw_closest_scale(4, 4, 4)': '(4.0, 4.0, 4.0)',
    },
    'closest raw czyx s0': {
        'get_raw_closest_scale(16, 16, 16)': '(16.0, 16.0, 16.0)',
    },
    'closest raw micrometer': {
        'get_raw_closest_scale(16, 8, 8)': '(16.0, 8.0, 8.0)',
    },
    'closest raw http': {
        'get_raw_closest_scale(16, 16, 16)': '(16.0, 16.0, 16.0)',
    },
    'closest raw plain array': {
        'get_raw_closest_scale(8, 4, 4)': 'None',
    },
    'idi v2 group': {
        'voxel_size': 'Coordinate(8, 8, 8)',
        'offset': 'Coordinate(96, 200, 304)',
        'roi': 'Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([96.0, 200.0, 304.0])',
        'path': '<root>/v2.zarr/em/s0',
    },
    'idi v2 group 16': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(96, 200, 304)',
        'roi': 'Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(16, 16, 16)',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([96.0, 200.0, 304.0])',
        'path': '<root>/v2.zarr/em/s1',
    },
    'idi v2 group 12': {
        'voxel_size': 'Coordinate(12, 12, 12)',
        'offset': 'Coordinate(144, 300, 456)',
        'roi': 'Roi(Coordinate(144, 300, 456), Coordinate(96, 96, 96))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'Coordinate(12, 12, 12)',
        'output_voxel_size': 'Coordinate(12, 12, 12)',
        '_voxel_size_f': 'ndarray[float64]([12.0, 12.0, 12.0])',
        '_offset_f': 'ndarray[float64]([144.0, 300.0, 456.0])',
        'path': '<root>/v2.zarr/em/s0',
    },
    'idi v2 group 32': {
        'voxel_size': 'Coordinate(32, 32, 32)',
        'offset': 'Coordinate(192, 400, 608)',
        'roi': 'Roi(Coordinate(192, 400, 608), Coordinate(128, 128, 128))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(32, 32, 32)',
        'output_voxel_size': 'Coordinate(32, 32, 32)',
        '_voxel_size_f': 'ndarray[float64]([32.0, 32.0, 32.0])',
        '_offset_f': 'ndarray[float64]([192.0, 400.0, 608.0])',
        'path': '<root>/v2.zarr/em/s1',
    },
    'idi v2 group 4': {
        'voxel_size': 'Coordinate(4, 4, 4)',
        'offset': 'Coordinate(48, 100, 152)',
        'roi': 'Roi(Coordinate(48, 100, 152), Coordinate(32, 32, 32))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'Coordinate(4, 4, 4)',
        'output_voxel_size': 'Coordinate(4, 4, 4)',
        '_voxel_size_f': 'ndarray[float64]([4.0, 4.0, 4.0])',
        '_offset_f': 'ndarray[float64]([48.0, 100.0, 152.0])',
        'path': '<root>/v2.zarr/em/s0',
    },
    'idi v2 s1 16': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(96, 200, 304)',
        'roi': 'Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(16, 16, 16)',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([96.0, 200.0, 304.0])',
        'path': '<root>/v2.zarr/em/s1',
    },
    'idi czyx group': {
        'voxel_size': 'Coordinate(8, 8, 8)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/czyx.zarr/raw/s0',
    },
    'idi czyx group 16': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(16, 16, 16)',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/czyx.zarr/raw/s1',
    },
    'idi v3 group': {
        'voxel_size': 'Coordinate(4, 4, 4)',
        'offset': 'Coordinate(8, 18, 28)',
        'roi': 'Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(4, 4, 4)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(4, 4, 4)',
        '_voxel_size_f': 'ndarray[float64]([4.0, 4.0, 4.0])',
        '_offset_f': 'ndarray[float64]([8.0, 18.0, 28.0])',
        'path': '<root>/v3.zarr/s0',
    },
    'idi v3 group 8': {
        'voxel_size': 'Coordinate(8, 8, 8)',
        'offset': 'Coordinate(8, 18, 28)',
        'roi': 'Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'Coordinate(8, 8, 8)',
        'output_voxel_size': 'Coordinate(8, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([8.0, 18.0, 28.0])',
        'path': '<root>/v3.zarr/s1',
    },
    'idi v3 s1': {
        'voxel_size': 'Coordinate(8, 8, 8)',
        'offset': 'Coordinate(8, 18, 28)',
        'roi': 'Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([8.0, 18.0, 28.0])',
        'path': '<root>/v3.zarr/s1',
    },
    'idi v3 czyx 16': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(16, 16, 16)',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/v3_czyx.zarr/s1',
    },
    'idi v3 plain res': {
        'voxel_size': 'Coordinate(8, 8, 8)',
        'offset': 'Coordinate(4, 4, 4)',
        'roi': 'Roi(Coordinate(4, 4, 4), Coordinate(32, 32, 32))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([4.0, 4.0, 4.0])',
        'path': '<root>/v3_plain_res',
    },
    'idi n5 tx': {
        'voxel_size': 'Coordinate(8, 4, 2)',
        'offset': 'Coordinate(80, 40, 20)',
        'roi': 'Roi(Coordinate(80, 40, 20), Coordinate(80, 80, 60))',
        'shape': 'Coordinate(10, 20, 30)',
        'chunk_shape': '(5, 10, 15)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'n5'",
        'actual_voxel_size': 'Coordinate(8, 4, 2)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 4, 2)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 4.0, 2.0])',
        '_offset_f': 'ndarray[float64]([80.0, 40.0, 20.0])',
        'path': '<root>/a.n5/tx',
    },
    'idi n5 pixres': {
        'voxel_size': 'Coordinate(1, 1, 1)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(10, 20, 30))',
        'shape': 'Coordinate(10, 20, 30)',
        'chunk_shape': '(10, 20, 30)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'n5'",
        'actual_voxel_size': 'Coordinate(1, 1, 1)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(1, 1, 1)',
        '_voxel_size_f': 'ndarray[float64]([1.0, 1.0, 1.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/a.n5/pixres',
    },
    'idi n5 pixres offset': {
        'voxel_size': 'Coordinate(8, 8, 4)',
        'offset': 'Coordinate(8, 24, 32)',
        'roi': 'Roi(Coordinate(8, 24, 32), Coordinate(80, 160, 120))',
        'shape': 'Coordinate(10, 20, 30)',
        'chunk_shape': '(10, 20, 30)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'n5'",
        'actual_voxel_size': 'Coordinate(8, 8, 4)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 8, 4)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 4.0])',
        '_offset_f': 'ndarray[float64]([8.0, 24.0, 32.0])',
        'path': '<root>/a.n5/pixres_off',
    },
    'idi precomputed': {
        'voxel_size': 'Coordinate(16, 8, 4)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(32, 80, 80))',
        'shape': 'Coordinate(2, 10, 20)',
        'chunk_shape': '(2, 5, 10)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'precomputed'",
        'actual_voxel_size': 'Coordinate(16, 8, 4)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(16, 8, 4)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 8.0, 4.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': 'precomputed://<root>/pc',
    },
    'idi float 10.48': {
        'voxel_size': '(10.48, 8.0, 8.0)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(105, 16, 16))',
        'shape': 'Coordinate(10, 2, 2)',
        'chunk_shape': '(10, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': '(10.48, 8.0, 8.0)',
        'requested_voxel_size': '(10.48, 8.0, 8.0)',
        'output_voxel_size': '(10.48, 8.0, 8.0)',
        '_voxel_size_f': 'ndarray[float64]([10.48, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/float.zarr/s1',
    },
    'idi float 10': {
        'voxel_size': 'Coordinate(10, 8, 8)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(200, 32, 32))',
        'shape': 'Coordinate(20, 4, 4)',
        'chunk_shape': '(20, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': '(5.24, 4.0, 4.0)',
        'requested_voxel_size': 'Coordinate(10, 8, 8)',
        'output_voxel_size': 'Coordinate(10, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([10.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/float.zarr/s0',
    },
    'idi micrometer 16': {
        'voxel_size': 'Coordinate(16, 8, 8)',
        'offset': 'Coordinate(-8, -4, -4)',
        'roi': 'Roi(Coordinate(-8, -4, -4), Coordinate(32, 16, 16))',
        'shape': 'Coordinate(2, 2, 2)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 8, 8)',
        'requested_voxel_size': 'Coordinate(16, 8, 8)',
        'output_voxel_size': 'Coordinate(16, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([-8.0, -4.0, -4.0])',
        'path': '<root>/um.zarr/s1',
    },
    'idi root array': {
        'voxel_size': 'Coordinate(8, 4, 4)',
        'offset': 'Coordinate(80, 40, 40)',
        'roi': 'Roi(Coordinate(80, 40, 40), Coordinate(32, 16, 16))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 4, 4)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 4, 4)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 4.0, 4.0])',
        '_offset_f': 'ndarray[float64]([80.0, 40.0, 40.0])',
        'path': '<root>/root.zarr',
    },
    'idi unaligned offset': {
        'voxel_size': 'Coordinate(8, 8, 8)',
        'offset': 'Coordinate(8, 8, 8)',
        'roi': 'Roi(Coordinate(8, 8, 8), Coordinate(32, 32, 32))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(4, 4, 4)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 8, 8)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 8, 8)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        '_offset_f': 'ndarray[float64]([8.0, 8.0, 8.0])',
        'path': '<root>/legacy.zarr/unaligned',
    },
    'idi F order': {
        'voxel_size': 'Coordinate(8, 4, 2)',
        'offset': 'Coordinate(0, 0, 0)',
        'roi': 'Roi(Coordinate(0, 0, 0), Coordinate(32, 24, 16))',
        'shape': 'Coordinate(4, 6, 8)',
        'chunk_shape': '(4, 6, 8)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(8, 4, 2)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(8, 4, 2)',
        '_voxel_size_f': 'ndarray[float64]([8.0, 4.0, 2.0])',
        '_offset_f': 'ndarray[float64]([0.0, 0.0, 0.0])',
        'path': '<root>/legacy.zarr/forder',
    },
    'idi janelia s2': {
        'voxel_size': 'Coordinate(32, 32, 32)',
        'offset': 'Coordinate(-4, -4, -4)',
        'roi': 'Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(8, 8, 8)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(32, 32, 32)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(32, 32, 32)',
        '_voxel_size_f': 'ndarray[float64]([32.0, 32.0, 32.0])',
        '_offset_f': 'ndarray[float64]([-4.0, -4.0, -4.0])',
        'path': '<root>/janelia.zarr/em/s2',
    },
    'idi janelia 16': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(-4, -4, -4)',
        'roi': 'Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256))',
        'shape': 'Coordinate(16, 16, 16)',
        'chunk_shape': '(8, 8, 8)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(16, 16, 16)',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([-4.0, -4.0, -4.0])',
        'path': '<root>/janelia.zarr/em/s1',
    },
    'idi janelia v3 32': {
        'voxel_size': 'Coordinate(32, 32, 32)',
        'offset': 'Coordinate(-4, -4, -4)',
        'roi': 'Roi(Coordinate(-4, -4, -4), Coordinate(256, 256, 256))',
        'shape': 'Coordinate(8, 8, 8)',
        'chunk_shape': '(8, 8, 8)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(32, 32, 32)',
        'requested_voxel_size': 'Coordinate(32, 32, 32)',
        'output_voxel_size': 'Coordinate(32, 32, 32)',
        '_voxel_size_f': 'ndarray[float64]([32.0, 32.0, 32.0])',
        '_offset_f': 'ndarray[float64]([-4.0, -4.0, -4.0])',
        'path': '<root>/janelia_v3.zarr/s2',
    },
    'idi http group 16': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(96, 200, 304)',
        'roi': 'Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'Coordinate(16, 16, 16)',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([96.0, 200.0, 304.0])',
        'path': 'http:/v2.zarr/em/s1',
    },
    'idi http s1': {
        'voxel_size': 'Coordinate(16, 16, 16)',
        'offset': 'Coordinate(96, 200, 304)',
        'roi': 'Roi(Coordinate(96, 200, 304), Coordinate(64, 64, 64))',
        'shape': 'Coordinate(4, 4, 4)',
        'chunk_shape': '(2, 2, 2)',
        'axes_names': "['z', 'y', 'x']",
        'filetype': "'zarr'",
        'actual_voxel_size': 'Coordinate(16, 16, 16)',
        'requested_voxel_size': 'None',
        'output_voxel_size': 'Coordinate(16, 16, 16)',
        '_voxel_size_f': 'ndarray[float64]([16.0, 16.0, 16.0])',
        '_offset_f': 'ndarray[float64]([96.0, 200.0, 304.0])',
        'path': 'http:/v2.zarr/em/s1',
    },
    'crop v2 group': {
        '_read_voxel_size_and_offset': "(('s0',), ndarray[float64]([4.0, 4.0, 4.0]), ndarray[float64]([8.0, 18.0, 28.0]))",
    },
    'crop v3 group': {
        '_read_voxel_size_and_offset': "(('s0',), ndarray[float64]([4.0, 4.0, 4.0]), ndarray[float64]([8.0, 18.0, 28.0]))",
    },
    'crop v3 group direct': {
        '_read_voxel_size_and_offset_v3': "(('s0',), ndarray[float64]([4.0, 4.0, 4.0]), ndarray[float64]([8.0, 18.0, 28.0]))",
    },
    'crop v2 transform': {
        '_read_voxel_size_and_offset': '((), ndarray[float64]([2.0, 2.0, 2.0]), ndarray[float64]([1.0, 1.0, 1.0]))',
    },
    'crop v2 resolution': {
        '_read_voxel_size_and_offset': '((), ndarray[float64]([4.0, 4.0, 4.0]), ndarray[float64]([2.0, 2.0, 2.0]))',
    },
    'crop v3 transform': {
        '_read_voxel_size_and_offset': '((), ndarray[float64]([8.0, 8.0, 8.0]), ndarray[float64]([100.0, 200.0, 300.0]))',
    },
    'crop v3 resolution': {
        '_read_voxel_size_and_offset_v3': '((), ndarray[float64]([8.0, 8.0, 8.0]), ndarray[float64]([4.0, 4.0, 4.0]))',
    },
    'crop v2 s0 only': {
        '_read_voxel_size_and_offset': "(('s0',), ndarray[float64]([1.0, 1.0, 1.0]), ndarray[float64]([0.0, 0.0, 0.0]))",
    },
    'crop v3 s0 only': {
        '_read_voxel_size_and_offset': "(('s0',), ndarray[float64]([1.0, 1.0, 1.0]), ndarray[float64]([0.0, 0.0, 0.0]))",
    },
    'crop micrometer': {
        '_read_voxel_size_and_offset': "(('s0',), ndarray[float64]([0.008, 0.004, 0.004]), ndarray[float64]([0.076, 0.038, 0.038]))",
    },
    'crop czyx': {
        '_read_voxel_size_and_offset': "(('s0',), ndarray[float64]([1.0, 8.0, 8.0, 8.0]), ndarray[float64]([-0.5, 0.0, 0.0, 0.0]))",
    },
    'crop janelia v3': {
        '_read_voxel_size_and_offset_v3': "(('s0',), ndarray[float64]([8.0, 8.0, 8.0]), ndarray[float64]([-4.0, -4.0, -4.0]))",
    },
    'raw layer v2 group': {
        'outputDimensions': "{'z': [float64(8e-09), 'm'], 'y': [float64(8e-09), 'm'], 'x': [float64(8e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, 12.0], [0.0, 1.0, 0.0, 25.0], [0.0, 0.0, 1.0, 38.0]]',
    },
    'raw layer v2 s1': {
        'outputDimensions': "{'z': [float64(8e-09), 'm'], 'y': [float64(8e-09), 'm'], 'x': [float64(8e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, 12.0], [0.0, 1.0, 0.0, 25.0], [0.0, 0.0, 1.0, 38.0]]',
    },
    'raw layer janelia': {
        'outputDimensions': "{'z': [float64(8e-09), 'm'], 'y': [float64(8e-09), 'm'], 'x': [float64(8e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, -0.5], [0.0, 1.0, 0.0, -0.5], [0.0, 0.0, 1.0, -0.5]]',
    },
    'raw layer v3 group': {
        'outputDimensions': "{'z': [float64(4e-09), 'm'], 'y': [float64(4e-09), 'm'], 'x': [float64(4e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 4.5], [0.0, 0.0, 1.0, 7.0]]',
    },
    'raw layer root array': {
        'outputDimensions': "{'z': [float64(8e-09), 'm'], 'y': [float64(4e-09), 'm'], 'x': [float64(4e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 10.0], [0.0, 0.0, 1.0, 10.0]]',
    },
    'raw layer float': {
        'outputDimensions': "{'z': [float64(5.24e-09), 'm'], 'y': [float64(4e-09), 'm'], 'x': [float64(4e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]',
    },
    'raw layer precomputed': {
        'outputDimensions': "{'z': [float64(1.6e-08), 'm'], 'y': [float64(8e-09), 'm'], 'x': [float64(4e-09), 'm']}",
        'matrix': '[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]',
    },
}
