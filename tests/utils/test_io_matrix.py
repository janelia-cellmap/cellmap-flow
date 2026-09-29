"""What every dataset reader reports, recorded as literals.

The metadata readers (zarr v2 local and over http, zarr v3, N5,
neuroglancer precomputed), scale selection, ImageDataInterface and the crop
loader each had their own copy of the parsing before io/ replaced them.
This table pins what their public functions return on one set of fixtures,
so that the consolidation changes nothing that is not written down here.

Values are rendered by ``describe``, which keeps the distinctions that
matter downstream: list vs tuple, Coordinate vs floats, int vs float. Where
a value looks wrong it is still recorded as it is, with a note.
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


def _space(unit="nanometer"):
    return [{"name": n, "type": "space", "unit": unit} for n in "zyx"]


CHANNEL = [{"name": "c", "type": "channel"}]


def _ome(axes, levels, version="0.4"):
    """``levels``: (path, scale, translation or None, shape)."""
    datasets = []
    for path, scale, translation, _ in levels:
        transforms = [{"type": "scale", "scale": list(scale)}]
        if translation is not None:
            transforms.append({"type": "translation", "translation": list(translation)})
        datasets.append({"path": path, "coordinateTransformations": transforms})
    return [{"version": version, "axes": axes, "datasets": datasets}]


def _data(shape):
    return np.zeros(shape, dtype=np.uint8)


def _v2_group(root, container, inner, axes, levels):
    group = zarr.open_group(os.path.join(root, container), mode="a")
    if inner:
        group = group.require_group(inner)
    for path, _, _, shape in levels:
        # Chunks of half the shape, so a chunk shape is not the array shape.
        group.create_dataset(path, data=_data(shape), chunks=tuple(max(1, s // 2) for s in shape))
    group.attrs["multiscales"] = _ome(axes, levels)


def _v3_array(path, shape, attributes=None):
    metadata = {
        "shape": list(shape),
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [max(1, s // 2) for s in shape]},
        },
    }
    if attributes:
        metadata["attributes"] = attributes
    ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}, "metadata": metadata},
        create=True,
    ).result()


def _v3_group(root, name, axes, levels):
    for path, _, _, shape in levels:
        _v3_array(os.path.join(root, name, path), shape)
    with open(os.path.join(root, name, "zarr.json"), "w") as f:
        json.dump(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"multiscales": _ome(axes, levels, "0.5")}},
            },
            f,
        )


def _v2_array(root, container, name, shape, attrs, order="C"):
    group = zarr.open_group(os.path.join(root, container), mode="a")
    group.create_dataset(name, data=_data(shape), order=order).attrs.update(attrs)


def _n5_array(root, name, shape, attrs):
    n5 = zarr.open(N5FSStore(os.path.join(root, "a.n5")), mode="a")
    n5.create_dataset(name, data=_data(shape), chunks=(5, 10, 15)).attrs.update(attrs)


V2 = [("s0", (8, 8, 8), (100, 204, 308), (8, 8, 8)), ("s1", (16, 16, 16), (104, 208, 312), (4, 4, 4))]
CZYX = [("s0", (1, 8, 8, 8), (0, 4, 4, 4), (2, 8, 8, 8)), ("s1", (1, 16, 16, 16), (0, 8, 8, 8), (2, 4, 4, 4))]
V3 = [("s0", (4, 4, 4), (10, 20, 30), (8, 8, 8)), ("s1", (8, 8, 8), (12, 22, 32), (4, 4, 4))]
FLOAT = [("s0", (5.24, 4, 4), (2.62, 2, 2), (20, 4, 4)), ("s1", (10.48, 8, 8), (5.24, 4, 4), (10, 2, 2))]
MICRON = [
    ("s0", (0.008, 0.004, 0.004), (0.08, 0.04, 0.04), (4, 4, 4)),
    ("s1", (0.016, 0.008, 0.008), None, (2, 2, 2)),
]


def build(root):
    """Every fixture, under ``root``."""
    _v2_group(root, "v2.zarr", "em", _space(), V2)
    _v2_group(root, "czyx.zarr", "raw", CHANNEL + _space(), CZYX)
    _v2_group(root, "float.zarr", "", _space(), FLOAT)
    _v2_group(root, "um.zarr", "", _space("micrometer"), MICRON)
    _v3_group(root, "v3.zarr", _space(), V3)
    _v3_group(root, "v3_czyx.zarr", CHANNEL + _space(), CZYX)
    _v3_group(root, "um_v3.zarr", _space("micrometer"), MICRON)
    _v3_array(
        os.path.join(root, "v3_plain_tx"),
        (4, 4, 4),
        {"transform": {"scale": [8.0, 8.0, 8.0], "translate": [100.0, 200.0, 300.0]}},
    )
    # A v3 array's own resolution/offset are not rounded onto the voxel grid
    # (the v2 reader's are).
    _v3_array(os.path.join(root, "v3_plain_res"), (4, 4, 4), {"resolution": [8, 8, 8], "offset": [4, 4, 4]})
    # COSEM-style N5: transform in C order, like Davis writes it.
    transform = {"ordering": "C", "scale": [8, 4, 2], "translate": [80, 40, 20], "units": ["nm"] * 3}
    _n5_array(root, "tx", (10, 20, 30), {"transform": transform})
    # BigDataViewer/Paintera-style N5: x, y, z. Without an offset attribute
    # the whole lookup falls through to voxel size 1 (recorded as it is).
    pixres = {"pixelResolution": {"dimensions": [2, 4, 8], "unit": "nm"}, "downsamplingFactors": [2, 2, 1]}
    _n5_array(root, "pixres", (10, 20, 30), pixres)
    _n5_array(root, "pixres_off", (10, 20, 30), {**pixres, "offset": [30, 20, 10]})
    ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": os.path.join(root, "pc")},
            "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
            "scale_metadata": {
                "size": [20, 10, 2],
                "resolution": [4, 8, 16],
                "encoding": "raw",
                "chunk_size": [10, 5, 2],
                # Not read: every reader reports offset 0.
                "voxel_offset": [3, 2, 1],
            },
        },
        create=True,
    ).result()
    zarr.open(os.path.join(root, "root.zarr"), mode="w", shape=(4, 4, 4), dtype=np.uint8).attrs.update(
        {"resolution": [8, 4, 4], "offset": [80, 40, 40]}
    )
    # An offset that is not a multiple of the voxel size is rounded onto it.
    _v2_array(root, "legacy.zarr", "unaligned", (4, 4, 4), {"resolution": [8, 8, 8], "offset": [4, 4, 4]})
    _v2_array(root, "legacy.zarr", "forder", (4, 6, 8), {"resolution": [8, 4, 2], "offset": [0, 0, 0]}, "F")
    _v2_array(root, "legacy.zarr", "bare", (4, 4, 4), {})
    _v2_array(root, "legacy.zarr", "tx", (4, 4, 4), {"transform": {"scale": [2, 2, 2], "translate": [1, 1, 1]}})
    zarr.open_group(os.path.join(root, "s0_only.zarr"), mode="w").create_dataset("s0", data=_data((2, 2, 2)))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(root):
    """An http server over ``root``; returns (server, base URL)."""
    handler = functools.partial(_QuietHandler, directory=root)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


# ---------------------------------------------------------------------------
# What is recorded
# ---------------------------------------------------------------------------


def _meta(path):
    from cellmap_flow.utils.ds import read_ds_meta

    return read_ds_meta(path)


def _levels(path):
    """The scale table and the level chosen for each voxel size in T."""
    from cellmap_flow.utils import zarr_v3
    from cellmap_flow.utils.ds import _open_zarr, find_closest_scale, get_scale_info

    if zarr_v3.is_v3_container(path):
        table = zarr_v3.get_scale_info_v3(path)
        return table, [zarr_v3.find_closest_scale_v3(path, t) for t in T]
    return get_scale_info(_open_zarr(path, mode="r")), [find_closest_scale(path, t) for t in T]


def _closest(path):
    from cellmap_flow.utils.neuroglancer_utils import get_raw_closest_scale

    return [get_raw_closest_scale(path, t) for t in T]


def _idi(path, voxel_size=None):
    from cellmap_flow.image_data_interface import ImageDataInterface

    idi = ImageDataInterface(path, voxel_size=voxel_size)
    names = ("path", "voxel_size", "offset", "roi", "shape", "chunk_shape", "axes_names", "filetype")
    names += ("actual_voxel_size", "requested_voxel_size", "_offset_f")
    return {name: getattr(idi, name) for name in names}


def _crop(path):
    from cellmap_flow.finetune.crop_loader import _read_voxel_size_and_offset

    return _read_voxel_size_and_offset(path)


def _raw_layer(path):
    from cellmap_flow.utils.scale_pyramid import get_raw_layer

    source = get_raw_layer(path, normalize=False).to_json()["source"]
    transform = (source[0] if isinstance(source, list) else source)["transform"]
    scales = {k: v[0] for k, v in transform["outputDimensions"].items()}
    return scales, transform.get("matrix")


T = (None, (8, 8, 8), (12, 12, 12), (16, 16, 16), (32, 32, 32), (4, 4, 4))

# row: (recorder, fixture path; "http:" is served over http, "pc:" is precomputed://)
ROWS = {
    "meta v2 level": (_meta, "v2.zarr/em/s1"),
    "meta v2 group": (_meta, "v2.zarr/em"),
    "meta czyx": (_meta, "czyx.zarr/raw/s0"),
    "meta v3 group": (_meta, "v3.zarr"),
    "meta v3 level": (_meta, "v3.zarr/s1"),
    "meta v3 czyx": (_meta, "v3_czyx.zarr/s1"),
    "meta v3 plain transform": (_meta, "v3_plain_tx"),
    "meta v3 plain resolution": (_meta, "v3_plain_res"),
    "meta n5 transform": (_meta, "a.n5/tx"),
    "meta n5 pixelResolution": (_meta, "a.n5/pixres"),
    "meta n5 pixelResolution offset": (_meta, "a.n5/pixres_off"),
    "meta precomputed": (_meta, "pc:pc"),
    "meta float": (_meta, "float.zarr/s1"),
    "meta micrometer": (_meta, "um.zarr/s0"),
    "meta micrometer v3": (_meta, "um_v3.zarr/s0"),
    "meta root array": (_meta, "root.zarr"),
    "meta unaligned offset": (_meta, "legacy.zarr/unaligned"),
    "meta F order": (_meta, "legacy.zarr/forder"),
    "meta no attrs": (_meta, "legacy.zarr/bare"),
    "meta http group": (_meta, "http:v2.zarr/em"),
    "meta http czyx": (_meta, "http:czyx.zarr/raw/s1"),
    "meta http legacy": (_meta, "http:legacy.zarr/unaligned"),
    "levels v2": (_levels, "v2.zarr/em"),
    "levels czyx": (_levels, "czyx.zarr/raw"),
    "levels float": (_levels, "float.zarr"),
    "levels http": (_levels, "http:v2.zarr/em"),
    "levels v3": (_levels, "v3.zarr"),
    "levels micrometer v3": (_levels, "um_v3.zarr"),
    "closest v2 group": (_closest, "v2.zarr/em"),
    "closest v2 level": (_closest, "czyx.zarr/raw/s0"),
    "closest v3 level": (_closest, "v3.zarr/s1"),
    "closest micrometer": (_closest, "um.zarr"),
    "closest http": (_closest, "http:v2.zarr/em"),
    "closest plain array": (_closest, "root.zarr"),
    "idi v2 relabelled": (_idi, "v2.zarr/em", (12, 12, 12)),
    "idi v3": (_idi, "v3.zarr", (8, 8, 8)),
    "idi http": (_idi, "http:v2.zarr/em", (16, 16, 16)),
    "idi n5": (_idi, "a.n5/pixres_off"),
    "idi precomputed": (_idi, "pc:pc"),
    "idi float": (_idi, "float.zarr", (10, 8, 8)),
    # The crop loader reads the first scale as written: no unit conversion,
    # every axis.
    "crop v2 group": (_crop, "czyx.zarr/raw"),
    "crop v3 group": (_crop, "um_v3.zarr"),
    "crop transform": (_crop, "legacy.zarr/tx"),
    "crop resolution": (_crop, "legacy.zarr/unaligned"),
    "crop s0 only": (_crop, "s0_only.zarr"),
    "raw layer v3": (_raw_layer, "v3.zarr"),
    "raw layer precomputed": (_raw_layer, "pc:pc"),
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("io_matrix"))
    build(root)
    server, url = serve(root)
    yield root, url
    server.shutdown()
    server.server_close()


def compute(world, row):
    """``describe`` of what ``row`` records, with the fixture root and the
    server URL written as <root> and http:."""
    root, url = world
    recorder, where, *args = ROWS[row]
    if where.startswith("http:"):
        path = f"{url}/{where[len('http:'):]}"
    elif where.startswith("pc:"):
        path = "precomputed://" + os.path.join(root, where[len("pc:"):])
    else:
        path = os.path.join(root, where)
    try:
        value = describe(recorder(path, *args))
    except Exception as e:
        # Only the type: messages carry temporary paths.
        value = f"raises {type(e).__name__}"
    return value.replace(url, "http:").replace(root, "<root>")


def test_matrix(world):
    assert {row: compute(world, row) for row in ROWS} == EXPECTED


EXPECTED = {
    'meta v2 level': (
        "([16.0, 16.0, 16.0], [96.0, 200.0, 304.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], "
        "'zarr')"
    ),
    'meta v2 group': 'raises AttributeError',
    'meta czyx': (
        "([8.0, 8.0, 8.0], [0.0, 0.0, 0.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta v3 group': (
        "([4.0, 4.0, 4.0], [8.0, 18.0, 28.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta v3 level': (
        "([8.0, 8.0, 8.0], [8.0, 18.0, 28.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta v3 czyx': (
        "([16.0, 16.0, 16.0], [0.0, 0.0, 0.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta v3 plain transform': (
        "([8.0, 8.0, 8.0], [100.0, 200.0, 300.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta v3 plain resolution': (
        "([8.0, 8.0, 8.0], [4.0, 4.0, 4.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta n5 transform': (
        "([8.0, 4.0, 2.0], [80.0, 40.0, 20.0], (5, 10, 15), (10, 20, 30), ['z', 'y', 'x'], 'n5')"
    ),
    'meta n5 pixelResolution': (
        "([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (5, 10, 15), (10, 20, 30), ['z', 'y', 'x'], 'n5')"
    ),
    'meta n5 pixelResolution offset': (
        "([8.0, 8.0, 4.0], [8.0, 24.0, 32.0], (5, 10, 15), (10, 20, 30), ['z', 'y', 'x'], 'n5')"
    ),
    'meta precomputed': (
        "([16.0, 8.0, 4.0], [0.0, 0.0, 0.0], (2, 5, 10), (2, 10, 20), ['z', 'y', 'x'], "
        "'precomputed')"
    ),
    'meta float': (
        "([10.48, 8.0, 8.0], [0.0, 0.0, 0.0], (5, 1, 1), (10, 2, 2), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta micrometer': (
        "([8.0, 4.0, 4.0], [76.0, 38.0, 38.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta micrometer v3': (
        "([8.0, 4.0, 4.0], [76.0, 38.0, 38.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta root array': (
        "([8.0, 4.0, 4.0], [80.0, 40.0, 40.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta unaligned offset': (
        "([8.0, 8.0, 8.0], [8.0, 8.0, 8.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta F order': (
        "([8.0, 4.0, 2.0], [0.0, 0.0, 0.0], (4, 6, 8), (4, 6, 8), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta no attrs': (
        "([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta http group': (
        "([8.0, 8.0, 8.0], [96.0, 200.0, 304.0], (4, 4, 4), (8, 8, 8), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta http czyx': (
        "([16.0, 16.0, 16.0], [0.0, 0.0, 0.0], (2, 2, 2), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'meta http legacy': (
        "([8.0, 8.0, 8.0], [8.0, 8.0, 8.0], (4, 4, 4), (4, 4, 4), ['z', 'y', 'x'], 'zarr')"
    ),
    'levels v2': (
        "(({'s0': [96.0, 200.0, 304.0], 's1': [96.0, 200.0, 304.0]}, {'s0': [8.0, 8.0, 8.0], "
        "'s1': [16.0, 16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)}), [('s0', [96.0, 200.0, "
        "304.0], (8, 8, 8)), ('s0', [96.0, 200.0, 304.0], (8, 8, 8)), ('s0', [96.0, 200.0, 304.0], "
        "(8, 8, 8)), ('s1', [96.0, 200.0, 304.0], (4, 4, 4)), ('s1', [96.0, 200.0, 304.0], (4, 4, "
        "4)), ('s0', [96.0, 200.0, 304.0], (8, 8, 8))])"
    ),
    'levels czyx': (
        "(({'s0': [0.0, 0.0, 0.0], 's1': [0.0, 0.0, 0.0]}, {'s0': [8.0, 8.0, 8.0], 's1': [16.0, "
        "16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)}), [('s0', [0.0, 0.0, 0.0], (8, 8, 8)), "
        "('s0', [0.0, 0.0, 0.0], (8, 8, 8)), ('s0', [0.0, 0.0, 0.0], (8, 8, 8)), ('s1', [0.0, 0.0, "
        "0.0], (4, 4, 4)), ('s1', [0.0, 0.0, 0.0], (4, 4, 4)), ('s0', [0.0, 0.0, 0.0], (8, 8, "
        '8))])'
    ),
    'levels float': (
        "(({'s0': [0.0, 0.0, 0.0], 's1': [0.0, 0.0, 0.0]}, {'s0': [5.24, 4.0, 4.0], 's1': [10.48, "
        "8.0, 8.0]}, {'s0': (20, 4, 4), 's1': (10, 2, 2)}), [('s0', [0.0, 0.0, 0.0], (20, 4, 4)), "
        "('s0', [0.0, 0.0, 0.0], (20, 4, 4)), ('s1', [0.0, 0.0, 0.0], (10, 2, 2)), ('s1', [0.0, "
        "0.0, 0.0], (10, 2, 2)), ('s1', [0.0, 0.0, 0.0], (10, 2, 2)), ('s0', [0.0, 0.0, 0.0], (20, "
        '4, 4))])'
    ),
    'levels http': (
        "(({'s0': [96.0, 200.0, 304.0], 's1': [96.0, 200.0, 304.0]}, {'s0': [8.0, 8.0, 8.0], "
        "'s1': [16.0, 16.0, 16.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)}), [('s0', [96.0, 200.0, "
        "304.0], (8, 8, 8)), ('s0', [96.0, 200.0, 304.0], (8, 8, 8)), ('s0', [96.0, 200.0, 304.0], "
        "(8, 8, 8)), ('s1', [96.0, 200.0, 304.0], (4, 4, 4)), ('s1', [96.0, 200.0, 304.0], (4, 4, "
        "4)), ('s0', [96.0, 200.0, 304.0], (8, 8, 8))])"
    ),
    'levels v3': (
        "(({'s0': [8.0, 18.0, 28.0], 's1': [8.0, 18.0, 28.0]}, {'s0': [4.0, 4.0, 4.0], 's1': [8.0, "
        "8.0, 8.0]}, {'s0': (8, 8, 8), 's1': (4, 4, 4)}), [('s0', [8.0, 18.0, 28.0], (8, 8, 8)), "
        "('s1', [8.0, 18.0, 28.0], (4, 4, 4)), ('s1', [8.0, 18.0, 28.0], (4, 4, 4)), ('s1', [8.0, "
        "18.0, 28.0], (4, 4, 4)), ('s1', [8.0, 18.0, 28.0], (4, 4, 4)), ('s0', [8.0, 18.0, 28.0], "
        '(8, 8, 8))])'
    ),
    'levels micrometer v3': (
        "(({'s0': [76.0, 38.0, 38.0], 's1': [-8.0, -4.0, -4.0]}, {'s0': [8.0, 4.0, 4.0], "
        "'s1': [16.0, 8.0, 8.0]}, {'s0': (4, 4, 4), 's1': (2, 2, 2)}), [('s0', [76.0, 38.0, 38.0], "
        "(4, 4, 4)), ('s0', [76.0, 38.0, 38.0], (4, 4, 4)), ('s0', [76.0, 38.0, 38.0], (4, 4, 4)), "
        "('s1', [-8.0, -4.0, -4.0], (2, 2, 2)), ('s1', [-8.0, -4.0, -4.0], (2, 2, 2)), ('s0', "
        '[76.0, 38.0, 38.0], (4, 4, 4))])'
    ),
    'closest v2 group': (
        '[(8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (16.0, 16.0, 16.0), (16.0, 16.0, '
        '16.0), (8.0, 8.0, 8.0)]'
    ),
    'closest v2 level': (
        '[(8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (16.0, 16.0, 16.0), (16.0, 16.0, '
        '16.0), (8.0, 8.0, 8.0)]'
    ),
    'closest v3 level': (
        '[(4.0, 4.0, 4.0), (8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (8.0, 8.0, 8.0), '
        '(4.0, 4.0, 4.0)]'
    ),
    'closest micrometer': (
        '[(8.0, 4.0, 4.0), (8.0, 4.0, 4.0), (8.0, 4.0, 4.0), (16.0, 8.0, 8.0), (16.0, 8.0, 8.0), '
        '(8.0, 4.0, 4.0)]'
    ),
    'closest http': (
        '[(8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (8.0, 8.0, 8.0), (16.0, 16.0, 16.0), (16.0, 16.0, '
        '16.0), (8.0, 8.0, 8.0)]'
    ),
    'closest plain array': '[None, None, None, None, None, None]',
    'idi v2 relabelled': (
        "{'path': '<root>/v2.zarr/em/s0', 'voxel_size': Coordinate(12, 12, 12), "
        "'offset': Coordinate(144, 300, 456), 'roi': Roi(Coordinate(144, 300, 456), Coordinate(96, "
        "96, 96)), 'shape': Coordinate(8, 8, 8), 'chunk_shape': (4, 4, 4), 'axes_names': ['z', "
        "'y', 'x'], 'filetype': 'zarr', 'actual_voxel_size': Coordinate(8, 8, 8), "
        "'requested_voxel_size': Coordinate(12, 12, 12), '_offset_f': ndarray[float64]([144.0, "
        '300.0, 456.0])}'
    ),
    'idi v3': (
        "{'path': '<root>/v3.zarr/s1', 'voxel_size': Coordinate(8, 8, 8), 'offset': Coordinate(8, "
        "18, 28), 'roi': Roi(Coordinate(8, 18, 28), Coordinate(32, 32, 32)), "
        "'shape': Coordinate(4, 4, 4), 'chunk_shape': (2, 2, 2), 'axes_names': ['z', 'y', 'x'], "
        "'filetype': 'zarr', 'actual_voxel_size': Coordinate(8, 8, 8), "
        "'requested_voxel_size': Coordinate(8, 8, 8), '_offset_f': ndarray[float64]([8.0, 18.0, "
        '28.0])}'
    ),
    'idi http': (
        "{'path': 'http:/v2.zarr/em/s1', 'voxel_size': Coordinate(16, 16, 16), "
        "'offset': Coordinate(96, 200, 304), 'roi': Roi(Coordinate(96, 200, 304), Coordinate(64, "
        "64, 64)), 'shape': Coordinate(4, 4, 4), 'chunk_shape': (2, 2, 2), 'axes_names': ['z', "
        "'y', 'x'], 'filetype': 'zarr', 'actual_voxel_size': Coordinate(16, 16, 16), "
        "'requested_voxel_size': Coordinate(16, 16, 16), '_offset_f': ndarray[float64]([96.0, "
        '200.0, 304.0])}'
    ),
    'idi n5': (
        "{'path': '<root>/a.n5/pixres_off', 'voxel_size': Coordinate(8, 8, 4), "
        "'offset': Coordinate(8, 24, 32), 'roi': Roi(Coordinate(8, 24, 32), Coordinate(80, 160, "
        "120)), 'shape': Coordinate(10, 20, 30), 'chunk_shape': (5, 10, 15), 'axes_names': ['z', "
        "'y', 'x'], 'filetype': 'n5', 'actual_voxel_size': Coordinate(8, 8, 4), "
        "'requested_voxel_size': None, '_offset_f': ndarray[float64]([8.0, 24.0, 32.0])}"
    ),
    'idi precomputed': (
        "{'path': 'precomputed://<root>/pc', 'voxel_size': Coordinate(16, 8, 4), "
        "'offset': Coordinate(0, 0, 0), 'roi': Roi(Coordinate(0, 0, 0), Coordinate(32, 80, 80)), "
        "'shape': Coordinate(2, 10, 20), 'chunk_shape': (2, 5, 10), 'axes_names': ['z', 'y', 'x'], "
        "'filetype': 'precomputed', 'actual_voxel_size': Coordinate(16, 8, 4), "
        "'requested_voxel_size': None, '_offset_f': ndarray[float64]([0.0, 0.0, 0.0])}"
    ),
    'idi float': (
        "{'path': '<root>/float.zarr/s0', 'voxel_size': Coordinate(10, 8, 8), "
        "'offset': Coordinate(0, 0, 0), 'roi': Roi(Coordinate(0, 0, 0), Coordinate(200, 32, 32)), "
        "'shape': Coordinate(20, 4, 4), 'chunk_shape': (10, 2, 2), 'axes_names': ['z', 'y', 'x'], "
        "'filetype': 'zarr', 'actual_voxel_size': (5.24, 4.0, 4.0), "
        "'requested_voxel_size': Coordinate(10, 8, 8), '_offset_f': ndarray[float64]([0.0, 0.0, "
        '0.0])}'
    ),
    'crop v2 group': (
        "(('s0',), ndarray[float64]([1.0, 8.0, 8.0, 8.0]), ndarray[float64]([-0.5, 0.0, 0.0, "
        '0.0]))'
    ),
    'crop v3 group': (
        "(('s0',), ndarray[float64]([0.008, 0.004, 0.004]), ndarray[float64]([0.076, 0.038, "
        '0.038]))'
    ),
    'crop transform': '((), ndarray[float64]([2.0, 2.0, 2.0]), ndarray[float64]([1.0, 1.0, 1.0]))',
    'crop resolution': '((), ndarray[float64]([8.0, 8.0, 8.0]), ndarray[float64]([4.0, 4.0, 4.0]))',
    'crop s0 only': (
        "(('s0',), ndarray[float64]([1.0, 1.0, 1.0]), ndarray[float64]([0.0, 0.0, 0.0]))"
    ),
    'raw layer v3': (
        "({'z': float64(4e-09), 'y': float64(4e-09), 'x': float64(4e-09)}, [[1.0, 0.0, 0.0, 2.0], "
        '[0.0, 1.0, 0.0, 4.5], [0.0, 0.0, 1.0, 7.0]])'
    ),
    'raw layer precomputed': (
        "({'z': float64(1.6e-08), 'y': float64(8e-09), 'x': float64(4e-09)}, [[1.0, 0.0, 0.0, "
        '0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])'
    ),
}
