"""Read support for Zarr **v3**-format stores (``zarr.json`` metadata).

The ``zarr`` package pinned in this project (2.18.4) has no awareness of the
Zarr v3 storage format at all — ``zarr.open()`` cannot even recognize a
``zarr.json``-only store, let alone read one. Bumping to zarr-python 3.x was
considered and rejected (it drops ``zarr.n5``, used elsewhere in this
codebase, and conflicts with the installed ``funlib.persistence`` pin).

Instead, this module reads v3 metadata directly with ``json.load`` and reads
v3 chunk data via ``tensorstore``'s ``zarr3`` driver, which is already a
project dependency. It mirrors the output *contracts* of the equivalent v2
helpers in :mod:`cellmap_flow.utils.ds` so callers can dispatch on format and
otherwise not care which one they got. Local filesystem stores only — remote
(s3/gs/http) v3 stores are not handled here.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from funlib.geometry import Coordinate, Roi

logger = logging.getLogger(__name__)


ZARR_JSON = "zarr.json"

# Everything downstream works in nanometers.
_NM_PER_UNIT = {
    "nanometer": 1.0,
    "nm": 1.0,
    "micrometer": 1e3,
    "micron": 1e3,
    "um": 1e3,
    "µm": 1e3,
    "millimeter": 1e6,
    "mm": 1e6,
    "centimeter": 1e7,
    "cm": 1e7,
    "meter": 1e9,
    "m": 1e9,
    "angstrom": 0.1,
    "å": 0.1,
    "picometer": 1e-3,
    "pm": 1e-3,
}
_CHANNEL_AXIS_NAMES = ("c", "c^", "channel")
_NON_SPATIAL_AXIS_NAMES = _CHANNEL_AXIS_NAMES + ("t", "time")


def nm_per_unit(unit) -> float:
    """How many nanometers one ``unit`` is; 1 for a missing or unknown unit."""
    if unit is None:
        return 1.0
    key = str(unit).strip().lower()
    if key in ("", "pixel", "pixels"):
        return 1.0
    factor = _NM_PER_UNIT.get(key)
    if factor is None:
        if key not in _warned_units:
            _warned_units.add(key)
            logger.warning(f"Unknown spatial unit {unit!r}; treating it as nanometers")
        return 1.0
    return factor


_warned_units = set()


def spatial_axes(axes):
    """Indices, names and units of the spatial axes of an OME ``axes`` list.

    Returns ``(None, None, None)`` when there is no axes list. Axes typed
    "space" are spatial; untyped ones are unless named like a channel or time
    axis (OME 0.3 lists bare names).
    """
    if not axes:
        return None, None, None
    indices, names, units = [], [], []
    for i, axis in enumerate(axes):
        if isinstance(axis, str):
            name, kind, unit = axis, None, None
        else:
            name, kind, unit = axis.get("name"), axis.get("type"), axis.get("unit")
        if kind == "space" or (kind is None and name not in _NON_SPATIAL_AXIS_NAMES):
            indices.append(i)
            names.append(name)
            units.append(unit)
    if not indices:
        return None, None, None
    return indices, names, units


def to_nm(values, units):
    """``values`` (one per spatial axis) converted to nanometers."""
    if units is None:
        return [float(v) for v in values]
    return [float(v) * nm_per_unit(u) for v, u in zip(values, units)]


def ome_corner(translation, scale):
    """The lower corner of voxel 0, from an OME-NGFF ``translation``.

    OME-NGFF places ``translation`` at the *centre* of voxel 0 (Neuroglancer's
    ome.ts subtracts half a voxel for the same reason), while everything in
    cellmap-flow works with corners. Janelia pyramids store
    ``translation = scale/2 - 4`` per level, so every level's corner is -4 nm;
    read as a corner, each level sat half its voxel off (s1 8 nm, s2 16 nm)
    and the levels did not even agree with each other. ``translation`` and
    ``scale`` must be in the same units.
    """
    return [float(t) - float(s) / 2 for t, s in zip(translation, scale)]


def ome_translation(corner, scale):
    """The OME-NGFF ``translation`` (voxel-0 centre) for a lower ``corner``."""
    return [float(c) + float(s) / 2 for c, s in zip(corner, scale)]


# Unit conversion leaves float noise (0.009 um -> 8.999999999999998 nm);
# anything this close to a whole number is that number.
_INTEGRAL_TOLERANCE = 1e-6

_warned_non_integral = set()


def snap_integral(values):
    """``values`` as floats, with near-whole numbers made exactly whole."""
    arr = np.asarray(values, dtype=float)
    rounded = np.round(arr)
    close = np.abs(arr - rounded) <= _INTEGRAL_TOLERANCE * np.maximum(1.0, np.abs(arr))
    return np.where(close, rounded, arr)


def is_integral(values) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(np.all(snap_integral(arr) == np.round(arr)))


def coordinate_or_floats(values, what="voxel size", where=""):
    """A Coordinate when every value is a whole number, else a tuple of floats.

    Coordinate truncates: Coordinate(5.24) is 5, and a dataset at 5.24 nm
    then read 5% of the wrong voxels. Non-integer values are kept as floats
    (with a one-time warning) instead.
    """
    if values is None:
        return None
    snapped = snap_integral(values)
    if is_integral(snapped):
        return Coordinate(int(v) for v in snapped)
    key = (what, where, tuple(snapped.tolist()))
    if key not in _warned_non_integral:
        _warned_non_integral.add(key)
        logger.warning(
            f"{where or 'dataset'}: {what} {tuple(snapped.tolist())} is not a whole "
            "number of nanometers; keeping it as floats"
        )
    return tuple(float(v) for v in snapped)


def same_voxel_size(a, b) -> bool:
    """Equal as nanometer floats. Coordinate() truncated both sides, so a
    target of (10, 8, 8) matched an actual (10.48, 8, 8)."""
    a, b = snap_integral(a), snap_integral(b)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-6, atol=1e-9))


def coarser_anywhere(resolution, target) -> bool:
    """``resolution`` is coarser than ``target`` along some axis."""
    return any(
        r > t and not np.isclose(r, t, rtol=1e-6, atol=1e-9)
        for r, t in zip(snap_integral(resolution), snap_integral(target))
    )


def covering_roi(offset, voxel_size, shape):
    """The integer-nm Roi covering ``shape`` voxels at ``offset``.

    Exactly Roi(offset, voxel_size * shape) when everything is integral.
    """
    begin = snap_integral(offset)
    end = snap_integral(begin + snap_integral(voxel_size) * np.asarray(shape, dtype=float))
    begin, end = np.floor(begin).astype(int), np.ceil(end).astype(int)
    return Roi(Coordinate(begin), Coordinate(end - begin))


def is_v3_container(path: str) -> bool:
    """True if ``path`` is a directory with a ``zarr.json`` at its root."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, ZARR_JSON))


def find_v3_container(path: str) -> Optional[str]:
    """Walk up from ``path`` looking for the nearest directory containing a
    ``zarr.json``. Returns ``None`` if none is found (e.g. a v2 store, or a
    remote URL)."""
    if path.startswith("http://") or path.startswith("https://") or "://" in path:
        return None
    normalized = os.path.normpath(path)
    current = normalized
    while current and current != os.path.dirname(current):
        if is_v3_container(current):
            return current
        current = os.path.dirname(current)
    return None


def read_zarr_json(path: str) -> dict:
    with open(os.path.join(path, ZARR_JSON)) as f:
        return json.load(f)


def attrs_from_meta(meta: dict) -> dict:
    """Merge OME-NGFF 0.5's nested ``attributes.ome`` with the top-level
    ``attributes`` dict (some writers emit OME metadata unnested)."""
    attributes = meta.get("attributes", {}) or {}
    merged = dict(attributes)
    ome = attributes.get("ome")
    if isinstance(ome, dict):
        merged.update(ome)
    return merged


def _child_names(group_path: str) -> list:
    return sorted(
        name
        for name in os.listdir(group_path)
        if os.path.isdir(os.path.join(group_path, name))
        and is_v3_container(os.path.join(group_path, name))
    )


class _TensorstoreArray:
    """Adapter so a v3 array reads like the ``zarr.Array`` callers expect:
    ``arr[:]``/``arr[a:b]`` returns a real ``numpy.ndarray``."""

    def __init__(self, ts_obj):
        self._ts = ts_obj

    @property
    def shape(self):
        return tuple(self._ts.shape)

    @property
    def ndim(self):
        return self._ts.ndim

    @property
    def dtype(self):
        return np.dtype(self._ts.dtype.numpy_dtype)

    @property
    def chunks(self):
        return tuple(self._ts.chunk_layout.read_chunk.shape)

    def __getitem__(self, key):
        return np.asarray(self._ts[key].read().result())


def open_array_v3(path: str) -> _TensorstoreArray:
    """Open a v3 array for reading. ``path`` must be the array's own
    directory (i.e. already descended into, not its parent group)."""
    import tensorstore as ts

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": os.path.normpath(path)},
    }
    ts_obj = ts.open(spec, read=True, write=False).result()
    return _TensorstoreArray(ts_obj)


def multiscales_from_group(group_path: str) -> Optional[dict]:
    meta = read_zarr_json(group_path)
    if meta.get("node_type") != "group":
        return None
    attrs = attrs_from_meta(meta)
    multiscales = attrs.get("multiscales")
    if not multiscales:
        return None
    return multiscales[0]


def get_scale_info_v3(group_path: str) -> Tuple[dict, dict, dict]:
    """Mirror of ``ds.py``'s ``get_scale_info`` for a v3 multiscale group.

    Returns ``(offsets, resolutions, shapes)`` keyed by dataset path.
    """
    ms = multiscales_from_group(group_path)
    if ms is None:
        raise ValueError(f"No multiscales attribute found at {group_path}")

    spatial_indices, _, units = spatial_axes(ms.get("axes", []))

    offsets, resolutions, shapes = {}, {}, {}
    for scale in ms["datasets"]:
        transforms = scale["coordinateTransformations"]
        full_res = next(t["scale"] for t in transforms if t["type"] == "scale")
        full_translation = next(
            (t["translation"] for t in transforms if t["type"] == "translation"),
            [0.0] * len(full_res),
        )
        array_path = os.path.join(group_path, scale["path"])
        full_shape = read_zarr_json(array_path)["shape"]

        if spatial_indices is not None:
            resolutions[scale["path"]] = to_nm([full_res[i] for i in spatial_indices], units)
            offsets[scale["path"]] = ome_corner(
                to_nm([full_translation[i] for i in spatial_indices], units),
                resolutions[scale["path"]],
            )
            shapes[scale["path"]] = tuple(full_shape[i] for i in spatial_indices)
        else:
            resolutions[scale["path"]] = full_res
            offsets[scale["path"]] = ome_corner(full_translation, full_res)
            shapes[scale["path"]] = tuple(full_shape)
    return offsets, resolutions, shapes


def find_closest_scale_v3(group_path: str, target_resolution) -> Tuple[str, list, tuple]:
    """Mirror of ``ds.py``'s ``find_closest_scale`` for a v3 multiscale group.

    ``target_resolution=None`` defaults to the finest (first) scale rather
    than raising, since callers sometimes ask for the "closest scale" without
    a specific target in mind.
    """
    offsets, resolutions, shapes = get_scale_info_v3(group_path)
    if target_resolution is None:
        target_scale = next(iter(resolutions))
        return target_scale, offsets[target_scale], shapes[target_scale]

    target_scale = None
    last_scale = None
    for scale, res in resolutions.items():
        if last_scale is None:
            last_scale = scale
        if same_voxel_size(res, target_resolution):
            target_scale = scale
            break
        elif coarser_anywhere(res, target_resolution):
            target_scale = last_scale
            break
        last_scale = scale
    if target_scale is None:
        target_scale = last_scale
    return target_scale, offsets[target_scale], shapes[target_scale]


def _ds_info_from_group_dataset(group_path: str, ms: dict, dataset_entry: dict):
    """Metadata of one dataset entry of a multiscale group's ``datasets`` list:
    ``(voxel_size, offset, chunk_shape, shape, axes_names, "zarr")`` with
    voxel size and offset as nanometer floats."""
    spatial_indices, spatial_names, units = spatial_axes(ms.get("axes", []))

    array_path = os.path.join(group_path, dataset_entry["path"])
    arr_meta = read_zarr_json(array_path)

    transforms = dataset_entry["coordinateTransformations"]
    scale = next(t["scale"] for t in transforms if t["type"] == "scale")
    translation = next(
        (t["translation"] for t in transforms if t["type"] == "translation"),
        [0.0] * len(scale),
    )
    chunk_shape = tuple(arr_meta["chunk_grid"]["configuration"]["chunk_shape"])
    if spatial_indices is not None:
        voxel_size = to_nm([scale[i] for i in spatial_indices], units)
        offset = ome_corner(to_nm([translation[i] for i in spatial_indices], units), voxel_size)
        shape = tuple(arr_meta["shape"][i] for i in spatial_indices)
        axes_names = spatial_names
        # Spatial like the shape: a (c, z, y, x) array reported a 4-D chunk
        # shape against a 3-D shape.
        chunk_shape = tuple(chunk_shape[i] for i in spatial_indices)
    else:
        voxel_size = [float(v) for v in scale]
        offset = ome_corner(translation, scale)
        shape = tuple(arr_meta["shape"])
        axes_names = ["z", "y", "x"][-len(shape):]
    return voxel_size, offset, chunk_shape, shape, axes_names, "zarr"


def _ds_info_from_plain_array(meta: dict):
    """Metadata of an array with no ancestor multiscale group referencing it:
    look for `transform`/`resolution` attrs, default to unit scale/zero
    offset otherwise (mirrors crop_loader.py's array handling)."""
    attrs = attrs_from_meta(meta)
    shape = tuple(meta["shape"])
    if "transform" in attrs:
        tx = attrs["transform"]
        voxel_size = tx.get("scale", [1] * len(shape))
        offset = tx.get("translate", [0] * len(shape))
    elif "resolution" in attrs:
        voxel_size = attrs["resolution"]
        offset = attrs.get("offset", [0] * len(shape))
    else:
        voxel_size = [1] * len(shape)
        offset = [0] * len(shape)

    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    axes_names = ["z", "y", "x"][-len(shape):]
    return (
        [float(v) for v in voxel_size],
        [float(v) for v in offset],
        chunk_shape,
        shape,
        axes_names,
        "zarr",
    )


def legacy_ds_info(meta, where=""):
    """``get_ds_info``'s ``(voxel_size, chunk_shape, shape, roi, axes_names,
    filetype)`` from a ``(voxel_size, offset, chunk_shape, shape, axes_names,
    filetype)`` metadata tuple with float voxel size and offset."""
    voxel_size, offset, chunk_shape, shape, axes_names, filetype = meta
    return (
        coordinate_or_floats(voxel_size, "voxel size", where),
        chunk_shape,
        Coordinate(shape),
        covering_roi(offset, voxel_size, shape),
        axes_names,
        filetype,
    )


def get_ds_info_v3(path: str):
    """Mirror of ``ds.py``'s ``get_ds_info`` return contract for a local v3
    store: ``(voxel_size, chunk_shape, shape, roi, axes_names, "zarr")``.
    """
    return legacy_ds_info(read_ds_meta_v3(path), path)


def read_ds_meta_v3(path: str):
    """``(voxel_size, offset, chunk_shape, shape, axes_names, "zarr")`` for a
    local v3 store, voxel size and offset as nanometer floats."""
    container = find_v3_container(path)
    if container is None:
        raise RuntimeError(f"Could not find a Zarr v3 container in path: {path}")

    meta = read_zarr_json(container)

    if meta.get("node_type") == "array":
        # `path` may point directly at one scale's array inside an ancestor
        # multiscale group (each v3 array has its own zarr.json, so
        # `find_v3_container` stops here rather than continuing up to the
        # group). Check the parent for multiscales metadata that names this
        # array, so its per-scale voxel size/offset is used instead of the
        # array's own (usually absent) attrs.
        parent = os.path.dirname(os.path.normpath(container))
        if is_v3_container(parent):
            parent_meta = read_zarr_json(parent)
            if parent_meta.get("node_type") == "group":
                ms = multiscales_from_group(parent)
                if ms is not None:
                    rel = os.path.basename(os.path.normpath(container))
                    dataset_entry = next(
                        (d for d in ms["datasets"] if d["path"].lstrip("/") == rel),
                        None,
                    )
                    if dataset_entry is not None:
                        return _ds_info_from_group_dataset(parent, ms, dataset_entry)
        return _ds_info_from_plain_array(meta)

    # Group: descend to the matching (or first) multiscale dataset.
    ms = multiscales_from_group(container)
    if ms is not None:
        rel = os.path.relpath(path, container)
        dataset_entry = next(
            (d for d in ms["datasets"] if d["path"].lstrip("/") == rel),
            ms["datasets"][0],
        )
        return _ds_info_from_group_dataset(container, ms, dataset_entry)

    # Group with no multiscales: descend into a single array child
    # (mirrors ds.py's fallback of picking the first array key).
    children = _child_names(container)
    for name in children:
        child_meta = read_zarr_json(os.path.join(container, name))
        if child_meta.get("node_type") == "array":
            return read_ds_meta_v3(os.path.join(container, name))
    raise RuntimeError(f"No array found under Zarr v3 group: {container}")
