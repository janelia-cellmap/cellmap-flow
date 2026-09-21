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

    axes = ms.get("axes", [])
    spatial_indices = [i for i, a in enumerate(axes) if a.get("type") == "space"]
    if not spatial_indices:
        spatial_indices = None

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
            resolutions[scale["path"]] = [full_res[i] for i in spatial_indices]
            offsets[scale["path"]] = [full_translation[i] for i in spatial_indices]
            shapes[scale["path"]] = tuple(full_shape[i] for i in spatial_indices)
        else:
            resolutions[scale["path"]] = full_res
            offsets[scale["path"]] = full_translation
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
        if Coordinate(res) == Coordinate(target_resolution):
            target_scale = scale
            break
        elif any(r > t for r, t in zip(res, target_resolution)):
            target_scale = last_scale
            break
        last_scale = scale
    if target_scale is None:
        target_scale = last_scale
    return target_scale, offsets[target_scale], shapes[target_scale]


def _ds_info_from_group_dataset(group_path: str, ms: dict, dataset_entry: dict):
    """Build the ``get_ds_info``-contract tuple for one dataset entry of a
    multiscale group's ``datasets`` list."""
    axes = ms.get("axes", [])
    spatial_indices = [i for i, a in enumerate(axes) if a.get("type") == "space"]
    if not spatial_indices:
        spatial_indices = None

    array_path = os.path.join(group_path, dataset_entry["path"])
    arr_meta = read_zarr_json(array_path)

    transforms = dataset_entry["coordinateTransformations"]
    scale = next(t["scale"] for t in transforms if t["type"] == "scale")
    translation = next(
        (t["translation"] for t in transforms if t["type"] == "translation"),
        [0.0] * len(scale),
    )
    if spatial_indices is not None:
        voxel_size = Coordinate(scale[i] for i in spatial_indices)
        offset = Coordinate(translation[i] for i in spatial_indices)
        shape = Coordinate(arr_meta["shape"][i] for i in spatial_indices)
        axes_names = [axes[i]["name"] for i in spatial_indices]
    else:
        voxel_size = Coordinate(scale)
        offset = Coordinate(translation)
        shape = Coordinate(arr_meta["shape"])
        axes_names = ["z", "y", "x"][-len(shape):]
    chunk_shape = tuple(arr_meta["chunk_grid"]["configuration"]["chunk_shape"])
    roi = Roi(offset, voxel_size * shape)
    return voxel_size, chunk_shape, shape, roi, axes_names, "zarr"


def _ds_info_from_plain_array(meta: dict):
    """Build the ``get_ds_info``-contract tuple for an array with no
    ancestor multiscale group referencing it: look for `transform`/
    `resolution` attrs, default to unit scale/zero offset otherwise (mirrors
    crop_loader.py's array handling)."""
    attrs = attrs_from_meta(meta)
    shape = Coordinate(meta["shape"])
    if "transform" in attrs:
        tx = attrs["transform"]
        voxel_size = Coordinate(tx.get("scale", [1] * len(shape)))
        offset = Coordinate(tx.get("translate", [0] * len(shape)))
    elif "resolution" in attrs:
        voxel_size = Coordinate(attrs["resolution"])
        offset = Coordinate(attrs.get("offset", [0] * len(shape)))
    else:
        voxel_size = Coordinate([1] * len(shape))
        offset = Coordinate([0] * len(shape))

    chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    roi = Roi(offset, voxel_size * shape)
    axes_names = ["z", "y", "x"][-len(shape):]
    return voxel_size, chunk_shape, shape, roi, axes_names, "zarr"


def get_ds_info_v3(path: str):
    """Mirror of ``ds.py``'s ``get_ds_info`` return contract for a local v3
    store: ``(voxel_size, chunk_shape, shape, roi, axes_names, "zarr")``.
    """
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
            return get_ds_info_v3(os.path.join(container, name))
    raise RuntimeError(f"No array found under Zarr v3 group: {container}")
