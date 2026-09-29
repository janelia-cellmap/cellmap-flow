"""Read support for Zarr **v3**-format stores (``zarr.json`` metadata).

The ``zarr`` package pinned in this project (2.18.4) has no awareness of the
Zarr v3 storage format at all — ``zarr.open()`` cannot even recognize a
``zarr.json``-only store, let alone read one. Bumping to zarr-python 3.x was
considered and rejected (it drops ``zarr.n5``, used elsewhere in this
codebase, and conflicts with the installed ``funlib.persistence`` pin).

v3 metadata is read with ``json.load`` in :mod:`cellmap_flow.io.metadata`,
which reads every format into one ``ArrayMeta``; the functions here keep the
old tuple-returning contracts over it. Chunk data is read through
``tensorstore``'s ``zarr3`` driver. Local filesystem stores only — remote
(s3/gs/http) v3 stores are not handled.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np
from funlib.geometry import Coordinate, Roi

from cellmap_flow.io import metadata, paths
from cellmap_flow.io.metadata import (  # noqa: F401  (kept names; see io.metadata)
    attrs_from_meta,
    is_integral,
    multiscales_from_group,
    nm_per_unit,
    read_zarr_json,
    snap_integral,
    spatial_axes,
)
from cellmap_flow.io.multiscale import (  # noqa: F401  (kept names; see io.multiscale)
    coarser_anywhere,
    same_voxel_size,
    select_level,
)
from cellmap_flow.io.ome import ome_corner, ome_translation  # noqa: F401  (kept names)
from cellmap_flow.io.paths import (  # noqa: F401  (kept names; see io.paths)
    ZARR_JSON,
    find_v3_container,
    is_v3_container,
)

logger = logging.getLogger(__name__)


def to_nm(values, units):
    """``values`` (one per spatial axis) converted to nanometers."""
    return list(metadata.to_nm(values, units))


_warned_non_integral = set()


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


def covering_roi(offset, voxel_size, shape):
    """The integer-nm Roi covering ``shape`` voxels at ``offset``.

    Exactly Roi(offset, voxel_size * shape) when everything is integral.
    """
    begin = snap_integral(offset)
    end = snap_integral(begin + snap_integral(voxel_size) * np.asarray(shape, dtype=float))
    begin, end = np.floor(begin).astype(int), np.ceil(end).astype(int)
    return Roi(Coordinate(begin), Coordinate(end - begin))


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


def scale_info(levels):
    """``(offsets, resolutions, shapes)`` keyed by level path, spatial axes
    only, from ``io.metadata.list_levels`` output."""
    offsets, resolutions, shapes = {}, {}, {}
    for level, meta in levels:
        spatial = meta.spatial()
        resolutions[level] = list(spatial.voxel_size)
        offsets[level] = list(spatial.translation)
        shapes[level] = tuple(spatial.shape)
    return offsets, resolutions, shapes


def get_scale_info_v3(group_path: str) -> Tuple[dict, dict, dict]:
    """Mirror of ``ds.py``'s ``get_scale_info`` for a v3 multiscale group.

    Returns ``(offsets, resolutions, shapes)`` keyed by dataset path.
    """
    if multiscales_from_group(group_path) is None:
        raise ValueError(f"No multiscales attribute found at {group_path}")
    return scale_info(metadata.list_levels(group_path))


def level_info(level):
    """``(path, offset, shape)`` of a ``(path, ArrayMeta)`` level: spatial
    axes, the offset a list of nanometer floats, as find_closest_scale
    returned them."""
    path, meta = level
    spatial = meta.spatial()
    return path, list(spatial.translation), tuple(spatial.shape)


def find_closest_scale_v3(group_path: str, target_resolution) -> Tuple[str, list, tuple]:
    """Mirror of ``ds.py``'s ``find_closest_scale`` for a v3 multiscale group
    (see io.multiscale.select_level, "floor").

    ``target_resolution=None`` defaults to the finest (first) scale rather
    than raising, since callers sometimes ask for the "closest scale" without
    a specific target in mind.
    """
    if multiscales_from_group(group_path) is None:
        raise ValueError(f"No multiscales attribute found at {group_path}")
    return level_info(select_level(metadata.list_levels(group_path), target_resolution))


def legacy_meta(meta: metadata.ArrayMeta):
    """``(voxel_size, offset, chunk_shape, shape, axes_names, filetype)``, the
    tuple the old readers returned, from an ArrayMeta: spatial axes only,
    voxel size and offset as lists of nanometer floats.

    The local zarr v2/N5 reader always took the *last* n axes (n spatial
    ones) and called them z, y, x, whatever the metadata said; the others
    take the spatial axes by name. Both are kept.
    """
    spatial = meta.spatial()
    n = len(spatial.voxel_size)
    if meta.format in ("zarr2", "n5") and not paths.is_remote(meta.path):
        chunk_shape = tuple(meta.chunk_shape[-n:])
        shape = tuple(meta.shape[-n:])
        axes_names = ["z", "y", "x"][-n:]
    else:
        chunk_shape, shape = spatial.chunk_shape, spatial.shape
        axes_names = [a for a in spatial.axes if a != ""]
    if meta.format == "precomputed":
        filetype = "gs" if meta.path.startswith("gs://") else "precomputed"
    else:
        filetype = "n5" if meta.format == "n5" else "zarr"
    return (
        list(spatial.voxel_size),
        list(spatial.translation),
        chunk_shape,
        shape,
        axes_names,
        filetype,
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
    if find_v3_container(path) is None:
        raise RuntimeError(f"Could not find a Zarr v3 container in path: {path}")
    return legacy_meta(metadata.read_array_meta(path))
