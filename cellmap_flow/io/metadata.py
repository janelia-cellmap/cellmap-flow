"""Array metadata for every format cellmap-flow reads, as one ``ArrayMeta``.

``read_array_meta(path)`` reads zarr v2 (local, and http(s)/s3 through
fsspec), zarr v3 (local), N5 and neuroglancer precomputed. Every axis is
kept, in the array's own (C) order; sizes and translations are nanometer
floats, and the translation is the lower corner of voxel 0 -- an OME
translation (voxel 0's centre) is converted exactly, with no rounding onto
the voxel grid. ``.spatial()`` drops the channel/time axes.

Four parsers produce it:

- ``_ome``: an OME-NGFF multiscales entry (v2 ``.zattrs`` and v3
  ``zarr.json`` alike), matched on ``datasets[].path``.
- ``_n5``: the N5/funlib attribute search -- ``resolution``, ``scale``,
  ``pixelResolution`` x ``downsamplingFactors``, ``transform``, ``offset``,
  ``units`` -- on an array and its parent group, with the axis reversal N5
  needs. zarr v2 arrays without OME metadata go through it too, and keep
  its rounding of the offset onto the voxel grid (``regularize_offset``).
- ``legacy_attrs``: ``transform`` or ``resolution``/``offset`` on a v3
  array itself (and on crops), taken as written.
- ``_precomputed``: the volume's ``info`` JSON, as tensorstore reads it.

``list_levels(path)`` gives the levels of an OME multiscale group, or the
scales of a precomputed volume.

Each format's reader keeps the lookup order and fallbacks of the
per-format reader it replaced; tests/utils/test_io_metadata.py pins them.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import zarr
from funlib.geometry import Coordinate

from cellmap_flow.io import paths
from cellmap_flow.io.ome import CHANNEL_AXIS_NAMES, ome_corner

logger = logging.getLogger(__name__)

NANOMETER = "nanometer"

# ---------------------------------------------------------------------------
# Units and axes
# ---------------------------------------------------------------------------

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
_NON_SPATIAL_AXIS_NAMES = CHANNEL_AXIS_NAMES + ("t", "time")

_warned_units = set()


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


def to_nm(values, units) -> Tuple[float, ...]:
    """``values`` (one per axis) in nanometers; ``units`` None means nm."""
    if units is None:
        return tuple(float(v) for v in values)
    return tuple(float(v) * nm_per_unit(u) for v, u in zip(values, units))


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


def _axis_name(axis) -> str:
    return axis if isinstance(axis, str) else (axis.get("name") or "")


def _unnamed_spatial(n: int) -> Tuple[str, ...]:
    """Names for ``n`` spatial axes that were not named: z, y, x from the
    end, and "" for any axis before z."""
    return ("",) * max(0, n - 3) + ("z", "y", "x")[-n:] if n else ()


# ---------------------------------------------------------------------------
# Float noise
# ---------------------------------------------------------------------------

# Unit conversion leaves float noise (0.009 um -> 8.999999999999998 nm);
# anything this close to a whole number is that number.
_INTEGRAL_TOLERANCE = 1e-6


def snap_integral(values):
    """``values`` as floats, with near-whole numbers made exactly whole."""
    arr = np.asarray(values, dtype=float)
    rounded = np.round(arr)
    close = np.abs(arr - rounded) <= _INTEGRAL_TOLERANCE * np.maximum(1.0, np.abs(arr))
    return np.where(close, rounded, arr)


def is_integral(values) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(np.all(snap_integral(arr) == np.round(arr)))


def regularize_offset(voxel_size_float, offset_float):
    """
        offset is not a multiple of voxel_size. This is often due to someone defining
        offset to the point source of each array element i.e. the center of the rendered
        voxel, vs the offset to the corner of the voxel.
        apparently this can be a heated discussion. See here for arguments against
        the convention we are using: http://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf

    Only the legacy (non-OME) attributes go through this; an OME translation
    is converted to a corner exactly instead.

    Args:
        voxel_size_float ([float]): float voxel size list
        offset_float ([float]): float offset list
    Returns:
        (Coordinate, Coordinate)): returned offset size that is multiple of voxel size.
        For a non-integer voxel size, two tuples of floats instead (the same
        rounding, without truncating the voxel size to an integer first).
    """
    snapped_voxel_size = snap_integral(voxel_size_float)
    if not is_integral(snapped_voxel_size):
        vs = snapped_voxel_size
        off = snap_integral(offset_float)
        if not np.allclose(np.round(off / vs) * vs, off):
            logger.debug(f"Offset: {off} being rounded to nearest voxel size: {vs}")
            off = snap_integral(np.trunc((off + vs / 2) / vs) * vs)
        return tuple(float(v) for v in vs), tuple(float(v) for v in off)

    voxel_size = Coordinate(int(v) for v in snapped_voxel_size)
    offset = Coordinate(offset_float)

    if voxel_size is not None and (offset / voxel_size) * voxel_size != offset:

        logger.debug(
            f"Offset: {offset} being rounded to nearest voxel size: {voxel_size}"
        )
        offset = (
            (Coordinate(offset) + (Coordinate(voxel_size) / 2)) / Coordinate(voxel_size)
        ) * Coordinate(voxel_size)
        logger.debug(f"Rounded offset: {offset}")

    return Coordinate(voxel_size), Coordinate(offset)


# ---------------------------------------------------------------------------
# ArrayMeta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrayMeta:
    """What one array is, independent of its format.

    Every per-axis tuple has one entry per array dimension, in the array's
    (C) order. ``units[i]`` is "nanometer" on a spatial axis -- its
    ``voxel_size`` and ``translation`` have been converted to nm -- and None
    on a channel or time axis, where ``voxel_size`` is 1.0 and
    ``translation`` 0.0. ``translation`` is the lower corner of voxel 0.
    An axis nobody named has the name "".

    ``format`` is "zarr2", "zarr3", "n5" or "precomputed".
    """

    path: str
    format: str
    shape: Tuple[int, ...]
    dtype: Optional[np.dtype]
    chunk_shape: Tuple[int, ...]
    axes: Tuple[str, ...]
    units: Tuple[Optional[str], ...]
    voxel_size: Tuple[float, ...]
    translation: Tuple[float, ...]
    fill_value: object = None

    @property
    def spatial_indices(self) -> Tuple[int, ...]:
        return tuple(i for i, unit in enumerate(self.units) if unit is not None)

    @property
    def channel_axis(self) -> Optional[int]:
        """The first non-spatial axis that is not time, or None."""
        for i, (name, unit) in enumerate(zip(self.axes, self.units)):
            if unit is None and name not in ("t", "time"):
                return i
        return None

    def spatial(self) -> "ArrayMeta":
        """This array's metadata with only its spatial axes."""
        keep = self.spatial_indices

        def pick(values):
            return tuple(values[i] for i in keep)

        return replace(
            self,
            shape=pick(self.shape),
            chunk_shape=pick(self.chunk_shape),
            axes=pick(self.axes),
            units=pick(self.units),
            voxel_size=pick(self.voxel_size),
            translation=pick(self.translation),
        )


def _snapped(meta: ArrayMeta) -> ArrayMeta:
    return replace(
        meta,
        voxel_size=tuple(float(v) for v in snap_integral(meta.voxel_size)),
        translation=tuple(float(v) for v in snap_integral(meta.translation)),
    )


@dataclass(frozen=True)
class _Header:
    """An array's own shape/dtype/chunks, from whatever store holds it."""

    shape: Tuple[int, ...]
    dtype: Optional[np.dtype]
    chunks: Tuple[int, ...]
    fill_value: object = None


def _zarr_header(array) -> _Header:
    # A zarr Group has none of these, so asking for a group's metadata raises
    # AttributeError, as it always has.
    return _Header(tuple(array.shape), array.dtype, tuple(array.chunks), array.fill_value)


def _v3_dtype(data_type):
    try:
        return np.dtype(data_type)
    except TypeError:
        return None


def _v3_header(meta: dict) -> _Header:
    # For a sharded array this is the shard shape, as it always was.
    return _Header(
        tuple(meta["shape"]),
        _v3_dtype(meta.get("data_type")),
        tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]),
        meta.get("fill_value"),
    )


def _meta(path, fmt, header, spatial, names, voxel_size, translation) -> ArrayMeta:
    """An ArrayMeta whose axes ``spatial`` (indices) carry ``voxel_size`` and
    ``translation`` (nm, one per spatial axis) and are named ``names``."""
    ndim = len(header.shape)
    by_axis = {i: k for k, i in enumerate(spatial)}
    axes, units, vs, tr = [], [], [], []
    for i in range(ndim):
        k = by_axis.get(i)
        if k is None:
            axes.append(names.get(i, "") if isinstance(names, dict) else "")
            units.append(None)
            vs.append(1.0)
            tr.append(0.0)
        else:
            axes.append(names[i] if isinstance(names, dict) else names[k])
            units.append(NANOMETER)
            vs.append(float(voxel_size[k]))
            tr.append(float(translation[k]))
    return ArrayMeta(
        path=path,
        format=fmt,
        shape=header.shape,
        dtype=header.dtype,
        chunk_shape=header.chunks,
        axes=tuple(axes),
        units=tuple(units),
        voxel_size=tuple(vs),
        translation=tuple(tr),
        fill_value=header.fill_value,
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def match_dataset(multiscale: dict, dataset_path: str) -> Optional[dict]:
    """The ``datasets`` entry of one multiscales item whose path is
    ``dataset_path`` (leading and trailing "/" ignored), or None."""
    want = dataset_path.strip("/")
    return next(
        (d for d in multiscale["datasets"] if d["path"].strip("/") == want), None
    )


def dataset_transforms(entry: dict):
    """``(scale, translation)`` of an OME ``datasets`` entry as written, each
    None when the entry has none."""
    scale = translation = None
    for transform in entry.get("coordinateTransformations", []):
        if transform.get("type") == "scale":
            scale = transform["scale"]
        elif transform.get("type") == "translation":
            translation = transform["translation"]
    return scale, translation


def _ome(multiscale: dict, entry: dict, header: _Header, path: str, fmt: str) -> ArrayMeta:
    """ArrayMeta for ``entry`` of the OME ``multiscale`` item (one element of
    ``multiscales``); ``header`` is the entry's array.

    Units are converted to nm before the centre-to-corner step. Without an
    ``axes`` list every axis is spatial.
    """
    ndim = len(header.shape)
    spatial, spatial_names, units = spatial_axes(multiscale.get("axes", []))
    scale, translation = dataset_transforms(entry)
    if scale is None:
        raise KeyError(f"dataset {entry.get('path')!r} has no scale transformation")
    if translation is None:
        translation = [0.0] * len(scale)
    if spatial is None:
        spatial = list(range(ndim))
        names = _unnamed_spatial(ndim)
    else:
        declared = multiscale.get("axes", [])
        names = {i: _axis_name(declared[i]) for i in range(min(ndim, len(declared)))}
        names.update(dict(zip(spatial, spatial_names)))
    voxel_size = to_nm([scale[i] for i in spatial], units)
    corner = ome_corner(to_nm([translation[i] for i in spatial], units), voxel_size)
    return _meta(path, fmt, header, spatial, names, voxel_size, corner)


def _reverse(values):
    return values[::-1]


def n5_voxel_size(items: Sequence[dict], order: str):
    """The voxel size in the first of ``items`` (array attrs, then its
    parent's) that has one: ``resolution``, ``scale``, ``pixelResolution``
    times ``downsamplingFactors``, or ``transform.scale``.

    ``order`` is the axis order the caller reads in ("C", or "F" for N5).
    ``transform`` states its own ``ordering`` (C unless it says otherwise:
    Davis writes C order whatever the store) and is reversed to ``order``.
    """
    for attrs in items:
        if "resolution" in attrs:
            return attrs["resolution"]
        elif "scale" in attrs:
            return attrs["scale"]
        elif "pixelResolution" in attrs:
            downsampling_factors = [1, 1, 1]
            if "downsamplingFactors" in attrs:
                downsampling_factors = attrs["downsamplingFactors"]
            if "dimensions" not in attrs["pixelResolution"]:
                base_resolution = attrs["pixelResolution"]
            else:
                base_resolution = attrs["pixelResolution"]["dimensions"]
            return list(np.array(base_resolution) * np.array(downsampling_factors))
        elif "transform" in attrs:
            voxel_size = attrs["transform"]["scale"]
            if attrs["transform"].get("ordering", "C") != order:
                voxel_size = _reverse(voxel_size)
            return voxel_size
    return None


def n5_offset(items: Sequence[dict], order: str):
    """``offset``, or ``transform.translate``, as ``n5_voxel_size`` finds it."""
    for attrs in items:
        if "offset" in attrs:
            return attrs["offset"]
        elif "transform" in attrs:
            offset = attrs["transform"]["translate"]
            if attrs["transform"].get("ordering", "C") != order:
                offset = _reverse(offset)
            return offset
    return None


def n5_units(items: Sequence[dict], order: str, ndim: int):
    """``units``, ``pixelResolution.unit`` or ``transform.units``, one per
    axis (see _per_axis); ``ndim`` "pixels" when there are none."""
    for attrs in items:
        if "units" in attrs:
            return _per_axis(attrs["units"], ndim)
        elif "pixelResolution" in attrs and "unit" in attrs["pixelResolution"]:
            return [attrs["pixelResolution"]["unit"]] * ndim
        elif "transform" in attrs:
            units = _per_axis(attrs["transform"]["units"], ndim)
            if attrs["transform"].get("ordering", "C") != order:
                units = _reverse(units)
            return units
    return ["pixels"] * ndim


def _per_axis(units, ndim: int):
    """``units`` as one per axis: a single unit, a string, is every axis's.

    N5's units are reversed with its axes, and that must reverse the list,
    never the letters of a string.
    """
    return [units] * ndim if isinstance(units, str) else units


def _n5_multiscale_level(multiscales, level_path):
    """``transform`` scale/translate/units of the level at ``level_path`` in
    an N5 group's ``multiscales``; Nones when it is not listed."""
    if multiscales is None:
        return None, None, None
    for level in multiscales[0]["datasets"]:
        if level["path"] == level_path:
            transform = level["transform"]
            return transform["scale"], transform["translate"], transform["units"]
    return None, None, None


def _n5(items, order, ndim, is_n5, units=None, multiscales=None, level_path=None):
    """``(voxel_size, offset, units)`` in C order from the N5/funlib attributes.

    ``units`` is used instead of looking them up when given. For N5, what
    the attributes don't give -- the voxel size, the offset, or both -- is
    taken from the level's entry in the group's ``multiscales`` (a voxel
    size with that entry's units); what they do give is kept. BigDataViewer
    and Paintera N5s have a ``pixelResolution`` and no offset. What is still
    missing defaults to voxel size 1 and offset 0.
    """
    voxel_size = n5_voxel_size(items, order)
    offset = n5_offset(items, order)
    if units is None:
        units = n5_units(items, order, ndim)
    if is_n5 and (voxel_size is None or offset is None):
        level_scale, level_offset, level_units = _n5_multiscale_level(multiscales, level_path)
        if voxel_size is None and level_scale is not None:
            voxel_size, units = level_scale, level_units
        if offset is None:
            offset = level_offset
    if voxel_size is not None and offset is not None:
        if order == "F" or is_n5:
            return _reverse(voxel_size), _reverse(offset), _reverse(units)
        return voxel_size, offset, units

    dims = min(ndim, 3)
    if voxel_size is None:
        voxel_size = (1,) * dims
    if offset is None:
        offset = (0,) * len(voxel_size)
    units = _per_axis("pixels" if units is None else units, dims)
    if order == "F":
        return _reverse(voxel_size), _reverse(offset), _reverse(units)
    return voxel_size, offset, units


def legacy_attrs(attrs: dict, ndim: int) -> Tuple[List[float], List[float]]:
    """``(voxel_size, offset)`` from an array's own ``transform`` or
    ``resolution``/``offset`` attributes, as written; 1 and 0 without them.

    For zarr v3 plain arrays and imported crops, which never went through
    the N5 search: no parent lookup, no units, no rounding.
    """
    if "transform" in attrs:
        transform = attrs["transform"]
        voxel_size = transform.get("scale", [1] * ndim)
        offset = transform.get("translate", [0] * ndim)
    elif "resolution" in attrs:
        voxel_size = attrs["resolution"]
        offset = attrs.get("offset", [0] * ndim)
    else:
        voxel_size = [1] * ndim
        offset = [0] * ndim
    return [float(v) for v in voxel_size], [float(v) for v in offset]


def _attr_meta(path, fmt, header, voxel_size, offset) -> ArrayMeta:
    """ArrayMeta for voxel size/offset attributes that cover the last
    ``len(voxel_size)`` axes; any before them are channels."""
    ndim = len(header.shape)
    n = min(len(voxel_size), ndim)
    voxel_size, offset = list(voxel_size)[-n:], list(offset)[-n:]
    lead = ndim - n
    names = {i: ("c^" if lead == 1 else f"c^{i}") for i in range(lead)}
    names.update({lead + k: name for k, name in enumerate(_unnamed_spatial(n))})
    return _meta(path, fmt, header, list(range(lead, ndim)), names, voxel_size, offset)


# ---------------------------------------------------------------------------
# Format readers
# ---------------------------------------------------------------------------


def open_zarr(path, mode="r"):
    """Open a zarr v2 node, handling HTTP/HTTPS and (anonymous) S3 URLs via fsspec."""
    path = paths.normalize_path(path)
    if paths.is_remote(path):
        import fsspec

        options = {"anon": True} if path.startswith("s3://") else {}
        return zarr.open(fsspec.get_mapper(path, **options), mode=mode)
    return zarr.open(path, mode=mode)


def _group_attrs(root, rel: str) -> dict:
    return dict((root[rel] if rel else root).attrs)


def _find_multiscales(root, rel: str):
    """``(multiscales, group path)`` of the nearest group at or above ``rel``
    with multiscales; ``(root's value, "")`` when there is none."""
    while True:
        multiscales = _group_attrs(root, rel).get("multiscales", None)
        if multiscales or rel == "":
            return multiscales, rel
        rel = posixpath.dirname(rel)


def _relative(rel: str, group_rel: str) -> str:
    return rel if not group_rel else posixpath.relpath(rel, group_rel)


def _v2_array_meta(array, root, rel, path, fmt) -> ArrayMeta:
    """Metadata of a zarr v2 or N5 ``array`` at ``rel`` inside ``root``.

    OME multiscales on the nearest group above it that has them, when they
    list the array; otherwise the N5/funlib attributes of the array and its
    parent. ``root`` is None when the array's container is not known.
    """
    header = _zarr_header(array)
    is_n5 = fmt == "n5"
    # N5 attributes are x, y, z; a zarr array's own ``order`` is its chunk
    # memory layout, not an axis order, and is not consulted (only an
    # explicit ``order`` attribute is).
    order = "F" if is_n5 else array.attrs.get("order", "C")
    try:
        parent_rel = None if (root is None or rel == "") else posixpath.dirname(rel)
        items = [dict(array.attrs)]
        if parent_rel is not None:
            try:
                items.append(_group_attrs(root, parent_rel))
            except (KeyError, ValueError, RuntimeError):
                # An implicit parent (no .zgroup) has no attributes to offer.
                parent_rel = None
        multiscales, ms_rel = (
            (None, None) if parent_rel is None else _find_multiscales(root, parent_rel)
        )
        level_path = None if multiscales is None else _relative(rel, ms_rel)
        units = None
        if not is_n5 and multiscales:
            entry = match_dataset(multiscales[0], level_path)
            if entry is not None and dataset_transforms(entry)[0] is not None:
                # Exact: an OME corner is usually not a multiple of the voxel
                # size (-4 nm at 8 nm for Janelia data), and rounding it onto
                # the grid would undo the centre-to-corner conversion.
                return _snapped(_ome(multiscales[0], entry, header, path, fmt))
            # Not listed: its own attributes are read in the multiscales' units.
            units = spatial_axes(multiscales[0].get("axes", []))[2]
        voxel_size, offset, units = _n5(
            items, order, len(header.shape), is_n5, units, multiscales, level_path
        )
        if isinstance(units, str) or units is None:
            units = [units] * len(voxel_size)
        # Everything downstream is in nanometers. Only the literal "um" was
        # once converted, so OME-NGFF's "micrometer" came through as 0.004
        # "nm", truncated to 0, and a divide by zero fell back to voxel size 1.
        voxel_size, offset = regularize_offset(to_nm(voxel_size, units), to_nm(offset, units))
    except Exception as e:
        logger.error(
            "failed to read voxel size and offset for %s (%s), will use default values"
            % (getattr(array, "path", array), e)
        )
        voxel_size, offset = (1,) * 3, (0,) * 3
    return _attr_meta(path, fmt, header, voxel_size, offset)


def _ome_group_level(group, leaf, array, path) -> Optional[ArrayMeta]:
    """``array``'s metadata from the multiscales of ``group``, which list it
    as ``leaf``; None when ``group`` has no multiscales or they don't list
    it. Another level's entry never stands in: an unlisted level is read
    from its own attributes, as it is on disk (``_v2_array_meta``)."""
    multiscales = group.attrs.get("multiscales", None)
    if not multiscales:
        return None
    entry = match_dataset(multiscales[0], leaf)
    if entry is None:
        return None
    return _ome(multiscales[0], entry, _zarr_header(array), path, "zarr2")


def _read_remote(path: str) -> ArrayMeta:
    """zarr v2 over http(s) or anonymous s3."""
    node = open_zarr(path, mode="r")
    rel_extra = None
    # A URL at a zarr group (e.g. a multiscale container): its first level.
    if isinstance(node, zarr.hierarchy.Group):
        multiscales = node.attrs.get("multiscales", None)
        if multiscales:
            leaf = multiscales[0]["datasets"][0]["path"]
            return _ome_group_level(node, leaf.strip("/"), node[leaf], path)
        for key in sorted(node.keys()):
            if isinstance(node[key], zarr.core.Array):
                node, rel_extra = node[key], key
                break

    root, rel = None, ""
    if paths.suffix_format(path) is not None:
        container, sub_path = paths.split_container(path)
        # A sub-array (e.g. .zarr/raw/s0) has its multiscales on the group
        # right above it (raw), or on the root naming "raw/s0" directly.
        if sub_path:
            sub_path = sub_path.strip("/")
            parent_path, _, leaf = sub_path.rpartition("/")
            candidates = [(parent_path, leaf)]
            if parent_path:
                candidates.append(("", sub_path))
            for group_path, entry_path in candidates:
                try:
                    group = open_zarr(
                        paths.join(container, group_path) if group_path else container,
                        mode="r",
                    )
                    meta = _ome_group_level(group, entry_path, node, path)
                except Exception as e:
                    logger.warning(
                        "failed to read parent multiscale metadata for %s: %s" % (path, e)
                    )
                    continue
                if meta is not None:
                    return meta
        try:
            root = open_zarr(container, mode="r")
            rel = "/".join(p for p in (sub_path.strip("/"), rel_extra) if p)
        except Exception:
            root = None

    # No multiscales: the array's own attributes.
    return _v2_array_meta(node, root, rel, path, "zarr2")


def _read_local_v2(path: str) -> ArrayMeta:
    """Local zarr v2 and N5."""
    filename, ds_name = paths.split_container(path)
    if filename.endswith(".zarr") or paths.is_zarr_container(filename):
        fmt, store = "zarr2", filename
    elif filename.endswith(".n5"):
        from zarr.n5 import N5FSStore

        fmt, store = "n5", N5FSStore(filename)
    else:
        logger.error("don't know data format of %s in %s", ds_name, filename)
        raise RuntimeError("Unknown file format for %s" % filename)
    try:
        root = zarr.open(store, mode="r")
        array = root[ds_name] if ds_name else root
    except Exception as e:
        logger.error("failed to open %s/%s" % (filename, ds_name))
        raise e
    return _v2_array_meta(array, root, getattr(array, "path", ds_name), path, fmt)


def read_zarr_json(path: str) -> dict:
    with open(os.path.join(path, paths.ZARR_JSON)) as f:
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


def multiscales_from_group(group_path: str) -> Optional[dict]:
    """The first multiscales item of a v3 group, or None (also for an array)."""
    meta = read_zarr_json(group_path)
    if meta.get("node_type") != "group":
        return None
    multiscales = attrs_from_meta(meta).get("multiscales")
    if not multiscales:
        return None
    return multiscales[0]


def _v3_level(group_path: str, multiscale: dict, entry: dict) -> ArrayMeta:
    array_path = os.path.join(group_path, entry["path"])
    header = _v3_header(read_zarr_json(array_path))
    return _ome(multiscale, entry, header, array_path, "zarr3")


def _read_v3(path: str) -> ArrayMeta:
    """Local zarr v3: an array a multiscale group lists, an array with its own
    attributes, or a group (its matching, else first, level)."""
    container = paths.find_v3_container(path)
    if container is None:
        raise RuntimeError(f"Could not find a Zarr v3 container in path: {path}")
    meta = read_zarr_json(container)

    if meta.get("node_type") == "array":
        # ``path`` may point directly at one scale's array inside a multiscale
        # group (each v3 array has its own zarr.json, so find_v3_container
        # stops there). The parent's multiscales, when they name this array,
        # give its voxel size and offset.
        parent = os.path.dirname(os.path.normpath(container))
        if paths.is_v3_container(parent):
            if read_zarr_json(parent).get("node_type") == "group":
                multiscale = multiscales_from_group(parent)
                if multiscale is not None:
                    rel = os.path.basename(os.path.normpath(container))
                    entry = match_dataset(multiscale, rel)
                    if entry is not None:
                        return _v3_level(parent, multiscale, entry)
        header = _v3_header(meta)
        voxel_size, offset = legacy_attrs(attrs_from_meta(meta), len(header.shape))
        return _attr_meta(path, "zarr3", header, voxel_size, offset)

    multiscale = multiscales_from_group(container)
    if multiscale is not None:
        rel = os.path.relpath(path, container)
        entry = match_dataset(multiscale, rel) or multiscale["datasets"][0]
        return _v3_level(container, multiscale, entry)

    # A group with no multiscales: its first array child.
    for name in sorted(os.listdir(container)):
        child = os.path.join(container, name)
        if os.path.isdir(child) and paths.is_v3_container(child):
            if read_zarr_json(child).get("node_type") == "array":
                return _read_v3(child)
    raise RuntimeError(f"No array found under Zarr v3 group: {container}")


def _precomputed_info(path: str) -> dict:
    """The ``info`` JSON of the precomputed volume ``path`` is, or is a scale
    of: one read."""
    import tensorstore as ts

    # Not GCE's metadata server: probing it for credentials stalls a gs://
    # open off Google Cloud (io.source sets the same before opening one).
    os.environ.setdefault("GCE_METADATA_ROOT", "metadata.google.internal.invalid")
    kvstore, _ = paths.precomputed_kvstore(path)
    result = (ts.KvStore.open(kvstore).result() / "info").read("").result()
    if result.state != "value":
        raise FileNotFoundError(f"{path} is not a precomputed volume: it has no info file")
    return json.loads(result.value)


def _precomputed(path: str, info: Optional[dict] = None) -> ArrayMeta:
    """The scale of a neuroglancer precomputed volume ``path`` names (the
    volume is scale 0, ``…/s<N>`` scale N; ``paths.precomputed_kvstore``),
    in C order (channel, z, y, x), from the volume's ``info`` (read here
    unless given) as tensorstore's neuroglancer_precomputed driver reads it,
    so that choosing a scale opens none of them:

    - the shape is ``num_channels`` and the scale's ``size``;
    - the voxel size is its ``resolution``, in nm;
    - the chunk shape is its first ``chunk_sizes`` entry, tensorstore's read
      chunk whether the scale is sharded or not;
    - ``voxel_offset`` (0 when left out) is where voxel 0 is, in voxels, so
      the translation is ``voxel_offset * resolution``. tensorstore starts
      the volume's domain at it; io.source opens it with the domain moved
      to 0.

    ``info`` lists each per-axis value x, y, z.
    """
    info = _precomputed_info(path) if info is None else info
    _, index = paths.precomputed_kvstore(path)
    if index >= len(info["scales"]):
        raise ValueError(f"{path}: the volume has {len(info['scales'])} scales")
    scale = info["scales"][index]
    channels = (int(info["num_channels"]),)
    header = _Header(
        channels + tuple(int(v) for v in reversed(scale["size"])),
        np.dtype(info["data_type"]),
        channels + tuple(int(v) for v in reversed(scale["chunk_sizes"][0])),
    )
    voxel_size = [float(v) for v in reversed(scale["resolution"])]
    voxel_offset = reversed(scale.get("voxel_offset", [0, 0, 0]))
    corner = [float(v) * size for v, size in zip(voxel_offset, voxel_size)]
    names = {0: "channel", 1: "z", 2: "y", 3: "x"}
    return _meta(path, "precomputed", header, [1, 2, 3], names, voxel_size, corner)


def read_array_meta(path: str) -> ArrayMeta:
    """Metadata of the array at ``path`` (see the module docstring).

    A path at an OME multiscale group reads its first level, except for a
    local zarr v2 group, which is an error.
    """
    path = paths.normalize_path(path)
    if paths.is_precomputed(path):
        return _precomputed(path)
    if paths.is_remote(path):
        return _read_remote(path)
    if paths.find_v3_container(path) is not None:
        return _read_v3(path)
    return _read_local_v2(path)


# ---------------------------------------------------------------------------
# Levels of a multiscale group (or of a precomputed volume)
# ---------------------------------------------------------------------------


def levels_from_zarr_group(group, group_path: str = "") -> List[Tuple[str, ArrayMeta]]:
    """``list_levels`` of an opened zarr v2 group."""
    multiscale = group.attrs["multiscales"][0]
    return [
        (
            entry["path"],
            _ome(
                multiscale,
                entry,
                _zarr_header(group[entry["path"]]),
                paths.join(group_path, entry["path"]) if group_path else entry["path"],
                "zarr2",
            ),
        )
        for entry in multiscale["datasets"]
    ]


def _precomputed_levels(path: str) -> List[Tuple[str, ArrayMeta]]:
    """``list_levels`` of a precomputed volume: every scale its ``info``
    lists, in that order, as ``s<N>``, the path under the volume that opens
    scale N, all from one read of the info (a gs:// volume can have a
    dozen scales, and opening each took about 50 ms). The path of one scale
    (``…/s2``) is not the volume, and raises ValueError."""
    _, scale = paths.precomputed_scale(path)
    if scale is not None:
        raise ValueError(f"{path} is scale {scale} of a precomputed volume, not the volume")
    info = _precomputed_info(path)
    return [
        (f"s{i}", _precomputed(paths.join(path, f"s{i}"), info))
        for i in range(len(info["scales"]))
    ]


def list_levels(group_path: str) -> List[Tuple[str, ArrayMeta]]:
    """``[(dataset path, ArrayMeta)]`` for each level of the OME multiscale
    group at ``group_path``, in the order its ``datasets`` list them, or of
    the precomputed volume there (``_precomputed_levels``).

    Raises when there is no such group (KeyError for a v2 group without
    multiscales, ValueError for a v3 one).
    """
    group_path = paths.normalize_path(group_path)
    if paths.is_precomputed(group_path):
        return _precomputed_levels(group_path)
    if not paths.is_remote(group_path) and paths.is_v3_container(group_path):
        multiscale = multiscales_from_group(group_path)
        if multiscale is None:
            raise ValueError(f"No multiscales attribute found at {group_path}")
        return [
            (entry["path"], _v3_level(group_path, multiscale, entry))
            for entry in multiscale["datasets"]
        ]
    return levels_from_zarr_group(open_zarr(group_path, mode="r"), group_path)
