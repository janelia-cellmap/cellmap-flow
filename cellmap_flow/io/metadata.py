"""Array metadata for every format cellmap-flow reads, as one ``ArrayMeta``.

worldarray parses the metadata: zarr v2 and v3, N5 and neuroglancer
precomputed, on disk or at an http(s), s3 or gs URL (``worldarray.metadata``,
its spec's Conventions 1 and 4). This module turns worldarray's record (the
store's own units, ``translation`` the *centre* of voxel 0) into
cellmap-flow's: every axis kept, in the array's own (C) order; spatial sizes
and translations in nanometer floats; ``translation`` the lower *corner* of
voxel 0; non-spatial axes with unit None, voxel size 1 and translation 0.
``.spatial()`` drops the channel/time axes.

What stays here because worldarray has no such thing:

- ``ArrayMeta`` itself (corner-based, ``spatial()`` a method, ``fill_value``);
- the record's shape for precomputed (a leading ``channel`` axis, even of one
  channel) and for unnamed axes (``c^`` before ``z, y, x``);
- the ``…/s<N>`` names of precomputed scales (worldarray names them by key);
- a group path reading its first level, and ``list_levels`` refusing a group
  without multiscales;
- ``nm_per_unit``'s lenient rule (a missing, "pixel" or unknown unit is 1 nm),
  over worldarray's unit table;
- the crop loader's attribute helpers (``read_zarr_json``, ``attrs_from_meta``,
  ``legacy_attrs``, ``dataset_transforms``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import worldarray as wa
from worldarray import units as wa_units
from worldarray.store import Location, detect

from cellmap_flow.io import paths, store

logger = logging.getLogger(__name__)

NANOMETER = "nanometer"

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def _nm(metres: float) -> float:
    """``metres`` in nm, without the float noise of the division (1e-6 / 1e-9 is
    999.9999999999999)."""
    return float(f"{metres / 1e-9:.12g}")


# The browser reads this table too (web/build/export_data.py exports it for
# web/src/infer/geometry.ts). It is worldarray's length table, which keeps it
# private (``worldarray.units._LENGTH_M``, metres per unit), in nm per unit.
# "µm" is the micro sign (U+00B5), "μm" the Greek mu (U+03BC).
_NM_PER_UNIT = {unit: _nm(metres) for unit, metres in wa_units._LENGTH_M.items()}

_warned_units = set()


def nm_per_unit(unit) -> float:
    """How many nanometers one ``unit`` is; 1 for a missing, "pixel" or unknown
    unit (worldarray raises on an unknown one; cellmap-flow warns and reads on)."""
    if unit is None:
        return 1.0
    key = str(unit).strip().lower()
    if key in ("", "pixel", "pixels"):
        return 1.0
    try:
        return float(f"{wa_units.factor(str(unit), NANOMETER):.12g}")
    except ValueError:
        if key not in _warned_units:
            _warned_units.add(key)
            logger.warning(f"Unknown spatial unit {unit!r}; treating it as nanometers")
        return 1.0


def to_nm(values, units) -> Tuple[float, ...]:
    """``values`` (one per axis) in nanometers; ``units`` None means nm."""
    if units is None:
        return tuple(float(v) for v in values)
    return tuple(float(v) * nm_per_unit(u) for v, u in zip(values, units))


def _unnamed_spatial(n: int) -> Tuple[str, ...]:
    """Names for ``n`` spatial axes that were not named: z, y, x from the
    end, and "" for any axis before z."""
    return ("",) * max(0, n - 3) + ("z", "y", "x")[-n:] if n else ()


# ---------------------------------------------------------------------------
# Float noise
# ---------------------------------------------------------------------------

# Unit conversion leaves float noise (0.009 um -> 8.999999999999998 nm);
# anything this close to a whole number is that number. Relative, unlike
# worldarray.geometry.snap_integral's absolute 1e-6 of a voxel.
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
    ``translation`` 0.0. ``translation`` is the lower corner of voxel 0
    (worldarray's ``corner``; its ``translation`` is the centre).
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


def _fill_value(store_):
    """The array's fill value as tensorstore reads it, a Python scalar; None when it has none."""
    value = store_.fill_value
    return None if value is None else np.asarray(value).item()


def from_world_array(array: "wa.WorldArray", path: str) -> ArrayMeta:
    """cellmap-flow's record of a worldarray array opened at ``path``.

    worldarray's spatial axes are those named z, y or x, else the last three;
    the others keep unit None. Unnamed axes get cellmap-flow's names (z, y, x
    from the end, ``c^`` before them); a precomputed volume's channel axis,
    which worldarray drops when there is one channel, is put back.
    """
    native = array.meta
    nm = native.in_units(space=NANOMETER)
    spatial = set(native.spatial)
    corner = nm.corner
    ndim = native.ndim
    if native.axes_named:
        axes = list(native.axes)
    else:
        n = len(spatial)
        lead = ndim - n
        axes = [("c^" if lead == 1 else f"c^{i}") for i in range(lead)] + list(_unnamed_spatial(n))
    units, voxel_size, translation = [], [], []
    for i in range(ndim):
        if i in spatial:
            units.append(NANOMETER)
            voxel_size.append(float(snap_integral(nm.voxel_size[i])))
            translation.append(float(snap_integral(corner[i])))
        else:
            units.append(None)
            voxel_size.append(1.0)
            translation.append(0.0)
    shape, chunk_shape = tuple(native.shape), tuple(native.chunk_shape)
    if native.format == "precomputed":
        axes = ["channel" if a == "c" else a for a in axes]
        if ndim == 3:
            axes, units = ["channel", *axes], [None, *units]
            voxel_size, translation = [1.0, *voxel_size], [0.0, *translation]
            shape, chunk_shape = (1, *shape), (1, *chunk_shape)
    return ArrayMeta(
        path=path,
        format=native.format,
        shape=shape,
        dtype=native.dtype,
        chunk_shape=chunk_shape,
        axes=tuple(axes),
        units=tuple(units),
        voxel_size=tuple(voxel_size),
        translation=tuple(translation),
        fill_value=_fill_value(array.ts),
    )


# ---------------------------------------------------------------------------
# Opening a path as worldarray reads it
# ---------------------------------------------------------------------------

_SCALE_INDEX = re.compile(r"^s(\d+)$")


def precomputed_level(path: str) -> Tuple[str, Optional[str]]:
    """``(volume, scale key)`` worldarray opens for cellmap-flow's precomputed
    ``path``: its ``…/s<N>`` names scale N (worldarray names a scale by its
    key), a bare volume is scale 0. ValueError for a scale it does not have."""
    location, index = paths.precomputed_volume(path)
    info = store.read_json(location, "info")
    if info is None:
        raise FileNotFoundError(f"{path} is not a precomputed volume: it has no info file")
    scales = info["scales"]
    if index >= len(scales):
        raise ValueError(f"{path}: the volume has {len(scales)} scales")
    return location, str(scales[index].get("key", index))


def first_level(path: str) -> Optional[str]:
    """The name of the level a group path reads (its multiscales' first, else
    ``s0``, worldarray's discovery); None when ``path`` is an array."""
    node = detect(store.prepare(path))
    if node.kind == "array":
        return None
    try:
        return wa.open_pyramid(path, units=None).names[0]
    except ValueError as e:
        raise RuntimeError(f"No array found under the group {path}: {e}") from None


def open_level(path: str, **kwargs) -> "wa.WorldArray":
    """The worldarray array ``path`` names, as cellmap-flow names it: a
    precomputed ``…/s<N>`` its scale N, a group its first level."""
    path = store.prepare(paths.normalize_path(path))
    if paths.is_precomputed(path):
        volume, key = precomputed_level(path)
        return wa.open(volume, level=key, **kwargs)
    return wa.open(path, level=first_level(path), **kwargs)


def read_array_meta(path: str) -> ArrayMeta:
    """Metadata of the array at ``path`` (see the module docstring). A path at
    a group reads its first level: the first its multiscales list, else s0."""
    path = paths.normalize_path(path)
    return from_world_array(open_level(path, units=None), path)


# ---------------------------------------------------------------------------
# Levels of a multiscale group (or of a precomputed volume)
# ---------------------------------------------------------------------------


def _has_multiscales(path: str) -> bool:
    node = detect(store.prepare(path))
    if node.format == "zarr3":
        attrs = node.docs["zarr.json"].get("attributes", {}) or {}
    elif node.format == "zarr2":
        attrs = Location(node.path).read_json(".zattrs") or {}
    else:
        attrs = node.docs.get("attributes.json", {}) or {}
    ome = attrs.get("ome")
    return bool((ome or {}).get("multiscales") if isinstance(ome, dict) else attrs.get("multiscales"))


def list_levels(group_path: str) -> List[Tuple[str, ArrayMeta]]:
    """``[(dataset path, ArrayMeta)]`` for each level of the OME multiscale
    group at ``group_path``, in the order its ``datasets`` list them, or of
    the precomputed volume there (``s<N>`` for scale N).

    Raises when there is no such group (KeyError for a v2 group without
    multiscales, ValueError for a v3 one, or for one scale of a precomputed
    volume).
    """
    group_path = paths.normalize_path(group_path)
    if paths.is_precomputed(group_path):
        _, scale = paths.precomputed_scale(group_path)
        if scale is not None:
            raise ValueError(f"{group_path} is scale {scale} of a precomputed volume, not the volume")
        volume, _ = paths.precomputed_volume(group_path)
        pyramid = wa.open_pyramid(store.prepare(volume), units=None)
        return [
            (f"s{i}", from_world_array(level, paths.join(group_path, f"s{i}")))
            for i, level in enumerate(pyramid.levels)
        ]
    if not _has_multiscales(group_path):
        node = detect(group_path)
        if node.format == "zarr3":
            raise ValueError(f"No multiscales attribute found at {group_path}")
        raise KeyError(f"No multiscales attribute found at {group_path}")
    pyramid = wa.open_pyramid(group_path, units=None)
    return [
        (name, from_world_array(level, paths.join(group_path, name)))
        for name, level in zip(pyramid.names, pyramid.levels)
    ]


def node_type(path: str) -> Tuple[str, str]:
    """``(path, "group" or "array")``: what worldarray finds at ``path``.
    FileNotFoundError when there is nothing there."""
    path = paths.normalize_path(path)
    return path, detect(store.prepare(path)).kind


# ---------------------------------------------------------------------------
# The crop loader's attribute helpers
# ---------------------------------------------------------------------------


def match_dataset(multiscale: dict, dataset_path: str) -> Optional[dict]:
    """The ``datasets`` entry of one multiscales item whose path is
    ``dataset_path`` (leading and trailing "/" ignored), or None."""
    want = dataset_path.strip("/")
    return next((d for d in multiscale["datasets"] if d["path"].strip("/") == want), None)


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


def legacy_attrs(attrs: dict, ndim: int) -> Tuple[List[float], List[float]]:
    """``(voxel_size, offset)`` from an array's own ``transform`` or
    ``resolution``/``offset`` attributes, as written; 1 and 0 without them.

    For imported crops: no parent lookup, no units, no rounding.
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


def read_zarr_json(path: str) -> dict:
    """The ``zarr.json`` of the v3 node at ``path``; FileNotFoundError when
    there is none."""
    meta = store.read_json(path, paths.ZARR_JSON)
    if meta is None:
        raise FileNotFoundError(f"No {paths.ZARR_JSON} at {path}")
    return meta


def attrs_from_meta(meta: dict) -> dict:
    """Merge OME-NGFF 0.5's nested ``attributes.ome`` with the top-level
    ``attributes`` dict (some writers emit OME metadata unnested)."""
    attributes = meta.get("attributes", {}) or {}
    merged = dict(attributes)
    ome = attributes.get("ome")
    if isinstance(ome, dict):
        merged.update(ome)
    return merged
