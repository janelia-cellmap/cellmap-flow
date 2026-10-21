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

- ``ArrayMeta`` itself (corner-based, ``spatial()`` a method);
- the record's shape for precomputed (a leading ``channel`` axis, even of one
  channel) and for unnamed axes (``c^`` before ``z, y, x``);
- precomputed levels named ``s<N>`` (worldarray names them by key; it opens
  ``…/s<N>`` as scale N);
- ``list_levels`` refusing a group without multiscales;
- ``nm_per_unit``, over worldarray's unit table (``worldarray.units.TABLE``);
- the crop loader's attribute helpers (``read_zarr_json``, ``attrs_from_meta``,
  ``legacy_attrs``, ``dataset_transforms``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import worldarray as wa
from worldarray import units as wa_units

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
# web/src/infer/geometry.ts): worldarray's length table, in nm per unit.
# "µm" is the micro sign (U+00B5), "μm" the Greek mu (U+03BC).
_NM_PER_UNIT = {unit: _nm(metres) for unit, metres in wa_units.TABLE["length"].items()}

_warned_units = set()


def nm_per_unit(unit) -> float:
    """How many nanometers one ``unit`` is; 1 for a missing, "pixel" or unknown
    unit, with a warning for an unknown one."""
    if unit is None:
        return 1.0
    key = str(unit).strip().lower()
    if key in ("", "pixel", "pixels"):
        return 1.0
    if wa_units.kind(str(unit)) != "space":
        if key not in _warned_units:
            _warned_units.add(key)
            logger.warning(f"Unknown spatial unit {unit!r}; treating it as nanometers")
        return 1.0
    return float(f"{wa_units.factor(str(unit), NANOMETER):.12g}")


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


def from_world_array(native: "wa.ArrayMeta", path: str) -> ArrayMeta:
    """cellmap-flow's record of worldarray's ``native`` metadata (an array's
    ``.meta``) of the array at ``path``.

    worldarray's spatial axes are those named z, y or x, else the last three;
    the others keep unit None. Unnamed axes get cellmap-flow's names (z, y, x
    from the end, ``c^`` before them); a precomputed volume's channel axis,
    which worldarray drops when there is one channel, is put back.
    """
    nm = native.in_units(space=NANOMETER, strict=False)
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
        fill_value=native.fill_value,
    )


# ---------------------------------------------------------------------------
# Opening a path as worldarray reads it
# ---------------------------------------------------------------------------


def open_level(path: str, voxel_size=None, **kwargs) -> "wa.WorldArray":
    """The worldarray array ``path`` names, as cellmap-flow reads a path: an
    array as it is, ``…/s<N>`` of a precomputed volume as scale N, and a group
    or precomputed volume at its level for ``voxel_size`` (worldarray chooses
    it from the group's document), its first without one."""
    path = paths.normalize_path(path)
    named_scale = paths.is_precomputed(path) and paths.precomputed_scale(path)[1] is not None
    level = None if (named_scale or voxel_size is not None) else "first"
    return wa.open(path, level=level, voxel_size=voxel_size, strict_units=False, **kwargs)


def read_array_meta(path: str) -> ArrayMeta:
    """Metadata of the array at ``path`` (see the module docstring). A path at
    a group reads its first level: the first its multiscales list, else s0."""
    path = paths.normalize_path(path)
    try:
        array = open_level(path, units=None)
    except ValueError as e:
        if "no levels found" in str(e):
            raise RuntimeError(f"No array found under the group {path}: {e}") from None
        raise
    return from_world_array(array.meta, path)


# ---------------------------------------------------------------------------
# Levels of a multiscale group (or of a precomputed volume)
# ---------------------------------------------------------------------------


def _multiscale_group(group_path: str):
    """Check that ``group_path`` is a multiscale group: KeyError for a v2 (or
    N5) group without multiscales, or an array; ValueError for a v3 one."""
    node = wa.detect(group_path)
    attrs = {}
    if node.kind == "group":
        if node.format == "zarr3":
            attrs = node.docs["zarr.json"].get("attributes", {}) or {}
        elif node.format == "zarr2":
            attrs = store.read_json(node.path, ".zattrs") or {}
        else:
            attrs = node.docs.get("attributes.json", {}) or {}
    ome = attrs.get("ome")
    if not ((ome or {}).get("multiscales") if isinstance(ome, dict) else attrs.get("multiscales")):
        error = ValueError if node.format == "zarr3" else KeyError
        raise error(f"No multiscales attribute found at {group_path}")


def level_names(group_path: str, pyramid: "wa.Pyramid") -> List[str]:
    """cellmap-flow's names of ``pyramid``'s levels: a precomputed volume's
    scales are ``s<N>``, a group's levels their dataset paths."""
    if paths.is_precomputed(group_path):
        return [f"s{i}" for i in range(len(pyramid))]
    return pyramid.names


def pyramid(group_path: str, units=None) -> Tuple[str, "wa.Pyramid"]:
    """``(group path, worldarray Pyramid)`` of the multiscale group or
    precomputed volume at ``group_path``, in ``units`` (native when None);
    raises as ``list_levels`` does."""
    group_path = paths.normalize_path(group_path)
    if paths.is_precomputed(group_path):
        _, scale = paths.precomputed_scale(group_path)
        if scale is not None:
            raise ValueError(f"{group_path} is scale {scale} of a precomputed volume, not the volume")
    else:
        _multiscale_group(group_path)
    return group_path, wa.open_pyramid(group_path, units=units, strict_units=False)


def list_levels(group_path: str) -> List[Tuple[str, ArrayMeta]]:
    """``[(dataset path, ArrayMeta)]`` for each level of the OME multiscale
    group at ``group_path``, in the order its ``datasets`` list them, or of
    the precomputed volume there (``s<N>`` for scale N).

    An ArrayMeta needs each level's shape, dtype and chunks, which the group's
    document does not hold (``Pyramid.placements`` gives only voxel sizes and
    translations), so every level is opened, each on a context of its own.

    Raises when there is no such group (KeyError for a v2 group without
    multiscales, ValueError for a v3 one, or for one scale of a precomputed
    volume).
    """
    group_path, pyramid_ = pyramid(group_path)
    return [
        (name, from_world_array(
            wa.open(pyramid_.path, level=key, units=None, strict_units=False,
                    context=wa.capped_context(network=False)).meta,
            paths.join(group_path, name)))
        for name, key in zip(level_names(group_path, pyramid_), pyramid_.names)
    ]


def node_type(path: str) -> Tuple[str, str]:
    """``(path, "group" or "array")``: what worldarray finds at ``path``.
    FileNotFoundError when there is nothing there."""
    path = paths.normalize_path(path)
    return path, wa.detect(path).kind


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
