"""Opening a path as cellmap-flow reads it, and nanometers per unit.

Array metadata is worldarray's: ``wa.open(path).meta`` or ``wa.metadata(path)``
give an ``ArrayMeta`` (every axis in the array's order, the store's units,
``translation`` the *centre* of voxel 0 and ``.corner`` its lower corner),
``wa.open_pyramid(path)`` a group's levels, and ``.in_nm()`` the spatial axes
in nanometers. This module keeps the two things cellmap-flow adds on top:

- ``open_level``: the one way every reader opens a path, so that a group is
  read at its first level when no voxel size asks for another;
- ``nm_per_unit`` and ``_NM_PER_UNIT``: worldarray's unit table as nanometers
  per unit, forgiving of a missing or unknown unit, which the browser shares
  (``web/build/export_data.py``).

``snap_integral`` takes the float noise out of unit conversions.
"""

from __future__ import annotations

import logging

import numpy as np
import worldarray as wa
from worldarray import units as wa_units

from cellmap_flow.io import paths

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


# ---------------------------------------------------------------------------
# Opening a path as worldarray reads it
# ---------------------------------------------------------------------------


def open_level(path: str, voxel_size=None, **kwargs) -> "wa.WorldArray":
    """The worldarray array ``path`` names, as cellmap-flow reads a path: an
    array, or one scale of a precomputed volume, as it is whatever is asked
    for; a group or precomputed volume at its level for ``voxel_size``
    (worldarray chooses it from the group's document), its first without one.
    ``kwargs`` are ``wa.open``'s (``on_mismatch``, ``padding``, ``units``,
    ``context``, ...); units it cannot convert are left as they are rather
    than refused."""
    path = paths.normalize_path(path)
    level = None if voxel_size is not None else "first"
    return wa.open(path, level=level, voxel_size=voxel_size, strict_units=False, **kwargs)
