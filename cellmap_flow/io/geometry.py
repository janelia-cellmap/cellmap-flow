"""Where an array's voxels are in the world, and boxes of voxels.

World coordinates are nanometers. A ``Grid`` is an array's voxel size and
the lower corner of its voxel 0 (``ArrayMeta.voxel_size``/``translation``
on the spatial axes): voxel ``i`` covers ``[corner + i*vs, corner +
(i+1)*vs)``. Voxel sizes can be fractional (5.24 nm) and a corner need not
lie on the grid (Janelia raw is at -4 nm with 8 nm voxels), so nothing here
goes through funlib's ``Coordinate``, which truncates both. A ``Box`` is a
range of voxel indices, which may start before the array or run past it.

- ``Grid.world_to_box(roi)``: the voxels a read of a world ``Roi`` returns.
- ``Grid.box_to_world(box)``: the whole-nm ``Roi`` around a box.
- ``coordinate_or_floats``: a voxel size or offset as a ``Coordinate`` when
  it is whole, else as floats.
- ``list_populated_chunks``: the chunks of a local zarr v2 array that have a
  file, by index.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from funlib.geometry import Coordinate, Roi

from cellmap_flow.io.metadata import is_integral, snap_integral

logger = logging.getLogger(__name__)

# A chunk file of a three-dimensional zarr v2 array with "." separators: z.y.x.
CHUNK_KEY_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class Box:
    """Voxel indices ``begin`` up to ``begin + shape``, per axis."""

    begin: Tuple[int, ...]
    shape: Tuple[int, ...]

    @property
    def end(self) -> Tuple[int, ...]:
        return tuple(b + s for b, s in zip(self.begin, self.shape))


@dataclass(frozen=True)
class Grid:
    """``voxel_size`` and the lower corner of voxel 0 (``translation``),
    nanometer floats, one per axis."""

    voxel_size: Tuple[float, ...]
    translation: Tuple[float, ...]

    def world_to_box(self, roi: Roi) -> Box:
        """The voxels a read of ``roi`` returns: as many whole voxels as fit in
        ``roi.shape``, from the one ``roi.begin`` is in.

        That is the voxel below an off-grid start on either side of voxel 0:
        half a voxel before the array starts at voxel -1. Float noise from
        the division (9.9999999 voxels) counts as the whole number it is.
        """
        voxel_size = np.asarray(self.voxel_size, dtype=float)
        begin = np.floor(
            snap_integral(
                (np.asarray(roi.begin, dtype=float) - np.asarray(self.translation, dtype=float))
                / voxel_size
            )
        )
        shape = np.trunc(snap_integral(np.asarray(roi.shape, dtype=float) / voxel_size))
        return Box(tuple(int(b) for b in begin), tuple(int(s) for s in shape))

    def box_to_world(self, box: Box) -> Roi:
        """The integer-nm ``Roi`` covering ``box``: exactly its extent when
        that is whole nanometers, else rounded out to them (a Roi holds
        integers)."""
        voxel_size = snap_integral(self.voxel_size)
        begin = snap_integral(
            np.asarray(self.translation, dtype=float) + np.asarray(box.begin) * voxel_size
        )
        end = snap_integral(begin + voxel_size * np.asarray(box.shape, dtype=float))
        begin, end = np.floor(begin).astype(int), np.ceil(end).astype(int)
        return Roi(Coordinate(begin), Coordinate(end - begin))


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


def list_populated_chunks(array_dir) -> List[Tuple[int, ...]]:
    """The indices of the chunks of a local zarr v2 array that have a file.

    ``array_dir`` is the array's directory; its chunk files are the names
    ``CHUNK_KEY_RE`` matches, and ``.zarray`` and the rest are not chunks.
    In index order, not ``os.listdir``'s: that order differs between
    filesystems (ext4 orders names by a per-filesystem hash), so the same
    seed would draw different patches on another machine.
    """
    return sorted(
        tuple(int(i) for i in name.split("."))
        for name in os.listdir(array_dir)
        if CHUNK_KEY_RE.match(name)
    )
