"""Choosing a level of a multiscale pyramid for a voxel size.

The levels come from ``io.metadata.list_levels``, in the order the group's
``datasets`` list them (finest first, by convention). Voxel sizes are
compared as nanometer floats on the spatial axes, so 10.48 nm is not 10 nm.
"""

import logging
import math
import os
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import zarr

from cellmap_flow.io import metadata, paths
from cellmap_flow.io.metadata import ArrayMeta, snap_integral

logger = logging.getLogger(__name__)

Level = Tuple[str, ArrayMeta]


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


def _voxel_size(level: Level) -> Tuple[float, ...]:
    return level[1].spatial().voxel_size


def select_level(
    levels: Sequence[Level],
    voxel_size=None,
    mode: Literal["floor", "exact", "nearest"] = "floor",
) -> Level:
    """The ``(path, ArrayMeta)`` of ``levels`` to read at ``voxel_size``.

    With no ``voxel_size``, the first (finest) level. Otherwise, by ``mode``:

    - "floor": the level at ``voxel_size`` if there is one, else the last
      level before the first one that is coarser than ``voxel_size`` along
      some axis -- the finest level that is not too coarse, and the first
      level when even that is. This is how cellmap-flow has always chosen.
    - "exact": the level at ``voxel_size``; ValueError when there is none.
    - "nearest": the level whose voxel size is closest on a log scale
      (summed over the axes), the finer one on a tie.
    """
    levels = list(levels)
    if not levels:
        raise ValueError("no levels to choose from")
    if voxel_size is None:
        return levels[0]

    if mode == "floor":
        last = None
        for level in levels:
            if last is None:
                last = level
            if same_voxel_size(_voxel_size(level), voxel_size):
                return level
            if coarser_anywhere(_voxel_size(level), voxel_size):
                return last
            last = level
        return last

    if mode == "exact":
        for level in levels:
            if same_voxel_size(_voxel_size(level), voxel_size):
                return level
        raise ValueError(
            f"no level at {tuple(voxel_size)} nm; the levels are "
            + ", ".join(f"{path} {_voxel_size((path, meta))}" for path, meta in levels)
        )

    if mode == "nearest":
        target = snap_integral(voxel_size)

        def distance(level):
            return sum(
                abs(math.log(float(v) / float(t))) for v, t in zip(_voxel_size(level), target)
            )

        return min(levels, key=distance)

    raise ValueError(f"mode must be 'floor', 'exact' or 'nearest', got {mode!r}")


def _level_group(dataset_path: str) -> Optional[str]:
    """The multiscale group ``dataset_path`` picks a level from: the path
    itself when it is a zarr v2 group, or the nearest zarr.json node when that
    is a v3 group. None when it is an array."""
    container = paths.find_v3_container(dataset_path)
    if container is not None:
        node_type = metadata.read_zarr_json(container).get("node_type")
        return container if node_type == "group" else None
    node = metadata.open_zarr(dataset_path, mode="r")
    return dataset_path if isinstance(node, zarr.hierarchy.Group) else None


def select_dataset(
    dataset_path: str,
    voxel_size=None,
    mode: Literal["floor", "exact", "nearest"] = "floor",
) -> Tuple[str, Optional[str]]:
    """``(path of the array to read, level chosen)`` for a dataset path.

    A multiscale group resolves to its level for ``voxel_size`` (see
    ``select_level``); an array is read as it is, with level None. Raises
    when a group has no multiscales it can read.
    """
    group = _level_group(dataset_path)
    if group is None:
        return dataset_path, None
    level, _ = select_level(metadata.list_levels(group), voxel_size, mode)
    return paths.join(group, level), level


def _pyramid_of(dataset_path: str) -> str:
    """The multiscale group a dataset path belongs to: itself, or the group
    above it when it is one level's array."""
    container = paths.find_v3_container(dataset_path)
    if container is not None:
        # A per-scale path (.../s1) finds the array's own zarr.json first;
        # the pyramid is described by the group above it.
        if metadata.multiscales_from_group(container) is None:
            parent = os.path.dirname(os.path.normpath(container))
            if paths.is_v3_container(parent):
                container = parent
        if metadata.multiscales_from_group(container) is not None:
            return container
    if isinstance(metadata.open_zarr(dataset_path, mode="r"), zarr.core.Array):
        # Same for a v2 per-scale path: use the multiscale group above it.
        if "://" in dataset_path:
            return dataset_path.rstrip("/").rsplit("/", 1)[0]
        return os.path.dirname(os.path.normpath(dataset_path))
    return dataset_path


def closest_raw_scale(dataset_path: str, target_voxel_size) -> Optional[tuple]:
    """The voxel size (nm, z/y/x) of the level of ``dataset_path``'s pyramid
    that ``select_level`` picks for ``target_voxel_size``, or None if it
    can't be determined.

    ``dataset_path`` may be the multiscale group or one of its levels.
    """
    try:
        levels: List[Level] = metadata.list_levels(_pyramid_of(dataset_path))
        return tuple(_voxel_size(select_level(levels, target_voxel_size)))
    except Exception as e:
        logger.warning(
            f"Could not determine closest raw scale for {dataset_path} at "
            f"target_resolution={target_voxel_size}: {e}"
        )
        return None
