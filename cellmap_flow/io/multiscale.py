"""Choosing a level of a multiscale pyramid for a voxel size.

The rule is worldarray's ``select_level`` (its spec's Behaviour 3: "floor",
"exact", "nearest", "resample"), on the levels ``io.metadata.list_levels``
gives, in the order the group's ``datasets`` list them; a precomputed
volume's levels are its scales. Voxel sizes are compared as nanometer floats
on the spatial axes, so 10.48 nm is not 10 nm. What stays here: no voxel size
picks the first level, and a path resolves to the path of its level.
"""

import logging
from types import SimpleNamespace
from typing import List, Literal, Optional, Sequence, Tuple

from worldarray.levels import coarser_anywhere, same_voxel_size  # noqa: F401
from worldarray.levels import select_level as _select_level

from cellmap_flow.io import metadata, paths
from cellmap_flow.io.metadata import ArrayMeta

logger = logging.getLogger(__name__)

Level = Tuple[str, ArrayMeta]

Mode = Literal["floor", "exact", "nearest", "resample"]


def select_level(levels: Sequence[Level], voxel_size=None, mode: Mode = "floor") -> Level:
    """The ``(path, ArrayMeta)`` of ``levels`` to read at ``voxel_size``: the
    first (finest) with no ``voxel_size``, else worldarray's ``select_level``
    by ``mode``. "floor" there is the coarsest level no coarser than
    ``voxel_size`` on any axis, else the first: whatever the listing order,
    where cellmap-flow's took the last level before the first too coarse one
    (the same level for a pyramid listed finest first)."""
    levels = list(levels)
    if not levels:
        raise ValueError("no levels to choose from")
    if voxel_size is None:
        return levels[0]
    # worldarray reads an ArrayMeta's spatial axes as the ``spatial`` indices.
    views = [(path, SimpleNamespace(spatial=meta.spatial_indices, voxel_size=meta.voxel_size))
             for path, meta in levels]
    name, _ = _select_level(views, voxel_size, mode)
    return next(level for level in levels if level[0] == name)


def select_dataset(dataset_path: str, voxel_size=None, mode: Mode = "floor") -> Tuple[str, Optional[str]]:
    """``(path of the array to read, level chosen)`` for a dataset path.

    A multiscale group resolves to its level for ``voxel_size`` (see
    ``select_level``); an array is read as it is, with level None. Raises
    when a group has no multiscales it can read.

    A precomputed volume is a group whose levels are its scales: it
    resolves to ``…/s<N>`` for ``voxel_size`` the same way, and the path of
    one scale is read as it is. Without a voxel size a volume is read at
    scale 0, without opening it.
    """
    if paths.is_precomputed(dataset_path):
        volume, scale = paths.precomputed_scale(dataset_path)
        if scale is not None or voxel_size is None:
            return dataset_path, None
        group = volume
    else:
        node, kind = metadata.node_type(dataset_path)
        if kind != "group":
            return dataset_path, None
        group = node
    level, _ = select_level(metadata.list_levels(group), voxel_size, mode)
    return paths.join(group, level), level


def _pyramid_of(dataset_path: str) -> str:
    """The multiscale group a dataset path belongs to: itself, or the group
    above it when it is one level's array (a precomputed scale's volume)."""
    if paths.is_precomputed(dataset_path):
        return paths.precomputed_scale(dataset_path)[0]
    node, kind = metadata.node_type(dataset_path)
    # A per-scale path (.../s1) is an array; the pyramid is described by the
    # group above it.
    return paths.parent(node) if kind == "array" else node


def closest_raw_scale(dataset_path: str, target_voxel_size) -> Optional[tuple]:
    """The voxel size (nm, z/y/x) of the level of ``dataset_path``'s pyramid
    that ``select_level`` picks for ``target_voxel_size``, or None if it
    can't be determined.

    ``dataset_path`` may be the multiscale group or one of its levels (a
    precomputed volume or one of its scales).
    """
    try:
        levels: List[Level] = metadata.list_levels(_pyramid_of(dataset_path))
        return tuple(select_level(levels, target_voxel_size)[1].spatial().voxel_size)
    except Exception as e:
        logger.warning(
            f"Could not determine closest raw scale for {dataset_path} at "
            f"target_resolution={target_voxel_size}: {e}"
        )
        return None
