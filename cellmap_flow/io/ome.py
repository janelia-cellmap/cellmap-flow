"""OME-NGFF coordinate conventions.

OME-NGFF places ``translation`` at the *centre* of voxel 0 (Neuroglancer's
ome.ts subtracts half a voxel for the same reason), while everything in
cellmap-flow works with the lower corner of voxel 0. Readers convert with
``ome_corner`` and writers with ``ome_translation``; nothing else should
add or subtract half a voxel.
"""

from typing import List, Sequence


def ome_corner(translation: Sequence[float], scale: Sequence[float]) -> List[float]:
    """The lower corner of voxel 0, from an OME-NGFF ``translation``.

    Janelia pyramids store ``translation = scale/2 - 4`` per level, so every
    level's corner is -4 nm; read as a corner, each level sat half its voxel
    off (s1 8 nm, s2 16 nm) and the levels did not even agree with each
    other. ``translation`` and ``scale`` must be in the same units.
    """
    return [float(t) - float(s) / 2 for t, s in zip(translation, scale)]


def ome_translation(corner: Sequence[float], scale: Sequence[float]) -> List[float]:
    """The OME-NGFF ``translation`` (voxel-0 centre) for a lower ``corner``."""
    return [float(c) + float(s) / 2 for c, s in zip(corner, scale)]
