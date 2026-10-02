"""OME-NGFF coordinate conventions.

OME-NGFF places ``translation`` at the *centre* of voxel 0 (Neuroglancer's
ome.ts subtracts half a voxel for the same reason), while everything in
cellmap-flow works with the lower corner of voxel 0. Readers convert with
``ome_corner`` and writers with ``ome_translation``; nothing else should
add or subtract half a voxel. ``multiscales_attrs`` writes the attributes.
"""

from typing import Any, Dict, List, Sequence, Tuple


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


# The axis names that are channels, never space, wherever cellmap-flow
# names axes: OME axes lists, tensorstore dimension labels, a model's
# chunk_output_axes. They are written with type "channel" (no unit, no
# half-voxel shift). It is defined here rather than in io.metadata, which
# imports this module; import it from here.
CHANNEL_AXIS_NAMES = ("c", "c^", "channel")


def _translation(corner, voxel_size, axes) -> List[float]:
    return [
        float(c) if axis in CHANNEL_AXIS_NAMES else float(c) + float(v) / 2
        for axis, c, v in zip(axes, corner, voxel_size)
    ]


def multiscales_attrs(
    axes: Sequence[str],
    units: Sequence[str],
    levels: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    name: str = "",
    version: str = "0.4",
) -> Dict[str, Any]:
    """OME-NGFF ``{"multiscales": [...]}`` attributes for a pyramid.

    ``levels`` are ``(path, voxel_size, corner)``, one entry per axis in
    ``axes`` order, with ``corner`` the lower corner of voxel 0. Spatial
    axes get ``translation = corner + voxel_size / 2`` and the unit from
    ``units``; the channel axes (named "c", "c^" or "channel") are written
    as they are, with type "channel" and no unit. ``voxel_size`` is written as given.
    """
    axes_list = []
    for axis, unit in zip(axes, units):
        if axis in CHANNEL_AXIS_NAMES:
            axes_list.append({"name": axis, "type": "channel"})
        else:
            axes_list.append({"name": axis, "type": "space", "unit": unit})
    datasets = [
        {
            "coordinateTransformations": [
                {"scale": list(voxel_size), "type": "scale"},
                {"translation": _translation(corner, voxel_size, axes), "type": "translation"},
            ],
            "path": path,
        }
        for path, voxel_size, corner in levels
    ]
    return {
        "multiscales": [
            {
                "axes": axes_list,
                "coordinateTransformations": [{"scale": [1.0] * len(axes), "type": "scale"}],
                "datasets": datasets,
                "name": name,
                "version": version,
            }
        ]
    }


def singlescale_attrs(
    arr_name: str,
    voxel_size: Sequence[float],
    offset: Sequence[float],
    units: Sequence[str],
    axes: Sequence[str],
) -> Dict[str, Any]:
    """``multiscales_attrs`` for one array ``arr_name`` whose voxel 0 has its
    lower corner at ``offset``.

    Writing the corner as the translation put every output half a voxel off
    in Neuroglancer and in any OME reader.
    """
    return multiscales_attrs(axes, units, [(arr_name, voxel_size, offset)])
