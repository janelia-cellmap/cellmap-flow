"""The voxel size of the raw level a model's output is drawn over.

Choosing a level for a voxel size is worldarray's (``select_level``, its
spec's Behaviour 3; ``wa.open(path, voxel_size=...)`` applies it). What stays
here is the viewer's question: given any path into a pyramid and a model's
declared voxel size, which stored level does the model actually read, so
that its output can be placed over the raw at that level's size.
"""

import logging
from typing import Optional

import worldarray as wa

logger = logging.getLogger(__name__)


def closest_raw_scale(dataset_path: str, target_voxel_size) -> Optional[tuple]:
    """The voxel size (nm, z/y/x) of the level of ``dataset_path``'s pyramid
    that a model at ``target_voxel_size`` reads (worldarray's "floor": the
    level at that size, else the coarsest no coarser than it on any axis,
    else the first), or None, with a warning, when it cannot be determined.

    ``dataset_path`` may be the multiscale group or one of its levels, a
    precomputed volume or one of its scales, or a lone array (its own size).
    The sizes come from the group's own document where it lists them
    (OME-NGFF, COSEM N5, precomputed); no level is opened then.
    """
    try:
        pyramid = wa.open_pyramid(dataset_path, units="nanometer", strict_units=False)
        _, voxel_size = wa.select_level(pyramid.voxel_sizes(), target_voxel_size, "floor")
        return tuple(float(v) for v in voxel_size)
    except Exception as e:
        logger.warning(
            f"Could not determine closest raw scale for {dataset_path} at "
            f"target_resolution={target_voxel_size}: {e}"
        )
        return None
