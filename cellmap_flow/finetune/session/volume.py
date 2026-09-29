"""Annotation volumes: where they lie over the raw data.

An annotation volume is a zarr v2 group, ``<id>.zarr/annotation/s0``, on the
grid predictions are made on. Its root attr ``dataset_offset_nm`` is voxel
0's *centre* and is also written as the OME translation, since that is
where Neuroglancer draws voxel 0 while it is painted.
"""

import numpy as np

from cellmap_flow.io.ome import ome_corner, ome_translation


def volume_corner_nm(dataset_offset_nm, output_voxel_size) -> np.ndarray:
    """The world position of an annotation volume's voxel-0 lower corner, in nm.

    ``dataset_offset_nm`` (a root attr of every volume) is also written as the
    volume's OME-NGFF translation, and a translation is voxel 0's *centre*.
    Neuroglancer drew the volume that way while it was painted, so that is
    where the labels are. Reading the value as a corner, as this code used to,
    put every label half an annotation voxel away from where it was drawn.
    """
    offset = np.zeros(3) if dataset_offset_nm is None else dataset_offset_nm
    return np.asarray(ome_corner(offset, output_voxel_size), dtype=float)


def new_volume_geometry(raw_dataset_path: str, output_voxel_size, chunk_size):
    """``(dataset_offset_nm, shape_voxels)`` for a new volume over a raw dataset.

    The volume lies on the grid of the raw level at ``output_voxel_size`` (the
    grid predictions are made on), from that level's corner, padded to whole
    chunks. ``dataset_offset_nm`` is voxel 0's centre; see volume_corner_nm.
    """
    from cellmap_flow.image_data_interface import ImageDataInterface

    output_voxel_size = np.asarray(output_voxel_size, dtype=float)
    chunk_size = np.asarray(chunk_size, dtype=int)
    idi = ImageDataInterface(raw_dataset_path, voxel_size=output_voxel_size)
    offset = np.asarray(ome_translation(np.asarray(idi.offset, dtype=float), output_voxel_size))
    shape = (np.asarray(idi.roi.shape, dtype=float) / output_voxel_size).astype(int)
    return offset, np.ceil(shape / chunk_size).astype(int) * chunk_size
