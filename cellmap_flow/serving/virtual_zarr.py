"""The zarr v2 array an inference server makes up, as data.

A server serves one model's output as ``<dataset>/.zattrs``,
``<dataset>/s0/.zarray`` and ``<dataset>/s0/<z>.<y>.<x>[.<c>]`` chunks, which
neuroglancer reads as an OME-NGFF 0.4 group with one level. What those say,
and which region each chunk key is, is plain arithmetic on the model's
geometry and the raw data's; these functions are that arithmetic, with no
Flask and no process-wide state, so the server's routes only call them.

Every client reads this format, including dashboards older than the server,
so it changes only additively.
"""

import logging

import numcodecs
import numpy as np
from funlib.geometry import Roi

from cellmap_flow.io.ome import multiscales_attrs

logger = logging.getLogger(__name__)

# The chunk codec, which .zarray announces and chunk_encoder() applies.
BLOSC = {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1}


def zattrs(axes, output_voxel_size, origin, has_channel: bool, name: str) -> dict:
    """The group's OME-NGFF attributes: one level, s0, at the output voxel size.

    ``axes`` are the spatial axes, ``origin`` the lower corner of output
    voxel 0 in nm; OME writes voxel 0's centre, so neuroglancer draws voxel 0
    at the origin. A channel axis comes last, with scale 1 and translation 0.
    """
    names = list(axes) + (["c"] if has_channel else [])
    scale = [float(v) for v in output_voxel_size] + ([1.0] if has_channel else [])
    corner = [float(v) for v in origin] + ([0.0] if has_channel else [])
    units = ["nanometer"] * len(axes) + ([""] if has_channel else [])
    return multiscales_attrs(names, units, [("s0", scale, corner)], name=name)


def zarray(shape, chunks, dtype) -> dict:
    """The array's zarr v2 metadata. ``dtype`` is anything np.dtype takes;
    its typestr is written ("<f4", "|u1", ...)."""
    return {
        "chunks": list(chunks),
        "compressor": dict(BLOSC),
        "dtype": np.dtype(dtype).str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(shape),
        "zarr_format": 2,
    }


def served_spatial_shape(raw_offset, raw_shape, raw_voxel_size, origin, output_voxel_size) -> list:
    """Output voxels from ``origin`` to the end of the raw data, rounded up."""
    raw_end = np.array(raw_offset, dtype=float) + np.array(raw_shape, dtype=float) * np.array(
        raw_voxel_size, dtype=float
    )
    voxel = np.array(output_voxel_size, dtype=float)
    return [int(v) for v in np.ceil((raw_end - np.array(origin, dtype=float)) / voxel)]


def chunk_roi(index, spatial_block, output_voxel_size, origin) -> Roi:
    """The region (nm) chunk ``index`` covers: ``spatial_block`` output voxels
    per chunk, the grid starting at ``origin``."""
    block = np.array(spatial_block)
    corner = block * np.array(index)
    box = np.array([corner, block]) * output_voxel_size
    return Roi(tuple(int(v) for v in origin + box[0]), tuple(int(v) for v in box[1]))


def reorder_to_zarr_axes(data, model_axes, spatial_axes) -> np.ndarray:
    """``data``, laid out as ``model_axes``, in zarr's order: the spatial axes, then "c"."""
    zarr_axes = tuple(spatial_axes) + ("c",)
    model_axes = tuple(model_axes)

    if len(model_axes) != data.ndim:
        logger.warning(
            f"Model output ndim ({data.ndim}) != declared axes {model_axes}, "
            "skipping reorder"
        )
        return data

    if model_axes == zarr_axes:
        return data

    # For single-channel output the byte layout is identical regardless of
    # where the size-1 channel axis sits, so skip the expensive copy.
    c_idx = model_axes.index("c")
    if data.shape[c_idx] == 1:
        return data.reshape([data.shape[model_axes.index(ax)] for ax in zarr_axes])

    perm = tuple(model_axes.index(ax) for ax in zarr_axes)
    return np.ascontiguousarray(data.transpose(perm))


def chunk_encoder() -> numcodecs.Blosc:
    """The codec for the chunks, as ``BLOSC`` describes it."""
    return numcodecs.Blosc(cname=BLOSC["cname"], clevel=BLOSC["clevel"], shuffle=BLOSC["shuffle"])
