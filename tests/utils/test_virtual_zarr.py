"""The served array's arithmetic, without a server (the server's answers are pinned in test_served_metadata_snapshot)."""

import numpy as np
import pytest
from funlib.geometry import Roi

from cellmap_flow.serving import virtual_zarr


def test_the_served_shape_rounds_up_and_chunks_start_at_the_origin():
    # 11 voxels of 8 nm from -4 nm end at 84 nm: 5.5 voxels of 16 nm, so 6.
    origin = np.array([-4] * 3)
    assert virtual_zarr.served_spatial_shape([-4] * 3, [11] * 3, [8] * 3, origin, [16] * 3) == [6] * 3
    assert virtual_zarr.chunk_roi((2, 2, 2), (4, 4, 4), (16, 16, 16), origin) == Roi((124,) * 3, (64,) * 3)


@pytest.mark.parametrize(
    "shape, model_axes, expected_shape",
    [
        ((2, 3, 4, 5), ("c", "z", "y", "x"), (3, 4, 5, 2)),  # transposed
        ((3, 4, 5, 2), ("z", "y", "x", "c"), (3, 4, 5, 2)),  # already in zarr's order
        ((3, 4, 5), ("c", "z", "y", "x"), (3, 4, 5)),  # not what it says it is: left alone
    ],
)
def test_chunks_are_put_in_zarr_order(shape, model_axes, expected_shape):
    data = np.arange(np.prod(shape)).reshape(shape)
    out = virtual_zarr.reorder_to_zarr_axes(data, model_axes, ("z", "y", "x"))
    assert out.shape == expected_shape and out.flags.c_contiguous
    if len(shape) == 4:  # each voxel keeps its own channel values
        where = {"z": 1, "y": 2, "x": 3}
        voxel = tuple(slice(None) if axis == "c" else where[axis] for axis in model_axes)
        assert np.array_equal(out[1, 2, 3], data[voxel])
