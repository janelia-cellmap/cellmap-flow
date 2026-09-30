"""ModelGeometry: a model's sizes read once, and what follows from them."""

from types import SimpleNamespace

import numpy as np
import pytest
from funlib.geometry import Coordinate

from cellmap_flow.models.geometry import ModelGeometry


@pytest.mark.parametrize(
    "config, expected",
    [
        (  # 8 nm in, 16 nm out, a declared block
            dict(input_voxel_size=Coordinate(8, 8, 8), output_voxel_size=Coordinate(16, 16, 16),
                 read_shape=Coordinate(96, 96, 96) * 8, write_shape=Coordinate(4, 4, 4) * 16,
                 output_channels=2, channels=["a", "b"], block_shape=np.array((4, 4, 4, 2))),
            dict(input_shape=(96, 96, 96), output_shape=(4, 4, 4), context=(352, 352, 352),
                 block_shape=(4, 4, 4, 2), has_channel_axis=True, channel_names=("a", "b")),
        ),
        (  # funlib arithmetic: (9 - 4) / 2 floors to 2
            dict(input_voxel_size=(1, 1, 1), output_voxel_size=(1, 1, 1), read_shape=(9, 9, 9),
                 write_shape=(4, 4, 4), output_channels=1, chunk_output_axes=("z", "y", "x")),
            dict(input_shape=(9, 9, 9), output_shape=(4, 4, 4), context=(2, 2, 2),
                 block_shape=(4, 4, 4, 1), has_channel_axis=False, channel_names=None),
        ),
        (  # fractional nm are kept; context truncates them, as Coordinate does
            dict(input_voxel_size=(5.24, 4, 4), output_voxel_size=(5.24, 4, 4),
                 read_shape=np.array([52.4, 40, 40]), write_shape=[26.2, 24.0, 24],
                 output_channels=np.int64(3)),
            dict(input_voxel_size=(5.24, 4, 4), read_shape=(52.4, 40, 40), input_shape=(10, 10, 10),
                 output_shape=(5, 6, 6), context=(13, 8, 8), block_shape=(5, 6, 6, 3)),
        ),
    ],
    ids=["declared", "floor", "fractional"],
)
def test_what_follows_from_a_configs_sizes(config, expected):
    geometry = ModelGeometry.from_config(SimpleNamespace(**config))
    got = {name: getattr(geometry, name) for name in expected}
    got["block_shape"] = geometry.block_shape()
    assert got == expected
    assert all(type(v) in (int, float) for v in geometry.read_shape + geometry.input_voxel_size)
    assert geometry.context == Coordinate(expected["context"])
