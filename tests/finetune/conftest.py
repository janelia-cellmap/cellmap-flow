"""Fixtures for the finetune tests.

- ``tiny_script``: a script model that the trainer and the CLI can load.
"""

import pytest


SCRIPT = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn

input_voxel_size = output_voxel_size = Coordinate(8, 8, 8)
read_shape = write_shape = Coordinate(4, 4, 4) * input_voxel_size
output_channels = {channels}
torch.manual_seed(0)
model = nn.Sequential(nn.Conv3d(1, 4, 1), nn.ReLU(), nn.Conv3d(4, {channels}, 1))
"""


@pytest.fixture
def tiny_script(tmp_path):
    """``tiny_script(channels=1)``: the path of a script model, 4^3 in and out at 8 nm."""

    def write(channels=1):
        path = tmp_path / f"model_{channels}.py"
        path.write_text(SCRIPT.format(channels=channels))
        return path

    return write
