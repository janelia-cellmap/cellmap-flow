# %%
from funlib.geometry.coordinate import Coordinate
import numpy as np

input_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate((10, 10, 10)) * Coordinate(input_voxel_size)
write_shape = Coordinate((10, 10, 10)) * Coordinate(input_voxel_size)
output_voxel_size = Coordinate(8, 8, 8)

# %%
import torch
import torch.nn as nn


class FakeModel(nn.Module):
    def __init__(self, expected_output: torch.Tensor):
        super().__init__()
        self.expected_output = expected_output

    def forward(self, x):
        return self.expected_output


# %%


classes = ["mito", "er", "nuc", "pm", "ves", "ld"]

output_channels = 8
block_shape = np.array((10, 10, 10, output_channels))
# Models return (batch, channel, z, y, x). Channel c is filled with c + 1 so a
# test can tell whether the server moved the channel axis last for zarr.
model = FakeModel(
    expected_output=torch.arange(1, output_channels + 1, dtype=torch.float32)
    .view(1, output_channels, 1, 1, 1)
    .expand(1, output_channels, 10, 10, 10)
)
