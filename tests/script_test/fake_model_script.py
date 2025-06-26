# %%
from funlib.geometry.coordinate import Coordinate
import numpy as np

input_voxel_size = (8, 8, 8)
read_shape = Coordinate((60, 60, 60)) * Coordinate(input_voxel_size)
write_shape = Coordinate((60, 60, 60)) * Coordinate(input_voxel_size)
output_voxel_size = Coordinate((8, 8, 8))

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
block_shape = np.array((60, 60, 60, output_channels))
model = FakeModel(expected_output=torch.ones(1, *block_shape))
