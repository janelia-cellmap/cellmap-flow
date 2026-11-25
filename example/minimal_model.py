"""A complete model script: a small demo network that needs no weights.

It marks the voxels that are clearly darker than their surroundings, which in
EM picks out dense structures. It expects intensities of 0 or more. Replace ``DarkStructures`` with your own network to run
your own model; see docs/source/own_model.rst.
"""

import torch
import torch.nn.functional as F

# What the network reads and writes.
input_voxel_size = (8, 8, 8)     # nm, in z, y, x order
output_voxel_size = (8, 8, 8)
input_size = (64, 64, 64)        # voxels in
output_size = (40, 40, 40)       # voxels out
output_channels = 1
channels = ["dark"]              # optional channel names


class DarkStructures(torch.nn.Module):
    """Whether each voxel is clearly darker than the 25-voxel cube around it, as a probability."""

    def forward(self, x):
        nearby = F.avg_pool3d(x, 5, stride=1)[..., 10:-10, 10:-10, 10:-10]
        surroundings = F.avg_pool3d(x, 25, stride=1)
        darker = (surroundings - nearby) / (surroundings.abs() + 1e-6)
        return torch.sigmoid(30 * (darker - 0.1))


# The network. Replace this with your own network and weights.
model = DarkStructures()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
