import torch

# What the network reads and writes.
input_voxel_size = (8, 8, 8)     # nm, in z, y, x order
output_voxel_size = (8, 8, 8)
input_size = (64, 64, 64)        # voxels in
output_size = (60, 60, 60)       # voxels out
output_channels = 1
channels = ["example"]           # optional channel names

# The network. Replace this with your own network and weights.
model = torch.nn.Sequential(
    torch.nn.Conv3d(1, 8, 3), torch.nn.ReLU(),
    torch.nn.Conv3d(8, 1, 3), torch.nn.Sigmoid(),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
