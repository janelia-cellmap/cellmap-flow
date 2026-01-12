# Model configuration for fish segmentation
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import logging
from funlib.geometry.coordinate import Coordinate
import numpy as np
from fish_trainer.model import Unet3d

logger = logging.getLogger(__name__)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model = Unet3d(num_features=32, feature_factor=2)

checkpoint_path = "/groups/fishemf/fishemf/zouinkhim/fish_model_project/setups/setup_3/model_checkpoint_200000"
logger.warning(f"Loading model weights from checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
logger.warning("Model weights loaded successfully.")

# model = torch.nn.Sequential(model, torch.nn.Softmax(dim=0))

model = model.to(device)
model.eval()

# Model configuration in its native coordinate system
# Model expects z,y,x input and returns c^,z,y,x output
axes_names = ["c^", "z", "y", "x"]
output_voxel_size = Coordinate((30, 8, 8))  # z,y,x order
input_voxel_size = Coordinate((30, 8, 8))
read_shape = Coordinate((36, 286, 286))
# * Coordinate(input_voxel_size)  # z,y,x order
write_shape = Coordinate((8, 56, 56))
# * Coordinate(output_voxel_size)
output_channels = 3
block_shape = np.array((3, 8, 56, 56))  # c^,z,y,x order to match model's output
