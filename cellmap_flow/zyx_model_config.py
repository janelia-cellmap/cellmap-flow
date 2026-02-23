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

checkpoint_path = "/groups/fishemf/fishemf/zouinkhim/fish_model_project/setups/setup_4/model_checkpoint_90000"
logger.warning(f"Loading model weights from checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
logger.warning("Model weights loaded successfully.")


model = model.to(device)
model.eval()
output_channels = 1
output_voxel_size = Coordinate((1, 1, 1))  # z,y,x order
input_voxel_size = Coordinate((1, 1, 1))

read_shape = Coordinate((286, 286, 36))
write_shape = Coordinate((56, 56, 8))
block_shape = np.array((56, 56, 8, output_channels))
read_shape = read_shape * input_voxel_size
write_shape = write_shape * output_voxel_size
# * Coordinate(output_voxel_size)


context = (read_shape - write_shape) / 2
logger.warning(f"Model context: {context}")

output_dtype = np.uint8
# def process_chunk(idi, input_roi):

#     chunk = idi.to_ndarray_ts(input_roi)
#     chunk = chunk[..., np.newaxis]
#     return chunk


def process_chunk(idi, input_roi):

    input_roi = input_roi.grow(context, context)
    logger.warning(f"Processing chunk with input ROI: {input_roi}")
    chunk = idi.to_ndarray_ts(input_roi)
    logger.warning(f"Original chunk shape: {chunk.shape}")
    chunk = np.transpose(chunk, (2, 1, 0))
    # logger.warning(f"Transposed chunk shape: {chunk.shape}")

    with torch.no_grad():
        chunk = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).float()
        chunk = chunk.to(device)

        logger.warning(f"Input chunk shape: {chunk.shape}")
        result = model(chunk)
        # argmax
        result = torch.argmax(result, dim=1, keepdim=True)
        # result = torch.softmax(result, dim=1)
        logger.warning(f"Output chunk shape after model: {result.shape}")

        result = result.squeeze(0)
        result = result.permute(3, 2, 1, 0)  # c^,z,y,x to z,y,x,c^
        # result = result.permute(1, 2, 3, 0)
        result = result.cpu().detach().numpy()
        result = np.ascontiguousarray(result)
        logger.warning(f"Output chunk shape after transpose: {result.shape}")
        # logger.warning(
        #     f"Min/Max of output chunk after transpose: {result.min()}/{result.max()}"
        # )
        # logger.warning(f"Output chunk shape: {result.shape}")
        return result
