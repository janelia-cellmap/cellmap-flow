# pip install onnxruntime
# pip install onnx2torch

#%%
import numpy as np
from funlib.geometry import Coordinate
# import onnxruntime as ort
from onnx2torch import convert
import torch

# output_voxel_size = Coordinate((8, 8, 8))
# # voxel_size = Coordinate((8, 8, 8))
# input_voxel_size = Coordinate((16, 16, 16))

# for hideo data
output_voxel_size = Coordinate((6, 6, 6))
input_voxel_size = Coordinate((12, 12, 12))

read_shape = Coordinate((216, 216, 216)) * input_voxel_size
write_shape = Coordinate((68, 68, 68)) * output_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 1
block_shape = np.array((68, 68, 68, output_channels))




# Load ONNX model
onnx_model_path = "/groups/cellmap/cellmap/zouinkhim/models/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1/model.onnx"
model = convert(onnx_model_path)
model.eval()
model.to(torch.device("cuda"))

