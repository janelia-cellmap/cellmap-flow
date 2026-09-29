# pip install onnx2torch

#%%
import numpy as np
from funlib.geometry import Coordinate
import onnx
from onnx2torch import convert
import torch

# The model's own geometry. model.onnx was exported for a single input shape,
# 288^3 voxels at 16 nm giving 212^3 at 8 nm (the inference shapes in the
# metadata.json next to it), and the converted graph only runs at that shape.
output_voxel_size = Coordinate((8, 8, 8))
input_voxel_size = Coordinate((16, 16, 16))

read_shape = Coordinate((288, 288, 288)) * input_voxel_size
write_shape = Coordinate((212, 212, 212)) * output_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 1
block_shape = np.array((212, 212, 212, output_channels))




# Load ONNX model
onnx_model_path = "/nrs/cellmap/models/cellmap/jrc_mus-livers_16nm_to_8nm_mito/model.onnx"
# Given a path, convert() writes a temporary file next to the model for shape
# inference, which fails where that directory is read-only, as the catalog's
# is. Given the loaded model, it does not write there.
model = convert(onnx.load(onnx_model_path))
model.eval()
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

