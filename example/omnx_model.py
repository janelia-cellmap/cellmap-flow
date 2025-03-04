# pip install onnxruntime

import numpy as np
from funlib.geometry import Coordinate
import onnxruntime as ort


output_voxel_size = Coordinate((16, 16, 16))
voxel_size = Coordinate((8, 8, 8))
input_voxel_size = Coordinate((8, 8, 8))



read_shape = Coordinate((216, 216, 216)) * voxel_size
write_shape = Coordinate((68, 68, 68)) * output_voxel_size
context = (read_shape - write_shape) / 2

output_channels = 1
block_shape = np.array((68, 68, 68, output_channels))
model = None




# Load ONNX model
onnx_model_path = "/groups/cellmap/cellmap/zouinkhim/models/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1/model.onnx"
session = ort.InferenceSession(onnx_model_path)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

def process_chunk(idi, input_roi):
    # read in raw
    input_roi = input_roi.grow(context, context)
    chunk = idi.to_ndarray_ts(input_roi)
    output_data = session.run(None, {input_name: chunk})

    output_data = output_data.clip(0, 1) * 255.0
    return output_data.astype(np.uint8)
