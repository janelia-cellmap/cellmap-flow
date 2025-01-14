# https://github.com/bioimage-io/core-bioimage-io-python/blob/53dfc45cf23351da61e8b22d100d77fb54c540e6/example/model_creation.ipynb
# %%
import onnx
import onnxruntime as ort
session = ort.InferenceSession("my_model.onnx")

# Prepare the input data
input_name = session.get_inputs()[0].name

session.get_inputs()[0].shape
import numpy as np
session.run([session.get_outputs()[0], {session.get_inputs()[0]: np.random(1, 1, 128, 128)}])

# %%
