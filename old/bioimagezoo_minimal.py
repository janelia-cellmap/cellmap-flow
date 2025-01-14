# %%
# try:
#     import matplotlib
#     import torch

#     import bioimageio.core
# except ImportError:
#     %pip install bioimageio.core==0.6.7 torch==2.3.1 matplotlib==3.9.0

# %%
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# determined-chipmunk : edges
# happy-elephant: cells
# kind-seashell: mito
# hiding-blowfish: mito also
BMZ_MODEL_ID = "hiding-blowfish"  # "happy-elephant"#"kind-seashell"  # "happy-elephant"# "happy-elephant" #"kind-seashell"#"happy-elephant"  # "famous-fish"  # "stupendous-sheep"#"kind-seashell"  # "stupendous-sheep"  # "kind-seashell"  # "affable-shark"
from bioimageio.core import load_description
from bioimageio.core import predict  # , predict_many

model = load_description(BMZ_MODEL_ID)
model.weights.pytorch_state_dict
from bioimageio.core.model_adapters._pytorch_model_adapter import PytorchModelAdapter
from bioimageio.core.digest_spec import get_member_ids

# %
# %%
from bioimageio.core import Tensor
import numpy as np
from bioimageio.core import Sample
from bioimageio.core.stat_measures import Stat
from image_data_interface import ImageDataInterface
from funlib.geometry import Roi
from skimage.measure import block_reduce

# create random input tensor

s = 2
idi = ImageDataInterface(
    f"/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s{s}"
)
input_image = idi.to_ndarray_ts(
    Roi((215516, 78499, 105000), (64 * 8 * 2**s, 64 * 8 * 2**s, 64 * 8 * 2**s))
)
# add dimensions to start of input image
# input_image = block_reduce(input_image, (4,1,1), np.mean, cval=0)
# print(input_image.shape)
# input_image = (input_image - np.min(input_image))/np.max(input_image)
# input_image = np.random.rand(1, 1, 64, 64, 64).astype(np.float32)
if len(model.outputs[0].axes) == 5:
    input_image = input_image[np.newaxis, np.newaxis, ...].astype(np.float32)

    test_input_tensor = Tensor.from_numpy(
        input_image, dims=["batch", "c", "z", "y", "x"]
    )
else:
    input_image = input_image[:, np.newaxis, ...].astype(np.float32)
    test_input_tensor = Tensor.from_numpy(input_image, dims=["batch", "c", "y", "x"])


sample_input_id = get_member_ids(model.inputs)[0]
sample_output_id = get_member_ids(model.outputs)[0]

sample = Sample(
    members={sample_input_id: test_input_tensor}, stat={}, id="sample-from-numpy"
)

# predict_many(model=model, inputs=[sample])

prediction: Sample = predict(
    model=model, inputs=sample, skip_preprocessing=sample.stat is not None
)

# show the prediction result
# %%
print(output)
# %%
from matplotlib import pyplot as plt

f, axarr = plt.subplots(1, 2)
# axarr[0].imshow(input_image[0,0,:,:,0],cmap="gray")
ndim = prediction.members[sample_output_id].data.ndim
output = prediction.members[sample_output_id].data.to_numpy()
if ndim < 5:
    # append along second axis
    if len(model.outputs) > 1:
        outputs = []
        for id in get_member_ids(model.outputs):
            output = prediction.members[id].data.to_numpy()
            if output.ndim == 3:
                output = output[:, np.newaxis, ...]
            outputs.append(output)
        output = np.concatenate(outputs, axis=1)
        output = np.swapaxes(output, 1, 0)
else:
    output = output[0, ...]
print(output.shape)
axarr[0].imshow(sample.members[sample_input_id].data[0,0, :, :], cmap="gray")
axarr[1].imshow(output[0, 0, :, :], cmap="gray")

# %%
input_image = idi.to_ndarray_ts(Roi((211159, 76931, 104999), (64 * 8, 64 * 8, 64 * 8)))
# add dimensions to start of input image
input_image = input_image[np.newaxis, np.newaxis, ...].astype(np.float32)
np.min(input_image)
# %%
prediction.members["output0"].data[0, 0, 15, :, :].to_numpy()

# %%
from bioimageio.core import create_prediction_pipeline

create_prediction_pipeline(model, devices=None, weight_format=None)

# %%
