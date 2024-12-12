# %%
from bioimageio.core import load_description
from bioimageio.core import predict  # , predict_many

# %%
from bioimageio.core import Tensor
import numpy as np
from bioimageio.core import Sample
from bioimageio.core.digest_spec import get_member_ids


# create random input tensor
def process_chunk(model, idi, roi):
    input_image = idi.to_ndarray_ts(roi)
    # add dimensions to start of input image
    # print(input_image.shape)
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

        test_input_tensor = Tensor.from_numpy(
            input_image, dims=["batch", "c", "y", "x"]
        )
    sample_input_id = get_member_ids(model.inputs)[0]
    sample_output_id = get_member_ids(model.outputs)[0]

    sample = Sample(
        members={sample_input_id: test_input_tensor}, stat={}, id="sample-from-numpy"
    )
    prediction: Sample = predict(
        model=model, inputs=sample, skip_preprocessing=sample.stat is not None
    )
    ndim = prediction.members[sample_output_id].data.ndim
    output = prediction.members[sample_output_id].data.to_numpy()
    if ndim < 5 and len(model.outputs) > 1:
        # append along second axis
        outputs = []
        for id in get_member_ids(model.outputs):
            output = prediction.members[id].data.to_numpy()
            if output.ndim == 3:
                output = output[:, np.newaxis, ...]
            outputs.append(output)
        output = np.concatenate(outputs, axis=1)
        output = np.ascontiguousarray(np.swapaxes(output, 1, 0))
    else:
        output = output[0, ...]
    # if ndim == 5:
    #     # then is b,c,z,y,x, and only want z,y,x
    #     output = 255 * output[0, ...]
    # elif ndim == 4:
    #     # then is b,c,y,x  since it is 2d which is really z,c,y,x
    #     output = 255 * output[0, 0, ...]
    output = 255 * output
    output = output.astype(np.uint8)
    return output
