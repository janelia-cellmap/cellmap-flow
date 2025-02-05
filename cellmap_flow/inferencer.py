# %%
import numpy as np
import torch
from cellmap_flow.utils.data import (
    ModelConfig,
    BioModelConfig,
    DaCapoModelConfig,
    ScriptModelConfig,
)
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from funlib.geometry import Coordinate
import logging

logger = logging.getLogger(__name__)


def normalize_output(data):
    # Default normalization if no one is provided
    data = data.clip(-1, 1)
    data = (data + 1) * 255.0 / 2.0
    return data.astype(np.uint8)


def predict(read_roi, write_roi, config, **kwargs):
    idi = kwargs.get("idi")
    if idi is None:
        raise ValueError("idi must be provided in kwargs")

    device = kwargs.get("device")
    if device is None:
        raise ValueError("device must be provided in kwargs")

    use_half_prediction = kwargs.get("use_half_prediction", False)

    raw_input = idi.to_ndarray_ts(read_roi)
    raw_input = config.input_normalizer.normalize(raw_input)
    raw_input = np.expand_dims(raw_input, (0, 1))

    with torch.no_grad():
        raw_input_torch = torch.from_numpy(raw_input).float()
        if use_half_prediction:
            raw_input_torch = raw_input_torch.half()
        raw_input_torch = raw_input_torch.to(device, non_blocking=True)
        return config.model.forward(raw_input_torch).detach().cpu().numpy()[0]


class Inferencer:
    def __init__(self, model_config: ModelConfig, use_half_prediction=False):
        self.use_half_prediction = use_half_prediction
        self.model_config = model_config
        # condig is lazy so one call is needed to get the config
        _ = self.model_config.config

        if hasattr(self.model_config.config, "read_shape") and hasattr(
            self.model_config.config, "write_shape"
        ):
            self.context = (
                Coordinate(self.model_config.config.read_shape)
                - Coordinate(self.model_config.config.write_shape)
            ) / 2

        self.optimize_model()

        if not hasattr(self.model_config.config, "input_normalizer"):
            logger.warning("No input normalization function provided, using default")
            self.model_config.config.input_normalizer = MinMaxNormalizer()

        if not hasattr(self.model_config.config, "normalize_output"):
            logger.warning("No output normalization function provided, using default")
            self.model_config.config.normalize_output = normalize_output
        if not hasattr(self.model_config.config, "predict"):
            logger.warning("No predict function provided, using default")
            self.model_config.config.predict = predict

    def optimize_model(self):
        if not hasattr(self.model_config.config, "model"):
            logger.error("Model is not loaded, cannot optimize")
            return
        if not isinstance(self.model_config.config.model, torch.nn.Module):
            logger.error("Model is not a nn.Module, we only optimize torch models")
            return

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.error("No GPU available, using CPU")
        self.model_config.config.model.to(self.device)
        if self.use_half_prediction:
            self.model_config.config.model.half()
        print(f"Using device: {self.device}")
        # DIDN'T WORK with unet model
        # if torch.__version__ >= "2.0":
        #     self.model_config.config.model = torch.compile(self.model_config.config.model)
        # print("Model compiled")
        self.model_config.config.model.eval()

    def process_chunk(self, idi, roi):
        if isinstance(self.model_config, BioModelConfig):
            return self.process_chunk_bioimagezoo(idi, roi)
        elif isinstance(self.model_config, DaCapoModelConfig) or isinstance(
            self.model_config, ScriptModelConfig
        ):
            # check if process_chunk is in self.config
            if getattr(self.model_config.config, "process_chunk", None) and callable(
                self.model_config.config.process_chunk
            ):
                return self.model_config.config.process_chunk(idi, roi)
            else:
                return self.process_chunk_basic(idi, roi)
        else:
            raise ValueError(f"Invalid model config type {type(self.model_config)}")

    def process_chunk_basic(self, idi, roi):
        output_roi = roi

        input_roi = output_roi.grow(self.context, self.context)
        result = self.model_config.config.predict(
            input_roi,
            output_roi,
            self.model_config.config,
            idi=idi,
            device=self.device,
            use_half_prediction=self.use_half_prediction,
        )
        write_data = self.model_config.config.normalize_output(result)

        if write_data.dtype != np.uint8:
            logger.error(
                f"Model output is not of type uint8, converting to uint8. Output type: {write_data.dtype}"
            )
        return write_data.astype(np.uint8)

    # create random input tensor
    def process_chunk_bioimagezoo(self, idi, roi):
        from bioimageio.core import predict  # , predict_many
        from bioimageio.core import Tensor
        from bioimageio.core import Sample
        from bioimageio.core.digest_spec import get_member_ids

        test_input_tensor = Tensor.from_numpy(
            input_image, dims=["batch", "c", "z", "y", "x"]
        )

        # assume that our input data is always 3d, zyx and only has one channel
        slicer = []
        input_axes = self.model_config.config.input_axes
        slicer = tuple(
            [
                (
                    np.newaxis
                    if a == "c" or a == "batch" and "z" in input_axes
                    else slice(None)
                )
                for a in input_axes
            ]
        )

        input_image = idi.to_ndarray_ts(roi)
        input_image = input_image[slicer].astype(np.float32)
        test_input_tensor = Tensor.from_numpy(input_image, dims=input_axes)

        sample_input_id = get_member_ids(self.model_config.config.model.inputs)[0]
        sample_output_id = get_member_ids(self.model_config.config.model.outputs)[0]

        sample = Sample(
            members={sample_input_id: test_input_tensor},
            stat={},
            id="sample-from-numpy",
        )
        prediction: Sample = predict(
            model=self.model_config.config.model,
            inputs=sample,
            skip_preprocessing=sample.stat is not None,
        )
        ndim = prediction.members[sample_output_id].data.ndim
        output = prediction.members[sample_output_id].data.to_numpy()
        if ndim < 5 and len(self.model_config.config.model.outputs) > 1:
            if len(self.model.outputs) > 1:
                outputs = []
                for id in get_member_ids(self.model_config.config.model.outputs):
                    output = prediction.members[id].data.to_numpy()
                    if output.ndim == 3:
                        output = output[:, np.newaxis, ...]
                    outputs.append(output)
                output = np.concatenate(outputs, axis=1)
            output = np.ascontiguousarray(np.swapaxes(output, 1, 0))

        else:
            output = output[0, ...]

        output = 255 * output
        output = output.astype(np.uint8)
        return output


# from cellmap_flow.utils.data import ScriptModelConfig, DaCapoModelConfig, BioModelConfig
# from bioimageio.core import load_description
# from bioimageio.spec.utils import download
# from bioimageio.core.digest_spec import create_sample_for_model

# model = load_description("kind-seashell")  # affable-shark")

# # model_config = BioModelConfig(model_name="happy-elephant")
# # model_config.config
# print(model.inputs[0].axes)
# print(model.outputs[0])
# print(model.inputs[0].preprocessing)


# def get_axes_names_from_model(model):
#     def get_axes_names(axes):
#         if type(axes) is str:
#             # assume "b" should be batch
#             return [axis if axis != "b" else "batch" for axis in list(axes)]
#         else:
#             return [axis.id for axis in axes]

#     input_axes = get_axes_names(model.inputs[0].axes)
#     output_axes = get_axes_names(model.outputs[0].axes)
#     return input_axes, output_axes


# input_axes, output_axes = get_axes_names_from_model(model)


# # %%

# print(input_axes)

# slicer = []
# for input_axis in input_axes:
#     if input_axis == "c":
#         slicer.append(np.newaxis)
#     elif input_axis == "batch":
#         if "z" not in input_axes:
#             slicer.append(slice(None))
#         else:
#             slicer.append(np.newaxis)
#     else:
#         slicer.append(slice(None))

# slicer = tuple(
#     [
#         np.newaxis if a == "c" or a == "batch" and "z" in input_axes else slice(None)
#         for a in input_axes
#     ]
# )
# # slicer = tuple(slicer)
# print(slicer)
# # create random 3d array
# a = np.random.rand(10, 10, 10).astype(np.float32)
# print(a[slicer])
# # %%
# print(slicer)
# # %%
# from importlib import reload
# import cellmap_flow.utils.data

# reload(cellmap_flow.utils.data)
# from cellmap_flow.utils.data import BioModelConfig
# from bioimageio.core import load_description
# from bioimageio.core.digest_spec import create_sample_for_model

# model = load_description("happy-elephant")
# BioModelConfig.get_axes_names_from_model(model)

# print(model.inputs[0].shape)
# print(model.outputs[0].shape)
# create_sample_for_model(model)
# # bmc = BioModelConfig(model_name="happy-elephant", voxel_size=Coordinate(5, 5, 5))
# # bmc.config
# # %%
# # %%
