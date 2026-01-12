# %%
import numpy as np
import torch
from funlib.geometry import Coordinate
import logging
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.globals import g


logger = logging.getLogger(__name__)


def apply_postprocess(data, **kwargs):

    for pross in g.postprocess:
        # logger.error(f"applying postprocess: {pross}")
        data = pross(data, **kwargs)
    return data


def predict(read_roi, write_roi, config, **kwargs):
    idi = kwargs.get("idi")
    if idi is None:
        raise ValueError("idi must be provided in kwargs")

    device = kwargs.get("device")
    if device is None:
        raise ValueError("device must be provided in kwargs")

    use_half_prediction = kwargs.get("use_half_prediction", True)

    # Compare spatial axes only (exclude "c^" if present in model axes)
    raw_axes = idi.axes_names  # e.g., ["x", "y", "z"]
    model_axes = [ax for ax in config.axes_names if ax != "c^"]  # e.g., ["z", "y", "x"]

    # Convert ROI from model axis order to raw data axis order
    if raw_axes != model_axes:
        # Map each model axis position to its corresponding raw axis position
        # For model=['z','y','x'] and raw=['x','y','z']:
        # model[0]='z' -> raw[2], model[1]='y' -> raw[1], model[2]='x' -> raw[0]
        # So roi_axis_map = [2, 1, 0]
        roi_axis_map = [raw_axes.index(ax) for ax in model_axes]
        # Reorder ROI coordinates from model order to raw order
        read_roi_begin = read_roi.get_begin()
        read_roi_shape = read_roi.get_shape()
        reordered_begin = Coordinate(
            [read_roi_begin[model_axes.index(ax)] for ax in raw_axes]
        )
        reordered_shape = Coordinate(
            [read_roi_shape[model_axes.index(ax)] for ax in raw_axes]
        )
        from funlib.geometry import Roi

        read_roi = Roi(reordered_begin, reordered_shape)

    # Get raw data in its native axes
    raw_input = idi.to_ndarray_ts(read_roi)

    # Check if we need to reorder axes from raw data to model's expected spatial order
    if raw_axes != model_axes:
        # Need to transpose from raw axes to model axes
        # Map each model axis to its position in raw axes
        axis_map = [raw_axes.index(ax) for ax in model_axes]
        raw_input = np.transpose(raw_input, axis_map)
        logger.debug(
            f"Transposed input from {raw_axes} to {model_axes} using map {axis_map}"
        )

    # Add batch and channel dimensions
    raw_input = np.expand_dims(raw_input, (0, 1))

    with torch.no_grad():
        raw_input_torch = torch.from_numpy(raw_input).float()
        if use_half_prediction:
            raw_input_torch = raw_input_torch.half()
        raw_input_torch = raw_input_torch.to(device, non_blocking=True)
        output = config.model.forward(raw_input_torch).detach().cpu().numpy()[0]

    # Transpose output back to raw data axes if needed
    if raw_axes != model_axes:
        # Output channels stay first, transpose spatial dimensions
        # Map each raw axis to its position in model axes
        inverse_axis_map = [model_axes.index(ax) for ax in raw_axes]
        # Build full transpose with channel dimension (0) first, then spatial dimensions (shifted by 1)
        full_transpose = [0] + [i + 1 for i in inverse_axis_map]
        output = np.transpose(output, full_transpose)
        # Make C-contiguous for encoding
        output = np.ascontiguousarray(output)
        logger.debug(
            f"Transposed output from {model_axes} back to {raw_axes} using {full_transpose}"
        )

    return output


class Inferencer:
    def __init__(self, model_config: ModelConfig, use_half_prediction=True):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.error("No GPU available, using CPU")
        # torch.backends.cudnn.allow_tf32 = True  # May help performance with newer cuDNN
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True  # Find best algorithm for the hardware

        self.use_half_prediction = use_half_prediction
        self.model_config = model_config
        # config is lazy so one call is needed to get the config
        _ = self.model_config.config

        if hasattr(self.model_config.config, "read_shape") and hasattr(
            self.model_config.config, "write_shape"
        ):
            self.context = (
                Coordinate(self.model_config.config.read_shape)
                - Coordinate(self.model_config.config.write_shape)
            ) / 2

        self.optimize_model()
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
        # check if process_chunk is in self.config
        if getattr(self.model_config.config, "process_chunk", None) and callable(
            self.model_config.config.process_chunk
        ):
            result = self.model_config.config.process_chunk(idi, roi)
        else:
            result = self.process_chunk_basic(idi, roi)

        postprocessed = apply_postprocess(
            result,
            chunk_corner=tuple(roi.get_begin() // roi.get_shape()),
            chunk_num_voxels=np.prod(roi.get_shape() // idi.output_voxel_size),
        )
        return postprocessed

    def process_chunk_basic(self, idi, roi):
        output_roi = roi

        # Convert context from voxels to world coordinates
        # Context is in model's voxel space, so use model's input_voxel_size
        context_in_world = self.context * self.model_config.config.input_voxel_size
        input_roi = output_roi.grow(context_in_world, context_in_world)
        result = self.model_config.config.predict(
            input_roi,
            output_roi,
            self.model_config.config,
            idi=idi,
            device=self.device,
            use_half_prediction=self.use_half_prediction,
        )
        return result
