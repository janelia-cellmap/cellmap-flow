# %%
import numpy as np
import torch
from funlib.geometry import Coordinate
import logging
from cellmap_flow.utils.data import ModelConfig

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

    raw_input = idi.to_ndarray_ts(read_roi)
    raw_input = np.expand_dims(raw_input, (0, 1))

    with torch.no_grad():
        raw_input_torch = torch.from_numpy(raw_input).float()
        if use_half_prediction:
            raw_input_torch = raw_input_torch.half()
        # raw_input_torch = raw_input_torch.to(device)
        raw_input_torch = raw_input_torch.to(device, non_blocking=True)
        return config.model.forward(raw_input_torch).detach().cpu().numpy()[0]


class Inferencer:
    def __init__(self, model_config: ModelConfig, use_half_prediction=True):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.error("No GPU available, using CPU")
        torch.backends.cudnn.allow_tf32 = True  # May help performance with newer cuDNN
        torch.backends.cudnn.enabled = True

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

        input_roi = output_roi.grow(self.context, self.context)
        result = self.model_config.config.predict(
            input_roi,
            output_roi,
            self.model_config.config,
            idi=idi,
            device=self.device,
            use_half_prediction=self.use_half_prediction,
        )
        return result
