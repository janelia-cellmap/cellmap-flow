import numpy as np
import torch
from cellmap_flow.utils.data import (
    ModelConfig,
    BioModelConfig,
    DaCapoModelConfig,
    ScriptModelConfig,
)
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from funlib.persistence import Array
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

    raw_input = idi.to_ndarray_ts(read_roi)
    # raw_input = np.expand_dims(raw_input, (0, 1))
    raw_input = config.input_normalizer.normalize(raw_input)
    raw_input = np.expand_dims(raw_input, (0, 1))

    with torch.no_grad():
        return (
            config.model.forward(torch.from_numpy(raw_input).float().to(device))
            .detach()
            .cpu()
            .numpy()[0]
        )


class Inferencer:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.load_model(model_config)

        if not hasattr(self.model_config.config, "input_normalizer"):
            logger.warning("No input normalization function provided, using default")
            self.model_config.config.input_normalizer = MinMaxNormalizer()

        if not hasattr(self.model_config.config, "normalize_output"):
            logger.warning("No output normalization function provided, using default")
            self.model_config.config.normalize_output = normalize_output
        if not hasattr(self.model_config.config, "predict"):
            logger.warning("No predict function provided, using default")
            self.model_config.config.predict = predict
        if self.model:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.model.to(self.device)
            print(f"Using device: {self.device}")

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
            input_roi, output_roi, self.model_config.config, idi=idi, device=self.device
        )

        predictions = Array(
            result,
            output_roi,
            self.output_voxel_size,
        )
        write_data = predictions.to_ndarray(output_roi)
        write_data = self.model_config.config.normalize_output(write_data)

        if write_data.dtype != np.uint8:
            logger.error(
                f"Model output is not of type uint8, converting to uint8. Output type: {write_data.dtype}"
            )
        return write_data.astype(np.uint8)

    # create random input tensor
    def process_chunk_bioimagezoo(self, idi, roi):
        input_image = idi.to_ndarray_ts(roi)
        if len(self.model.outputs[0].axes) == 5:
            input_image = input_image[np.newaxis, np.newaxis, ...].astype(np.float32)
            test_input_tensor = Tensor.from_numpy(
                input_image, dims=["batch", "c", "z", "y", "x"]
            )
        else:
            input_image = input_image[:, np.newaxis, ...].astype(np.float32)

            test_input_tensor = Tensor.from_numpy(
                input_image, dims=["batch", "c", "y", "x"]
            )
        sample_input_id = get_member_ids(self.model.inputs)[0]
        sample_output_id = get_member_ids(self.model.outputs)[0]

        sample = Sample(
            members={sample_input_id: test_input_tensor},
            stat={},
            id="sample-from-numpy",
        )
        prediction: Sample = predict(
            model=self.model, inputs=sample, skip_preprocessing=sample.stat is not None
        )
        ndim = prediction.members[sample_output_id].data.ndim
        output = prediction.members[sample_output_id].data.to_numpy()
        if ndim < 5 and len(self.model.outputs) > 1:
            if len(self.model.outputs) > 1:
                outputs = []
                for id in get_member_ids(self.model.outputs):
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

    def load_model(self, config: ModelConfig):
        if isinstance(config, DaCapoModelConfig):
            # self.load_dacapo_model(config.run_name, iteration=config.iteration)
            self.load_script_model(config)
        elif isinstance(config, ScriptModelConfig):
            self.load_script_model(config)
        elif isinstance(config, BioModelConfig):
            self.load_bio_model(config.model_name)
        else:
            raise ValueError(f"Invalid model config type {type(config)}")

    def load_bio_model(self, bio_model_name):
        from bioimageio.core import load_description
        from bioimageio.core import predict  # , predict_many
        from bioimageio.core import Tensor
        from bioimageio.core import Sample
        from bioimageio.core.digest_spec import get_member_ids

        self.model = load_description(bio_model_name)

    def load_script_model(self, model_config: ScriptModelConfig):
        config = model_config.config
        self.model = config.model
        self.read_shape = config.read_shape
        self.write_shape = config.write_shape
        self.output_voxel_size = config.output_voxel_size
        self.context = (self.read_shape - self.write_shape) / 2

