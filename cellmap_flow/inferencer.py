import numpy as np
import torch
from cellmap_flow.utils.data import (
    ModelConfig,
    BioModelConfig,
    DaCapoModelConfig,
    ScriptModelConfig,
)
from funlib.persistence import Array


class Inferencer:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.load_model(model_config)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def process_chunk(self, idi, roi):
        if isinstance(self.config, BioModelConfig):
            return self.process_chunk_bioimagezoo(idi, roi)
        elif isinstance(self.config, DaCapoModelConfig) or isinstance(
            self.config, ScriptModelConfig
        ):
            return self.process_chunk_basic(idi, roi)
        else:
            raise ValueError(f"Invalid model config type {type(self.config)}")

    def process_chunk_basic(self, idi, roi):
        output_roi = roi

        input_roi = output_roi.grow(self.context, self.context)
        # input_roi = output_roi + context
        raw_input = idi.to_ndarray_ts(input_roi).astype(np.float32) / 255.0
        raw_input = np.expand_dims(raw_input, (0, 1))

        with torch.no_grad():
            predictions = Array(
                self.model.forward(torch.from_numpy(raw_input).float().to(self.device))
                .detach()
                .cpu()
                .numpy()[0],
                output_roi,
                self.output_voxel_size,
            )
        write_data = predictions.to_ndarray(output_roi).clip(-1, 1)
        write_data = (write_data + 1) * 255.0 / 2.0
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
            self.load_dacapo_model(config.run_name, iteration=config.iteration)
        elif isinstance(config, ScriptModelConfig):
            self.load_script_model(config)
        elif isinstance(config, BioModelConfig):
            self.load_bio_model(config.model_name)
        else:
            raise ValueError(f"Invalid model config type {type(config)}")

    def load_dacapo_model(self, bio_model_name, iteration="best"):
        from dacapo.store.create_store import create_config_store, create_weights_store
        from dacapo.experiments import Run

        config_store = create_config_store()

        weights_store = create_weights_store()
        run_config = config_store.retrieve_run_config(bio_model_name)

        run = Run(run_config)
        self.model = run.model

        weights = weights_store.retrieve_weights(
            bio_model_name,
            iteration,
        )
        self.model.load_state_dict(weights.model)
        self.model.eval()
        # output_voxel_size = self.model.scale(input_voxel_size)
        # input_shape = Coordinate(model.eval_input_shape)
        # output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

        # context = (input_size - output_size) / 2
        # TODO load this part from dacapo
        # self.read_shape = config.read_shape
        # self.write_shape = config.write_shape
        # self.output_voxel_size = config.output_voxel_size
        # self.context = (self.read_shape - self.write_shape) / 2

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


# %%
