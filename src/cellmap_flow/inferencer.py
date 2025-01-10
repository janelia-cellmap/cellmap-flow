# %%
# original dacapo i used: dacapo-ml                 0.3.1.dev428+g3abcf96          pypi_0    pypi
# had to add skip_on_failure=True to @root_validators in cellmap-schema
import numpy as np
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run
from bioimageio.core import load_description
from bioimageio.core import predict  # , predict_many
from bioimageio.core import Tensor
from bioimageio.core import Sample
from bioimageio.core.digest_spec import get_member_ids


class Inferencer:
    def __init__(self, model_name=None):
        if model_name:
            self.load_model(model_name)

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

    def process_chunk_dacapo(self, idi, roi):
        return True

    def process_chunk(self, idi, roi):
        if self.is_bioimage_model:
            return self.process_chunk_bioimagezoo(idi, roi)
        else:
            return self.process_chunk_dacapo(idi, roi)

    def load_model_dacapo(self, model_name, iteration="best"):
        config_store = create_config_store()

        weights_store = create_weights_store()
        run_config = config_store.retrieve_run_config(model_name)

        run = Run(run_config)
        model = run.model

        weights = weights_store.retrieve_weights(
            model_name,
            iteration,
        )
        model.load_state_dict(weights.model)
        model.to("cuda")
        model.eval()

    def load_model(self, model_name):
        try:
            self.model = load_description(model_name)
            self.is_bioimage_model = True
        except:
            self.model = self.load_model_dacapo(model_name, iteration="best")
            self.is_bioimage_model = False


# %%
