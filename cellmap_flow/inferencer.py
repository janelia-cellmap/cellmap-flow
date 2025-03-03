# %%
import numpy as np
import torch
from funlib.geometry import Coordinate
import logging
from cellmap_flow.utils.data import (
    ModelConfig,
    BioModelConfig,
    DaCapoModelConfig,
    ScriptModelConfig,
    CellMapModelConfig,
)
import cellmap_flow.globals as g
import neuroglancer
from scipy import spatial

logger = logging.getLogger(__name__)


def apply_postprocess(data, **kwargs):
    for pross in g.postprocess:
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
        if isinstance(self.model_config, BioModelConfig):
            result = self.process_chunk_bioimagezoo(idi, roi)
        elif (
            isinstance(self.model_config, DaCapoModelConfig)
            or isinstance(self.model_config, ScriptModelConfig)
            or isinstance(self.model_config, CellMapModelConfig)
        ):
            # check if process_chunk is in self.config
            if getattr(self.model_config.config, "process_chunk", None) and callable(
                self.model_config.config.process_chunk
            ):
                result = self.model_config.config.process_chunk(idi, roi)
            else:
                result = self.process_chunk_basic(idi, roi)
        else:
            raise ValueError(f"Invalid model config type {type(self.model_config)}")

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
        # result = apply_postprocess(result)
        return result

    # create random input tensor
    def process_chunk_bioimagezoo(self, idi, roi):
        from bioimageio.core import predict  # , predict_many
        from bioimageio.core import Tensor
        from bioimageio.core import Sample
        from bioimageio.core.digest_spec import get_member_ids

        input_image = idi.to_ndarray_ts(roi)
        if len(self.model_config.config.model.outputs[0].axes) == 5:
            input_image = input_image[np.newaxis, np.newaxis, ...].astype(np.float32)
            test_input_tensor = Tensor.from_numpy(
                input_image, dims=["batch", "c", "z", "y", "x"]
            )
        else:
            input_image = input_image[:, np.newaxis, ...].astype(np.float32)

            test_input_tensor = Tensor.from_numpy(
                input_image, dims=["batch", "c", "y", "x"]
            )
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

    # def calculate_equivalences(self):
    #     g.postprocess[-1].calculate_equivalences(self.equivalences)
    #     # ids = list(self.edge_voxel_position_to_id_dict.values())
    #     # tree = spatial.cKDTree(positions)
    #     # neighbors = tree.query_ball_tree(tree, 1)  # distance of 1 voxel
    #     # for i in range(len(neighbors)):
    #     #     for j in neighbors[i]:
    #     #         self.equivalences.union(ids[i], ids[j])


# %%
import numpy as np

# rreate random array of size 66352 x 3
data = np.random.random((66352, 3))
tree = spatial.cKDTree(data)
# neighbors = tree.query_ball_tree(tree, 1)  # distance of 1 voxel

# %%
# a = {1: 2}
# b = a.copy()
# a.update({3: 4})
# print(b)
# %%
