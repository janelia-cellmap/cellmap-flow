# %%
import logging
import warnings

from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi
import copy
from typing import List
import yaml
from funlib.geometry.coordinate import Coordinate
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelConfig:
    def __init__(self):
        self._config = None

    def __str__(self) -> str:
        attributes = vars(self)
        elms = ", ".join(f"{key}: {value}" for key, value in attributes.items())
        return f"{type(self)} : {elms}"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
            check_config(self._config)
        return self._config

    @property
    def output_dtype(self):
        """
        Returns the output dtype of the model.
        If not defined, defaults to np.float32.
        """
        if hasattr(self.config, "output_dtype"):
            return self.config.output_dtype
        logger.warning(
            f"Model {self.name} does not define output_dtype, defaulting to np.float32"
        )
        return np.float32


class ScriptModelConfig(ModelConfig):

    def __init__(self, script_path, name=None, scale=None):
        super().__init__()
        self.script_path = script_path
        self.name = name
        self.scale = scale

    @property
    def command(self):
        return f"script -s {self.script_path}"

    def _get_config(self):
        from cellmap_flow.utils.load_py import load_safe_config

        config = load_safe_config(self.script_path)
        return config


class DaCapoModelConfig(ModelConfig):

    def __init__(self, run_name: str, iteration: int, name=None):
        super().__init__()
        self.run_name = run_name
        self.iteration = iteration
        self.name = name

    @property
    def command(self):
        return f"dacapo -r {self.run_name} -i {self.iteration}"

    def _get_config(self):

        config = Config()

        run = get_dacapo_run_model(self.run_name, self.iteration)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device:", device)

        run.model.to(device)
        run.model.eval()
        config.model = run.model

        in_shape = run.model.eval_input_shape
        out_shape = run.model.compute_output_shape(in_shape)[1]

        voxel_size = run.datasplit.train[0].raw.voxel_size
        config.input_voxel_size = voxel_size
        config.read_shape = Coordinate(in_shape) * Coordinate(voxel_size)
        config.write_shape = Coordinate(out_shape) * Coordinate(voxel_size)
        config.output_voxel_size = Coordinate(run.model.scale(voxel_size))
        channels = get_dacapo_channels(run.task)
        config.output_channels = len(
            channels
        )  # 0:all_mem,1:organelle,2:mito,3:er,4:nucleus,5:pm,6:vs,7:ld
        config.block_shape = np.array(tuple(out_shape) + (len(channels),))

        return config


class BioModelConfig(ModelConfig):
    def __init__(
        self,
        model_name: str,
        voxel_size,
        edge_length_to_process=None,
        name=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.voxel_size = voxel_size
        self.name = name
        self.voxels_to_process = None
        if edge_length_to_process:
            self.voxels_to_process = edge_length_to_process**3

    @property
    def command(self):
        return f"bioimage -m {self.model_name}"

    def _get_config(self):
        from bioimageio.core import load_description
        from funlib.geometry import Coordinate
        from types import MethodType

        config = Config()
        config.model = load_description(self.model_name)

        (
            config.input_name,
            config.input_axes,
            config.input_spatial_dims,
            config.input_slicer,
            is_2d_with_batch,
        ) = self.load_input_information(config.model)

        (
            config.output_names,
            config.output_axes,
            config.block_shape,
            config.output_spatial_dims,
            config.output_channels,
        ) = self.load_output_information(config.model)

        if self.voxels_to_process and not is_2d_with_batch:
            warnings.warn("edge_length_to_process is only supported for 2D models")

        if self.voxels_to_process and is_2d_with_batch:
            batch_size = max(
                1,
                self.voxels_to_process // np.prod(config.input_spatial_dims),
            )
            config.input_spatial_dims[config.input_axes.index("z")] = batch_size
            config.output_spatial_dims[0] = batch_size  # output dims is always zyx
            config.block_shape[0] = batch_size  # block shape is always zyxc

        config.input_voxel_size = Coordinate(self.voxel_size)
        config.output_voxel_size = Coordinate(self.voxel_size)
        config.read_shape = (
            Coordinate(config.input_spatial_dims) * config.input_voxel_size
        )
        config.write_shape = (
            Coordinate(config.output_spatial_dims) * config.output_voxel_size
        )
        config.context = (config.read_shape - config.write_shape) / 2
        config.process_chunk = MethodType(process_chunk_bioimage, config)
        config.format_output_bioimage = MethodType(format_output_bioimage, config)
        return config

    def load_input_information(self, model):
        from bioimageio.core.digest_spec import get_test_inputs

        input_sample = get_test_inputs(model)
        if len(input_sample.members) > 1:
            raise ValueError("Only one input tensor is supported")

        input_name, input_axes, input_dims, is_2d_with_batch = self.get_axes_and_dims(
            input_sample
        )
        input_spatial_dims = self.get_spatial_dims(input_axes, input_dims)

        input_slicer = self.get_input_slicer(input_axes)
        return (
            input_name,
            input_axes,
            input_spatial_dims,
            input_slicer,
            is_2d_with_batch,
        )

    def load_output_information(self, model):
        from bioimageio.core.digest_spec import get_test_outputs

        output_sample = get_test_outputs(model)

        output_names, output_axes, _, _ = self.get_axes_and_dims(output_sample)
        finalized_output, finalized_output_axes = format_output_bioimage(
            None, output_sample, output_names, copy.deepcopy(output_axes)
        )
        # finalized_output_axes should be czyx
        output_dims = finalized_output.shape
        ouput_spatial_dims = [
            output_dims[finalized_output_axes.index(a)] for a in ["z", "y", "x"]
        ]
        output_channels = output_dims[finalized_output_axes.index("c")]
        block_shape = [
            output_dims[finalized_output_axes.index(a)] for a in ["z", "y", "x", "c"]
        ]
        return (
            output_names,
            output_axes,
            block_shape,
            ouput_spatial_dims,
            output_channels,
        )

    def get_axes_and_dims(self, sample):
        sample_names = list(sample.shape.keys())
        sample_axis_to_dims_dicts = list(sample.shape.values())
        sample_axes = []
        sample_dims = []
        is_2d_with_batch = False
        for sample_axis_to_dim_dict in sample_axis_to_dims_dicts:
            # simplify batches--> 'b' and channels--> 'c'. if 'z' isn't present use it instead of batch
            current_sample_axes = sample_axis_to_dim_dict.keys()
            if [
                "b" in current_sample_axes or "batch" in current_sample_axes
            ] and "z" not in current_sample_axes:
                is_2d_with_batch = True

            sample_axes.append(
                [
                    ("z" if (a[0] == "b" and "z" not in current_sample_axes) else a[0])
                    for a in current_sample_axes
                ]
            )
            sample_dims.append(list(sample_axis_to_dim_dict.values()))
        if len(sample_names) == 1:
            return sample_names[0], sample_axes[0], sample_dims[0], is_2d_with_batch
        return sample_names, sample_axes, sample_dims, is_2d_with_batch

    def get_spatial_dims(self, axes, dims):
        spatial_axes = []
        for a, d in zip(axes, dims):
            if a in ["x", "y", "z"]:
                spatial_axes.append(d)
        return spatial_axes

    def get_input_slicer(self, input_axes):
        slicer = tuple(
            [
                (
                    np.newaxis
                    if a.startswith("c") or (a == "b" and "z" in input_axes)
                    else slice(None)
                )
                for a in input_axes
            ]
        )

        return slicer


def check_config(config):
    assert hasattr(config, "model") or hasattr(
        config, "predict"
    ), f"Model or predict not found in config"
    assert hasattr(config, "read_shape"), f"read_shape not found in config"
    assert hasattr(config, "write_shape"), f"write_shape not found in config"
    assert hasattr(config, "input_voxel_size"), f"input_voxel_size not found in config"
    assert hasattr(
        config, "output_voxel_size"
    ), f"output_voxel_size not found in config"
    assert hasattr(config, "output_channels"), f"output_channels not found in config"
    assert hasattr(config, "block_shape"), f"block_shape not found in config"


class Config:
    pass


def get_dacapo_channels(task):
    if hasattr(task, "channels"):
        return task.channels
    elif type(task).__name__ == "AffinitiesTask":
        return ["x", "y", "z"]
    else:
        return ["membrane"]


def get_dacapo_run_model(run_name, iteration):
    from dacapo.experiments import Run
    from dacapo.store.create_store import create_config_store, create_weights_store

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)
    if iteration > 0:

        weights_store = create_weights_store()
        weights = weights_store.retrieve_weights(run, iteration)
        run.model.load_state_dict(weights.model)

    return run


def concat_along_c(arrs, axes_list, channel_axis_name="c"):
    """
    Concatenate a list of arrays along the axis named `channel_axis_name`.

    Parameters
    ----------
    arrs : list of np.ndarray
        The list of arrays to concatenate.
    axes_list : list of list of str
        The list of list-of-axis-names. axes_list[i] is the axis names
        corresponding to arrs[i].
    channel_axis_name : str
        The name of the "channel" axis. Default is "c".

    Returns
    -------
    out : np.ndarray
        The concatenated array.
    out_axes : list of str
        The list of axis names for the output array.
    """
    # 1. Find the channel axis index if it exists in any of the arrays
    c_index = None
    for i, axes in enumerate(axes_list):
        if channel_axis_name in axes:
            c_index = axes.index(channel_axis_name)
            break

    # 2. If no array contains the channel axis, define a default index
    #    for insertion (say 0) so that channel is the first dimension.
    if c_index is None:
        c_index = 0

        # Insert 'c' axis in *every* array
        for i, arr in enumerate(arrs):
            arrs[i] = np.expand_dims(arr, axis=c_index)  # shape (..., 1, ...)
            axes_list[i].insert(c_index, channel_axis_name)
    else:
        # 3. For arrays that lack the channel axis, insert a singleton dimension
        for i, axes in enumerate(axes_list):
            if channel_axis_name not in axes:
                # Expand the dimensions at c_index
                arrs[i] = np.expand_dims(arrs[i], axis=c_index)
                axes_list[i].insert(c_index, channel_axis_name)

    # 4. Concatenate along the channel axis index
    out = np.concatenate(arrs, axis=c_index)
    # Axes are consistent now, so we can just pick the axes from one of them
    out_axes = axes_list[0]

    return out, out_axes


def reorder_axes(
    arr: np.ndarray, axes: list[str], desired_order: list[str] = ["z", "y", "x", "c"]
) -> tuple[np.ndarray, list[str]]:
    """
    Reorder/remove axes so that the final array has axes in the desired order.

    - Any axis not in desired_order is removed IF its size == 1,
    otherwise a ValueError is raised.
    - If an axis from desired_order is missing, we insert a size-1 dimension
    in the correct position so the final shape has exactly 4 dimensions.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    axes : list of str
        Axis labels corresponding to arr.shape in order.

    Returns
    -------
    arr : np.ndarray
        The reshaped/reordered array with axes in desired_order.
    out_axes : list of str
        The final axis labels, which should be exactly desired_order.
    """

    # 1) Remove any unwanted axes (not in desired) if size==1
    for i in reversed(range(len(axes))):
        ax = axes[i]
        if ax not in desired_order:
            if arr.shape[i] != 1:
                raise ValueError(
                    f"Cannot remove axis '{ax}' with size {arr.shape[i]} (must be 1)."
                )
            # Remove this axis by squeezing
            arr = np.squeeze(arr, axis=i)
            del axes[i]

    # 2) Reorder the axes we have so they appear in the sequence desired_order
    #    Build a list of indices in the order desired.
    perm = []
    for ax in desired_order:
        if ax in axes:
            perm.append(axes.index(ax))

    arr = arr.transpose(perm)
    axes = [axes[i] for i in perm]

    # 3) For any missing axis among desired_order,
    #    insert a size-1 dimension at the correct position.
    for i, ax in enumerate(desired_order):
        if ax not in axes:
            arr = np.expand_dims(arr, axis=i)
            axes.insert(i, ax)

    # Now axes should match desired_order exactly.
    return arr, axes


def process_chunk_bioimage(self, idi: ImageDataInterface, input_roi: Roi):
    from bioimageio.core import predict, Sample, Tensor

    input_image = idi.to_ndarray_ts(input_roi.grow(self.context, self.context))
    input_image = input_image[self.input_slicer].astype(np.float32)
    input_sample = Sample(
        members={self.input_name: Tensor.from_numpy(input_image, dims=self.input_axes)},
        stat={},
        id="sample",
    )
    output: Sample = predict(
        model=self.model,
        inputs=input_sample,
        skip_preprocessing=input_sample.stat is not None,
    )
    output, _ = self.format_output_bioimage(output)
    return output


def process_chunk_bioimage_test(self, idi: ImageDataInterface, input_roi: Roi):
    from bioimageio.core import predict, create_prediction_pipeline, Sample, Tensor
    from bioimageio.core.digest_spec import get_io_sample_block_metas

    input_image = idi.to_ndarray_ts(input_roi.grow(self.context, self.context))
    input_image = input_image[self.input_slicer].astype(np.float32)
    input_axes = ["batch", "channel", "z", "y", "x"]
    input_sample = Sample(
        members={self.input_name: Tensor.from_numpy(input_image, dims=input_axes)},
        stat={},
        id="sample",
    )
    pp = create_prediction_pipeline(self.model)
    print(pp._default_input_halo)
    output = pp.predict_sample_without_blocking(input_sample)
    print(output.shape)
    output, _ = self.format_output_bioimage(output)
    return output


def format_output_bioimage(self, output_sample, output_names=None, output_axes=None):
    if output_names is None:
        output_names = self.output_names
    if output_axes is None:
        output_axes = copy.deepcopy(self.output_axes)
    if type(output_names) == list:
        outputs = []
        for output_name in output_names:
            outputs.append(output_sample.members[output_name].data.to_numpy())
        output, output_axes = concat_along_c(outputs, output_axes)
    else:
        output = output_sample.members[output_names].data.to_numpy()
    output, reordered_axes = reorder_axes(
        output, output_axes, desired_order=["c", "z", "y", "x"]
    )
    output = np.ascontiguousarray(output).clip(0, 1) * 255.0
    return output.astype(np.uint8), reordered_axes


# from funlib.geometry import Coordinate, Roi
# from bioimageio.core import predict, create_prediction_pipeline, Sample, Tensor

# for n in [
#     "impartial-shrimp",
#     "affable-shark",
#     "happy-elephant",
#     "kind-seashell",
# ]:  # "impartial-shrimp", "affable-shark",
#     print("starting", n)
#     b = BioModelConfig(n, Coordinate(16, 16, 16))
#     b = b.config
#     from bioimageio.core.digest_spec import get_test_inputs, get_test_outputs

#     pp = create_prediction_pipeline(b.model)

#     print(b.model.outputs[0].axes)

#     o = b.process_chunk(
#         ImageDataInterface(
#             "/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s1"
#         ),
#         input_roi=Roi((16000, 16000, 16000), b.read_shape),
#     )
#     print(b, b.output_voxel_size, b.read_shape, b.write_shape, b.block_shape, o.shape)
#     # print("stopped", n)

# # %%
# b.model.inputs
# # %%


def parse_model_configs(yaml_file_path: str) -> List[ModelConfig]:
    """
    Reads a YAML file that defines a list of model configs.
    Validates them manually, then returns a list of constructed ModelConfig objects.
    """
    with open(yaml_file_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Top-level YAML structure must be a list.")

    configs: List[ModelConfig] = []

    for idx, model_def in enumerate(data):
        # Common checks:
        if "type" not in model_def:
            raise ValueError(f"Missing 'type' field in model definition #{idx+1}")

        model_type = model_def["type"]
        name = model_def.get("name")

        if model_type == "bio":
            # Expect "model_name"
            if "model_name" not in model_def:
                raise ValueError(f"Missing 'model_name' in bio model #{idx+1}")
            config = BioModelConfig(
                model_name=model_def["model_name"],
                name=name,
            )

        elif model_type == "script":
            # Expect "script_path"
            if "script_path" not in model_def:
                raise ValueError(f"Missing 'script_path' in script model #{idx+1}")
            config = ScriptModelConfig(
                script_path=model_def["script_path"],
                name=name,
            )

        elif model_type == "dacapo":
            # Expect "run_name" and "iteration"
            if "run_name" not in model_def or "iteration" not in model_def:
                raise ValueError(
                    f"Missing 'run_name' or 'iteration' in dacapo model #{idx+1}"
                )
            config = DaCapoModelConfig(
                run_name=model_def["run_name"],
                iteration=model_def["iteration"],
                name=name,
            )

        else:
            raise ValueError(
                f"Invalid 'type' field '{model_type}' in model definition #{idx+1}"
            )

        configs.append(config)

    return configs


from cellmap_flow.models.cellmap_models import CellmapModel
from typing import Optional


class CellMapModelConfig(ModelConfig):
    """
    Configuration class for a CellmapModel.
    Similar to DaCapoModelConfig, but uses a CellmapModel object
    to populate the necessary metadata and define a prediction function.
    """

    def __init__(self, folder_path, name, scale=None):
        """
        :param cellmap_model: An instance of CellmapModel containing metadata
                              and references to ONNX, TorchScript, or PyTorch models.
        :param name: Optional name for this configuration.
        """
        super().__init__()
        self.cellmap_model = CellmapModel(folder_path=folder_path)
        self.name = name
        self.scale = scale

    @property
    def command(self) -> str:
        """
        You can either return a placeholder command or remove this property if not needed.
        For consistency with your DaCapoModelConfig, we return something minimal here.
        """
        return "cellmap-model -f {self.cellmap_model.folder_path} -n {self.name}"

    def _get_config(self) -> Config:
        """
        Build and return a `Config` object populated using the CellmapModel's metadata and ONNX runtime.
        """
        config = Config()

        # Access metadata from the CellmapModel
        metadata = self.cellmap_model.metadata

        # If you want to store any of these metadata fields into your config object, do so here:
        config.model_name = metadata.model_name
        config.model_type = metadata.model_type
        config.framework = metadata.framework
        config.spatial_dims = metadata.spatial_dims
        config.in_channels = metadata.in_channels
        config.output_channels = metadata.out_channels
        config.iteration = metadata.iteration
        config.input_voxel_size = Coordinate(metadata.input_voxel_size)
        config.output_voxel_size = Coordinate(metadata.output_voxel_size)
        config.channels_names = metadata.channels_names
        read_shape = metadata.inference_input_shape
        write_shape = metadata.inference_output_shape
        config.read_shape = Coordinate(read_shape) * config.input_voxel_size
        config.write_shape = Coordinate(write_shape) * config.output_voxel_size
        config.block_shape = [*write_shape, metadata.out_channels]
        config.model = self.cellmap_model.ts_model
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("device:", device)

        config.model.to(device)
        config.model.eval()
        return config
