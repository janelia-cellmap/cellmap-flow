import logging
import warnings
import copy

from cellmap_models.model_export.cellmap_model import CellmapModel, get_huggingface_model
from cellmap_flow.image_data_interface import ImageDataInterface
from funlib.geometry import Roi, Coordinate
import numpy as np
import torch
from cellmap_flow.utils.serialize_config import Config

logger = logging.getLogger(__name__)


def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


class ModelConfig:
    def __init__(self):
        self._config = None

    def __str__(self) -> str:
        elms = []
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                elms.append(f"{k}: type={type(v)} shape={v.shape}\n")
            else:
                elms.append(f"{k}: {v}\n")
        return f"{type(self).__name__}({', '.join(elms)})"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
            self._validate_config()
        return self._config

    def _validate_config(self):
        """Ensure config has required attributes and shapes are consistent."""
        required = [
            "read_shape",
            "write_shape",
            "input_voxel_size",
            "output_voxel_size",
            "output_channels",
            "block_shape",
        ]
        if not (hasattr(self._config, "model") or hasattr(self._config, "predict")):
            required.insert(0, "model or predict")

        for attr in required:
            if not hasattr(self._config, attr):
                raise AttributeError(f"{attr} not found in config")

        self._validate_model_shapes()
        logger.warning(f"Model config validated: {self.__str__()}")

    def _validate_model_shapes(self):
        """Run a dummy forward pass to verify declared shapes match actual model output."""
        config = self._config
        if not hasattr(config, "model"):
            return

        model = config.model
        input_size = np.array(config.read_shape) // np.array(config.input_voxel_size)
        declared_output_size = np.array(config.write_shape) // np.array(
            config.output_voxel_size
        )
        declared_block_spatial = np.array(config.block_shape)[:3]

        try:
            device = next(model.parameters()).device
            dummy = torch.zeros(
                (1, 1, *[int(s) for s in input_size]), device=device
            )
            was_training = model.training
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            if was_training:
                model.train()

            actual_output = np.array(out.shape[1:])  # drop batch dim
            # Determine actual spatial shape (skip channel dim if present)
            if len(actual_output) == 4:
                actual_channels = actual_output[0]
                actual_spatial = actual_output[1:]
            elif len(actual_output) == 3:
                actual_channels = 1
                actual_spatial = actual_output
            else:
                logger.warning(
                    f"Unexpected model output ndim={len(actual_output)}, "
                    "skipping shape validation"
                )
                return

            errors = []
            if not np.array_equal(actual_spatial, declared_output_size):
                errors.append(
                    f"write_shape mismatch: declared write_shape / output_voxel_size = "
                    f"{declared_output_size.tolist()} but model actually outputs "
                    f"spatial shape {actual_spatial.tolist()}. "
                    f"Expected write_shape = "
                    f"{(actual_spatial * np.array(config.output_voxel_size)).tolist()}"
                )
            if not np.array_equal(actual_spatial, declared_block_spatial):
                errors.append(
                    f"block_shape mismatch: declared block_shape spatial dims = "
                    f"{declared_block_spatial.tolist()} but model actually outputs "
                    f"spatial shape {actual_spatial.tolist()}. "
                    f"Expected block_shape = "
                    f"{[*actual_spatial.tolist(), int(actual_channels)]}"
                )
            if int(actual_channels) != int(config.output_channels):
                errors.append(
                    f"output_channels mismatch: declared {config.output_channels} "
                    f"but model actually outputs {int(actual_channels)} channels"
                )
            if errors:
                msg = (
                    f"Script config shape validation failed for "
                    f"{getattr(self, 'script_path', 'unknown')}:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )
                raise ValueError(msg)

            logger.info(
                f"Shape validation passed: input {input_size.tolist()} -> "
                f"output spatial {actual_spatial.tolist()}, "
                f"channels {int(actual_channels)}"
            )
        except (RuntimeError, TypeError) as e:
            logger.warning(f"Could not validate model shapes (forward pass failed): {e}")

    @property
    def chunk_output_axes(self) -> tuple[str, ...]:
        """Returns the axes order of processed chunk output. Defaults to ('c', 'z', 'y', 'x').

        Models can override by setting config.chunk_output_axes.
        Note: this is distinct from config.output_axes used by BioModelConfig
        for raw bioimageio model axes.
        """
        if hasattr(self.config, "chunk_output_axes"):
            return tuple(self.config.chunk_output_axes)
        return ("c", "z", "y", "x")

    @property
    def output_dtype(self):
        """Returns the output dtype of the model. Defaults to np.float32."""
        if hasattr(self.config, "output_dtype"):
            return self.config.output_dtype
        logger.warning(
            f"Model {self.name} does not define output_dtype, defaulting to np.float32"
        )
        return np.float32

    def to_dict(self):
        """
        Export model configuration as a dict that can be used with build_model_from_entry.

        Returns:
            Dictionary containing model type and all init parameters.
        """
        raise NotImplementedError("Subclasses must implement to_dict()")


class ScriptModelConfig(ModelConfig):

    cli_name = "script"

    def __init__(self, script_path, name=None, scale=None):
        super().__init__()
        self.script_path = script_path
        self.name = name
        self.scale = scale

    @property
    def command(self):
        return f"script --script-path {self.script_path}"

    def _get_config(self):
        from cellmap_flow.utils.load_py import load_safe_config

        config = load_safe_config(self.script_path)
        if not hasattr(config, "read_shape"):
            config.read_shape = Coordinate(config.input_size) * Coordinate(
                config.input_voxel_size
            )
        if not hasattr(config, "write_shape"):
            config.write_shape = Coordinate(config.output_size) * Coordinate(
                config.output_voxel_size
            )
        if not hasattr(config, "block_shape"):
            config.block_shape = np.array(
                tuple(config.output_size) + (config.output_channels,)
            )
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {"type": "script", "script_path": self.script_path}
        if self.name is not None:
            result["name"] = self.name
        if self.scale is not None:
            result["scale"] = self.scale
        return result


class DaCapoModelConfig(ModelConfig):

    cli_name = "dacapo"

    def __init__(self, run_name: str, iteration: int, name=None, scale=None):
        super().__init__()
        self.run_name = run_name
        self.iteration = iteration
        self.name = name
        self.scale = scale

    @property
    def command(self):
        return f"dacapo --run-name {self.run_name} --iteration {self.iteration}"

    def _get_config(self):
        from dacapo.experiments import Run
        from dacapo.store.create_store import create_config_store, create_weights_store

        config = Config()
        run = self._load_dacapo_run()

        run.model.to(_get_device())
        run.model.eval()
        config.model = run.model

        in_shape = run.model.eval_input_shape
        out_shape = run.model.compute_output_shape(in_shape)[1]
        voxel_size = run.datasplit.train[0].raw.voxel_size

        config.input_voxel_size = Coordinate(voxel_size)
        config.output_voxel_size = Coordinate(run.model.scale(voxel_size))
        config.read_shape = Coordinate(in_shape) * config.input_voxel_size
        config.write_shape = Coordinate(out_shape) * config.input_voxel_size
        config.channels = self._get_channels(run.task)
        config.output_channels = len(config.channels)
        config.block_shape = np.array(tuple(out_shape) + (config.output_channels,))
        return config

    def _load_dacapo_run(self):
        """Load DaCapo run with optional weights."""
        from dacapo.experiments import Run
        from dacapo.store.create_store import create_config_store, create_weights_store

        config_store = create_config_store()
        run_config = config_store.retrieve_run_config(self.run_name)
        run = Run(run_config)

        if self.iteration > 0:
            weights_store = create_weights_store()
            weights = weights_store.retrieve_weights(run, self.iteration)
            run.model.load_state_dict(weights.model)
        return run

    @staticmethod
    def _get_channels(task):
        """Extract channel names from task."""
        if hasattr(task, "channels"):
            return task.channels
        elif type(task).__name__ == "AffinitiesTask":
            return ["x", "y", "z"]
        return ["membrane"]

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {
            "type": "dacapo",
            "run_name": self.run_name,
            "iteration": self.iteration,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.scale is not None:
            result["scale"] = self.scale
        return result


class FlyModelConfig(ModelConfig):

    cli_name = "fly"

    def __init__(
        self,
        checkpoint_path: str,
        channels: list[str],
        input_voxel_size: tuple,
        output_voxel_size: tuple,
        name: str = None,
        input_size=None,
        output_size=None,
        scale=None,
    ):
        super().__init__()
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.channels = channels
        self.input_voxel_size = input_voxel_size
        self.output_voxel_size = output_voxel_size
        self.scale = scale
        self._model = None
        if input_size is None or output_size is None:
            input_size = (178, 178, 178)
            output_size = (56, 56, 56)
            logger.warning(
                "Input and output size not provided, defaulting to (178, 178, 178) and (56, 56, 56)"
            )
        self.input_size = input_size
        self.output_size = output_size

    @property
    def command(self):
        return f"fly --checkpoint-path {self.checkpoint_path} --channels {','.join(self.channels)} --input-voxel-size {','.join(map(str,self.input_voxel_size))} --output-voxel-size {','.join(map(str,self.output_voxel_size))}"

    def load_eval_model(self, num_channels, checkpoint_path):
        """Load evaluation model from checkpoint (TorchScript or PyTorch)."""
        device = _get_device()

        if checkpoint_path.endswith(".ts"):
            model_backbone = torch.jit.load(checkpoint_path, map_location=device)
        else:
            from fly_organelles.model import StandardUnet

            model_backbone = StandardUnet(num_channels)
            checkpoint = torch.load(
                checkpoint_path, weights_only=True, map_location="cpu"
            )
            model_backbone.load_state_dict(checkpoint["model_state_dict"])

        model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
        model.to(device)
        model.eval()
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_eval_model(len(self.channels), self.checkpoint_path)
        return self._model

    def _get_config(self):
        config = Config()
        config.model = self.model
        config.input_voxel_size = Coordinate(self.input_voxel_size)
        config.output_voxel_size = Coordinate(self.output_voxel_size)
        config.read_shape = Coordinate(self.input_size) * config.input_voxel_size
        config.write_shape = Coordinate(self.output_size) * config.input_voxel_size
        config.channels = self.channels
        config.output_channels = len(self.channels)
        config.block_shape = np.array(
            tuple(self.output_size) + (config.output_channels,)
        )
        # Add axes_names for server compatibility
        config.axes_names = ["x", "y", "z", "c^"]
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {
            "type": "fly",
            "checkpoint_path": self.checkpoint_path,
            "channels": self.channels,
            "input_voxel_size": list(self.input_voxel_size),
            "output_voxel_size": list(self.output_voxel_size),
        }
        if self.name is not None:
            result["name"] = self.name
        if self.input_size is not None:
            result["input_size"] = list(self.input_size)
        if self.output_size is not None:
            result["output_size"] = list(self.output_size)
        if self.scale is not None:
            result["scale"] = self.scale
        return result


class BioModelConfig(ModelConfig):

    cli_name = "bioimage"

    def __init__(
        self,
        model_name: str,
        voxel_size,
        edge_length_to_process=None,
        name=None,
        scale=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.voxel_size = voxel_size
        self.name = name
        self.scale = scale
        self.voxels_to_process = None
        if edge_length_to_process:
            self.voxels_to_process = edge_length_to_process**3

    @property
    def command(self):
        return f"bioimage --model-name {self.model_name}"

    def _get_config(self):
        from bioimageio.core import load_description
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

        if self.voxels_to_process:
            if not is_2d_with_batch:
                warnings.warn("edge_length_to_process is only supported for 2D models")
            else:
                batch_size = max(
                    1, self.voxels_to_process // np.prod(config.input_spatial_dims)
                )
                config.input_spatial_dims[config.input_axes.index("z")] = batch_size
                config.output_spatial_dims[0] = batch_size
                config.block_shape[0] = batch_size

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

        input_name, input_axes, input_dims, is_2d_with_batch = self.get_and_dims(
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

        output_dims = finalized_output.shape
        output_spatial_dims = [
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
            output_spatial_dims,
            output_channels,
        )

    def get_axes_and_dims(self, sample):
        sample_names = list(sample.shape.keys())
        sample_axis_to_dims_dicts = list(sample.shape.values())
        sample_axes = []
        sample_dims = []
        is_2d_with_batch = False

        for sample_axis_to_dim_dict in sample_axis_to_dims_dicts:
            current_sample_axes = sample_axis_to_dim_dict.keys()
            if (
                "b" in current_sample_axes or "batch" in current_sample_axes
            ) and "z" not in current_sample_axes:
                is_2d_with_batch = True

            # Use 'z' instead of 'b' if z is not present (for 2D models)
            sample_axes.append(
                [
                    "z" if (a[0] == "b" and "z" not in current_sample_axes) else a[0]
                    for a in current_sample_axes
                ]
            )
            sample_dims.append(list(sample_axis_to_dim_dict.values()))

        if len(sample_names) == 1:
            return sample_names[0], sample_axes[0], sample_dims[0], is_2d_with_batch
        return sample_names, sample_axes, sample_dims, is_2d_with_batch

    def get_spatial_dims(self, axes, dims):
        return [d for a, d in zip(axes, dims) if a in ["x", "y", "z"]]

    def get_input_slicer(self, input_axes):
        return tuple(
            (
                np.newaxis
                if a.startswith("c") or (a == "b" and "z" in input_axes)
                else slice(None)
            )
            for a in input_axes
        )

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {
            "type": "bioimage",
            "model_name": self.model_name,
            "voxel_size": list(self.voxel_size) if hasattr(self.voxel_size, '__iter__') else self.voxel_size,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.scale is not None:
            result["scale"] = self.scale
        if self.voxels_to_process is not None:
            # Reconstruct edge_length_to_process from voxels_to_process
            edge_length = round(self.voxels_to_process ** (1/3))
            result["edge_length_to_process"] = edge_length
        return result


def concat_along_c(arrs, axes_list, channel_axis_name="c"):
    """Concatenate arrays along the channel axis, adding channel dim if missing."""
    # Find channel axis index (default to 0 if not found)
    c_index = next(
        (
            axes.index(channel_axis_name)
            for axes in axes_list
            if channel_axis_name in axes
        ),
        0,
    )

    # Ensure all arrays have channel axis at c_index
    for i, axes in enumerate(axes_list):
        if channel_axis_name not in axes:
            arrs[i] = np.expand_dims(arrs[i], axis=c_index)
            axes_list[i].insert(c_index, channel_axis_name)

    return np.concatenate(arrs, axis=c_index), axes_list[0]


def reorder_axes(
    arr: np.ndarray, axes: list[str], desired_order: list[str] = ["z", "y", "x", "c"]
) -> tuple[np.ndarray, list[str]]:
    """Reorder/remove axes to match desired_order, removing size-1 unwanted axes."""
    # Remove unwanted axes (not in desired_order) if size==1
    for i in reversed(range(len(axes))):
        if axes[i] not in desired_order:
            if arr.shape[i] != 1:
                raise ValueError(
                    f"Cannot remove axis '{axes[i]}' with size {arr.shape[i]} (must be 1)."
                )
            arr = np.squeeze(arr, axis=i)
            del axes[i]

    # Reorder existing axes to match desired_order
    perm = [axes.index(ax) for ax in desired_order if ax in axes]
    arr = arr.transpose(perm)
    axes = [axes[i] for i in perm]

    # Add missing axes as size-1 dimensions
    for i, ax in enumerate(desired_order):
        if ax not in axes:
            arr = np.expand_dims(arr, axis=i)
            axes.insert(i, ax)

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
    output = predict(
        model=self.model,
        inputs=input_sample,
        skip_preprocessing=bool(input_sample.stat),
    )
    output, _ = self.format_output_bioimage(output)
    return output


def format_output_bioimage(self, output_sample, output_names=None, output_axes=None):
    output_names = output_names or self.output_names
    output_axes = copy.deepcopy(output_axes or self.output_axes)

    if isinstance(output_names, list):
        outputs = [output_sample.members[name].data.to_numpy() for name in output_names]
        output, output_axes = concat_along_c(outputs, output_axes)
    else:
        output = output_sample.members[output_names].data.to_numpy()

    output, reordered_axes = reorder_axes(
        output, output_axes, desired_order=["c", "z", "y", "x"]
    )
    output = np.ascontiguousarray(output).clip(0, 1) * 255.0
    return output.astype(np.uint8), reordered_axes


class CellMapModelConfig(ModelConfig):
    """Configuration class for a CellmapModel."""

    cli_name = "cellmap"

    def __init__(self, folder_path, name=None, scale=None):
        super().__init__()
        self.cellmap_model = CellmapModel(folder_path=folder_path)
        if name is None:
            # folder name 
            name = folder_path.rstrip("/").split("/")[-1]
        self.name = name
        self.scale = scale

    @property
    def command(self) -> str:
        return f"cellmap --folder-path {self.cellmap_model.folder_path} --name {self.name}"

    def _get_config(self) -> Config:
        config = Config()
        metadata = self.cellmap_model.metadata

        # Populate config from metadata
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
        config.channels = metadata.channels_names  # alias for compatibility

        config.read_shape = Coordinate(metadata.input_shape) * config.input_voxel_size
        config.write_shape = Coordinate(metadata.output_shape) * config.output_voxel_size
        config.inference_input_shape = Coordinate(metadata.inference_input_shape)* config.input_voxel_size
        config.inference_output_shape = Coordinate(metadata.inference_output_shape)* config.output_voxel_size
        
        config.block_shape = [*metadata.output_shape, metadata.out_channels]

        config.model = self.cellmap_model.ts_model
        config.model.to(_get_device())
        config.model.eval()
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {
            "type": "cellmap",
            "folder_path": self.cellmap_model.folder_path,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.scale is not None:
            result["scale"] = self.scale
        return result

class HuggingFaceModelConfig(ModelConfig):
    """Configuration class for a Hugging Face model."""

    cli_name = "huggingface"

    def __init__(self, repo, revision=None, name=None, scale=None):
        super().__init__()
        self.repo = repo
        self.revision = revision
        if name is None:
            # Use repo name as default
            name = repo.split("/")[-1]
        self.name = name
        self.scale = scale
        self._metadata = None

    def _load_metadata(self):
        """Load metadata.json from the HuggingFace repo (cached after first call)."""
        if self._metadata is not None:
            return self._metadata
        try:
            from huggingface_hub import hf_hub_download
            import json
            path = hf_hub_download(self.repo, "metadata.json", revision=self.revision)
            with open(path) as f:
                self._metadata = json.load(f)
        except Exception:
            self._metadata = {}
        return self._metadata

    @property
    def command(self) -> str:
        cmd = f"huggingface --repo {self.repo}"
        if self.revision:
            cmd += f" --revision {self.revision}"
        return cmd

    def _get_config(self) -> Config:
        cellmap_model = get_huggingface_model(self.repo, self.revision)
        config = CellMapModelConfig(folder_path=cellmap_model.folder_path)._get_config()
        return config

    def to_dict(self):
        """Export configuration for use with build_model_from_entry."""
        result = {
            "type": "huggingface",
            "repo": self.repo,
        }
        if self.revision is not None:
            result["revision"] = self.revision
        if self.name is not None:
            result["name"] = self.name
        if self.scale is not None:
            result["scale"] = self.scale

        # Include metadata from HuggingFace model for pipeline builder display
        metadata = self._load_metadata()
        if metadata:
            for key in ("channels_names", "input_voxel_size", "output_voxel_size",
                        "inference_input_shape", "inference_output_shape",
                        "in_channels", "out_channels", "model_type", "framework",
                        "spatial_dims", "iteration", "model_name", "description"):
                if key in metadata:
                    result[key] = metadata[key]

        return result