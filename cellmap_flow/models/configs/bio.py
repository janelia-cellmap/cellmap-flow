"""``BioModelConfig``: a bioimage.io model, run through ``bioimageio.core``.

The shapes come from the model's test input and output. A 2-D model's
batch axis stands in for z. Chunks go through ``process_chunk_bioimage``,
bound onto the built Config, whose output ``format_output_bioimage``
puts channels first and scales to uint8.
"""

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np
from funlib.geometry import Coordinate, Roi

from cellmap_flow.models.configs.base import Config, ModelConfig, _as_int_tuple

if TYPE_CHECKING:
    from cellmap_flow.image_data_interface import ImageDataInterface


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
        # The server CLI passes both of these as strings ("8,8,8", "64").
        self.voxel_size = (
            _as_int_tuple(voxel_size) if isinstance(voxel_size, str) else voxel_size
        )
        self.name = name
        self.scale = scale
        self.voxels_to_process = None
        if edge_length_to_process:
            self.voxels_to_process = int(edge_length_to_process) ** 3

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
        # format_output_bioimage clips to [0, 1] and scales to uint8; saying so
        # stops the server advertising (and casting to) float32.
        config.output_dtype = np.uint8
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
        """This config as a model entry, which ``registry.build_model`` rebuilds."""
        result = self._with_name_scale({
            "type": "bioimage",
            "model_name": self.model_name,
            "voxel_size": list(self.voxel_size) if hasattr(self.voxel_size, '__iter__') else self.voxel_size,
        })
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


def process_chunk_bioimage(self, idi: "ImageDataInterface", input_roi: Roi):
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
