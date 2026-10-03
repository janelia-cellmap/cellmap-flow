"""``BioModelConfig``: a BioImage Model Zoo model, run through ``bioimageio.core``.

``model`` is any source ``bioimageio.core`` loads: a zoo id or nickname
("conscientious-dromedary"), a DOI or URL, or the path of a model's
``rdf.yaml`` (``bioimageio.yaml``) or packaged ``.zip``. Its description (the
RDF, read as format 0.5; a 0.4 one is converted) gives the rest:

- What a chunk is. A 3D model (space axes z, y, x) is given one tile a
  chunk. A 2D model (y, x) is given ``slices_per_chunk`` z slices a chunk, each
  segmented on its own: in one call when its batch axis takes any size, a
  call a slice when the RDF fixes the batch at 1 or has no batch axis.
- The tile's size. An axis of fixed size keeps it; a parameterized one
  (``min + n * step``) gets the smallest size it takes of at least
  ``input_size``: 256 a side for a 2D model and 128 for a 3D one by default.
- The context: the halo of the model's output, cut off on each side so that
  chunks meet where the model saw both sides of the seam. ``context`` sets
  it; the zoo's EM U-Nets give none.
- The voxel size: the input's space axes' ``scale`` and ``unit``, in nm,
  unless ``voxel_size`` is given. No zoo model gives a unit, so the zoo's EM
  models read at the voxel size they were trained at, from cellmap-flow's
  own table (``bioimage_voxel_sizes.yaml``); any other needs ``voxel_size``.
- The output: every output tensor's channels, one tensor after another, as
  the model's own postprocessing leaves them (a sigmoid, if it has one;
  nothing clipped or rescaled), served as float32 unless the RDF says the
  output is integers (labels), which keep an integer type.
- The weights: whichever ``bioimageio.core`` prefers of those the model has,
  or ``weight_format``.

The model's preprocessing (normalization) and postprocessing run in
``bioimageio.core``'s prediction pipeline, which is built with the config,
once, and reused for every chunk.

What it cannot run: a model with more than one required input, an input with
more than one channel (RGB), or an output that is not a map over the input's
space, such as micro-SAM's, whose masks have an object axis of data-dependent
size and need prompts.

``bioimageio`` is imported only when the model is built: the CLIs and the
dashboard's model form import every type. The server runs in pixi.toml's
``bioimageio`` environment unless the entry names another ``env``.
"""

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from cellmap_flow.models.configs.base import Config, ModelConfig, _voxel_size
from cellmap_flow.models.geometry import _numbers

if TYPE_CHECKING:
    from cellmap_flow.image_data_interface import ImageDataInterface

logger = logging.getLogger(__name__)

# What bioimageio.core runs (bioimageio.core.common.SupportedWeightsFormat).
WEIGHT_FORMATS = (
    "pytorch_state_dict",
    "torchscript",
    "onnx",
    "tensorflow_saved_model_bundle",
    "keras_hdf5",
    "keras_v3",
)
# The tile edge, in voxels, asked of a parameterized axis when no
# input_size is given: a 2D model's tile is one slice, so it can be larger.
DEFAULT_INPUT_SIZE = {2: 256, 3: 128}
DEFAULT_SLICES_PER_CHUNK = 8
# nm per unit, for the space units of the 0.5 spec that a voxel is measured
# in; any other (a foot, a parsec) is taken for no unit at all.
NM_PER_UNIT = {
    "picometer": 1e-3,
    "angstrom": 0.1,
    "nanometer": 1.0,
    "micrometer": 1e3,
    "millimeter": 1e6,
    "centimeter": 1e7,
    "meter": 1e9,
}


def _sizes(value):
    """None, or ``value`` (one number, "20,256,256" or a list) as a tuple of ints."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [v for v in value.replace("(", "").replace(")", "").split(",") if v.strip()]
    if np.ndim(value) == 0:
        value = [value]
    sizes = tuple(int(float(v)) for v in value)
    if any(s < 0 for s in sizes):
        raise ValueError(f"sizes must not be negative, got {list(sizes)}")
    return sizes


def _per_axis(sizes, n: int, what: str) -> tuple:
    """``sizes`` for each of ``n`` axes: one number is every axis's."""
    if len(sizes) == 1:
        return tuple(sizes) * n
    if len(sizes) != n:
        raise ValueError(f"{what} needs one number or {n} (one per space axis of the model), got {list(sizes)}")
    return tuple(sizes)


def _as_given(sizes):
    """``sizes`` as a model entry writes them: one number as itself, more as a list."""
    return sizes[0] if len(sizes) == 1 else list(sizes)


def _space_axes(axes) -> list:
    """The space axes of ``axes``, as z, y, x: by id when they are called so, else in the RDF's order."""
    space = [a for a in axes if a.type == "space"]
    ids = [str(a.id) for a in space]
    named = [i for i in ("z", "y", "x") if i in ids]
    if len(named) == len(ids):
        return [space[ids.index(i)] for i in named]
    return space


def _check_axes(tensor, what: str):
    """Raise for a tensor axis that is no axis of an image: time, index."""
    other = [f"{a.id} ({a.type})" for a in tensor.axes if a.type not in ("batch", "channel", "space")]
    if other:
        raise ValueError(
            f"The model's {what} {tensor.id} has axes cellmap-flow cannot map onto an image: {', '.join(other)}"
        )


def _channel_axis(tensor):
    return next((a for a in tensor.axes if a.type == "channel"), None)


def _input_shape(tensor, space, wanted) -> tuple:
    """The size of each of ``space`` (the input's space axes) for a tile of about ``wanted``."""
    sizes = {}
    for axis, want in zip(space, wanted):
        size = axis.size
        if isinstance(size, int):
            if want != size:
                logger.info(f"Input axis {axis.id} has a fixed size of {size}; using it, not {want}")
            sizes[str(axis.id)] = size
        elif hasattr(size, "step"):  # ParameterizedSize: min + n * step, for any n
            n = max(0, math.ceil((want - size.min) / size.step))
            sizes[str(axis.id)] = size.min + n * size.step
    by_id = {str(a.id): a for a in space}
    for axis in space:
        size = axis.size
        if str(axis.id) in sizes:
            continue
        if hasattr(size, "axis_id") and str(size.tensor_id) == str(tensor.id) and str(size.axis_id) in sizes:
            # SizeReference to another of its space axes (y as large as x),
            # as bioimageio.spec's SizeReference.get_size computes it.
            ref = by_id[str(size.axis_id)]
            sizes[str(axis.id)] = int(sizes[str(ref.id)] * ref.scale / axis.scale + size.offset)
        else:
            raise ValueError(f"Input axis {axis.id} has a size cellmap-flow cannot choose: {size}")
    return tuple(sizes[str(a.id)] for a in space)


def _output_size(axis, input_axes, tile, input_id) -> int:
    """An output space axis's size, before any of it is cut off, for a tile of the input of ``tile``.

    ``input_axes`` and ``tile`` map the input's space axes' ids to the axes
    and to their sizes.
    """
    size = axis.size
    if isinstance(size, int):
        return size
    if hasattr(size, "axis_id") and str(size.tensor_id) == input_id and str(size.axis_id) in tile:
        # SizeReference: as bioimageio.spec's SizeReference.get_size computes it.
        ref = input_axes[str(size.axis_id)]
        return int(tile[str(ref.id)] * ref.scale / axis.scale + size.offset)
    raise ValueError(f"Output axis {axis.id} has a size cellmap-flow cannot tell from the input's: {size}")


def _data_types(tensor) -> list:
    """The dtypes a tensor's description gives: one, or one per channel."""
    data = tensor.data if isinstance(tensor.data, (list, tuple)) else [tensor.data]
    return [np.dtype(str(d.type)) for d in data]


def served_dtype(outputs) -> np.dtype:
    """What chunks of ``outputs`` (output tensor descriptions) are served as.

    float32 unless every output is declared as integers: the model's own
    postprocessing (a sigmoid, labels from a watershed) has run by then, so
    the RDF's dtype is what the values are. bool is served as uint8 and
    int64 as uint64, which neuroglancer shows; integer outputs are ids or
    classes, which are not negative.
    """
    types = [t for tensor in outputs for t in _data_types(tensor)]
    if not types or any(t.kind not in "biu" for t in types):
        return np.dtype(np.float32)
    dtype = np.result_type(*types)
    if dtype.kind == "b":
        return np.dtype(np.uint8)
    if dtype == np.int64:
        return np.dtype(np.uint64)
    # uint64 with a signed type has no common integer type.
    return dtype if dtype.kind in "iu" else np.dtype(np.float32)


def _arrange(array, dims, order):
    """``array``, whose axes are ``dims``, with its axes in ``order``.

    An axis of ``order`` that ``dims`` lacks is added with size 1; one of
    ``dims`` that ``order`` lacks must be of size 1, and is dropped.
    """
    dims = [str(d) for d in dims]
    order = [str(d) for d in order]
    for i in reversed(range(len(dims))):
        if dims[i] not in order:
            if array.shape[i] != 1:
                raise ValueError(f"Cannot drop axis {dims[i]} of size {array.shape[i]}")
            array = array.squeeze(i)
            del dims[i]
    for axis in order:
        if axis not in dims:
            array = array[..., np.newaxis]
            dims.append(axis)
    return array.transpose([dims.index(a) for a in order])


class BioModelConfig(ModelConfig):
    """A BioImage Model Zoo model, run through bioimageio.core.

    Args:
        model: a zoo id or nickname ("conscientious-dromedary"), a DOI or
            URL, or the path of a model's rdf.yaml (bioimageio.yaml) or
            packaged .zip.
        voxel_size: nm per input voxel (one number, or z, y, x): the level
            the model reads. By default the RDF's, from its input's space
            axes' scale and unit; for a model whose RDF gives no unit (all of
            the zoo's), the one it was trained at when cellmap-flow's table
            knows it (``bioimage_catalog.trained_at``: the zoo's EM models);
            else it is needed. A 2D model's z is the spacing of its slices,
            its y by default.
        input_size: voxels a side of the tile the model is given, one number
            or one per space axis of the model (z, y, x; y, x for a 2D
            model). An axis the RDF fixes keeps its size; a parameterized
            one gets the smallest size it takes of at least this. By default
            256 for a 2D model and 128 for a 3D one.
        context: voxels cut off each side of the model's output (one number,
            or one per space axis of the model), so chunks meet where the
            model saw both sides of the seam; the input is read that much
            larger. By default the RDF's halo, 0 where it gives none.
        slices_per_chunk: z slices in a chunk of a 2D model; 8 by default. A
            3D model ignores it.
        weight_format: the weights to run ("torchscript",
            "pytorch_state_dict", "onnx", ...); by default bioimageio.core's
            pick of those the model has.
    """

    cli_name = "bioimage"
    # bioimageio.core and its backends are not in cellmap-flow's own
    # environment; pixi.toml's `bioimageio` environment has them.
    default_env = "bioimageio"

    def __init__(
        self,
        model: str,
        voxel_size=None,
        input_size=None,
        context=None,
        slices_per_chunk: int = None,
        weight_format: str = None,
        name=None,
        scale=None,
    ):
        super().__init__()
        self.model = str(model)
        # The server CLI and the model form pass sizes as strings ("8,8,8",
        # "256"); a YAML gives a number or a list. The voxel size is not
        # _as_int_tuple's, which truncates a 5.24 nm voxel to 5.
        self.voxel_size = None if voxel_size is None else _voxel_size(voxel_size)
        self.input_size = _sizes(input_size)
        self.context = _sizes(context)
        self.slices_per_chunk = None if slices_per_chunk is None else int(slices_per_chunk)
        if self.slices_per_chunk is not None and self.slices_per_chunk < 1:
            raise ValueError(f"slices_per_chunk must be at least 1, got {self.slices_per_chunk}")
        if weight_format is not None and weight_format not in WEIGHT_FORMATS:
            raise ValueError(f"weight_format must be one of {', '.join(WEIGHT_FORMATS)}, not {weight_format!r}")
        self.weight_format = weight_format
        self.name = name
        self.scale = scale

    def _get_config(self):
        from bioimageio.core import create_prediction_pipeline, load_model_description

        # Without the description's IO checks, which download and hash every
        # file it names (covers, documentation, test tensors: 14 s of 15 for
        # "conscientious-dromedary") and refuse a model whose test tensor's
        # dtype is not the one described ("stupendous-sheep": int16 for
        # uint16). Those are bioimageio's test of a model, not what running
        # it needs; the weights are still checked against their sha256 when
        # the pipeline downloads them.
        description = load_model_description(self.model, format_version="latest", perform_io_checks=False)
        config = Config()
        self._describe(config, description)
        # Built here, once: it loads the weights onto the device, which
        # bioimageio.core.predict() did again for every chunk. Its adapter
        # hands each call a model of its own from a queue, so chunks served
        # at once (more than one GPU slot) do not share one.
        config.model = create_prediction_pipeline(description, weights_format=self.weight_format)
        config.process_chunk = self.process_chunk
        logger.info(
            f"{self.model}: {config.ndim}D, tiles of {list(config.tile_shape)} voxels, "
            f"{config.block_shape[:-1].tolist()} written a chunk, as {np.dtype(config.output_dtype)}"
        )
        return config

    def _describe(self, config, description):
        """Set ``config``'s geometry, and what process_chunk needs, from the model's description."""
        inputs = [t for t in description.inputs if not getattr(t, "optional", False)]
        if len(inputs) != 1:
            raise ValueError(f"{self.model} needs {len(inputs)} inputs ({', '.join(str(t.id) for t in inputs)}); "
                             "cellmap-flow gives a model one image")
        tensor = inputs[0]
        _check_axes(tensor, "input")
        channel = _channel_axis(tensor)
        if channel is not None and len(channel.channel_names) != 1:
            raise ValueError(f"{self.model} takes {len(channel.channel_names)} input channels "
                             f"({', '.join(channel.channel_names)}); cellmap-flow gives it one")
        space = _space_axes(tensor.axes)
        ndim = len(space)
        if ndim not in (2, 3):
            raise ValueError(f"{self.model}'s input has {ndim} space axes; cellmap-flow runs 2D and 3D models")
        batch = next((a for a in tensor.axes if a.type == "batch"), None)

        tile = _input_shape(tensor, space, _per_axis(self.input_size or (DEFAULT_INPUT_SIZE[ndim],), ndim,
                                                     "input_size"))
        crop = self._crop(description, space)
        out_shape, ratio, channels, channel_axes = self._outputs(description, tensor, space, tile, crop)

        in_voxel = np.asarray(self._input_voxel_size(space), dtype=float)
        if ndim == 2:
            # Each slice is segmented on its own: no context in z, and a z
            # voxel out is a z voxel in.
            slices = self.slices_per_chunk or DEFAULT_SLICES_PER_CHUNK
            read, write, ratio = (slices, *tile), (slices, *out_shape), (1.0, *ratio)
        else:
            read, write = tile, out_shape
        out_voxel = in_voxel * np.asarray(ratio, dtype=float)
        read_shape = np.asarray(read) * in_voxel
        write_shape = np.asarray(write) * out_voxel
        context = (read_shape - write_shape) / 2
        in_voxels = context / in_voxel
        if (context < 0).any() or not np.allclose(in_voxels, np.rint(in_voxels)):
            # The chunk is read around the written region on the input's
            # grid; half a voxel off, every output would be shifted.
            raise ValueError(
                f"{self.model}'s output ({list(write)} voxels of {list(_numbers(out_voxel))} nm) is not centred "
                f"on whole voxels of its input ({list(read)} of {list(_numbers(in_voxel))} nm); give an "
                "input_size or context that makes the difference even"
            )

        # Plain numbers rather than Coordinates, which are integers: a
        # 5.24 nm voxel would be 5, and every chunk placed on the wrong grid.
        config.input_voxel_size = _numbers(in_voxel)
        config.output_voxel_size = _numbers(out_voxel)
        config.read_shape = _numbers(read_shape)
        config.write_shape = _numbers(write_shape)
        config.context = _numbers(context)
        config.output_channels = len(channels)
        config.channels = channels
        config.block_shape = np.array((*write, len(channels)))
        config.output_dtype = served_dtype(description.outputs)

        config.ndim = ndim
        config.tile_shape = tile
        config.input_id = str(tensor.id)
        config.input_axes = [str(a.id) for a in tensor.axes]
        config.input_dtype = _data_types(tensor)[0]
        config.space_axes = [str(a.id) for a in space]
        # A 2D model's slices go in as one batch when its batch axis takes
        # any size. The zoo's EM models fix it at 1, or have none, and are
        # given one slice a call.
        config.batch_axis = str(batch.id) if ndim == 2 and batch is not None and batch.size is None else None
        config.output_channel_axes = channel_axes
        config.crop = crop

    def _outputs(self, description, tensor, space, tile, crop):
        """(size once cropped, output over input voxel size, channel names, channel axes), from the outputs.

        The sizes and ratios are per space axis; the channel axes are each
        output's, by its id (None for an output without one). Every output is
        served, its channels after the one before's, so they must all be the
        same size at the same scale.
        """
        input_axes = {str(a.id): a for a in space}
        tile = dict(zip(input_axes, tile))
        geometry, channels, channel_axes = None, [], {}
        for out in description.outputs:
            _check_axes(out, "output")
            out_space = {str(a.id): a for a in out.axes if a.type == "space"}
            if set(out_space) != set(input_axes):
                raise ValueError(f"{self.model}'s output {out.id} has space axes {sorted(out_space)}, "
                                 f"its input {sorted(input_axes)}")
            sizes, ratio = [], []
            for axis, cut in zip(space, crop[str(out.id)]):
                out_axis = out_space[str(axis.id)]
                sizes.append(_output_size(out_axis, input_axes, tile, str(tensor.id)) - 2 * cut)
                ratio.append(out_axis.scale / axis.scale)
            if min(sizes) < 1:
                raise ValueError(f"{self.model}'s output {out.id} is {sizes} voxels once its context is cut off; "
                                 "give a larger input_size or a smaller context")
            if geometry is not None and geometry != (sizes, ratio):
                raise ValueError(f"{self.model}'s outputs differ in size or scale; cellmap-flow serves them as "
                                 "the channels of one array")
            geometry = (sizes, ratio)
            channel = _channel_axis(out)
            if channel is None:
                names = [str(out.id)]
            elif len(description.outputs) > 1:
                names = [f"{out.id}_{n}" for n in channel.channel_names]
            else:
                names = list(channel.channel_names)
            channels += names
            channel_axes[str(out.id)] = None if channel is None else str(channel.id)
        return tuple(geometry[0]), tuple(geometry[1]), channels, channel_axes

    def _crop(self, description, space) -> dict:
        """Voxels cut off each side of each output, by its id, per space axis: context, else its halo."""
        if self.context is not None:
            context = _per_axis(self.context, len(space), "context")
            return {str(out.id): context for out in description.outputs}
        crop = {}
        for out in description.outputs:
            halo = {str(a.id): getattr(a, "halo", None) or 0 for a in out.axes if a.type == "space"}
            crop[str(out.id)] = tuple(halo.get(str(a.id), 0) for a in space)
        return crop

    def _input_voxel_size(self, space) -> tuple:
        """nm per input voxel, z, y, x: voxel_size, else the RDF's space axes'
        scale and unit, else the one it was trained at (``trained_at``)."""
        if self.voxel_size is not None:
            return self.voxel_size
        nm = []
        for axis in space:
            per_unit = NM_PER_UNIT.get(axis.unit) if axis.unit else None
            if per_unit is None:
                from cellmap_flow.models.bioimage_catalog import trained_at

                trained = trained_at(str(self.model)) or {}
                if trained.get("voxel_size"):
                    logger.info(f"{self.model}: at {trained['voxel_size']} nm, the voxel size it was trained at "
                                f"({trained['trained_on']}); its description gives none")
                    return _numbers(trained["voxel_size"])
                raise ValueError(
                    f"{self.model}'s description gives its input no physical voxel size (axis {axis.id} "
                    f"has unit {axis.unit!r}); give voxel_size, the nm per voxel of the level it should read"
                )
            # 0.008 micrometer is 8.000000000000002 nm in floats.
            nm.append(round(axis.scale * per_unit, 6))
        if len(nm) == 2:
            nm = [nm[0], *nm]  # a 2D model's slices: as far apart as its y voxels
        return _numbers(nm)

    def process_chunk(self, idi: "ImageDataInterface", output_roi):
        """``output_roi`` predicted: ``(channels, z, y, x)``, in ``output_dtype``."""
        from bioimageio.core import Sample, Tensor

        config = self.config
        data = np.asarray(idi.to_ndarray_ts(output_roi.grow(config.context, config.context)))
        data = data.astype(config.input_dtype, copy=False)
        space = config.space_axes
        if config.ndim == 3:
            calls = [(data, space)]
        elif config.batch_axis is not None:
            calls = [(data, [config.batch_axis, *space])]
        else:
            calls = [(image, space) for image in data]

        results = []
        for array, dims in calls:
            image = Tensor.from_numpy(_arrange(array, dims, config.input_axes), dims=config.input_axes)
            sample = Sample(members={config.input_id: image}, stat={}, id="chunk")
            # The tile is read with its context already, so it is not padded
            # again; the context is cut off below, as the RDF's halo or the
            # one asked for instead.
            output = config.model.predict_sample_without_blocking(
                sample, skip_input_padding=True, skip_output_cropping=True
            )
            results.append(self._output(output, dims))
        if config.ndim == 2 and config.batch_axis is None:
            result = np.stack(results, axis=1)  # each slice's (c, y, x) into (c, z, y, x)
        else:
            result = results[0]
        return np.ascontiguousarray(result.astype(config.output_dtype, copy=False))

    def _output(self, sample, dims):
        """A predicted sample's outputs, each (channels, *dims) and cropped, one after another."""
        config = self.config
        crop = {output_id: dict(zip(config.space_axes, cut)) for output_id, cut in config.crop.items()}
        parts = []
        for output_id, channel_axis in config.output_channel_axes.items():
            tensor = sample.members[output_id]
            # An output without a channel axis is one channel.
            array = _arrange(np.asarray(tensor.data), tensor.dims, [channel_axis or "__channel__", *dims])
            cut = [crop[output_id].get(d, 0) for d in dims]
            array = array[(slice(None), *(slice(c, n - c) for c, n in zip(cut, array.shape[1:])))]
            parts.append(array)
        return np.concatenate(parts, axis=0)

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds.

        What the model's description decides unless told otherwise (its
        voxel size, tile and context) is written only when it was given, so
        that an exported entry keeps following the model.
        """
        result = {"type": "bioimage", "model": self.model}
        if self.voxel_size is not None:
            result["voxel_size"] = list(self.voxel_size)
        if self.input_size is not None:
            result["input_size"] = _as_given(self.input_size)
        if self.context is not None:
            result["context"] = _as_given(self.context)
        if self.slices_per_chunk is not None:
            result["slices_per_chunk"] = self.slices_per_chunk
        if self.weight_format is not None:
            result["weight_format"] = self.weight_format
        return self._with_name_scale(result)
