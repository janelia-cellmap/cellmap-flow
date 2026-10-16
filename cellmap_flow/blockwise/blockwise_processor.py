import logging

# No logging configuration here: the CLIs set it up, and the dashboard,
# which imports this lazily, keeps its own.
logger = logging.getLogger(__name__)

import os
from pathlib import Path
from typing import NamedTuple
import zarr
import daisy
import numpy as np
from funlib.geometry.coordinate import Coordinate
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.persistence.arrays.datasets import ArrayNotFoundError
from zarr.storage import NestedDirectoryStore
from zarr.hierarchy import open_group
from functools import partial
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.config.yaml import ConfigError, load_config, resolve_data_path
from cellmap_flow.models.registry import build_models
from cellmap_flow.io.ome import singlescale_attrs
from cellmap_flow.models.model_merger import get_model_merger
from cellmap_flow.jobs.launch import submit_bsub_job
from cellmap_flow.jobs.spec import shell_quote
from cellmap_flow.jobs.settings import launcher_settings
from cellmap_flow.jobs.site import current_site
from cellmap_flow.process_chain import process_chain


def _validate_settings(config):
    """The checks on a task YAML that need neither the model nor the data.

    Shared by the processor and precheck(), so both reject the same files
    with the same messages.
    """
    if "output_path" not in config:
        raise ConfigError("Missing required field in YAML: output_path")
    if ".zarr" not in str(config["output_path"]):
        raise ConfigError("output_path should be a zarr with .zarr on it")
    if "task_name" not in config:
        raise ConfigError("Missing required field in YAML: task_name")
    if "workers" not in config:
        raise ConfigError("Missing required field in YAML: workers")
    if not isinstance(config["workers"], int) or config["workers"] < 1:
        raise ConfigError(
            f"workers should be an integer greater than 0, got {config['workers']!r}"
        )
    if config.get("track_progress", False) and "tmp_dir" not in config:
        raise ConfigError(
            "Missing required field in YAML: tmp_dir, it is mandatory to track progress"
        )
    if config.get("bounding_boxes"):
        _check_bounding_boxes(config["bounding_boxes"])
    try:
        get_model_merger(str(config.get("model_mode", "AND")).upper())
    except ValueError as e:
        raise ConfigError(str(e))
    if config.get("cross_channels"):
        try:
            get_model_merger(str(config["cross_channels"]).upper())
        except ValueError as e:
            raise ConfigError(f"Invalid cross_channels setting: {e}")
    output_channels = config.get("output_channels")
    if output_channels:
        names = list(output_channels) if isinstance(output_channels, (dict, list)) else [output_channels]
        if len(names) != len(set(names)):
            raise ConfigError(
                f"output_channels has duplicated channel names. channels: {names}"
            )


def _check_bounding_boxes(boxes):
    """Refuse ``bounding_boxes`` that are not a list of ``{offset, shape}``.

    Each is three numbers, nm, z, y, x: ``offset`` the box's lower corner
    (0, 0, 0 when left out) and ``shape`` its size, every side above 0.
    """
    if not isinstance(boxes, list):
        raise ConfigError(f"bounding_boxes should be a list of {{offset, shape}}, got {boxes!r}")
    for n, box in enumerate(boxes, start=1):
        if not isinstance(box, dict) or "shape" not in box:
            raise ConfigError(f"bounding box {n} should be {{offset, shape}} in nm, got {box!r}")
        for key, value in (("offset", box.get("offset", [0, 0, 0])), ("shape", box["shape"])):
            numbers = isinstance(value, (list, tuple)) and len(value) == 3 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            )
            if not numbers:
                raise ConfigError(
                    f"bounding box {n}: {key} should be three numbers, nm, z, y, x; got {value!r}"
                )
        if any(v <= 0 for v in box["shape"]):
            raise ConfigError(f"bounding box {n}: every side of its shape should be above 0, got {box['shape']}")


class _Output:
    """One zarr a run writes, ``<path>/<output channel>/s0``: ``roi`` (nm,
    z, y, x) is where it lies, and ``arrays`` its arrays, one per output
    channel, once created or opened."""

    def __init__(self, path, roi):
        self.path, self.roi, self.arrays = Path(path), roi, []


class _Region(NamedTuple):
    """One daisy task of a run: ``name`` is its task id, and its blocks
    write ``roi`` (nm, z, y, x; a box, or the whole volume) into
    ``outputs[output]``."""

    name: str
    roi: daisy.Roi
    output: int


def _load_task(yaml_config) -> dict:
    """The task YAML at the path ``yaml_config`` (load_config refuses
    anything else), checked by _validate_settings."""
    config = load_config(yaml_config)
    _validate_settings(config)
    return config


def precheck(yaml_config: str) -> dict:
    """Check a blockwise task YAML without side effects.

    Loads no model, writes nothing, and leaves ``g`` alone -- unlike
    constructing CellMapFlowBlockwiseProcessor, which the dashboard used to
    do for this: that created the output arrays, loaded every model's weights
    into the dashboard process (onto its GPU, if it had one), and replaced the
    dashboard's live normalization and postprocessing with the task's.

    Checks the settings, that every model entry builds a model config, that
    json_data builds its normalizers and postprocessors, and that a local
    data_path exists. What needs the model's geometry (output shapes, the
    channels) is left to the run itself.

    Raises:
        ConfigError: with the reason, when the file is not usable.
    """
    config = _load_task(yaml_config)

    # Constructing a ModelConfig does not load the model; that happens when
    # .config is first read, which nothing here does.
    models = build_models(config["models"], base_dir=os.path.dirname(os.path.abspath(yaml_config)))
    if len(models) == 0:
        raise ConfigError("No models found in the configuration.")

    data_path = resolve_data_path(config["data_path"], getattr(models[0], "scale", None))
    if "://" not in str(data_path) and not os.path.exists(data_path):
        raise ConfigError(f"data_path does not exist: {data_path}")

    if config.get("json_data"):
        try:
            PipelineSpec.from_json_data(config["json_data"], strict=True).build()  # and discarded
        except Exception as e:
            raise ConfigError(f"Invalid json_data: {e}") from e

    return {
        "data_path": data_path,
        "output_path": str(config["output_path"]),
        "models": [getattr(m, "name", None) or type(m).__name__ for m in models],
    }


class CellMapFlowBlockwiseProcessor:

    def __init__(self, yaml_config: str, create=False):
        """Run the CellMapFlow server with a Fly model."""
        self.config = _load_task(yaml_config)
        self.yaml_config = yaml_config

        self.input_path = self.config["data_path"]
        self.charge_group = self.config["charge_group"]
        self.queue = self.config["queue"]

        logger.info(f"Data path: {self.input_path}")

        self.output_path = self.config["output_path"]
        z_con = self.output_path.split(".zarr")[0]+".zarr"
        zarr.open(z_con,mode="a")  # this is to create the zarr if it does not exist
        self.output_path = Path(self.output_path)

        output_channels = None
        if "output_channels" in self.config:
            output_channels = self.config["output_channels"]

        json_data = None
        if "json_data" in self.config:
            json_data = self.config["json_data"]

        task_name = self.config["task_name"]
        self.workers = self.config["workers"]

        # Determine if output_channels is dict or list format
        self.output_channels_is_dict = (
            isinstance(output_channels, dict) if output_channels else False
        )
        if self.output_channels_is_dict:
            self.output_channel_names = list(output_channels.keys())
            self.output_channel_indices = output_channels
        else:
            self.output_channel_names = output_channels if output_channels else None
            self.output_channel_indices = None
        self.cpu_workers = self.config.get("cpu_workers", 12)
        # LSF run limit for each worker: the task's, else the saved launcher
        # setting, else the site's. Without -W the GPU queues kill a worker
        # at two hours; see jobs/site.py's default_walltime.
        self.walltime = (
            self.config.get("walltime")
            or launcher_settings().walltime
            or current_site().default_walltime
        )
        # Added and create == True to fix client error when create: True in the yaml, so when it is a client it will not be changed
        if "create" in self.config and create == True:
            create = self.config["create"]
            if isinstance(create, str):
                logger.warning(
                    f"Type config[create] is str = {create}, better set a bool"
                )
                create = create.lower() == "true"

        self.track_progress = self.config.get("track_progress", False)

        if self.track_progress:
            self.tmp_dir = (
                Path(self.config["tmp_dir"]) / f"tmp_flow_daisy_progress_{task_name}"
            )
            if not self.tmp_dir.exists():
                self.tmp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.tmp_dir = None

        # Build model configuration objects
        models = build_models(self.config["models"], base_dir=os.path.dirname(os.path.abspath(yaml_config)))
        # The master (create) only needs the models' geometry, to schedule
        # blocks and create the outputs, so it runs no forward at all. The
        # workers run the model: each Inferencer's warmup forward checks the
        # declared shapes, and building a config (the first one is read below,
        # before its Inferencer exists) runs no forward of its own.
        for model in models:
            model.validate_model_shapes = not create
            model.check_shapes_on_warmup = not create
            logger.info(str(model))

        if len(models) == 0:
            raise ConfigError("No models found in the configuration.")

        # The same data_path + scale rule as cellmap_flow infer and yaml.
        # All models read through one ImageDataInterface, so the first
        # model's scale is the one that applies.
        scale = getattr(models[0], "scale", None)
        other_scales = [
            getattr(m, "scale", None) for m in models[1:] if getattr(m, "scale", None) != scale
        ]
        if other_scales:
            logger.warning(
                f"Models give different scales ({[scale] + other_scales}); blockwise "
                f"reads one dataset, using {models[0].name!r}'s scale {scale!r}"
            )
        self.input_path = resolve_data_path(self.input_path, scale)
        logger.info(f"Reading: {self.input_path}")

        # Support multiple models with model_mode
        self.models = models
        self.model_mode_str = self.config.get("model_mode", "AND").upper()
        self.model_merger = get_model_merger(self.model_mode_str)

        if len(models) > 1:
            logger.info(
                f"Using {len(models)} models with merge mode: {self.model_mode_str}"
            )

        # Support cross-channel processing
        self.process_only = self.config.get("process_only", None)
        self.cross_channels_mode = self.config.get("cross_channels", None)
        
        if self.cross_channels_mode:
            self.cross_channels_mode = self.cross_channels_mode.upper()
            self.cross_channels_merger = get_model_merger(self.cross_channels_mode)
        else:
            self.cross_channels_merger = None
        
        if self.process_only:
            logger.info(
                f"Processing only channels: {self.process_only} with merge mode: {self.cross_channels_mode}"
            )

        self.model_config = models[0]
        geometry = self.model_config.geometry

        # this is zyx
        block_shape = list(geometry.block_shape())[:3]
        self.block_shape = tuple(self.config.get("block_size", block_shape))

        self.input_voxel_size = Coordinate(geometry.input_voxel_size)
        self.output_voxel_size = Coordinate(geometry.output_voxel_size)
        # The model's channel names, by whichever name it gives them
        # (channels, channels_names or classes). They name the outputs when
        # the task gives no output_channels, and a list of output_channels
        # picks its channels by them; a dict gives indices instead.
        self.channels = list(geometry.channel_names) if geometry.channel_names else None
        if self.channels is None and not self.output_channels_is_dict:
            raise ConfigError(
                f"Model {self.model_config.name!r} names no channels (channels, "
                "channels_names or classes): give output_channels as a mapping "
                "of output name to channel index"
            )

        self.task_name = task_name
        if output_channels:
            if self.output_channels_is_dict:
                self.output_channels = self.output_channel_names
            else:
                self.output_channels = output_channels
        else:
            self.output_channels = self.channels
            self.output_channels_is_dict = False
            self.output_channel_names = self.channels
            self.output_channel_indices = None

        if not isinstance(self.output_channels, list):
            self.output_channels = [self.output_channels]

        # The chain every block runs: the task's json_data, also installed as
        # the process's, as before; without one (None), the process's own.
        self.input_norms = self.postprocess = None
        if json_data:
            spec = PipelineSpec.from_json_data(json_data, strict=True)
            self.input_norms, self.postprocess = spec.build()
            process_chain().set(spec, built=(self.input_norms, self.postprocess))
        self.dtype = process_chain().output_dtype(geometry.output_dtype, self.postprocess)

        self.inferencers = []
        self.inferencer = None
        if not create:
            self.inferencers = [
                Inferencer(model, use_half_prediction=False) for model in self.models
            ]
            self.inferencer = self.inferencers[0]  # Keep for backward compatibility

        # resample (true unless the task says false) reads the data resampled
        # to the model's input voxel size when it has no level at it; the
        # grid below is then the resampled one, starting at the level's own
        # corner.
        self.idi_raw = ImageDataInterface(
            self.input_path,
            voxel_size=self.input_voxel_size,
            on_voxel_size_mismatch="resample" if self.config.get("resample", True) else "relabel",
        )

        # The output grid starts at the corner of the raw level the model
        # reads (whole nanometers), so every output voxel sits exactly on the
        # input voxels it is computed from; anchored at 0 it was half a raw
        # voxel off Janelia data, whose corner is -4 nm. The whole-volume
        # output is the raw extent, shrunk to whole output voxels.
        self.grid_origin = Coordinate(
            int(v) for v in np.round(np.array(self.idi_raw.offset, dtype=float))
        )
        raw_roi = daisy.Roi(
            self.grid_origin,
            Coordinate(self.idi_raw.shape) * self.input_voxel_size,
        )
        self.full_output_roi = self._snap_to_output_grid(raw_roi)

        self.bounding_boxes = self.config.get("bounding_boxes", None)
        self.separate_zarrs = self.config.get("separate_bounding_boxes_zarrs", False)
        # What the run writes (self.outputs) and the daisy tasks that write
        # it (self.regions); a worker works them out as the master does, to
        # open the outputs and to know which a block of which task fills.
        self.outputs, self.regions = self._plan()

        logger.info(f"type: {self.dtype}")
        for output in self.outputs:
            logger.info(f"output: {output.path}, {output.roi} nm, {output.roi.shape / self.output_voxel_size} voxels")

        # _validate_settings checks a task's own output_channels; the model's
        # channel names, used when it gives none, can repeat too.
        if len(self.output_channels) != len(set(self.output_channels)):
            raise Exception(f"output_channels has duplicated channel names. channels: {self.output_channels}")

        for output in self.outputs:
            if create and output.path != Path(z_con):
                # A group, with the groups between it and the zarr, so that
                # listing the zarr finds it (box_1, box_2, ...).
                zarr.open_group(z_con, mode="a", path=str(output.path.relative_to(z_con)))
            for channel in self.output_channels:
                if create:
                    array = self._create_output(output, channel)
                else:
                    # A worker opens what the master created.
                    try:
                        array = open_ds(NestedDirectoryStore(output.path / channel / "s0"), "a")
                    except Exception as e:
                        raise Exception(f"Failed to open {output.path / channel}\n{e}")
                output.arrays.append(array)
        # The first output's arrays: with separate_bounding_boxes_zarrs and
        # several boxes, the first box's; otherwise the only output's.
        self.output_arrays = self.outputs[0].arrays

    def _channel_indices(self, name):
        """The model channels a dict of output_channels gives output ``name``, as a list."""
        indices = self.output_channel_indices[name]
        return [indices] if isinstance(indices, int) else indices

    def _plan(self):
        """The outputs this run writes, and the daisy tasks that write them.

        Without bounding_boxes, one output over the raw data's extent and
        one task over it. With them, each box is snapped out to every output
        voxel it touches and cut to the raw data's extent, a task per box;
        the boxes share one output over their bounding box, or with
        separate_bounding_boxes_zarrs each has its own,
        ``<output_path>/box_<n>`` counting from 1 in the list's order. Either
        way an output covers only what its boxes need, and its corner (the
        funlib offset; OME's translation is half a voxel in) is where they
        start.
        """
        name = f"predict_{self.model_config.name}_{self.task_name}"
        boxes = [self._box_roi(n, box) for n, box in enumerate(self.bounding_boxes or [], start=1)]
        if not boxes:
            outputs = [_Output(self.output_path, self.full_output_roi)]
            regions = [(self.full_output_roi, 0)]
        elif self.separate_zarrs:
            outputs = [_Output(self.output_path / f"box_{n}", box) for n, box in enumerate(boxes, start=1)]
            regions = [(box, n) for n, box in enumerate(boxes)]
        else:
            union = boxes[0]
            for box in boxes[1:]:
                union = union.union(box)
            outputs = [_Output(self.output_path, union)]
            regions = [(box, 0) for box in boxes]
        names = [name] if len(regions) == 1 else [f"{name}_roi{n}" for n in range(1, len(regions) + 1)]
        return outputs, [_Region(task, roi, output) for task, (roi, output) in zip(names, regions)]

    def _box_roi(self, n, box):
        """Bounding box ``n``'s ROI: every output voxel it touches, inside the data.

        Its corners are snapped outward onto the output voxel grid (anchored
        at the raw data's corner), so a box drawn across a voxel keeps it,
        and the result is cut to the raw data's extent, where every output
        voxel has data to be computed from.
        """
        offset = np.array(box.get("offset", [0, 0, 0]), dtype=float)
        begin = np.floor(offset).astype(int)
        end = np.ceil(offset + np.array(box["shape"], dtype=float)).astype(int)
        given = daisy.Roi(Coordinate(begin), Coordinate(end - begin))
        snapped = self._snap_to_output_grid(given, mode="grow")
        inside = snapped.intersect(self.full_output_roi)
        if inside.empty:
            raise ConfigError(
                f"bounding box {n} ({given} nm) is outside the data, which covers {self.full_output_roi} nm"
            )
        if inside != snapped:
            logger.warning(f"Bounding box {n} ({given} nm) reaches past the data; cut to {inside}")
        elif snapped != given:
            logger.info(f"Bounding box {n} ({given} nm) snapped out to whole output voxels: {snapped}")
        return inside

    def _create_output(self, output, channel):
        """Create ``output``'s output ``channel``: its array s0, and its group's OME attributes.

        Spatially, z, y, x, the array is ``output.roi`` in output voxels,
        voxel 0's corner at its offset. An output that a dict of
        output_channels gives several model channels stacks them on a
        leading axis "c", with voxel size 1 and offset 0, and each chunk
        holds all of them; any other output holds one channel and is 3-D.
        Resuming into an output that covers another region, or whose
        attributes differ, is refused.
        """
        path = output.path / channel / "s0"
        shape = tuple(int(s) for s in output.roi.shape / self.output_voxel_size)
        offset = tuple(int(o) for o in output.roi.offset)
        _refuse_another_region(path, output.roi)
        chunk_shape, voxel_size = tuple(self.block_shape), tuple(self.output_voxel_size)
        axes, units = ["z", "y", "x"], ["nanometer"] * 3
        if self.output_channels_is_dict and self.output_channel_indices:
            stacked = len(self._channel_indices(channel))
            if stacked > 1:
                shape, chunk_shape = (stacked, *shape), (stacked, *chunk_shape)
                voxel_size, offset = (1, *voxel_size), (0, *offset)
                axes, units = ["c", *axes], ["", *units]
        try:
            array = prepare_ds(
                NestedDirectoryStore(path),
                shape,
                dtype=self.dtype,
                chunk_shape=chunk_shape,
                voxel_size=Coordinate(voxel_size),
                axis_names=axes,
                units=units,
                offset=Coordinate(offset),
            )
        except Exception as e:
            raise Exception(
                f"Failed to prepare {path} \n try deleting it manually and run again ! {e}"
            )
        try:
            z_store = NestedDirectoryStore(output.path / channel)
            zg = open_group(store=z_store, mode="a")
            zattrs = singlescale_attrs(
                arr_name="s0",
                voxel_size=voxel_size,
                offset=offset,
                units=units,
                axes=axes,
            )
            if "multiscales" in list(zg.attrs):
                old_multiscales = zg.attrs["multiscales"]
                if old_multiscales != zattrs["multiscales"]:
                    logger.info(f"Old multiscales: {old_multiscales}")
                    logger.info(f"New multiscales: {zattrs['multiscales']}")
                    raise ValueError(
                        f"multiscales attribute already exists in {z_store.path} and is "
                        "different from the new one. If it was written by an older "
                        "cellmap-flow, which placed outputs half a voxel off (OME "
                        "translation is a voxel centre), its blocks are on a different "
                        "grid: write to a new output path instead of resuming."
                    )
            zg.attrs["multiscales"] = zattrs["multiscales"]
        except Exception as e:
            raise Exception(
                f"Failed to prepare ome-ngff metadata for {path}, {e}"
            )
        return array

    def _snap_to_output_grid(self, roi, mode="shrink"):
        """``roi`` snapped onto the output voxel grid anchored at grid_origin:
        shrunk to the voxels it holds whole, or with ``mode="grow"`` grown to
        every voxel it touches."""
        shifted = daisy.Roi(roi.offset - self.grid_origin, roi.shape)
        snapped = shifted.snap_to_grid(self.output_voxel_size, mode=mode)
        return daisy.Roi(snapped.offset + self.grid_origin, snapped.shape)

    def _region_of(self, block):
        """The region (the daisy task) ``block`` is of, by its task id; the
        only one when there is one."""
        if len(self.regions) == 1:
            return self.regions[0]
        for region in self.regions:
            if region.name == block.task_id:
                return region
        raise ValueError(
            f"A block of task {block.task_id!r}, which is none of this run's: {[r.name for r in self.regions]}"
        )

    def _blocks_roi(self, region, block_shape):
        """The total write ROI of ``region``'s task, whose blocks are
        ``block_shape`` (nm).

        daisy lays a task's blocks from the corner of this ROI, so it starts
        on its output's block grid, anchored at the output's corner: at the
        box itself for a box with an output of its own, at or before it for
        one of several boxes sharing one. It ends where the region does;
        the last blocks overhang that end (fit="overhang"), and process_fn
        writes only what they hold inside the region.

        A block's label ids (Morton, Affinity) are keyed to its index on
        the volume's grid (Inferencer._chunk_index, from the raw data's
        corner), floored when the block is off that grid. Two blocks of one
        output share this grid, so the floored indices differ, and so do
        their ids; a box on the volume's grid gets the ids the server does.
        """
        corner = np.array(self.outputs[region.output].roi.offset)
        shape = np.array(block_shape)
        begin = corner + (np.array(region.roi.offset) - corner) // shape * shape
        return daisy.Roi(Coordinate(begin), region.roi.end - Coordinate(begin))

    def process_fn(self, block):
        if not self.inferencers:
            raise RuntimeError("Only a blockwise worker (--client) builds the models' inferencers")

        # A block writes only what is inside its region, the box or the
        # whole volume: the last blocks along each axis overhang its end, and
        # are cut there.
        region = self._region_of(block)
        arrays = self.outputs[region.output].arrays
        write_roi = block.write_roi.intersect(region.roi)
        if write_roi.empty:
            logger.warning(f"empty write roi: {write_roi}")
            return

        # Process chunk with all models
        if len(self.inferencers) == 1:
            # Single model - original behavior
            chunk_data = self.inferencers[0].process_chunk(
                self.idi_raw,
                block.write_roi,
                input_norms=self.input_norms,
                postprocess=self.postprocess,
                grid_origin=self.grid_origin,
            )
        else:
            # Multiple models - merge outputs based on model_mode
            model_outputs = []
            for inferencer in self.inferencers:
                output = inferencer.process_chunk(
                    self.idi_raw,
                    block.write_roi,
                    input_norms=self.input_norms,
                    postprocess=self.postprocess,
                    grid_origin=self.grid_origin,
                )
                if self.process_only and self.cross_channels_merger:
                    # Extract only the specified channels
                    channel_outputs = [output[ch_idx] for ch_idx in self.process_only]
                    # Merge the extracted channels based on cross_channels mode
                    output = self.cross_channels_merger.merge(channel_outputs)
                model_outputs.append(output)

            # Merge outputs based on model_mode
            chunk_data = self.model_merger.merge(model_outputs)

        chunk_data = chunk_data.astype(self.dtype)

        for channel, array in zip(self.output_channels, arrays):
            prediction = Array(
                self._output_data(chunk_data, channel), block.write_roi.offset, self.output_voxel_size
            )
            array_roi = write_roi
            if array.roi.dims == 4:  # stacked: its whole channel axis, which starts at 0
                array_roi = daisy.Roi((0, *write_roi.offset), (array.roi.shape[0], *write_roi.shape))
            array[array_roi] = prediction.to_ndarray(write_roi)

    def _output_data(self, chunk_data, channel):
        """The part of a block's prediction ``chunk_data`` (channels first)
        that output ``channel`` holds.

        With a dict of output_channels, the model channels its entry
        indexes: an entry listing several stacks them, in its order, on a
        leading axis. Otherwise, the model channel of that name. A
        prediction the chain has already made 3-D is one channel, and can
        fill only one output.
        """
        if chunk_data.ndim == 3:
            if len(self.output_channels) > 1:
                raise ValueError("output channels should be 1")
            return chunk_data
        if self.output_channels_is_dict and self.output_channel_indices:
            indices = self._channel_indices(channel)
            if len(indices) == 1:
                return chunk_data[indices[0]]
            return np.stack([chunk_data[index] for index in indices])
        return chunk_data[self.channels.index(channel)]

    def client(self):
        client = daisy.Client()
        while True:
            with client.acquire_block() as block:
                if block is None:
                    break
                try:
                    self.process_fn(block)

                    block.status = daisy.BlockStatus.SUCCESS
                    if self.track_progress:
                        marker = progress_marker(self.tmp_dir, block)
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.touch()
                except Exception:
                    logger.exception(f"Error processing block {block}")
                    block.status = daisy.BlockStatus.FAILED

    def run(self) -> bool:
        """Process every ROI; True only if every block of every ROI succeeded."""

        geometry = self.model_config.geometry
        context = geometry.context
        read_roi = daisy.Roi((0, 0, 0), Coordinate(geometry.read_shape))
        write_roi = read_roi.grow(-context, -context)

        if self.bounding_boxes:
            logger.info(f"Processing {len(self.bounding_boxes)} bounding box(es)")
        else:
            logger.info(f"Processing entire dataset: {self.full_output_roi}")

        # Process each ROI
        failures = []
        for roi_idx, region in enumerate(self.regions):
            total_write_roi = self._blocks_roi(region, write_roi.shape)
            total_read_roi = total_write_roi.grow(context, context)
            logger.info(f"ROI {roi_idx+1}/{len(self.regions)}: {region.roi}, blocks from {total_write_roi.offset}")

            # The daisy task id, and so each worker's LSF job name and log.
            name = region.name

            # Its blocks start on the output's chunk grid, so each writes
            # chunks of its own and reads only the raw data: none waits on
            # another (read_write_conflict).
            task = daisy.Task(
                name,
                total_roi=total_read_roi,
                read_roi=read_roi,
                write_roi=write_roi,
                process_function=spawn_worker(
                    name,
                    self.yaml_config,
                    self.charge_group,
                    self.queue,
                    ncpu=self.cpu_workers,
                    walltime=self.walltime,
                ),
                check_function=partial(check_block, self.tmp_dir) if self.track_progress else None,
                read_write_conflict=False,
                fit="overhang",
                max_retries=0,
                timeout=None,
                num_workers=self.workers,
            )

            task_state = _run_blockwise([task]).get(task.task_id)
            logger.info(f"ROI {roi_idx+1}/{len(self.regions)} - Task state: {task_state}")
            unfinished = _blocks_not_done(task_state)
            if unfinished:
                failures.append(f"{name}: {unfinished}")

        if failures:
            logger.error(
                "Blocks failed or were never processed -- "
                + "; ".join(failures)
                + ". Their errors are in the worker logs."
            )
            return False
        return True


def _refuse_another_region(path, roi):
    """Raise ConfigError if an array at ``path`` covers anything but ``roi`` (nm, z, y, x).

    Resuming writes into what is there, block by block, so it must be the
    same region. funlib's prepare_ds refuses another shape saying only that
    the dataset is not compatible, and takes another offset with the same
    shape, keeping the old one, which the OME check after it then blamed on
    an older cellmap-flow's half-voxel placement.
    """
    try:
        existing = open_ds(NestedDirectoryStore(path), "r")
    except ArrayNotFoundError:
        return
    there = daisy.Roi(existing.roi.offset[-3:], existing.roi.shape[-3:])
    if there != roi:
        raise ConfigError(
            f"{path} already holds an output covering {there} nm, and this task's covers {roi} nm: "
            "write to a new output path, or delete it. (A run over bounding_boxes by an older "
            "cellmap-flow wrote an array the size of the whole volume; it now covers only the boxes.)"
        )


def _run_blockwise(tasks, server_factory=None):
    """daisy.run_blockwise, but returning each task's TaskState.

    daisy.run_blockwise returns ``all(state.is_done())``, and is_done() counts
    failed blocks as done, so it said True however many blocks failed and the
    master exited 0. This runs the same sequence -- a server in a worker
    thread so Ctrl+C can stop it, with daisy's progress monitor -- and keeps
    the states.
    """
    from multiprocessing import Event
    from multiprocessing.pool import ThreadPool

    from daisy.cl_monitor import CLMonitor
    from daisy.tcp import IOLooper

    if server_factory is None:
        server_factory = daisy.Server
    stop_event = Event()

    def run():
        server = server_factory(stop_event=stop_event)
        CLMonitor(server)
        return server.run_blockwise(tasks)

    IOLooper.clear()
    with ThreadPool(processes=1) as pool:
        result = pool.apply_async(run)
        try:
            return result.get()
        except KeyboardInterrupt:
            stop_event.set()
            return result.get()


def _blocks_not_done(task_state) -> int:
    """Blocks of a task that failed, were orphaned, or never ran."""
    if task_state is None:
        return 1
    never_ran = (
        task_state.total_block_count
        - task_state.completed_count
        - task_state.failed_count
        - task_state.orphaned_count
    )
    return task_state.failed_count + task_state.orphaned_count + max(never_ran, 0)


def progress_marker(tmp_dir, block: daisy.Block) -> Path:
    """The file that records ``block`` as done: <tmp_dir>/<task>/<write ROI>.

    One directory per daisy task, and named by the block's write ROI rather
    than its index. The index is only unique within a task -- daisy counts it
    from each task's own total_roi, and every bounding box is its own task --
    so with one shared directory, finishing ROI 1 marked ROI 2's blocks with
    the same indices as done. An index also names a different region once the
    block size changes; a write ROI does not.
    """
    roi = block.write_roi
    name = "_".join(str(int(v)) for v in roi.offset) + "-" + "_".join(
        str(int(v)) for v in roi.shape
    )
    return Path(tmp_dir) / str(block.task_id) / name


def check_block(tmp_dir, block: daisy.Block) -> bool:
    return progress_marker(tmp_dir, block).exists()


def spawn_worker(name, yaml_config, charge_group, queue, ncpu=12, walltime=None, log_dir=None):
    """A daisy spawn function that submits one blockwise worker to LSF.

    Goes through jobs.launch.submit_bsub_job, so each worker gets a run limit
    (-W; the GPU queues' own default is two hours), its own log file named
    after its job id, and a submission that fails loudly: a bsub error
    raises here, and daisy stops the run, rather than the master waiting
    forever for a worker that never started.

    Logs go to ``log_dir``, by default ./daisy_logs as before, resolved now so
    that every worker writes to the same place.
    """
    log_dir = Path(log_dir or "daisy_logs").resolve()

    def run_worker():
        submit_bsub_job(
            f"cellmap_flow blockwise {shell_quote(yaml_config)} --client",
            queue=queue,
            charge_group=charge_group,
            job_name=str(name),
            num_gpus=1,
            num_cpus=ncpu,
            walltime=walltime or current_site().default_walltime,
            log_dir=log_dir,
            # A worker asking for more cores per GPU than the queue's ratio
            # is held by LSF for minutes before bsub returns; wait for it
            # rather than time out and lose track of a job that still lands.
            bsub_timeout=None,
        )

    return run_worker
