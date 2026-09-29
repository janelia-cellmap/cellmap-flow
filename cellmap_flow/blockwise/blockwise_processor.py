import logging

# No logging configuration here: the CLIs set it up, and the dashboard,
# which imports this lazily, keeps its own.
logger = logging.getLogger(__name__)

import os
import shlex
from pathlib import Path
import zarr
import daisy
import numpy as np
from funlib.geometry.coordinate import Coordinate
from funlib.persistence import Array, open_ds, prepare_ds
from zarr.storage import NestedDirectoryStore
from zarr.hierarchy import open_group
from functools import partial
from cellmap_flow.globals import g
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.utils.config_utils import (
    ConfigError,
    build_models,
    load_config,
    resolve_data_path,
)
from cellmap_flow.utils.serilization_utils import get_process_dataset
from cellmap_flow.utils.ds import generate_singlescale_metadata
from cellmap_flow.models.model_merger import get_model_merger
from cellmap_flow.utils.bsub_utils import DEFAULT_WALLTIME, submit_bsub_job


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
    bounding_boxes = config.get("bounding_boxes", None)
    if (
        bounding_boxes
        and config.get("separate_bounding_boxes_zarrs", False)
        and len(bounding_boxes) > 1
    ):
        raise ConfigError(
            "separate_bounding_boxes_zarrs can only be used with one bounding box"
        )
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
    config = load_config(yaml_config)
    _validate_settings(config)

    # Constructing a ModelConfig does not load the model; that happens when
    # .config is first read, which nothing here does.
    models = build_models(config["models"])
    if len(models) == 0:
        raise ConfigError("No models found in the configuration.")

    data_path = resolve_data_path(config["data_path"], getattr(models[0], "scale", None))
    if "://" not in str(data_path) and not os.path.exists(data_path):
        raise ConfigError(f"data_path does not exist: {data_path}")

    if config.get("json_data"):
        try:
            get_process_dataset(config["json_data"])  # built and discarded
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
        self.config = load_config(yaml_config)
        _validate_settings(self.config)
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
        # LSF run limit for each worker. Without -W the GPU queues kill a
        # worker at two hours; see bsub_utils.DEFAULT_WALLTIME.
        self.walltime = (
            self.config.get("walltime")
            or getattr(g, "walltime", None)
            or DEFAULT_WALLTIME
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
        models = build_models(self.config["models"])
        # The master (create) only needs the models' geometry, to schedule
        # blocks and create the outputs; the workers run the model, so only
        # they build Inferencers and run the dummy forward pass.
        for model in models:
            model.validate_model_shapes = not create
            logger.info(str(model))

        if len(models) == 0:
            raise ConfigError("No models found in the configuration.")

        # The same data_path + scale rule as cellmap_flow and cellmap_flow_yaml.
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

        # this is zyx

        block_shape = [int(x) for x in self.model_config.config.block_shape][:3]
        self.block_shape = tuple(self.config.get("block_size", block_shape))

        self.input_voxel_size = Coordinate(self.model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(self.model_config.config.output_voxel_size)
        # self.output_channels = self.model_config.config.output_channels
        self.channels = self.model_config.config.channels

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

        if json_data:
            g.input_norms, g.postprocess = get_process_dataset(json_data)

        self.dtype = g.get_output_dtype(self.model_config.output_dtype)

        self.inferencers = []
        self.inferencer = None
        if not create:
            self.inferencers = [
                Inferencer(model, use_half_prediction=False) for model in self.models
            ]
            self.inferencer = self.inferencers[0]  # Keep for backward compatibility

        self.idi_raw = ImageDataInterface(
            self.input_path, voxel_size=self.input_voxel_size
        )
        self.output_arrays = []

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

        if self.bounding_boxes and self.separate_zarrs:
            bounding_box = self.bounding_boxes[0]
            offset = tuple(bounding_box.get("offset", [0, 0, 0]))
            shape = tuple(bounding_box.get("shape", [0, 0, 0]))
            roi = daisy.Roi(offset, shape)
            roi2 = self._snap_to_output_grid(roi)
            if roi2 != roi:
                logger.warning(f"Bounding box ROI {roi} was not aligned to output voxel size grid {self.output_voxel_size}, it has been adjusted to {roi2} to avoid misalignment issues.")
            roi = roi2
            output_shape = (np.array(roi.shape)
                        / np.array(self.output_voxel_size)
                        ).astype(int)
            offset = (np.array(roi.offset)
                    #   /np.array(self.output_voxel_size)
                      ).astype(int)
        else:
            # Cover the raw data where it actually is: its offset (a corner)
            # and extent, snapped inward to the output voxel grid. The whole-
            # volume task in run() uses the same ROI, so its blocks start on
            # the output's chunk grid instead of straddling chunks.
            output_shape = (
                np.array(self.full_output_roi.shape) / np.array(self.output_voxel_size)
            ).astype(int)
            offset = tuple(int(o) for o in self.full_output_roi.offset)

        logger.info(f"output_shape: {output_shape}")
        logger.info(f"type: {self.dtype}")
        logger.info(f"output_path: {self.output_path}")

        # Ensure we have output channels to iterate over
        channels_to_create = self.output_channels if self.output_channels else []
        if not isinstance(channels_to_create, list):
            channels_to_create = [channels_to_create]

        # check if there is two channels_to_create with same name
        if len(channels_to_create) != len(set(channels_to_create)):
            raise Exception(f"output_channels has duplicated channel names. channels: {channels_to_create}")

        for channel in channels_to_create:
            if create:
                try:
                    # Determine output shape - for dict format with multiple channels, we need 4D
                    if self.output_channels_is_dict and self.output_channel_indices:
                        channel_indices = self.output_channel_indices[channel]
                        if isinstance(channel_indices, int):
                            channel_indices = [channel_indices]

                        if len(channel_indices) > 1:
                            # Multi-channel output - need 4D array (channels, z, y, x)
                            final_output_shape = (len(channel_indices),) + tuple(
                                output_shape.astype(int)
                            )
                            final_offset = (0,) + tuple(offset)
                            
                        else:
                            # Single channel output - 3D array
                            final_output_shape = tuple(output_shape.astype(int))
                            final_offset = tuple(offset)
                    else:
                        # List format - 3D array
                        final_output_shape = tuple(output_shape.astype(int))
                        final_offset = tuple(offset)
                    array = prepare_ds(
                        NestedDirectoryStore(self.output_path / channel / "s0"),
                        final_output_shape,
                        dtype=self.dtype,
                        chunk_shape=(
                            self.block_shape
                            if len(final_output_shape) == 3
                            else (len(channel_indices),) + self.block_shape
                        ),
                        voxel_size=Coordinate(
                            self.output_voxel_size
                            if len(final_output_shape) == 3
                            else (1,) + tuple(self.output_voxel_size)
                        ),
                        axis_names=(
                            ["z", "y", "x"]
                            if len(final_output_shape) == 3
                            else ["c", "z", "y", "x"]
                        ),
                        units=(
                            ["nanometer"] * 3
                            if len(final_output_shape) == 3
                            else [""] + ["nanometer"] * 3
                        ),
                        offset=Coordinate(
                            final_offset
                        ),
                    )
                except Exception as e:
                    raise Exception(
                        f"Failed to prepare {self.output_path/channel/'s0'} \n try deleting it manually and run again ! {e}"
                    )
                try:
                    z_store = NestedDirectoryStore(self.output_path / channel)
                    zg = open_group(store=z_store, mode="a")

                    # Determine metadata parameters based on dimensionality
                    if self.output_channels_is_dict and self.output_channel_indices:
                        channel_indices = self.output_channel_indices[channel]
                        if isinstance(channel_indices, int):
                            channel_indices = [channel_indices]

                        if len(channel_indices) > 1:
                            # 4D metadata
                            metadata_voxel_size = (1,) + tuple(self.output_voxel_size)
                            metadata_offset = list(final_offset)
                            metadata_units = [""] + ["nanometer"] * 3
                            metadata_axes = ["c", "z", "y", "x"]
                        else:
                            # 3D metadata
                            metadata_voxel_size = self.output_voxel_size
                            metadata_offset = list(final_offset)
                            metadata_units = ["nanometer"] * 3
                            metadata_axes = ["z", "y", "x"]
                    else:
                        # List format - 3D metadata
                        metadata_voxel_size = self.output_voxel_size
                        metadata_offset = list(final_offset)
                        metadata_units = ["nanometer"] * 3
                        metadata_axes = ["z", "y", "x"]

                    zattrs = generate_singlescale_metadata(
                        arr_name="s0",
                        voxel_size=metadata_voxel_size,
                        offset=metadata_offset,
                        units=metadata_units,
                        axes=metadata_axes,
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
                        f"Failed to prepare ome-ngff metadata for {self.output_path/channel/'s0'}, {e}"
                    )
            else:
                try:
                    array = open_ds(
                        NestedDirectoryStore(self.output_path / channel / "s0"),
                        "a",
                    )
                except Exception as e:
                    raise Exception(f"Failed to open {self.output_path/channel}\n{e}")
            self.output_arrays.append(array)

    def _snap_to_output_grid(self, roi):
        """``roi`` shrunk onto the output voxel grid anchored at grid_origin."""
        shifted = daisy.Roi(roi.offset - self.grid_origin, roi.shape)
        snapped = shifted.snap_to_grid(self.output_voxel_size, mode="shrink")
        return daisy.Roi(snapped.offset + self.grid_origin, snapped.shape)

    def process_fn(self, block):
        if not self.inferencers:
            raise RuntimeError("Only a blockwise worker (--client) builds the models' inferencers")

        # Handle 4D vs 3D array ROI intersection
        first_array = self.output_arrays[0]
        if len(first_array.roi.shape) == 4:
            # For 4D arrays, create spatial ROI by skipping the channel dimension
            array_spatial_roi = daisy.Roi(
                first_array.roi.offset[1:], first_array.roi.shape[1:]
            )
            write_roi = block.write_roi.intersect(array_spatial_roi)
        else:
            # For 3D arrays, use normal intersection
            write_roi = block.write_roi.intersect(first_array.roi)

        if write_roi.empty:
            logger.warning(f"empty write roi: {write_roi}")
            return

        # Process chunk with all models
        if len(self.inferencers) == 1:
            # Single model - original behavior
            chunk_data = self.inferencers[0].process_chunk(
                self.idi_raw, block.write_roi
            )
        else:
            # Multiple models - merge outputs based on model_mode
            model_outputs = []
            for inferencer in self.inferencers:
                output = inferencer.process_chunk(self.idi_raw, block.write_roi)
                if self.process_only and self.cross_channels_merger:
                    # Extract only the specified channels
                    channel_outputs = [output[ch_idx] for ch_idx in self.process_only]
                    # Merge the extracted channels based on cross_channels mode
                    output = self.cross_channels_merger.merge(channel_outputs)
                model_outputs.append(output)

            # Merge outputs based on model_mode
            chunk_data = self.model_merger.merge(model_outputs)

        chunk_data = chunk_data.astype(self.dtype)

        for i, array in enumerate(self.output_arrays):
            if not self.output_channels or i >= len(self.output_channels):
                continue

            channel_name = self.output_channels[i]

            if chunk_data.ndim == 3:
                if len(self.output_channels) > 1:
                    raise ValueError("output channels should be 1")
                predictions = Array(
                    chunk_data,
                    block.write_roi.offset,
                    self.output_voxel_size,
                )
            else:
                if self.output_channels_is_dict and self.output_channel_indices:
                    # Dictionary format: extract multiple channels for this output
                    channel_indices = self.output_channel_indices[channel_name]
                    if isinstance(channel_indices, int):
                        channel_indices = [channel_indices]

                    if len(channel_indices) == 1:
                        # Single channel output
                        channel_data = chunk_data[channel_indices[0]]
                    else:
                        # Multi-channel output - stack channels
                        channel_data = np.stack(
                            [chunk_data[idx] for idx in channel_indices], axis=0
                        )

                    predictions = Array(
                        channel_data,
                        block.write_roi.offset,
                        self.output_voxel_size,
                    )
                else:
                    # List format: original behavior
                    index = self.channels.index(channel_name)
                    predictions = Array(
                        chunk_data[index],
                        block.write_roi.offset,
                        self.output_voxel_size,
                    )
            # Handle writing to 4D vs 3D arrays
            if len(array.roi.shape) == 4:
                # For 4D arrays, create spatial ROI and then full ROI for writing
                array_spatial_roi = daisy.Roi(array.roi.offset[1:], array.roi.shape[1:])
                spatial_write_roi = write_roi.intersect(array_spatial_roi)
                if spatial_write_roi.empty:
                    continue
                # For 4D array writing, we need to include the channel dimension
                full_write_roi = daisy.Roi(
                    (0,) + spatial_write_roi.offset,
                    (array.roi.shape[0],) + spatial_write_roi.shape,
                )
                array[full_write_roi] = predictions.to_ndarray(spatial_write_roi)
            else:
                # For 3D arrays, use normal intersection and writing
                array_write_roi = write_roi.intersect(array.roi)
                if array_write_roi.empty:
                    continue
                array[array_write_roi] = predictions.to_ndarray(array_write_roi)

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

        read_shape = self.model_config.config.read_shape
        write_shape = self.model_config.config.write_shape

        context = (Coordinate(read_shape) - Coordinate(write_shape)) / 2

        read_roi = daisy.Roi((0, 0, 0), read_shape)
        write_roi = read_roi.grow(-context, -context)

        # Check if bounding boxes are specified
        bounding_boxes = self.config.get("bounding_boxes", None)

        conflicts = False
        
        if bounding_boxes and len(bounding_boxes) > 0:
            # Process specific ROIs from bounding boxes
            logger.info(f"Processing {len(bounding_boxes)} bounding box(es)")
            rois_to_process = []
            # If there is ROI the ROI can be different than the block order which can cause read-write conflicts
            # best way is to align the ROI with the block shape, but for now we will just warn the user about potential conflicts
            if not self.separate_zarrs:
                conflicts = True
            
            for i, bbox in enumerate(bounding_boxes):
                offset = tuple(bbox.get("offset", [0, 0, 0]))
                shape = tuple(bbox.get("shape", [0, 0, 0]))
                roi = daisy.Roi(offset, shape)
                roi = self._snap_to_output_grid(roi)
                rois_to_process.append(roi)
                logger.info(f"Bounding box {i+1}: offset={offset}, shape={shape}")
        else:
            # Process entire dataset: the extent of the output array.
            total_write_roi = self.full_output_roi
            rois_to_process = [total_write_roi]
            logger.info(f"Processing entire dataset: {total_write_roi}")

        # Process each ROI
        failures = []
        for roi_idx, total_write_roi in enumerate(rois_to_process):
            total_read_roi = total_write_roi.grow(context, context)
            
            name = f"predict_{self.model_config.name}{self.task_name}"
            if len(rois_to_process) > 1:
                name = f"{name}_roi{roi_idx+1}"

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
                read_write_conflict=conflicts,
                fit="overhang",
                max_retries=0,
                timeout=None,
                num_workers=self.workers,
            )

            task_state = _run_blockwise([task]).get(task.task_id)
            logger.info(f"ROI {roi_idx+1}/{len(rois_to_process)} - Task state: {task_state}")
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

    Goes through bsub_utils.submit_bsub_job, so each worker gets a run limit
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
            f"cellmap_flow_blockwise {shlex.quote(str(yaml_config))} --client",
            queue=queue,
            charge_group=charge_group,
            job_name=str(name),
            num_gpus=1,
            num_cpus=ncpu,
            walltime=walltime or DEFAULT_WALLTIME,
            log_dir=log_dir,
            # A worker asking for more cores per GPU than the queue's ratio
            # is held by LSF for minutes before bsub returns; wait for it
            # rather than time out and lose track of a job that still lands.
            bsub_timeout=None,
        )

    return run_worker
