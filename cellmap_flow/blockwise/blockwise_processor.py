import logging
import subprocess
from pathlib import Path

import daisy
import numpy as np
from funlib.geometry.coordinate import Coordinate
from funlib.persistence import Array, open_ds, prepare_ds
from zarr.storage import NestedDirectoryStore
from zarr.hierarchy import open_group
from zarr.storage import DirectoryStore
from functools import partial
from cellmap_flow.globals import g
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.utils.config_utils import build_models, load_config
from cellmap_flow.utils.serilization_utils import get_process_dataset
from cellmap_flow.utils.ds import generate_singlescale_metadata

logger = logging.getLogger(__name__)


class CellMapFlowBlockwiseProcessor:

    def __init__(self, yaml_config: str, create=False):
        """Run the CellMapFlow server with a Fly model."""
        self.config = load_config(yaml_config)
        self.yaml_config = yaml_config

        self.input_path = self.config["data_path"]
        self.charge_group = self.config["charge_group"]
        self.queue = self.config["queue"]

        logger.info(f"Data path: {self.input_path}")

        if "output_path" not in self.config:
            raise Exception("Missing required field in YAML: output_path")
        self.output_path = self.config["output_path"]
        self.output_path = Path(self.output_path)

        output_channels = None
        if "output_channels" in self.config:
            output_channels = self.config["output_channels"]

        json_data = None
        if "json_data" in self.config:
            json_data = self.config["json_data"]

        if "task_name" not in self.config:
            raise Exception("Missing required field in YAML: task_name")
        if "workers" not in self.config:
            raise Exception("Missing required field in YAML: workers")

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
        if self.workers < 1:
            raise Exception("Workers should be greater than 0.")
        self.cpu_workers = self.config.get("cpu_workers", 12)
        # Added and create == True to fix client error when create: True in the yaml, so when it is a client it will not be changed
        if "create" in self.config and create == True:
            create = self.config["create"]
            if isinstance(create, str):
                logger.warning(
                    f"Type config[create] is str = {create}, better set a bool"
                )
                create = create.lower() == "true"

        if "tmp_dir" not in self.config:
            raise Exception(
                "Missing required field in YAML: tmp_dir, it is mandatory to track progress"
            )

        self.tmp_dir = (
            Path(self.config["tmp_dir"]) / f"tmp_flow_daisy_progress_{task_name}"
        )
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Build model configuration objects
        models = build_models(self.config["models"])
        # For debugging, print each model config
        for model in models:
            logger.info(str(model))

        if len(models) == 0:
            raise Exception("No models found in the configuration.")
        if len(models) > 1:
            raise Exception(
                "Multiple models is not currently supported by blockwise processor."
            )
        self.model_config = models[0]

        # this is zyx

        self.block_shape = [int(x) for x in self.model_config.config.block_shape][:3]

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

        self.dtype = g.get_output_dtype(self.model_config.output_dtype)

        if json_data:
            g.input_norms, g.postprocess = get_process_dataset(json_data)

        self.inferencer = Inferencer(self.model_config, use_half_prediction=False)

        self.idi_raw = ImageDataInterface(
            self.input_path, voxel_size=self.input_voxel_size
        )
        self.output_arrays = []

        output_shape = (
            np.array(self.idi_raw.shape)
            * np.array(self.input_voxel_size)
            / np.array(self.output_voxel_size)
        )

        logger.info(f"output_shape: {output_shape}")
        logger.info(f"type: {self.dtype}")
        logger.info(f"output_path: {self.output_path}")

        # Ensure we have output channels to iterate over
        channels_to_create = self.output_channels if self.output_channels else []

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
                        else:
                            # Single channel output - 3D array
                            final_output_shape = tuple(output_shape.astype(int))
                    else:
                        # List format - 3D array
                        final_output_shape = tuple(output_shape.astype(int))

                    array = prepare_ds(
                        NestedDirectoryStore(self.output_path / channel / "s0"),
                        final_output_shape,
                        dtype=self.dtype,
                        chunk_shape=(
                            self.block_shape
                            if len(final_output_shape) == 3
                            else (len(channel_indices),) + tuple(self.block_shape)
                        ),
                        voxel_size=(
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
                        offset=(
                            (0, 0, 0) if len(final_output_shape) == 3 else (0, 0, 0, 0)
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
                            metadata_translation = [0.0] * 4
                            metadata_units = [""] + ["nanometer"] * 3
                            metadata_axes = ["c", "z", "y", "x"]
                        else:
                            # 3D metadata
                            metadata_voxel_size = self.output_voxel_size
                            metadata_translation = [0.0] * 3
                            metadata_units = ["nanometer"] * 3
                            metadata_axes = ["z", "y", "x"]
                    else:
                        # List format - 3D metadata
                        metadata_voxel_size = self.output_voxel_size
                        metadata_translation = [0.0] * 3
                        metadata_units = ["nanometer"] * 3
                        metadata_axes = ["z", "y", "x"]

                    zattrs = generate_singlescale_metadata(
                        arr_name="s0",
                        voxel_size=metadata_voxel_size,
                        translation=metadata_translation,
                        units=metadata_units,
                        axes=metadata_axes,
                    )
                    if "multiscales" in list(zg.attrs):
                        old_multiscales = zg.attrs["multiscales"]
                        if old_multiscales != zattrs["multiscales"]:
                            raise ValueError(
                                f"multiscales attribute already exists in {z_store.path} and is different from the new one"
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

    def process_fn(self, block):
        logger.error(f"Processing block {block}")

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

        chunk_data = self.inferencer.process_chunk(self.idi_raw, block.write_roi)

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
                    (self.tmp_dir / f"{block.block_id[1]}").touch()
                except Exception as e:
                    logger.error(f"Error processing block {block}: {e}")
                    block.status = daisy.BlockStatus.FAILED

    def run(self):

        read_shape = self.model_config.config.read_shape
        write_shape = self.model_config.config.write_shape

        context = (Coordinate(read_shape) - Coordinate(write_shape)) / 2

        read_roi = daisy.Roi((0, 0, 0), read_shape)
        write_roi = read_roi.grow(-context, -context)

        total_write_roi = self.idi_raw.roi
        # .snap_to_grid(self.output_voxel_size)
        total_read_roi = total_write_roi.grow(context, context)

        name = f"predict_{self.model_config.name}{self.task_name}"

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
            ),
            check_function=partial(check_block, self.tmp_dir),
            read_write_conflict=False,
            fit="overhang",
            max_retries=0,
            timeout=None,
            num_workers=self.workers,
        )

        task_state = daisy.run_blockwise([task])
        logger.info(f"Task state: {task_state}")


def check_block(tmp_dir, block: daisy.Block) -> bool:
    return (tmp_dir / f"{block.block_id[1]}").exists()


def spawn_worker(name, yaml_config, charge_group, queue, ncpu=12):
    def run_worker():
        if not Path("prediction_logs").exists():
            Path("prediction_logs").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "bsub",
                "-P",
                charge_group,
                "-J",
                str(name),
                "-q",
                queue,
                "-n",
                str(ncpu),
                "-gpu",
                "num=1",
                "-o",
                f"prediction_logs/out.out",
                "-e",
                f"prediction_logs/out.err",
                "cellmap_flow_blockwise_processor",
                f"{yaml_config}",
                "--client",
            ]
        )

    return run_worker
