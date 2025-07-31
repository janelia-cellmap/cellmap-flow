import logging
import daisy
from zarr.storage import DirectoryStore
import numpy as np
from funlib.geometry.coordinate import Coordinate
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer

from cellmap_flow.utils.config_utils import load_config, build_models

from cellmap_flow.utils.serilization_utils import get_process_dataset

from funlib.persistence import prepare_ds, open_ds, Array
from pathlib import Path

from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


class CellMapFlowBlockwiseProcessor:

    def __init__(self, yaml_config: str, create=True):
        """Run the CellMapFlow server with a Fly model."""
        self.config = load_config(yaml_config)
        self.yaml_config = yaml_config

        self.input_path = self.config["data_path"]
        self.charge_group = self.config["charge_group"]
        self.queue = self.config["queue"]

        print("Data path:", self.input_path)

        if "output_path" not in self.config:
            logger.error("Missing required field in YAML: output_path")
            return
        self.output_path = self.config["output_path"]
        self.output_path = Path(self.output_path)

        output_channels = None
        if "output_channels" in self.config:
            output_channels = self.config["output_channels"].split(",")

        json_data = None
        if "json_data" in self.config:
            json_data = self.config["json_data"]

        if "task_name" not in self.config:
            logger.error("Missing required field in YAML: task_name")
            return
        if "workers" not in self.config:
            logger.error("Missing required field in YAML: workers")
            return
        self.workers = self.config["workers"]
        if self.workers <= 1:
            logger.error("Workers should be greater than 1.")
            return
        if "create" in self.config:
            create = self.config["create"]
            if isinstance(create, str):
                logger.warning(
                    f"Type config[create] is str = {create}, better set a bool"
                )
                create = create.lower() == "true"

        task_name = self.config["task_name"]

        # Build model configuration objects
        models = build_models(self.config["models"])
        # For debugging, print each model config
        for model in models:
            print(model)

        if len(models) == 0:
            logger.error("No models found in the configuration.")
            return
        if len(models) > 1:
            logger.error(
                "Multiple models is not currently supported by blockwise processor."
            )
            return
        self.model_config = models[0]

        # this is zyx

        self.block_shape = [int(x) for x in self.model_config.config.block_shape][:3]

        self.input_voxel_size = Coordinate(self.model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(self.model_config.config.output_voxel_size)
        self.output_channels = self.model_config.config.output_channels
        self.channels = self.model_config.config.channels

        self.task_name = task_name
        if output_channels:
            self.output_channels = output_channels
        else:
            self.output_channels = self.channels

        self.dtype = g.get_output_dtype(self.model_config.output_dtype)

        if json_data:
            g.input_norms, g.postprocess = get_process_dataset(json_data)

        self.inferencer = Inferencer(self.model_config)

        self.idi_raw = ImageDataInterface(
            self.input_path, voxel_size=self.input_voxel_size
        )
        self.output_arrays = []

        output_shape = (
            np.array(self.idi_raw.shape)
            * np.array(self.input_voxel_size)
            / np.array(self.output_voxel_size)
        )

        print(f"output_shape: {output_shape}")
        print(f"type: {self.dtype}")
        print(f"output_path: {self.output_path}")
        for channel in self.output_channels:
            if create:
                try:
                    array = prepare_ds(
                        DirectoryStore(self.output_path / channel / "s0"),
                        output_shape,
                        dtype=self.dtype,
                        chunk_shape=self.block_shape,
                        voxel_size=self.output_voxel_size,
                        axis_names=["z", "y", "x"],
                        units=["nm", "nm", "nm"],
                        offset=(0, 0, 0),
                    )
                except Exception as e:
                    raise Exception(
                        f"Failed to prepare {self.output_path/channel/'s0'} \n try deleting it manually and run again ! {e}"
                    )
            else:
                try:
                    array = open_ds(
                        DirectoryStore(self.output_path / channel / "s0"),
                        "a",
                    )
                except Exception as e:
                    raise Exception(f"Failed to open {self.output_path/channel}\n{e}")
            self.output_arrays.append(array)

    def process_fn(self, block):

        write_roi = block.write_roi.intersect(self.output_arrays[0].roi)

        if write_roi.empty:
            print(f"empty write roi: {write_roi}")
            return

        # Check if block is already processed before expensive inference computation
        fill_value = getattr(self.output_arrays[0], 'fill_value', self.dtype(0))
        if not (self.output_arrays[0][write_roi] == fill_value).all():
            return

        chunk_data = self.inferencer.process_chunk(self.idi_raw, block.write_roi)

        chunk_data = chunk_data.astype(self.dtype)

        for i, array in enumerate(self.output_arrays):

            if chunk_data.shape == 3:
                if len(self.output_channels) > 1:
                    raise ValueError("output channels should be 1")
                predictions = Array(
                    chunk_data,
                    block.write_roi.offset,
                    self.output_voxel_size,
                )
            else:
                index = self.channels.index(self.output_channels[i])
                predictions = Array(
                    chunk_data[index],
                    block.write_roi.offset,
                    self.output_voxel_size,
                )
            array[write_roi] = predictions.to_ndarray(write_roi)

    def client(self):
        client = daisy.Client()
        while True:
            with client.acquire_block() as block:
                if block is None:
                    break
                self.process_fn(block)

                block.status = daisy.BlockStatus.SUCCESS

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
            ),
            read_write_conflict=True,
            fit="overhang",
            max_retries=2,
            timeout=1800,
            num_workers=self.workers,
        )

        daisy.run_blockwise([task])
        # , multiprocessing= False


import subprocess


def spawn_worker(name, yaml_config, charge_group, queue, ncpu=12):
    def run_worker():
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
                "run",
                "-y",
                f"{yaml_config}",
                "--client",
            ]
        )

    return run_worker
