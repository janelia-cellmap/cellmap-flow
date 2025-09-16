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
        
        task_name = self.config["task_name"]
        self.workers = self.config["workers"]
        if self.workers <= 1:
            logger.error("Workers should be greater than 1.")
            return
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
            logger.error("Missing required field in YAML: tmp_dir, it is mandatory to track progress")
            return

        self.tmp_dir = Path(self.config["tmp_dir"]) / f"tmp_flow_daisy_progress_{task_name}"
        if not self.tmp_dir.exists():
            self.tmp_dir.mkdir(parents=True, exist_ok=True)

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

        print(f"output_shape: {output_shape}")
        print(f"type: {self.dtype}")
        print(f"output_path: {self.output_path}")
        for channel in self.output_channels:
            if create:
                try:
                    array = prepare_ds(
                        NestedDirectoryStore(self.output_path / channel / "s0"),
                        output_shape,
                        dtype=self.dtype,
                        chunk_shape=self.block_shape,
                        voxel_size=self.output_voxel_size,
                        axis_names=["z", "y", "x"],
                        units=["nanometer",]*3,
                        offset=(0, 0, 0),
                    )
                except Exception as e:
                    raise Exception(
                        f"Failed to prepare {self.output_path/channel/'s0'} \n try deleting it manually and run again ! {e}"
                    )
                try:
                    z_store = NestedDirectoryStore(self.output_path / channel)
                    zg = open_group(store=z_store, mode='a')
                    if 'multiscales' in zg.attrs:
                        raise ValueError(f'multiscales attribute already exists in {z_store.path}')
                    else:
                        zattrs = generate_singlescale_metadata(arr_name='s0',
                                                               voxel_size=self.output_voxel_size,
                                                               translation=[0.0,]*3,
                                                               units=['nanometer',]*3,
                                                               axes=['z', 'y', 'x'])
                        zg.attrs['multiscales'] = zattrs['multiscales']
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

        write_roi = block.write_roi.intersect(self.output_arrays[0].roi)

        if write_roi.empty:
            print(f"empty write roi: {write_roi}")
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
                "run",
                "-y",
                f"{yaml_config}",
                "--client",
            ]
        )

    return run_worker
