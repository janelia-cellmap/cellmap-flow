import logging
import shutil
import daisy
from zarr.storage import DirectoryStore
import numpy as np
from funlib.geometry.coordinate import Coordinate
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.utils.data import ModelConfig
from cellmap_flow.utils.web_utils import (
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
)
from cellmap_flow.norm.input_normalize import get_normalizations
from cellmap_flow.post.postprocessors import get_postprocessors

from funlib.persistence import prepare_ds, open_ds, Array
from pathlib import Path

from cellmap_flow.globals import Flow
from cellmap_flow.utils.web_utils import encode_to_str, decode_to_json

logger = logging.getLogger(__name__)

def get_output_dtype():
    dtype = np.float32
    # if len(g.input_norms) > 0:
    #     for norm in g.input_norms[::-1]:
    #         if norm.dtype:
    #             dtype = norm.dtype
    #             break
    g = Flow()
    if len(g.postprocess) > 0:
        for postprocess in g.postprocess[::-1]:
            if postprocess.dtype:
                dtype = postprocess.dtype
                break
    return dtype


def get_process_dataset(json_data: str):
    logger.error(f"json data: {json_data}")
    input_norm_fns = get_normalizations(json_data[INPUT_NORM_DICT_KEY])
    postprocess_fns = get_postprocessors(json_data[POSTPROCESS_DICT_KEY])
    return input_norm_fns, postprocess_fns


class CellMapFlowBlockwiseProcessor:

    def __init__(self, dataset_name: str, model_config: ModelConfig, output_path,json_data: str = None, create=True,overwrite=False,output_channels=None,task_name=""):
        # this is zyx
        self.model_config = model_config
        self.input_path = dataset_name
        self.output_path = output_path
        self.chpoint_path = model_config.chpoint_path
        self.block_shape = [int(x) for x in model_config.config.block_shape][:3]

        self.input_voxel_size = Coordinate(model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels
        self.channels = model_config.config.channels
        self.task_name = task_name
        if output_channels:
            self.output_channels = output_channels
        else:
            self.output_channels = self.channels
        
        self.dtype = get_output_dtype()

        g = Flow()

        self.json_str = None

        if json_data:
            if type(json_data) == str:
                json_data = decode_to_json(json_data)
            g.input_norms, g.postprocess = get_process_dataset(json_data)
            self.json_str = encode_to_str(json_data)

        self.inferencer = Inferencer(model_config)

        self.idi_raw = ImageDataInterface(
            dataset_name, target_resolution=self.input_voxel_size
        )
        self.output_arrays = []

        output_path = Path(output_path)

        output_shape = (
            np.array(self.idi_raw.shape)
            * np.array(self.input_voxel_size)
            / np.array(self.output_voxel_size)
        )

        print(f"output_shape: {output_shape}")
        print(f"type: {self.dtype}")
        print(f"output_path: {output_path}")
        for channel in self.output_channels:
            path = output_path / channel
            exists = path.exists()

            if exists:
                if create and overwrite:
                    # case (1,1,1): delete then prepare
                    print(f"Deleting existing directory: {path}")
                    shutil.rmtree(path)
                    print(f"Creating directory: {path}")
                    array = prepare_ds(
                        DirectoryStore(path),
                        output_shape,
                        dtype=self.dtype,
                        chunk_shape=self.block_shape,
                        voxel_size=self.output_voxel_size,
                        axis_names=["z", "y", "x"],
                        units=["nm", "nm", "nm"],
                        offset=(0, 0, 0),
                    )
                else:
                    # all other exists=1 cases (0,0,1), (1,0,1), (0,1,1): just open
                    print(f"Opening existing dataset: {path}")
                    try:
                        array = open_ds(
                            DirectoryStore(path),
                            "a",
                        )
                    except Exception as e:
                        raise Exception(f"Failed to open {path}\n{e}")
            else:
                if create:
                    # both (1,0,0) and (1,1,0): prepare
                    print(f"Creating directory: {path}")
                    array = prepare_ds(
                        DirectoryStore(path),
                        output_shape,
                        dtype=self.dtype,
                        chunk_shape=self.block_shape,
                        voxel_size=self.output_voxel_size,
                        axis_names=["z", "y", "x"],
                        units=["nm", "nm", "nm"],
                        offset=(0, 0, 0),
                    )
                else:
                    # both (0,0,0) and (0,1,0): error
                    raise FileNotFoundError(
                        f"Directory {path} does not exist. Use create=True to create it."
                    )

            self.output_arrays.append(array)
        

    def process_fn(self, block):

        write_roi = block.write_roi.intersect(self.output_arrays[0].roi)

        if write_roi.empty:
            print(f"empty write roi: {write_roi}")
            return
        
        if self.output_arrays[0].to_ndarray(write_roi).any():
            print(f"write roi already filled: {write_roi}")
            return

        chunk_data = self.inferencer.process_chunk(self.idi_raw, block.write_roi)

        chunk_data = chunk_data.astype(self.dtype)

        # if self.output_arrays[0][block.write_roi].any():
        #     return

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

    def run(self, workers=15,timeout=60):

        read_shape = self.model_config.config.read_shape
        write_shape = self.model_config.config.write_shape

        context = (
                Coordinate(read_shape)
                - Coordinate(write_shape)
            ) / 2

        read_roi = daisy.Roi((0,0,0), read_shape)
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
        process_function=spawn_worker(name,
        self.chpoint_path,
        self.channels,
        self.input_voxel_size,
        self.output_voxel_size,
        self.input_path,
        self.output_path,
        self.output_channels,
        self.json_str
        ),
        read_write_conflict=True,
        fit="overhang",
        max_retries=2,
        timeout=timeout,
        num_workers=workers,
        )

        daisy.run_blockwise([task])
        # , multiprocessing= False

        

import subprocess
def spawn_worker(name,
   checkpoint, 
   channels, 
   input_voxel_size, 
   output_voxel_size, 
   data_path, 
   output_path,
   output_channels,
   json_str
):
    def run_worker():
        subprocess.run(
            ["bsub",
             "-P",
             "cellmap",
             "-J",
             str(name),
             "-q",
             "gpu_h100",
             "-n",
             "12",
             "-gpu",
             "num=1",
             "-o",
             f"prediction_logs/out.out",
             "-e",
             f"prediction_logs/out.err",
             "fly_processor",
             "run",
                "-c",
                f"{checkpoint}",
                "-ch",
                f"{','.join(channels)}",
                "-ivs",
                f"{','.join(map(str, input_voxel_size))}",
                "-ovs",
                f"{','.join(map(str, output_voxel_size))}",
                "-d",
                f"{data_path}",
                "-o",
                f"{output_path}",
             ]
             +(["-outc",f"{','.join(output_channels)}"] if output_channels else [])
             +(["-json",f"{json_str}"] if json_str else [])
             )


    return run_worker