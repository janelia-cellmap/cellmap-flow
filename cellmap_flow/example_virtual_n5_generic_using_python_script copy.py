# %% load image zoo

from cellmap_flow.utils import load_safe_config
import argparse
from http import HTTPStatus
from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np
import numcodecs
from scipy import spatial
from zarr.n5 import N5ChunkWrapper
from funlib.geometry import Roi
import numpy as np
import logging

logger = logging.getLogger(__name__)

# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
import socket
from cellmap_flow.image_data_interface import ImageDataInterface


from cellmap_flow.inferencer import Inferencer


# %%
script_path = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py"
config = load_safe_config(script_path)
SCALE_LEVEL = 0
IDI_RAW = None
OUTPUT_VOXEL_SIZE = None
VOL_SHAPE_ZYX = None
VOL_SHAPE = None
VOL_SHAPE_ZYX_IN_BLOCKS = None
VOXEL_SIZE = None
BLOCK_SHAPE = config.block_shape
MAX_SCALE = None
CHUNK_ENCODER = None
EQUIVALENCES = None
DS = None
INFERENCER = Inferencer(script_path=script_path)
dataset_name = "/nrs/cellmap/data/jrc_mus-cerebellum-1/jrc_mus-cerebellum-1.zarr/recon-1/em/fibsem-uint8"
IDI_RAW = ImageDataInterface(f"{dataset_name}/s{SCALE_LEVEL}")
OUTPUT_VOXEL_SIZE = config.output_voxel_size

# determined-chimpmunk is edges


# %%
MAX_SCALE = 0

VOL_SHAPE_ZYX = np.array(IDI_RAW.shape)
VOL_SHAPE = np.array([*VOL_SHAPE_ZYX[::-1], 8])
# VOL_SHAPE_ZYX_IN_BLOCKS = np.ceil(VOL_SHAPE_ZYX / BLOCK_SHAPE[:3]).astype(int)
# VOXEL_SIZE = IDI_RAW.voxel_size

CHUNK_ENCODER = N5ChunkWrapper(np.uint8, BLOCK_SHAPE, compressor=numcodecs.GZip())

scales = [[2**s, 2**s, 2**s, 1] for s in range(MAX_SCALE + 1)]
attr = {
    "pixelResolution": {
        "dimensions": [*OUTPUT_VOXEL_SIZE, 1],
        "unit": "nm",
    },
    "ordering": "C",
    "scales": scales,
    "axes": ["x", "y", "z", "c^"],
    "units": ["nm", "nm", "nm", ""],
    "translate": [0, 0, 0, 0],
}


attr

# %%

attr = {
    "transform": {
        "ordering": "C",
        "axes": ["x", "y", "z", "c^"],
        "scale": [
            *OUTPUT_VOXEL_SIZE,
            1,
        ],
        "units": ["nm", "nm", "nm", ""],
        "translate": [0.0, 0.0, 0.0, 0.0],
    },
    "compression": {"type": "gzip", "useZlib": False, "level": -1},
    "blockSize": BLOCK_SHAPE[:].tolist(),
    "dataType": "uint8",
    "dimensions": VOL_SHAPE.tolist(),
}
attr

chunk_x = 2
chunk_y = 2
chunk_z = 2


corner = BLOCK_SHAPE[:3] * np.array([chunk_z, chunk_y, chunk_x])
box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE
roi = Roi(box[0], box[1])
print("about_to_process_chunk")
chunk = INFERENCER.process_chunk_basic(IDI_RAW, roi)
# logger.error(f"chunk {chunk}")
print(chunk)
x = (
    # Encode to N5 chunk format (header + compressed data)
    CHUNK_ENCODER.encode(chunk),
    HTTPStatus.OK,
    {"Content-Type": "application/octet-stream"},
)


# %%
