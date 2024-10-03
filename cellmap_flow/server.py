# import argparse
# from http import HTTPStatus
# from flask import Flask, jsonify
# from flask_cors import CORS

import numpy as np
import numcodecs
import json
from zarr.n5 import N5ChunkWrapper
import torch
from fastapi.middleware.cors import CORSMiddleware
from funlib.persistence import open_ds
from funlib.geometry import Roi
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
from fastapi.encoders import jsonable_encoder
from cellmap_flow.settings import KEY_SSL, CERT_SSL

# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
from funlib.geometry import Coordinate
from dacapo.experiments.architectures import CNNectomeUNetConfig

from dacapo.experiments.tasks import DistanceTaskConfig
import gc
import uvicorn



# This demo produces an RGB volume for aesthetic purposes.
# Note that this is 3 (virtual) teravoxels per channel.
NUM_CHANNELS = 1
BLOCK_SHAPE = np.array([68, 68, 68, NUM_CHANNELS])
MAX_SCALE = 0

CHUNK_ENCODER = N5ChunkWrapper(np.float32, BLOCK_SHAPE, compressor=numcodecs.GZip())

MODEL = None
DS = None
VOL_SHAPE = None
INPUT_VOXEL_SIZE = None
description = """
This is a CellMap Flow Neuroglancer data server that serves up real time predictions from a trained model.
"""

class App(FastAPI):

    def __init__(self, model_path, ds_container,ds_dataset) -> None:
        super().__init__(
            title="Cellmap Flow backend server API",
            description=description,
            version="0.0.1",
            terms_of_service="http://example.com/terms/",
            license_info={
                "name": "Apache 2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
            },
            docs_url="/",
        )
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # self.model = model
        # self.ds = ds

        # global DS, CONFIG_STORE, WEIGHTS_STORE, MODEL
        # CONFIG_STORE = create_config_store()
        # WEIGHTS_STORE = create_weights_store()
        DS = open_ds(ds_container, ds_dataset)
        VOL_SHAPE = np.array([20_000 * 8, 20_000 * 8, 20_000 * 8, NUM_CHANNELS])

        INPUT_VOXEL_SIZE = [16, 16, 16]
        OUTPUT_VOXEL_SIZE = [8, 8, 8]
        task_config = DistanceTaskConfig(
            name="cosem_distance_task_mito_8nm_v3",
            channels=["mito"],
            clip_distance=10 * OUTPUT_VOXEL_SIZE,
            tol_distance=10 * OUTPUT_VOXEL_SIZE,
            scale_factor=20 * OUTPUT_VOXEL_SIZE,
            mask_distances=False,
        )

        architecture_config = CNNectomeUNetConfig(
            name="attention-upsample-unet",
            input_shape=Coordinate(216, 216, 216),
            eval_shape_increase=Coordinate(72, 72, 72),
            fmaps_in=1,
            num_fmaps=12,
            fmaps_out=72,
            fmap_inc_factor=6,
            downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
            constant_upsample=True,
            upsample_factors=[(2, 2, 2)],
            use_attention=True,
        )
        task = task_config.task_type(task_config)
        architecture = architecture_config.architecture_type(architecture_config)
        MODEL = task.create_model(architecture)
        # weights = WEIGHTS_STORE.retrieve_weights(
        #     "finetuned_3d_lsdaffs_weight_ratio_0.5_jrc_22ak351-leaf-3m_plasmodesmata_all_training_points_unet_default_trainer_lr_0.00005_bs_2__0",
        #     265000,
        # )
        path_to_weights = model_path
        # weights = torch.load(path_to_weights, map_location="cuda")
        weights = torch.load(path_to_weights,torch.device('cpu'))
        MODEL.load_state_dict(weights.model)
        # MODEL.to("cuda")
        MODEL.eval()

        @self.get("/attributes.json")
        async def top_level_attributes():
            scales = [[2**s, 2**s, 2**s, 1] for s in range(MAX_SCALE + 1)]
            attr = {
                "pixelResolution": {
                    "dimensions": [*OUTPUT_VOXEL_SIZE, 1.0],
                    "unit": "nm",
                },
                "ordering": "C",
                "scales": scales,
                "axes": ["x", "y", "z", "c"],
                "units": ["nm", "nm", "nm", ""],
                "translate": [0, 0, 0, 0],
            }
            return jsonable_encoder(attr)


        @self.get("/s<int:scale>/attributes.json")
        def attributes(scale):
            attr = {
                "transform": {
                    "ordering": "C",
                    "axes": ["x", "y", "z", "c"],
                    "scale": [
                        *OUTPUT_VOXEL_SIZE,
                        1,
                    ],
                    "units": ["nm", "nm", "nm"],
                    "translate": [0.0, 0.0, 0.0],
                },
                "compression": {"type": "gzip", "useZlib": False, "level": -1},
                "blockSize": BLOCK_SHAPE.tolist(),
                "dataType": "float32",
                "dimensions": (VOL_SHAPE[:3] // 2**scale).tolist()
                + [int(VOL_SHAPE[3])],
            }
            return jsonable_encoder(attr)

        @self.get(
            "/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>"
        )
        def chunk(scale, chunk_x, chunk_y, chunk_z, chunk_c):
            """
            Serve up a single chunk at the requested scale and location.

            This 'virtual N5' will just display a color gradient,
            fading from black at (0,0,0) to white at (max,max,max).
            """
            assert (
                chunk_c == 0
            ), "neuroglancer requires that all blocks include all channels"
            print(chunk_x, chunk_y, chunk_z, chunk_c)
            corner = BLOCK_SHAPE[:3] * np.array([chunk_x, chunk_y, chunk_z])
            box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE
            block_vol = inference_for_chunk(scale, box)
            return Response(
                CHUNK_ENCODER.encode(block_vol), media_type="binary/octet-stream"
            )
            # return (
            #     # Encode to N5 chunk format (header + compressed data)

            #     HTTPStatus.OK,
            #     {"Content-Type": "application/octet-stream"},
            # )


def start(app, port=5007):

    return uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        ssl_keyfile=KEY_SSL,
        ssl_certfile=CERT_SSL,
    )

def gradient_data_for_chunk(scale, box):
    box = box.copy()
    box[1] = np.minimum(box[1], VOL_SHAPE[:3] // 2**scale)
    rel_box = box - box[0]
    box_s0 = (2**scale) * box
    shape_czyx = BLOCK_SHAPE[::-1]
    block_vol_czyx = np.zeros(shape_czyx, np.float32)
    block_vol = block_vol_czyx.T
    for c in [0, 1, 2]:
        v0, v1 = np.interp(box_s0[:, c], [0, VOL_SHAPE[c]], [0, 1.0])
        i0, i1 = rel_box[:, c]
        view = np.moveaxis(block_vol[..., c], c, -1)
        view[..., i0:i1] = np.linspace(v0, v1, i1 - i0, False)
    return block_vol_czyx

def inference_for_chunk(scale, box):
    box = box.copy()
    print(f"{box=}")
    grow_by = 91 * INPUT_VOXEL_SIZE[0]
    roi = Roi(box[0][::-1], box[1]).grow(grow_by, grow_by)
    print(f"{roi=} after grow")
    data = DS.to_ndarray(roi) / 255.0
    data = data[np.newaxis, np.newaxis, ...].astype(np.float32)
    data = torch.from_numpy(data)
    with torch.no_grad():
        block_vol_czyx = MODEL(data)
        block_vol_czyx = block_vol_czyx.cpu().numpy()
        block_vol_czyx = block_vol_czyx[0, :NUM_CHANNELS, ...]
        print(block_vol_czyx.shape)
    del data
    gc.collect()
    return block_vol_czyx
