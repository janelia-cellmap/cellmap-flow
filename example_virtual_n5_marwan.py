"""
# Example Virtual N5

Example service showing how to host a virtual N5,
suitable for browsing in neuroglancer.

Neuroglancer is capable of browsing N5 files, as long as you store them on
disk and then host those files over http (with a CORS-friendly http server).
But what if your data doesn't exist on disk yet?

This server hosts a "virtual" N5.  Nothing is stored on disk,
but neuroglancer doesn't need to know that.  This server provides the
necessary attributes.json files and chunk files on-demand, in the
"locations" (url patterns) that neuroglancer expects.

For simplicity, this file uses Flask. In a production system,
you'd probably want to use something snazzier, like FastAPI.

To run the example, install a few dependencies:

    conda create -n example-virtual-n5 -c conda-forge zarr flask flask-cors
    conda activate example-virtual-n5

Then just execute the file:

    python example_virtual_n5.py

Or, for better performance, use a proper http server:

    conda install -c conda-forge gunicorn
    gunicorn --bind 0.0.0.0:8000 --workers 8 --threads 1 example_virtual_n5:app

You can browse the data in neuroglancer after configuring the viewer with the appropriate layer [settings][1].

[1]: http://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%7D%2C%22position%22:%5B5000.5%2C7500.5%2C10000.5%5D%2C%22crossSectionScale%22:25%2C%22projectionScale%22:32767.999999999996%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22n5://http://127.0.0.1:8000%22%2C%22transform%22:%7B%22outputDimensions%22:%7B%22x%22:%5B1e-9%2C%22m%22%5D%2C%22y%22:%5B1e-9%2C%22m%22%5D%2C%22z%22:%5B1e-9%2C%22m%22%5D%2C%22c%5E%22:%5B1%2C%22%22%5D%7D%7D%7D%2C%22tab%22:%22rendering%22%2C%22opacity%22:0.42%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitRGB%28%5Cn%20%20%20%20vec3%28%5Cn%20%20%20%20%20%20getDataValue%280%29%2C%5Cn%20%20%20%20%20%20getDataValue%281%29%2C%5Cn%20%20%20%20%20%20getDataValue%282%29%5Cn%20%20%20%20%29%5Cn%20%20%29%3B%5Cn%7D%5Cn%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22colorful-data%22%7D%5D%2C%22layout%22:%224panel%22%7D
"""

# NOTE: To generate host key and host cert do the following: https://serverfault.com/questions/224122/what-is-crt-and-key-files-and-how-to-generate-them
# openssl genrsa 2048 > host.key
# chmod 400 host.key
# openssl req -new -x509 -nodes -sha256 -days 365 -key host.key -out host.cert
# Then can run like this:
# gunicorn --certfile=host.cert --keyfile=host.key --bind 0.0.0.0:8000 --workers 2 --threads 1 example_virtual_n5:app
# NOTE: You will probably have to access the host:8000 separately and say it is safe to go there

import argparse
from http import HTTPStatus
from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np
import numcodecs
from zarr.n5 import N5ChunkWrapper
import torch
from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from dacapo.store.create_store import create_config_store, create_weights_store

# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
from funlib.geometry import Coordinate
from dacapo.experiments.tasks import AffinitiesTaskConfig, AffinitiesTask, DistanceTask
from dacapo.experiments.architectures import CNNectomeUNetConfig, CNNectomeUNet

from dacapo.experiments.tasks import DistanceTaskConfig
import gc

app = Flask(__name__)
CORS(app)


# This demo produces an RGB volume for aesthetic purposes.
# Note that this is 3 (virtual) teravoxels per channel.
NUM_CHANNELS = 1
BLOCK_SHAPE = np.array([68, 68, 68, NUM_CHANNELS])
MAX_SCALE = 0

CHUNK_ENCODER = N5ChunkWrapper(np.float32, BLOCK_SHAPE, compressor=numcodecs.GZip())

MODEL = None
DS = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--port", default=8000)
    args = parser.parse_args()
    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=args.debug,
        threaded=not args.debug,
        use_reloader=args.debug,
    )


# global DS, CONFIG_STORE, WEIGHTS_STORE, MODEL
# CONFIG_STORE = create_config_store()
# WEIGHTS_STORE = create_weights_store()
DS = open_ds(
    "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr",
    "recon-1/em/fibsem-uint8/s1",
)
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
path_to_weights = "/nrs/cellmap/zouinkhim/crop_num_experiment_v2/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1/checkpoints/iterations/345000"
weights = torch.load(path_to_weights, map_location="cuda")
MODEL.load_state_dict(weights.model)
MODEL.to("cuda")
MODEL.eval()


@app.route("/attributes.json")
def top_level_attributes():
    scales = [[2**s, 2**s, 2**s, 1] for s in range(MAX_SCALE + 1)]
    attr = {
        "pixelResolution": {"dimensions": [*OUTPUT_VOXEL_SIZE, 1.0], "unit": "nm"},
        "ordering": "C",
        "scales": scales,
        "axes": ["x", "y", "z", "c"],
        "units": ["nm", "nm", "nm", ""],
        "translate": [0, 0, 0, 0],
    }
    return jsonify(attr), HTTPStatus.OK


@app.route("/s<int:scale>/attributes.json")
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
        "dimensions": (VOL_SHAPE[:3] // 2**scale).tolist() + [int(VOL_SHAPE[3])],
    }
    return jsonify(attr), HTTPStatus.OK


@app.route("/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>")
def chunk(scale, chunk_x, chunk_y, chunk_z, chunk_c):
    """
    Serve up a single chunk at the requested scale and location.

    This 'virtual N5' will just display a color gradient,
    fading from black at (0,0,0) to white at (max,max,max).
    """
    assert chunk_c == 0, "neuroglancer requires that all blocks include all channels"
    print(chunk_x, chunk_y, chunk_z, chunk_c)
    corner = BLOCK_SHAPE[:3] * np.array([chunk_x, chunk_y, chunk_z])
    box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE
    block_vol = inference_for_chunk(scale, box)

    return (
        # Encode to N5 chunk format (header + compressed data)
        CHUNK_ENCODER.encode(block_vol),
        HTTPStatus.OK,
        {"Content-Type": "application/octet-stream"},
    )


def gradient_data_for_chunk(scale, box):
    """
    Return the demo gradient data for a single chunk.

    Args:
        scale:
            Which downscale level is being requested
        box:
            The bounding box of the requested chunk,
            specified in units of the chunk's own scale.
    """
    # Compute the portion of the box that is actually populated.
    # It will differ from [(0,0,0), BLOCK_SHAPE] at higher scales,
    # where the chunk may extend beyond the bounding box of the entire volume.
    box = box.copy()
    box[1] = np.minimum(box[1], VOL_SHAPE[:3] // 2**scale)

    # Same as box, but in chunk-relative coordinates.
    rel_box = box - box[0]

    # Same as box, but in scale-0 coordinates.
    box_s0 = (2**scale) * box

    # Allocate the chunk.
    shape_czyx = BLOCK_SHAPE[::-1]
    block_vol_czyx = np.zeros(shape_czyx, np.float32)

    # For convenience below, we want to address the
    # chunk via [X,Y,Z,C] indexing (F-order).
    block_vol = block_vol_czyx.T

    # Interpolate along each axis and write the results
    # into separate channels (X=red, Y=green, Z=blue).
    for c in [0, 1, 2]:
        # This is the min/max color value in the chunk for this channel/axis.
        v0, v1 = np.interp(box_s0[:, c], [0, VOL_SHAPE[c]], [0, 1.0])

        # Write the gradient for this channel.
        i0, i1 = rel_box[:, c]
        view = np.moveaxis(block_vol[..., c], c, -1)
        view[..., i0:i1] = np.linspace(v0, v1, i1 - i0, False)

    # Return the C-order view
    return block_vol_czyx


def inference_for_chunk(scale, box):
    # Compute the portion of the box that is actually populated.
    # It will differ from [(0,0,0), BLOCK_SHAPE] at higher scales,
    # where the chunk may extend beyond the bounding box of the entire volume.
    box = box.copy()
    # box[1] = np.minimum(box[0] + box[1], VOL_SHAPE[:3] // 2**scale)
    print(f"{box=}")
    grow_by = 91 * INPUT_VOXEL_SIZE[0]
    roi = Roi(box[0][::-1], box[1]).grow(grow_by, grow_by)
    print(f"{roi=} after grow")
    data = DS.to_ndarray(roi) / 255.0
    # prepend batch and channel dimensions
    data = data[np.newaxis, np.newaxis, ...].astype(np.float32)
    # move to cuda
    data = torch.from_numpy(data).to("cuda")
    with torch.no_grad():
        block_vol_czyx = MODEL(data)
        block_vol_czyx = block_vol_czyx.cpu().numpy()
        block_vol_czyx = block_vol_czyx[0, :NUM_CHANNELS, ...]
        print(block_vol_czyx.shape)
    # block_vol_czyx = np.swapaxes(block_vol_czyx, 1, 3).copy()
    del data
    torch.cuda.empty_cache()
    gc.collect()
    return block_vol_czyx


if __name__ == "__main__":
    main()
