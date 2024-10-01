# %%
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

from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run

import argparse
from http import HTTPStatus
from flask import Flask, jsonify, render_template
from flask_cors import CORS

import numpy as np
import numcodecs
from zarr.n5 import N5ChunkWrapper
import torch
from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np

from skimage.measure import label
from scipy import spatial
import neuroglancer
import time

# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
import gc
import socket

app = Flask(__name__)
CORS(app)


# This demo produces an RGB volume for aesthetic purposes.
# Note that this is 3 (virtual) teravoxels per channel.
NUM_CHANNELS = 1
BLOCK_SHAPE = np.array([36, 36, 36, NUM_CHANNELS])
MAX_SCALE = 0

CHUNK_ENCODER = N5ChunkWrapper(np.uint64, BLOCK_SHAPE, compressor=numcodecs.GZip())

MODEL = None
DS = None
EDGE_VOXEL_POSITION_TO_VAL_DICT = {}
EQUIVALENCES = neuroglancer.equivalence_map.EquivalenceMap()

# %%
ZARR_PATH = "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1_normalized.zarr"
DATASET = "recon-1/em/fibsem-uint8"
# load raw data
DS = open_ds(
    ZARR_PATH,
    f"{DATASET}/s2",
)
VOL_SHAPE_ZYX = np.array(DS.shape)
VOL_SHAPE = np.array([*VOL_SHAPE_ZYX[::-1], NUM_CHANNELS])
VOL_SHAPE_ZYX_IN_BLOCKS = np.ceil(VOL_SHAPE_ZYX / BLOCK_SHAPE[:3]).astype(int)


neuroglancer.set_server_bind_address("0.0.0.0")
VIEWER = neuroglancer.Viewer()
ip_address = socket.getfqdn()

with VIEWER.txn() as s:
    s.layers["raw"] = neuroglancer.ImageLayer(
        source=f'zarr://http://cellmap-vm1.int.janelia.org/{ZARR_PATH.replace("/nrs/cellmap", "/nrs")}/{DATASET}',
    )
    s.layers[f"inference and postprocessing"] = neuroglancer.SegmentationLayer(
        source=f"n5://http://{ip_address}:8000/test.n5/test",
        equivalences=EQUIVALENCES.to_json(),
    )
    s.cross_section_scale = 1e-9
    s.projection_scale = 500e-9
print(VIEWER)


PREVIOUS_UPDATE_TIME = 0


def update_state():
    global PREVIOUS_UPDATE_TIME
    if time.time() - PREVIOUS_UPDATE_TIME > 2:
        with VIEWER.txn() as s:
            s.layers[f"inference and postprocessing"].equivalences = (
                EQUIVALENCES.to_json()
            )
        PREVIOUS_UPDATE_TIME = time.time()


update_state()
# global DS, CONFIG_STORE, WEIGHTS_STORE, MODEL

INPUT_VOXEL_SIZE = [16, 16, 16]
OUTPUT_VOXEL_SIZE = [16, 16, 16]
CONFIG_STORE = create_config_store()
WEIGHTS_STORE = create_weights_store()
run_name = "20240925_mito_setup04_no_upsample_16_16_0"
run_config = CONFIG_STORE.retrieve_run_config(run_name)

run = Run(run_config)  # , load_starter_model=False)
task = run.task
MODEL = run.model
# print(MODEL.architecture)
# path_to_weights = "/nrs/cellmap/zouinkhim/crop_num_experiment_v2/v21_mito_attention_finetuned_distances_8nm_mito_jrc_mus-livers_mito_8nm_attention-upsample-unet_default_one_label_1/checkpoints/iterations/345000"
# weights = torch.load(path_to_weights, map_location="cuda")
weights = WEIGHTS_STORE.retrieve_weights(
    run_name,
    80000,
)
MODEL.load_state_dict(weights.model)
# MODEL.load_state_dict(weights.model)
MODEL.to("cuda")
MODEL.eval()
# %%


# @app.route("/home")
# def home():

#     print(VIEWER)
#     # print(neuroglancer.to_url(viewer.state))
#     # s.position = VOL_SHAPE_ZYX[::-1] / 2

#     return render_template("iframe.html", url=VIEWER)


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


@app.route("/test.n5/test/attributes.json")
def top_level_attributes():
    scales = [[2**s, 2**s, 2**s, 1] for s in range(MAX_SCALE + 1)]
    attr = {
        "pixelResolution": {"dimensions": [*OUTPUT_VOXEL_SIZE, 1.0], "unit": "nm"},
        "ordering": "C",
        "scales": scales,
        "axes": ["x", "y", "z", "c^"],
        "units": ["nm", "nm", "nm", ""],
        "translate": [0, 0, 0, 0],
    }
    return jsonify(attr), HTTPStatus.OK


@app.route("/test.n5/test/s<int:scale>/attributes.json")
def attributes(scale):
    attr = {
        "transform": {
            "ordering": "C",
            "axes": ["x", "y", "z", "c^"],
            "scale": [
                *OUTPUT_VOXEL_SIZE,
                1,
            ],
            "units": ["nm", "nm", "nm"],
            "translate": [0.0, 0.0, 0.0],
        },
        "compression": {"type": "gzip", "useZlib": False, "level": -1},
        "blockSize": BLOCK_SHAPE.tolist(),
        "dataType": "uint64",
        "dimensions": (VOL_SHAPE[:3] // 2**scale).tolist() + [int(VOL_SHAPE[3])],
    }
    return jsonify(attr), HTTPStatus.OK


@app.route(
    "/test.n5/test/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>"
)
def chunk(scale, chunk_x, chunk_y, chunk_z, chunk_c):
    """
    Serve up a single chunk at the requested scale and location.

    This 'virtual N5' will just display a color gradient,
    fading from black at (0,0,0) to white at (max,max,max).
    """

    assert chunk_c == 0, "neuroglancer requires that all blocks include all channels"
    print(chunk_x, chunk_y, chunk_z, chunk_c)
    corner = BLOCK_SHAPE[:3] * np.array([chunk_z, chunk_y, chunk_x])
    box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE

    global_id_offset = np.prod(BLOCK_SHAPE[:3]) * (
        VOL_SHAPE_ZYX_IN_BLOCKS[0] * VOL_SHAPE_ZYX_IN_BLOCKS[1] * chunk_x
        + VOL_SHAPE_ZYX_IN_BLOCKS[0] * chunk_y
        + chunk_z
    )
    block_vol = postprocess_for_chunk(
        inference_for_chunk(scale, box), global_id_offset, corner
    )

    return (
        # Encode to N5 chunk format (header + compressed data)
        CHUNK_ENCODER.encode(block_vol),
        HTTPStatus.OK,
        {"Content-Type": "application/octet-stream"},
    )


def inference_for_chunk(scale, box):

    # Compute the portion of the box that is actually populated.
    # It will differ from [(0,0,0), BLOCK_SHAPE] at higher scales,
    # where the chunk may extend beyond the bounding box of the entire volume.
    box = box.copy()
    # box[1] = np.minimum(box[0] + box[1], VOL_SHAPE[:3] // 2**scale)
    print(f"{box=}")
    grow_by = 90 * INPUT_VOXEL_SIZE[0]
    roi = Roi(box[0], box[1]).grow(grow_by, grow_by)
    print(f"{roi=} after grow")
    data = DS.to_ndarray(roi) / 255.0
    # create random array with floats between 0 and 1
    # prepend batch and channel dimensions
    data = data[np.newaxis, np.newaxis, ...].astype(np.float32)
    # move to cuda
    data = torch.from_numpy(data).to("cuda")
    with torch.no_grad():
        block_vol_czyx = MODEL(data)
        block_vol_czyx = block_vol_czyx.cpu().numpy()
        block_vol_czyx = block_vol_czyx[0, :NUM_CHANNELS, ...]
    # block_vol_czyx = np.swapaxes(block_vol_czyx, 1, 3).copy()
    del data

    torch.cuda.empty_cache()
    gc.collect()
    return block_vol_czyx


# %%
def postprocess_for_chunk(chunk, id_offset, corner):
    global EDGE_VOXEL_POSITION_TO_VAL_DICT
    # do connected components on thresholded chunk
    thresholded = chunk > 0
    postprocessed, num = label(thresholded, return_num=True)
    postprocessed = postprocessed.astype(np.uint64)
    if num == 0:
        return postprocessed.astype(np.uint64)

    postprocessed[postprocessed > 0] += id_offset

    postprocessed_squeezed = postprocessed[0, ...]
    mask = np.zeros_like(postprocessed_squeezed, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    postprocessed_squeezed_ma = np.ma.masked_array(postprocessed_squeezed, mask)
    z, y, x = np.ma.where(postprocessed_squeezed_ma > 0)
    values = postprocessed_squeezed_ma[z, y, x]
    EDGE_VOXEL_POSITION_TO_VAL_DICT.update(
        dict(
            zip(
                zip(
                    z + corner[0],
                    y + corner[1],
                    x + corner[2],
                ),
                values,
            )
        )
    )
    update_equivalences()
    return postprocessed


# %%
def update_equivalences():
    global EQUIVALENCES

    positions = list(EDGE_VOXEL_POSITION_TO_VAL_DICT.keys())
    ids = list(EDGE_VOXEL_POSITION_TO_VAL_DICT.values())
    tree = spatial.cKDTree(positions)
    neighbors = tree.query_ball_tree(tree, 1)  # distance of 1 voxel
    for i in range(len(neighbors)):
        for j in neighbors[i]:
            EQUIVALENCES.union(ids[i], ids[j])
    update_state()


# %%
if __name__ == "__main__":
    main()
