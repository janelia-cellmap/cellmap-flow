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
"""

# %%
# NOTE: To generate host key and host cert do the following: https://serverfault.com/questions/224122/what-is-crt-and-key-files-and-how-to-generate-them
# openssl genrsa 2048 > host.key
# chmod 400 host.key
# openssl req -new -x509 -nodes -sha256 -days 365 -key host.key -out host.cert
# Then can run like this:
# gunicorn --certfile=host.cert --keyfile=host.key --bind 0.0.0.0:8000 --workers 1 --threads 1 example_virtual_n5:app
# NOTE: You will probably have to access the host:8000 separately and say it is safe to go there

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

app = Flask(__name__)
CORS(app)

from cellmap_flow.inferencer import Inferencer

import socket

# Get the hostname
hostname = socket.gethostname()

# Get the local IP address

print(f"Host name: {hostname}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--port", default=8000)
    args = parser.parse_args()
    app.run(
        host="0.0.0.0",
        # port=args.port,
        debug=args.debug,
        threaded=not args.debug,
        use_reloader=args.debug,
    )


# %%
script_path = "/groups/cellmap/cellmap/zouinkhim/cellmap-flow/example/model_spec.py"
config = load_safe_config(script_path)
SCALE_LEVEL = None
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


# determined-chimpmunk is edges
# kind-seashell is mito
# happy-elephant is cells
@app.route("/<path:dataset>/attributes.json")
def top_level_attributes(dataset):
    if "__" not in dataset:
        return jsonify({"n5": "2.1.0"}), HTTPStatus.OK

    if not (dataset.startswith("gs://") or dataset.startswith("s3://")):
        dataset = "/" + dataset

    dataset_name, s, BMZ_MODEL_ID = dataset.split("__")

    global OUTPUT_VOXEL_SIZE, BLOCK_SHAPE, VOL_SHAPE, CHUNK_ENCODER, IDI_RAW, INFERENCER
    print(dataset_name, s, BMZ_MODEL_ID)
    
    
    

    # self.read_shape = config.read_shape
    # self.write_shape = config.write_shape
    
    # self.context = (self.read_shape - self.write_shape) / 2
    SCALE_LEVEL = int(s[1:])
    
    
    # MODEL = Inferencer(BMZ_MODEL_ID)

    IDI_RAW = ImageDataInterface(f"{dataset_name}/s{SCALE_LEVEL}")
    OUTPUT_VOXEL_SIZE = config.output_voxel_size

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
    return jsonify(attr), HTTPStatus.OK


@app.route("/<path:dataset>/s<int:scale>/attributes.json")
def attributes(dataset, scale):
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
    return jsonify(attr), HTTPStatus.OK


@app.route(
    "/<path:dataset>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>/"
)
def chunk(dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c):
    """
    Serve up a single chunk at the requested scale and location.

    This 'virtual N5' will just display a color gradient,
    fading from black at (0,0,0) to white at (max,max,max).
    """
    try:
        # assert chunk_c == 0, "neuroglancer requires that all blocks include all channels"
        corner = BLOCK_SHAPE[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE
        roi = Roi(box[0], box[1])
        print("about_to_process_chunk")
        chunk = INFERENCER.process_chunk_basic(IDI_RAW, roi)
        # logger.error(f"chunk {chunk}")
        print(chunk)
        return (
            # Encode to N5 chunk format (header + compressed data)
            CHUNK_ENCODER.encode(chunk),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
    except Exception as e:
        return jsonify(error=str(e)), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == "__main__":
    main()
