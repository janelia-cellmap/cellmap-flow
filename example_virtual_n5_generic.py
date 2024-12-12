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

# %%
# NOTE: To generate host key and host cert do the following: https://serverfault.com/questions/224122/what-is-crt-and-key-files-and-how-to-generate-them
# openssl genrsa 2048 > host.key
# chmod 400 host.key
# openssl req -new -x509 -nodes -sha256 -days 365 -key host.key -out host.cert
# Then can run like this:
# gunicorn --certfile=host.cert --keyfile=host.key --bind 0.0.0.0:8000 --workers 1 --threads 1 example_virtual_n5:app
# NOTE: You will probably have to access the host:8000 separately and say it is safe to go there

# %% load image zoo
from bioimageio.core import load_description
from bioimageio.core import predict  # , predict_many


import argparse
from http import HTTPStatus
from flask import Flask, jsonify
from flask_cors import CORS

import numpy as np
import numcodecs
from scipy import spatial
from zarr.n5 import N5ChunkWrapper
from funlib.persistence import open_ds
from funlib.geometry import Roi
import numpy as np
from funlib.geometry import Coordinate
from skimage.morphology import erosion
from scipy.ndimage import binary_dilation


# NOTE: Normally we would just load in run but here we have to recreate it to save time since our run has so many points
import neuroglancer
import socket
import skimage
from image_data_interface import ImageDataInterface

app = Flask(__name__)
CORS(app)

from bioimageio.core import Tensor
from bioimagezoo_processor import process_chunk

import socket

# Get the hostname
hostname = socket.gethostname()

# Get the local IP address
print(f"Host name: {hostname}:8000")


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


# OUTPUT_VOXEL_SIZE = [8 * 2**SCALE_LEVEL, 8 * 2**SCALE_LEVEL, 8 * 2**SCALE_LEVEL]


# %%
SCALE_LEVEL = None
IDI_RAW = None
OUTPUT_VOXEL_SIZE = None
VOL_SHAPE_ZYX = None
VOL_SHAPE = None
VOL_SHAPE_ZYX_IN_BLOCKS = None
VOXEL_SIZE = None
BLOCK_SHAPE = None
MAX_SCALE = None
CHUNK_ENCODER = None
EQUIVALENCES = None
DS = None
MODEL = None


# determined-chimpmunk is edges
# kind-seashell is mito
# happy-elephant is cells
@app.route("/test.n5/<string:dataset>/attributes.json")
def top_level_attributes(dataset):
    global OUTPUT_VOXEL_SIZE, BLOCK_SHAPE, VOL_SHAPE, CHUNK_ENCODER, IDI_RAW, MODEL
    # SCALE_LEVEL = 2

    dataset_name, BMZ_MODEL_ID, s = dataset.split("__")
    SCALE_LEVEL = int(s)
    # BMZ_MODEL_ID = "determined-chipmunk"  # "happy-elephant"  # "determined-chipmunk"  # "kind-seashell"  # "affable-shark"
    MODEL = load_description(BMZ_MODEL_ID)

    # global SCALE_LEVEL, IDI_RAW, OUTPUT_VOXEL_SIZE, VOL_SHAPE_ZYX, VOL_SHAPE, VOL_SHAPE_ZYX_IN_BLOCKS, VOXEL_SIZE, BLOCK_SHAPE, MAX_SCALE, CHUNK_ENCODER
    # SCALE_LEVEL = 1
    # IDI_RAW = ImageDataInterface(f"/nrs/cellmap/data/jrc_mus-liver-zon-2/jrc_mus-liver-zon-2.zarr/recon-1/em/fibsem-uint8/s{SCALE_LEVEL}")
    IDI_RAW = ImageDataInterface(
        f"/nrs/cellmap/data/{dataset_name}/{dataset_name}.zarr/recon-1/em/fibsem-uint8/s{SCALE_LEVEL}"
    )
    OUTPUT_VOXEL_SIZE = IDI_RAW.voxel_size

    # %%
    BLOCK_SHAPE = np.array([128, 128, 128, 9])
    MAX_SCALE = 0

    VOL_SHAPE_ZYX = np.array(IDI_RAW.ds.shape)
    print(VOL_SHAPE_ZYX)
    VOL_SHAPE = np.array([*VOL_SHAPE_ZYX[::-1], 9])
    VOL_SHAPE_ZYX_IN_BLOCKS = np.ceil(VOL_SHAPE_ZYX / BLOCK_SHAPE[:3]).astype(int)
    VOXEL_SIZE = IDI_RAW.ds.voxel_size

    CHUNK_ENCODER = N5ChunkWrapper(np.uint8, BLOCK_SHAPE, compressor=numcodecs.GZip())

    scales = [[2**s, 2**s, 2**s, 1] for s in range(MAX_SCALE + 1)]
    attr = {
        "pixelResolution": {
            "dimensions": [*OUTPUT_VOXEL_SIZE, 1],
            "unit": "nm",
        },
        "ordering": "C",
        "scales": scales,
        "axes": ["x", "y", "z", "c"],
        "units": ["nm", "nm", "nm", ""],
        "translate": [0, 0, 0, 0],
    }
    return jsonify(attr), HTTPStatus.OK


@app.route("/test.n5/<string:dataset>/s<int:scale>/attributes.json")
def attributes(dataset, scale):
    attr = {
        "transform": {
            "ordering": "C",
            "axes": ["x", "y", "z", "c"],
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
    "/test.n5/<string:dataset>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>/"
)
def chunk(dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c):
    """
    Serve up a single chunk at the requested scale and location.

    This 'virtual N5' will just display a color gradient,
    fading from black at (0,0,0) to white at (max,max,max).
    """
    # assert chunk_c == 0, "neuroglancer requires that all blocks include all channels"
    corner = BLOCK_SHAPE[:3] * np.array([chunk_z, chunk_y, chunk_x])
    box = np.array([corner, BLOCK_SHAPE[:3]]) * OUTPUT_VOXEL_SIZE
    roi = Roi(box[0], box[1])
    chunk = process_chunk(MODEL, IDI_RAW, roi)
    print(chunk.shape)
    return (
        # Encode to N5 chunk format (header + compressed data)
        CHUNK_ENCODER.encode(chunk),
        HTTPStatus.OK,
        {"Content-Type": "application/octet-stream"},
    )


# %%
if __name__ == "__main__":
    main()
