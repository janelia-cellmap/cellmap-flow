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

import argparse
from http import HTTPStatus
from flask import Flask, jsonify, render_template
from flask_cors import CORS

import numpy as np
import numcodecs
from zarr.n5 import N5ChunkWrapper

app = Flask(__name__)
CORS(app)

# This demo produces an RGB volume for aesthetic purposes.
# Note that this is 3 (virtual) teravoxels per channel.
VOL_SHAPE = np.array([10_000, 15_000, 20_000, 3])
BLOCK_SHAPE = np.array([128, 96, 64, 3])
MAX_SCALE = 9

CHUNK_ENCODER = N5ChunkWrapper(np.float32, BLOCK_SHAPE, compressor=numcodecs.GZip())

import os

environ = os.environ


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


@app.route("/tester")
def tester():
    return render_template("iframe.html", url="https://wikipedia.org")


@app.route("/home")
def home():
    return render_template("iframe.html")


@app.route("/glb_loader")
def glb_loader():
    return render_template("glbs.html")


@app.route("/attributes.json")
def top_level_attributes():
    # if environ["REQUEST_METHOD"] == "OPTIONS":
    #     # Respuesta para las solicitudes preflight (OPTIONS)
    #     response_headers = [
    #         ("Access-Control-Allow-Origin", "*"),
    #         ("Access-Control-Allow-Methods", "POST, OPTIONS"),
    #         ("Access-Control-Allow-Headers", "Content-Type"),
    #     ]
    #     start_response("200 OK", response_headers)
    #     print("a responder")
    #     return []
    scales = [[2**s, 2**s, 2**s, 1] for s in range(MAX_SCALE + 1)]
    attr = {
        "pixelResolution": {"dimensions": [1.0, 1.0, 1.0, 1.0], "unit": "nm"},
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
            "scale": [2**scale, 2**scale, 2**scale, 1],
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

    corner = BLOCK_SHAPE[:3] * np.array([chunk_x, chunk_y, chunk_z])
    box = np.array([corner, corner + BLOCK_SHAPE[:3]])
    block_vol = gradient_data_for_chunk(scale, box)

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


if __name__ == "__main__":
    main()
