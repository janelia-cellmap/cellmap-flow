import argparse
import logging
import socket
from http import HTTPStatus

import numpy as np
import numcodecs
from flask import Flask, jsonify
from flask_cors import CORS
from zarr.n5 import N5ChunkWrapper
from funlib.geometry import Roi

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from funlib.geometry.coordinate import Coordinate
from cellmap_flow.utils.data import (
    ModelConfig,
    BioModelConfig,
    DaCapoModelConfig,
    ScriptModelConfig,
    IP_PATTERN,
)
from cellmap_flow.utils.web_utils import get_free_port, get_public_ip
import click

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Example usage:
#     conda install -c conda-forge gunicorn
#     gunicorn --bind 0.0.0.0:8000 --workers 8 --threads 1 example_virtual_n5:app
#
# Or just run:
#     python example_virtual_n5.py
# ------------------------------------------------------------------------------


class CellMapFlowServer:
    """
    Flask application hosting a "virtual N5" for neuroglancer.

    Attributes:
        script_path (str): Path to a Python script containing the model specification
        block_shape (tuple): The block shape for chunking
        app (Flask): The Flask application instance
        inferencer (Inferencer): Your CellMapFlow inferencer object
    """

    def __init__(self, dataset_name: str, model_config: ModelConfig):
        """
        Initialize the server.

        Args:
            script_path (str): Path to the Python script containing model specification
            block_shape (tuple): Shape of the blocks used for chunking
        """
        self.block_shape = [(int(x)) for x in model_config.config.block_shape]
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels

        self.inferencer = Inferencer(model_config)

        self.idi_raw = ImageDataInterface(dataset_name)
        if ".zarr" in dataset_name:
            self.vol_shape = np.array(
                [*np.array(self.idi_raw.shape)[::-1], self.output_channels]
            )  # converting from z,y,x order to x,y,z order zarr to n5
            self.axis = ["x", "y", "z", "c^"]
        else:
            self.vol_shape = np.array(
                [*np.array(self.idi_raw.shape), self.output_channels]
            )
            self.axis = ["z", "y", "x", "c^"]
        self.chunk_encoder = N5ChunkWrapper(
            np.uint8, self.block_shape, compressor=numcodecs.GZip()
        )

        # Create and configure Flask
        self.app = Flask(__name__)
        CORS(self.app)

        # To help debug which machine we're on
        hostname = socket.gethostname()
        print(f"Host name: {hostname}", flush=True)

        # Register Flask routes
        self._register_routes()

    def _register_routes(self):
        """
        Register all routes for the Flask application.
        """

        # Top-level attributes (and dataset-level attributes)
        self.app.add_url_rule(
            "/<path:dataset>/attributes.json",
            view_func=self.top_level_attributes,
            methods=["GET"],
        )

        # Scale-level attributes
        self.app.add_url_rule(
            "/<path:dataset>/s<int:scale>/attributes.json",
            view_func=self.attributes,
            methods=["GET"],
        )

        # Chunk data route
        chunk_route = "/<path:dataset>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>/"
        self.app.add_url_rule(chunk_route, view_func=self.chunk, methods=["GET"])

    def top_level_attributes(self, dataset):
        """
        Return top-level N5 attributes, or dataset-level attributes.

        The Neuroglancer N5 data source expects '/attributes.json' at the dataset root.
        """
        # For simplicity, let's say we only allow a single scale (s0), or up to some MAX_SCALE
        max_scale = 0
        # We define the chunk encoder
        # Prepare scales array
        scales = [[2**s, 2**s, 2**s, 1] for s in range(max_scale + 1)]

        # Construct top-level attributes
        attr = {
            "pixelResolution": {
                "dimensions": [*self.output_voxel_size, 1],
                "unit": "nm",
            },
            "ordering": "C",
            "scales": scales,
            "axes": self.axis,
            "units": ["nm", "nm", "nm", ""],
            "translate": [0, 0, 0, 0],
        }

        return jsonify(attr), HTTPStatus.OK

    def attributes(self, dataset, scale):
        """
        Return the attributes of a specific scale (like /s0/attributes.json).
        """
        attr = {
            "transform": {
                "ordering": "C",
                "axes": self.axis,
                "scale": [*self.output_voxel_size, 1],
                "units": ["nm", "nm", "nm", ""],
                "translate": [0.0, 0.0, 0.0, 0.0],
            },
            "compression": {"type": "gzip", "useZlib": False, "level": -1},
            "blockSize": list(self.block_shape),
            "dataType": "uint8",
            "dimensions": self.vol_shape.tolist(),
        }
        print(f"Attributes: {attr}", flush=True)
        return jsonify(attr), HTTPStatus.OK

    def chunk(self, dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c):
        """
        Serve up a single chunk at the requested scale and location.
        This 'virtual N5' will just run an inference function and return the data.
        """
        # try:
        # assert chunk_c == 0, "neuroglancer requires that all blocks include all channels"
        corner = self.block_shape[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, self.block_shape[:3]]) * self.output_voxel_size
        roi = Roi(box[0], box[1])

        chunk = self.inferencer.process_chunk_basic(self.idi_raw, roi)
        return (
            # Encode to N5 chunk format (header + compressed data)
            self.chunk_encoder.encode(chunk),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
        # except Exception as e:
        #     return jsonify(error=str(e)), HTTPStatus.INTERNAL_SERVER_ERROR

    def run(self, debug=False, port=8000, certfile=None, keyfile=None):
        """
        Run the Flask development server with optional SSL cert/key.
        """
        ssl_context = None
        if certfile and keyfile:
            # (certfile, keyfile) tuple enables HTTPS in the built-in dev server
            ssl_context = (certfile, keyfile)

        self.app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            use_reloader=debug,
            ssl_context=ssl_context,  # <-- pass SSL context to Flask dev server
        )
        address = f"{'https' if ssl_context else 'http'}://{get_public_ip()}:{port}"
        logger.error(IP_PATTERN.format(ip_address=address))
        print(IP_PATTERN.format(ip_address=address), flush=True)


# Create a global instance (so that gunicorn can point to `app`


@click.command()
@click.option(
    "-d", "--dataset_name", type=str, required=True, help="Name of the dataset."
)
@click.option(
    "-c", "--config", type=str, required=True, help="Path to the model config file."
)
@click.option("--debug", is_flag=True, help="Run in debug mode.")
@click.option("-p", "--port", default=0, type=int, help="Port to listen on.")
@click.option("--certfile", default=None, help="Path to SSL certificate file.")
@click.option("--keyfile", default=None, help="Path to SSL private key file.")
def main(dataset_name, config, debug, port, certfile, keyfile):
    dataset = dataset_name
    model_config = ScriptModelConfig(script_path=config)
    server = CellMapFlowServer(dataset, model_config)
    if port == 0:
        port = get_free_port()

    server.run(
        debug=debug,
        port=port,
        certfile=certfile,
        keyfile=keyfile,
    )


if __name__ == "__main__":
    main()
