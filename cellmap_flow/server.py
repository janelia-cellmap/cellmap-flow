import json
import logging
import socket
from http import HTTPStatus
from flask import request
import neuroglancer
import numpy as np
import numcodecs
from flask import Flask, jsonify, redirect
from flask_cors import CORS
from flasgger import Swagger
from zarr.n5 import N5ChunkWrapper
from funlib.geometry import Roi
from funlib.geometry.coordinate import Coordinate

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.utils.data import ModelConfig
from cellmap_flow.utils.web_utils import (
    get_public_ip,
    decode_to_json,
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
    IP_PATTERN,
    get_free_port,
)
from cellmap_flow.norm.input_normalize import get_normalizations
from cellmap_flow.post.postprocessors import get_postprocessors

from cellmap_flow.globals import g
import requests
import time

logger = logging.getLogger(__name__)


def get_process_dataset(dataset: str):
    if ARGS_KEY not in dataset:
        return None, [], []  # No normalization or postprocessing
    norm_data = dataset.split(ARGS_KEY)
    if len(norm_data) != 3:
        raise ValueError(
            f"Invalid dataset format. Expected two occurrences of {ARGS_KEY}. found {len(norm_data)} {dataset}"
        )
    encoded_data = norm_data[1]
    result = decode_to_json(encoded_data)
    logger.error(f"Decoded data: {result}")
    dashboard_url = result.get("dashboard_url", None)
    input_norm_fns = get_normalizations(result[INPUT_NORM_DICT_KEY])
    postprocess_fns = get_postprocessors(result[POSTPROCESS_DICT_KEY])
    logger.error(f"Normalized data: {result}")
    return dashboard_url, input_norm_fns, postprocess_fns


class CellMapFlowServer:
    """
    Flask application hosting a "virtual N5" for Neuroglancer.
    All routes are defined via Flask decorators for convenience.
    """

    def __init__(self, dataset_name: str, model_config: ModelConfig):
        """
        Initialize the server and set up routes via decorators.
        """

        # this is zyx
        self.read_block_shape = [int(x) for x in model_config.config.block_shape]
        # this needs to have z and x swapped
        self.n5_block_shape = self.read_block_shape.copy()
        self.n5_block_shape[0], self.n5_block_shape[2] = (
            self.n5_block_shape[2],
            self.n5_block_shape[0],
        )

        self.input_voxel_size = Coordinate(model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels

        self.inferencer = Inferencer(model_config)

        # Load or initialize your dataset
        self.idi_raw = ImageDataInterface(
            dataset_name, target_resolution=self.input_voxel_size
        )

        # Refresh rate for custom state updates
        self.refresh_rate_seconds = 5
        self.previous_refresh_time = 0
        output_shape = (
            np.array(self.idi_raw.shape)
            * np.array(self.input_voxel_size)
            / np.array(self.output_voxel_size)
        )

        if ".zarr" in dataset_name:
            # Convert from (z, y, x) -> (x, y, z) plus channels
            self.default_vol_shape = [
                *output_shape[::-1],
                self.output_channels,
            ]
            self.axis = ["x", "y", "z", "c^"]
            self.vol_shape = self.default_vol_shape.copy()
        else:
            # For non-Zarr data
            self.default_vol_shape = [*output_shape, self.output_channels]
            self.axis = ["z", "y", "x", "c^"]
            self.vol_shape = self.default_vol_shape.copy()
        # Chunk encoding for N5
        self.chunk_encoder = self._initialize_chunk_encoder()

    def _initialize_chunk_encoder(self):
        return N5ChunkWrapper(
            g.get_output_dtype(), self.n5_block_shape, compressor=numcodecs.Zstd()
        )
        # Create and configure Flask
        self.app = Flask(__name__)
        CORS(self.app)
        self._configure_swagger()

        hostname = socket.gethostname()
        print(f"Host name: {hostname}", flush=True)

        # ------------------------------------------------------
        # Routes using @self.app.route -- no add_url_rule calls!
        # ------------------------------------------------------

        @self.app.route("/")
        def home():
            """
            Redirects to Swagger UI at /apidocs/ for documentation.
            ---
            tags:
              - Documentation
            responses:
              302:
                description: Redirect to API docs
            """
            return redirect("/apidocs/")

        @self.app.route("/<path:dataset>/attributes.json", methods=["GET"])
        def top_level_attributes(dataset):
            """
            Return top-level or dataset-level N5 attributes.
            ---
            tags:
              - Attributes
            parameters:
              - in: path
                name: dataset
                schema:
                  type: string
                required: true
                description: Dataset name or path
            responses:
              200:
                description: Attributes in JSON
            """
            g.dashboard_url, g.input_norms, g.postprocess = get_process_dataset(dataset)
            self.vol_shape = self.default_vol_shape.copy()
            self.n5_block_shape[-1] = self.default_vol_shape[-1]

            for postprocess in g.postprocess:
                if hasattr(postprocess, "num_channels"):
                    self.vol_shape[-1] = postprocess.num_channels
                    self.n5_block_shape[-1] = postprocess.num_channels

            self.chunk_encoder = N5ChunkWrapper(
                g.get_output_dtype(),
                self.n5_block_shape,
                compressor=numcodecs.Zstd(),
            )

            return self._top_level_attributes_impl(dataset)

        @self.app.route("/<path:dataset>/s<int:scale>/attributes.json", methods=["GET"])
        def attributes(dataset, scale):
            """
            Return attributes of a specific scale (e.g. /s0/attributes.json).
            ---
            tags:
              - Attributes
            parameters:
              - in: path
                name: dataset
                schema:
                  type: string
              - in: path
                name: scale
                schema:
                  type: integer
            responses:
              200:
                description: Scale-level attributes in JSON
            """
            g.dashboard_url, g.input_norms, g.postprocess = get_process_dataset(dataset)
            self.vol_shape = self.default_vol_shape.copy()
            self.n5_block_shape[-1] = self.default_vol_shape[-1]

            for postprocess in g.postprocess:
                if hasattr(postprocess, "num_channels"):
                    self.vol_shape[-1] = postprocess.num_channels
                    self.n5_block_shape[-1] = postprocess.num_channels

            self.chunk_encoder = N5ChunkWrapper(
                g.get_output_dtype(),
                self.n5_block_shape,
                compressor=numcodecs.Zstd(),
            )
            return self._attributes_impl(dataset, scale)

        @self.app.route(
            "/<path:dataset>/s<int:scale>/<int:chunk_x>/<int:chunk_y>/<int:chunk_z>/<int:chunk_c>/",
            methods=["GET"],
        )
        def chunk(dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c):
            """
            Serve a single chunk at the requested scale and location.
            ---
            tags:
              - Chunks
            parameters:
              - in: path
                name: dataset
                schema:
                  type: string
              - in: path
                name: scale
                schema:
                  type: integer
              - in: path
                name: chunk_x
                schema:
                  type: integer
              - in: path
                name: chunk_y
                schema:
                  type: integer
              - in: path
                name: chunk_z
                schema:
                  type: integer
              - in: path
                name: chunk_c
                schema:
                  type: integer
            responses:
              200:
                description: Compressed chunk
              500:
                description: Internal server error
            """
            return self._chunk_impl(dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c)

    def _configure_swagger(self):
        """
        Configure Flasgger/Swagger settings for auto-generated docs.
        """
        self.app.config["SWAGGER"] = {
            "title": "CellMapFlow Virtual N5 API",
            "uiversion": 3,  # Use Swagger UI 3.x
        }
        swagger_config = {
            "headers": [],
            "specs": [
                {
                    "version": "0.0.1",
                    "title": "CellMapFlow Virtual N5 API",
                    "endpoint": "api_spec",
                    "description": "API to serve a virtual N5 interface for Neuroglancer.",
                    "route": "/api_spec.json",
                }
            ],
            "static_url_path": "/flasgger_static",
            "swagger_ui": True,
            "specs_route": "/apidocs/",
        }
        self.swagger = Swagger(self.app, config=swagger_config)

    #
    # --- Implementation (called by the decorated routes) ---
    #
    def _top_level_attributes_impl(self, dataset):
        max_scale = 0
        scales = [[2**s, 2**s, 2**s, 1] for s in range(max_scale + 1)]
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

    def _attributes_impl(self, dataset, scale):
        dtype = g.get_output_dtype().__name__
        attr = {
            "transform": {
                "ordering": "C",
                "axes": self.axis,
                "scale": [*self.output_voxel_size, 1],
                "units": ["nm", "nm", "nm", ""],
                "translate": [0.0, 0.0, 0.0, 0.0],
            },
            "compression": {"type": "zstd"},
            "blockSize": list(self.n5_block_shape),
            "dataType": dtype,
            "dimensions": self.vol_shape,
        }
        print(f"Attributes (scale={scale}): {attr}", flush=True)
        return jsonify(attr), HTTPStatus.OK

    def _chunk_impl(
        self, dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c, get_encoded=True
    ):
        corner = self.read_block_shape[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, self.read_block_shape[:3]]) * self.output_voxel_size
        roi = Roi(box[0], box[1])
        chunk_data = self.inferencer.process_chunk(self.idi_raw, roi)

        chunk_data = chunk_data.astype(g.get_output_dtype())

        current_time = time.time()

        # assume only one has equivalences
        for postprocess in g.postprocess:
            if (
                hasattr(postprocess, "equivalences")
                and postprocess.equivalences is not None
                and (current_time - self.previous_refresh_time)
                > self.refresh_rate_seconds
            ):
                equivalences = {
                    "dataset": dataset,
                    "equivalences": [
                        [int(item) for item in sublist]
                        for sublist in postprocess.equivalences.to_json()
                    ],
                }

                response = requests.post(
                    g.dashboard_url + "/update/equivalences",
                    json=equivalences,
                )
                self.previous_refresh_time = current_time
                continue

        if get_encoded:
            return (
                self.chunk_encoder.encode(chunk_data),
                HTTPStatus.OK,
                {"Content-Type": "application/octet-stream"},
            )
        else:
            return chunk_data

    #
    # --- Server Runner ---
    #
    def run(self, debug=False, port=None, certfile=None, keyfile=None):
        """
        Run the Flask dev server with optional SSL certificate.
        """
        ssl_context = None
        if certfile and keyfile:
            ssl_context = (certfile, keyfile)

        if port is None or port == 0:
            port = get_free_port()

        address = f"{'https' if ssl_context else 'http'}://{get_public_ip()}:{port}"
        output = f"{IP_PATTERN[0]}{address}{IP_PATTERN[1]}"
        logger.error(output)
        print(output, flush=True)

        self.app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            use_reloader=debug,
            ssl_context=ssl_context,
        )


# ------------------------------------
# Example usage (if run directly):
#
#   python your_server.py
#
# Then visit:
#   http://localhost:8000/
#   http://localhost:8000/apidocs/
# ------------------------------------
if __name__ == "__main__":
    # Dummy ModelConfig example; replace with real config
    class DummyConfig:
        block_shape = (32, 32, 32)
        output_voxel_size = (4, 4, 4)
        output_channels = 1

    dummy_model_config = ModelConfig(config=DummyConfig())

    server = CellMapFlowServer("example.zarr", dummy_model_config)
    server.run(debug=True, port=8000)

# # %%
# import neuroglancer
# neuroglancer.set_server_bind_address("http://h06u01.int.janelia.org:19821/v/733c608c2ad97d2340bfc83f1f9459d5be4d9d49/")
# with neuroglancer.Viewer().txn() as s:
#     print(s.layers)
# %%
