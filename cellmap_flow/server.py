import logging
import socket
from http import HTTPStatus
from flask import request
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
from cellmap_flow.utils.data import ModelConfig, IP_PATTERN
from cellmap_flow.utils.web_utils import get_public_ip
from cellmap_flow.norm.input_normalize import MinMaxNormalizer

logger = logging.getLogger(__name__)


class CellMapFlowServer:
    """
    Flask application hosting a "virtual N5" for Neuroglancer.
    All routes are defined via Flask decorators for convenience.
    """

    def __init__(self, dataset_name: str, model_config: ModelConfig):
        """
        Initialize the server and set up routes via decorators.
        """
        self.block_shape = [int(x) for x in model_config.config.block_shape]
        self.input_voxel_size = Coordinate(model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels

        self.inferencer = Inferencer(model_config)

        # Load or initialize your dataset
        self.idi_raw = ImageDataInterface(
            dataset_name, target_resolution=self.input_voxel_size
        )
        if ".zarr" in dataset_name:
            # Convert from (z, y, x) -> (x, y, z) plus channels
            self.vol_shape = np.array(
                [*np.array(self.idi_raw.shape)[::-1], self.output_channels]
            )
            self.axis = ["x", "y", "z", "c^"]
        else:
            # For non-Zarr data
            self.vol_shape = np.array(
                [*np.array(self.idi_raw.shape), self.output_channels]
            )
            self.axis = ["z", "y", "x", "c^"]

        # Chunk encoding for N5
        self.chunk_encoder = N5ChunkWrapper(
            np.uint8, self.block_shape, compressor=numcodecs.GZip()
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
            return self._attributes_impl(dataset, scale)

        @self.app.route("/input_normalize", methods=["POST"])
        def input_normalize():
            """
            Update input normalization parameters for inference.
            ---
            tags:
              - Normalization
            requestBody:
              required: true
              content:
                application/json:
                  schema:
                    type: object
                    properties:
                      norm_type:
                        type: string
                        description: Normalization type
                        example: "scale"
                      min_value:
                        type: number
                        description: Minimum value
                        example: 0.0
                      max_value:
                        type: number
                        description: Maximum value
                        example: 1.0
            responses:
              200:
                description: Success
              400:
                description: Invalid request or missing parameters
            """
            data = request.get_json()

            # Extract parameters from JSON body
            norm_type = data.get("norm_type")
            min_value = data.get("min_value")
            max_value = data.get("max_value")

            if not all([norm_type, min_value is not None, max_value is not None]):
                return {"error": "Missing one or more required parameters"}, 400

            return self._input_normalize_impl(
                norm_type, float(min_value), float(max_value)
            )

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
        print(f"Attributes (scale={scale}): {attr}", flush=True)
        return jsonify(attr), HTTPStatus.OK

    def _input_normalize_impl(self, norm_type, min_value, max_value):
        print(f"Input normalization: {norm_type}, {min_value}, {max_value}", flush=True)
        if norm_type == MinMaxNormalizer.name():
            self.inferencer.model_config.input_normalizer = MinMaxNormalizer(
                min_value, max_value
            )
            return jsonify(success=True), HTTPStatus.OK
        else:
            return (
                jsonify(
                    error=f"Invalid normalization type, only supports {MinMaxNormalizer.name()}"
                ),
                HTTPStatus.BAD_REQUEST,
            )

    def _chunk_impl(self, dataset, scale, chunk_x, chunk_y, chunk_z, chunk_c):
        corner = self.block_shape[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, self.block_shape[:3]]) * self.output_voxel_size
        roi = Roi(box[0], box[1])
        chunk_data = self.inferencer.process_chunk(self.idi_raw, roi)
        return (
            self.chunk_encoder.encode(chunk_data),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )

    #
    # --- Server Runner ---
    #
    def run(self, debug=False, port=8000, certfile=None, keyfile=None):
        """
        Run the Flask dev server with optional SSL certificate.
        """
        ssl_context = None
        if certfile and keyfile:
            ssl_context = (certfile, keyfile)

        address = f"{'https' if ssl_context else 'http'}://{get_public_ip()}:{port}"
        logger.error(IP_PATTERN.format(ip_address=address))
        print(IP_PATTERN.format(ip_address=address), flush=True)

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
