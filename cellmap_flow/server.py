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
from cellmap_flow.models.models_config import ModelConfig
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


def get_output_shape(idi, model_config, output_channels):
    input_shape = np.array(idi.shape)
    input_axes = idi.axes_names
    input_voxel_size = np.array(model_config.config.input_voxel_size)
    output_voxel_size = np.array(model_config.config.output_voxel_size)
    output_axes = model_config.config.axes_names

    output_shape = input_shape * input_voxel_size / output_voxel_size
    output_shape = np.ceil(output_shape).astype(int)
    # swap axes to match output
    spatial_output_axes = [x for x in output_axes if x != "c^"]
    permitation_axes = [input_axes.index(ax) for ax in spatial_output_axes]
    output_shape = output_shape[permitation_axes]
    index_channel = output_axes.index("c^")
    full_output_shape = list(output_shape)
    full_output_shape.insert(index_channel, output_channels)
    return tuple([int(x) for x in full_output_shape])


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
        self.block_shape = list([int(b) for b in model_config.config.block_shape])

        self.input_voxel_size = Coordinate(model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels

        self.inferencer = Inferencer(model_config)

        # Load or initialize your dataset
        self.idi_raw = ImageDataInterface(
            dataset_name
            # , voxel_size=self.input_voxel_size
        )

        # Refresh rate for custom state updates
        self.refresh_rate_seconds = 5
        self.previous_refresh_time = 0

        self.shape = get_output_shape(
            self.idi_raw,
            model_config,
            self.output_channels,
        )

        self.axes_names = model_config.config.axes_names
        self.channels_index = self.axes_names.index("c^")
        self.units = ["nm" if ax != "c^" else "" for ax in self.axes_names]
        self.voxel_size = list(self.output_voxel_size)
        self.voxel_size.insert(self.channels_index, 1)
        self.voxel_size = tuple(self.voxel_size)

        # spacial_block_shape is self.block_shape without channel dimension
        self.spacial_block_shape = self.block_shape.copy()
        self.spacial_block_shape.pop(self.channels_index)

        # Chunk encoding for N5
        self.chunk_encoder = self._initialize_chunk_encoder()

        # Create and configure Flask
        self.app = Flask(__name__)
        CORS(self.app)
        self._configure_swagger()

        hostname = socket.gethostname()
        print(f"Host name: {hostname}", flush=True)

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
            self.process_dataset_url(dataset)
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
            self.process_dataset_url(dataset)

            return self._attributes_impl(dataset, scale)

        @self.app.route(
            "/<path:dataset>/s<int:scale>/<int:chunk_1>/<int:chunk_2>/<int:chunk_3>/<int:chunk_4>/",
            methods=["GET"],
        )
        @self.app.route(
            "/<path:dataset>/s<int:scale>/<int:chunk_1>/<int:chunk_2>/<int:chunk_3>/<int:chunk_4>",
            methods=["GET"],
        )
        def chunk(dataset, scale, chunk_1, chunk_2, chunk_3, chunk_4):
            return self._chunk_impl(dataset, scale, chunk_1, chunk_2, chunk_3, chunk_4)

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
        scales = [[1, 1, 1, 1]]
        attr = {
            "pixelResolution": {
                "dimensions": self.voxel_size,
                "unit": "nm",
            },
            "ordering": "C",
            "scales": [[1, 1, 1, 1]],
            "axes": self.axes_names,
            "units": self.units,
            "translate": [0, 0, 0, 0],
        }
        return jsonify(attr), HTTPStatus.OK

    def _attributes_impl(self, dataset, scale):
        dtype = g.get_output_dtype().__name__
        attr = {
            "transform": {
                "ordering": "C",
                "axes": self.axes_names,
                "scale": self.voxel_size,
                "units": self.units,
                "translate": [0.0, 0.0, 0.0, 0.0],
            },
            "compression": {"type": "zstd"},
            "blockSize": list(self.block_shape),
            "dataType": dtype,
            "dimensions": self.shape,
        }
        print(f"Attributes (scale={scale}): {attr}", flush=True)
        return jsonify(attr), HTTPStatus.OK

    def process_dataset_url(self, dataset_url):
        g.dashboard_url, g.input_norms, g.postprocess = get_process_dataset(dataset_url)

        for postprocess in g.postprocess:
            if hasattr(postprocess, "num_channels"):
                # Convert to list to allow modification
                shape_list = list(self.shape)
                block_shape_list = list(self.block_shape)
                shape_list[self.channels_index] = postprocess.num_channels
                block_shape_list[self.channels_index] = postprocess.num_channels
                self.shape = tuple(shape_list)
                self.block_shape = tuple(block_shape_list)

        self.chunk_encoder = N5ChunkWrapper(
            g.get_output_dtype(),
            self.block_shape,
            compressor=numcodecs.Zstd(),
        )

    def _chunk_impl(self, dataset, scale, chunk_1, chunk_2, chunk_3, chunk_4):
        chunk_indices = [chunk_1, chunk_2, chunk_3, chunk_4]
        chunk_dict = {}
        for i, axis in enumerate(self.axes_names):
            chunk_dict[axis] = chunk_indices[i]

        chunk_c = chunk_dict["c^"]
        chunk_z = chunk_dict["z"]
        chunk_y = chunk_dict["y"]
        chunk_x = chunk_dict["x"]
        corner = self.spacial_block_shape * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, self.spacial_block_shape]) * self.output_voxel_size
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

        return (
            self.chunk_encoder.encode(chunk_data),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )

    def _initialize_chunk_encoder(self):
        return N5ChunkWrapper(
            g.get_output_dtype(), self.block_shape, compressor=numcodecs.Zstd()
        )

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
