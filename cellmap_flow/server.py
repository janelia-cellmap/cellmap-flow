import logging
import socket
from http import HTTPStatus
import numpy as np
import numcodecs
from flask import Flask, jsonify, redirect
from flask_cors import CORS
from flasgger import Swagger
from funlib.geometry import Roi
from funlib.geometry.coordinate import Coordinate

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.utils.web_utils import (
    get_public_ip,
    IP_PATTERN,
    get_free_port,
)
from cellmap_flow.utils.serilization_utils import get_process_dataset_url

from cellmap_flow.globals import g

import requests
import time

logger = logging.getLogger(__name__)


class CellMapFlowServer:
    """
    Flask application hosting a "virtual Zarr" for Neuroglancer.
    All routes are defined via Flask decorators for convenience.
    """

    def __init__(self, dataset_name: str, model_config: ModelConfig):
        """
        Initialize the server and set up routes via decorators.
        """

        self.zarr_block_shape = [int(x) for x in model_config.config.block_shape]

        self.input_voxel_size = Coordinate(model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels
        self.output_dtype = model_config.output_dtype
        self.model_output_axes = model_config.chunk_output_axes

        self.inferencer = Inferencer(model_config)

        # Load or initialize your dataset
        self.idi_raw = ImageDataInterface(
            dataset_name, voxel_size=self.input_voxel_size
        )
        self.axes = self.idi_raw.axes_names.copy()
        # remove channel axis if present can be c^, c, or channel
        for axis_name in ["c^", "c", "channel"]:
            if axis_name in self.axes:
                self.axes.remove(axis_name)

        # Determine whether the model output includes a channel axis
        self.has_channel = any(
            ax in model_config.chunk_output_axes for ax in ("c", "c^", "channel")
        )

        if self.has_channel:
            # The model output spatial axes match the input data axes (not the
            # hardcoded default which assumes z,y,x).  Override so that
            # _reorder_to_zarr_axes applies the correct permutation.
            self.model_output_axes = ("c",) + tuple(self.axes)
        else:
            self.model_output_axes = tuple(self.axes)

        # Refresh rate for custom state updates
        self.refresh_rate_seconds = 5
        self.previous_refresh_time = 0
        output_shape = (
            np.array(self.idi_raw.shape)
            * np.array(self.input_voxel_size)
            / np.array(self.output_voxel_size)
        )
        if self.has_channel:
            self.vol_shape = [*output_shape, self.output_channels]
        else:
            self.vol_shape = [int(x) for x in output_shape]

        # Chunk encoding for Zarr
        self.chunk_encoder = self._initialize_chunk_encoder()

        # Create and configure Flask
        self.app = Flask(__name__)
        CORS(self.app)
        self._configure_swagger()

        hostname = socket.gethostname()
        print(f"Host name: {hostname}", flush=True)

        @self.app.route("/")
        def home():
            return redirect("/apidocs/")

        @self.app.route("/<path:dataset>/.zattrs", methods=["GET"])
        def top_level_attributes(dataset):
            self.refresh_dataset(dataset)

            return self._top_level_attributes_impl(dataset)

        @self.app.route("/<path:dataset>/s<int:scale>/.zarray", methods=["GET"])
        def attributes(dataset, scale):
            self.refresh_dataset(dataset)
            return self._attributes_impl(dataset, scale)

        @self.app.route(
            "/<path:dataset>/s<int:scale>/<int:chunk_z>.<int:chunk_y>.<int:chunk_x>.<int:chunk_c>",
            methods=["GET"],
            strict_slashes=False,
        )
        def chunk_4d(dataset, scale, chunk_z, chunk_y, chunk_x, chunk_c):
            return self._chunk_impl(dataset, scale, chunk_z, chunk_y, chunk_x)

        @self.app.route(
            "/<path:dataset>/s<int:scale>/<int:chunk_z>.<int:chunk_y>.<int:chunk_x>",
            methods=["GET"],
            strict_slashes=False,
        )
        def chunk_3d(dataset, scale, chunk_z, chunk_y, chunk_x):
            return self._chunk_impl(dataset, scale, chunk_z, chunk_y, chunk_x)

    def _configure_swagger(self):
        self.app.config["SWAGGER"] = {
            "title": "CellMapFlow Virtual Zarr API",
            "uiversion": 3,  # Use Swagger UI 3.x
        }
        swagger_config = {
            "headers": [],
            "specs": [
                {
                    "version": "0.0.1",
                    "title": "CellMapFlow Virtual Zarr API",
                    "endpoint": "api_spec",
                    "description": "API to serve a virtual Zarr interface for Neuroglancer.",
                    "route": "/api_spec.json",
                }
            ],
            "static_url_path": "/flasgger_static",
            "swagger_ui": True,
            "specs_route": "/apidocs/",
        }
        self.swagger = Swagger(self.app, config=swagger_config)

    def refresh_dataset(self, dataset):
        g.dashboard_url, g.input_norms, g.postprocess = get_process_dataset_url(dataset)

        for postprocess in g.postprocess:
            if hasattr(postprocess, "num_channels") and self.has_channel:
                self.vol_shape[-1] = postprocess.num_channels
                self.zarr_block_shape[-1] = postprocess.num_channels

        # Update chunk encoder for Zarr
        self.chunk_encoder = numcodecs.Blosc(
            cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE
        )

    def _top_level_attributes_impl(self, dataset):
        max_scale = 0
        datasets = []
        for s in range(max_scale + 1):
            scale_factor = 2**s
            scale_values = [
                float(self.output_voxel_size[i] * scale_factor)
                for i in range(len(self.output_voxel_size))
            ]
            translation_values = [0.0] * len(self.output_voxel_size)
            if self.has_channel:
                scale_values.append(1.0)
                translation_values.append(0.0)
            datasets.append(
                {
                    "coordinateTransformations": [
                        {"type": "scale", "scale": scale_values},
                        {"type": "translation", "translation": translation_values},
                    ],
                    "path": f"s{s}",
                }
            )

        axes_list = []
        for axis_name in self.axes:
            axes_list.append({"name": axis_name, "type": "space", "unit": "nanometer"})
        if self.has_channel:
            axes_list.append({"name": "c", "type": "channel"})

        top_scale = [1.0] * len(self.axes)
        if self.has_channel:
            top_scale.append(1.0)

        attr = {
            "multiscales": [
                {
                    "version": "0.4",
                    "name": dataset,
                    "axes": axes_list,
                    "datasets": datasets,
                    "coordinateTransformations": [
                        {"type": "scale", "scale": top_scale}
                    ],
                }
            ]
        }
        return jsonify(attr), HTTPStatus.OK

    def _attributes_impl(self, dataset, scale):
        dtype = g.get_output_dtype(self.output_dtype).__name__
        # Map numpy dtypes to Zarr dtypes
        dtype_map = {
            "uint8": "|u1",
            "uint16": "<u2",
            "uint32": "<u4",
            "uint64": "<u8",
            "int8": "<i1",
            "int16": "<i2",
            "int32": "<i4",
            "int64": "<i8",
            "float32": "<f4",
            "float64": "<f8",
        }
        zarr_dtype = dtype_map.get(dtype, dtype)

        attr = {
            "chunks": list(self.zarr_block_shape),
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1},
            "dtype": zarr_dtype,
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "shape": self.vol_shape,
            "zarr_format": 2,
        }
        print(f"Array metadata (scale={scale}): {attr}", flush=True)
        return jsonify(attr), HTTPStatus.OK

    def _chunk_impl(self, dataset, scale, chunk_z, chunk_y, chunk_x):
        corner = self.zarr_block_shape[:3] * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, self.zarr_block_shape[:3]]) * self.output_voxel_size
        roi = Roi(box[0], box[1])
        chunk_data = self.inferencer.process_chunk(self.idi_raw, roi)

        # Reorder model output axes to Zarr-expected order
        if self.has_channel:
            chunk_data = self._reorder_to_zarr_axes(chunk_data)

        chunk_data = chunk_data.astype(g.get_output_dtype(self.output_dtype))

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

        # Encode using Zarr format
        encoded = self.chunk_encoder.encode(chunk_data)

        return (
            encoded,
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )

    def _reorder_to_zarr_axes(self, data: np.ndarray) -> np.ndarray:
        """Reorder data from model output axes to Zarr-expected order matching self.axes + channel."""
        zarr_axes = tuple(self.axes) + ("c",)
        model_axes = self.model_output_axes

        if len(model_axes) != data.ndim:
            logger.warning(
                f"Model output ndim ({data.ndim}) != declared axes {model_axes}, "
                "skipping reorder"
            )
            return data

        if tuple(model_axes) == zarr_axes:
            return data

        # For single-channel output the byte layout is identical regardless of
        # where the size-1 channel axis sits, so skip the expensive copy.
        c_idx = model_axes.index("c")
        if data.shape[c_idx] == 1:
            return data.reshape([data.shape[model_axes.index(ax)] for ax in zarr_axes])

        # Build permutation from model axes order to zarr axes order
        perm = tuple(model_axes.index(ax) for ax in zarr_axes)
        return np.ascontiguousarray(data.transpose(perm))

    def _initialize_chunk_encoder(self):
        return numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)

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
