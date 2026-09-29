import logging
import os
import select
import socket
import ssl
import threading
from collections import OrderedDict
from http import HTTPStatus
from typing import NamedTuple, Optional

import numpy as np
import numcodecs
from flask import Flask, has_request_context, jsonify, redirect, request
from flask_cors import CORS
from funlib.geometry import Roi
from funlib.geometry.coordinate import Coordinate

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import ChunkCancelled, DeviceSlots, Inferencer
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    get_public_ip,
    IP_PATTERN,
    get_free_port,
)
from cellmap_flow.utils.restart_token import TOKEN_HEADER, tokens_match
from cellmap_flow.utils import zarr_v3
from cellmap_flow.utils.serilization_utils import get_process_dataset_url

from cellmap_flow.globals import g

import requests
import time

logger = logging.getLogger(__name__)

# How many distinct chains (layer URLs) one server keeps built at once.
CHAIN_CACHE_SIZE = 32

# How the server reads its raw data; see CellMapFlowServer.__init__.
RAW_CACHE_BYTES_ENV = "CELLMAP_FLOW_RAW_CACHE_BYTES"
RAW_CACHE_BYTES_DEFAULT = 1 << 30
RAW_READ_CONCURRENCY_ENV = "CELLMAP_FLOW_RAW_READ_CONCURRENCY"


def _env_count(name, default):
    """A whole number from the environment (``1e9`` allowed), or ``default``."""
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _client_gone_check():
    """A callable telling whether this request's client has hung up, or None.

    The werkzeug dev server puts the connection's socket in the environ. A
    browser that drops a request, as neuroglancer does for chunks a pan took
    out of view, closes that connection, and its socket then reads
    end-of-file. While a request waits for its answer the client sends
    nothing else, so any other readable state means it is still there.
    Another WSGI server, a TLS socket, a platform without poll() or a call
    outside a request (the CLI's server check) gives no signal: every
    chunk is then computed, as before.
    """
    if not has_request_context():
        return None
    sock = request.environ.get("werkzeug.socket")
    if sock is None or isinstance(sock, ssl.SSLSocket) or not hasattr(select, "poll"):
        return None

    def gone():
        try:
            poller = select.poll()
            poller.register(sock, select.POLLIN)
            if not poller.poll(0):
                return False
            return sock.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):  # reset, or already closed
            return True

    return gone


class ServedChain(NamedTuple):
    """The normalization/postprocessing a layer URL asks for."""

    dashboard_url: Optional[str]
    input_norms: Optional[list]  # None: the process default (g.input_norms)
    postprocess: Optional[list]  # None: the process default (g.postprocess)

    def effective_postprocess(self):
        return g.postprocess if self.postprocess is None else self.postprocess


class CellMapFlowServer:
    """
    Flask application hosting a "virtual Zarr" for Neuroglancer.
    All routes are defined via Flask decorators for convenience.
    """

    def __init__(
        self,
        dataset_name: str,
        model_config: ModelConfig,
        restart_callback=None,
        restart_token=None,
    ):
        """
        Initialize the server and set up routes via decorators.

        ``restart_callback`` enables POST /__control__/restart, which then
        only accepts requests carrying ``restart_token`` in the
        X-Restart-Token header.

        ``CELLMAP_FLOW_GPU_SLOTS`` (default 1), read here, is how many chunk
        requests may use the device at once; the rest wait their turn in
        arrival order. See inferencer.DeviceSlots.

        ``CELLMAP_FLOW_RAW_CACHE_BYTES`` (default 1 GiB; 0 for none) and
        ``CELLMAP_FLOW_RAW_READ_CONCURRENCY`` (default: tensorstore's, one
        decode per core) set how the raw data is read.
        """
        if restart_callback is not None and not restart_token:
            raise ValueError("restart_callback requires a restart_token")

        # Before anything reads model_config.config: the Inferencer builds it
        # so that the declared shapes are checked on its warmup forward, on
        # the GPU, rather than by a separate forward on the CPU. A mismatch
        # raises here, before the server announces itself.
        self.inferencer = Inferencer(model_config, device_slots=DeviceSlots.from_env())

        block_shape = [int(x) for x in model_config.config.block_shape]

        self.input_voxel_size = Coordinate(model_config.config.input_voxel_size)
        self.output_voxel_size = Coordinate(model_config.config.output_voxel_size)
        self.output_channels = model_config.config.output_channels
        self.output_dtype = model_config.output_dtype
        self.model_output_axes = model_config.chunk_output_axes

        # Kept so /__control__/model_info can report geometry without the
        # dashboard having to build the model itself.
        self.model_config = model_config

        self.restart_callback = restart_callback
        self.restart_token = restart_token

        # Every chunk request reads its input here, and neuroglancer sends
        # them several at a time. The IDI's default single reader thread
        # queued those reads behind one another (a cold 178^3 read from /nrs
        # took 0.3-0.5 s instead of 0.2-0.3), and without a cache neighbouring
        # chunks, whose inputs overlap by two thirds, each read and decoded
        # it all again.
        self.idi_raw = ImageDataInterface(
            dataset_name,
            voxel_size=self.input_voxel_size,
            concurrency_limit=_env_count(RAW_READ_CONCURRENCY_ENV, None),
            cache_bytes=_env_count(RAW_CACHE_BYTES_ENV, RAW_CACHE_BYTES_DEFAULT),
        )
        # The output grid starts at the corner of the raw level the model
        # reads, so every output voxel sits exactly on the input voxels it is
        # computed from. Anchored at 0 it was half a raw voxel off Janelia
        # data, whose corner is -4 nm. Whole nanometers: Roi is integral.
        self.origin = np.round(np.array(self.idi_raw.offset, dtype=float)).astype(int)
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

        # Each layer URL carries its own chain; they are built once per URL
        # and never written to g, so layers (tabs, users) sharing this server
        # don't get each other's normalization.
        self._chains = OrderedDict()
        self._chain_lock = threading.Lock()
        self._warned_no_chain = False

        n_spatial = len(self.axes)
        # block_shape is (*spatial, channels); only the spatial part is the
        # chunk grid. The channel count comes from the model (or the chain).
        self._spatial_block = block_shape[:n_spatial]
        if self.has_channel and len(block_shape) > n_spatial:
            if block_shape[n_spatial] != self.output_channels:
                logger.warning(
                    f"block_shape {block_shape} ends in {block_shape[n_spatial]} "
                    f"channels but output_channels is {self.output_channels}; "
                    "serving output_channels"
                )
        self._spatial_shape = self._served_spatial_shape()
        self.vol_shape, self.zarr_block_shape = self._zarr_geometry(
            ServedChain(None, None, None)
        )

        # Chunk encoding for Zarr
        self.chunk_encoder = self._initialize_chunk_encoder()

        # Create and configure Flask
        self.app = Flask(__name__)
        CORS(self.app)

        hostname = socket.gethostname()
        print(f"Host name: {hostname}", flush=True)

        # Opening a server's address in a browser shows what it serves.
        @self.app.route("/")
        def home():
            return redirect("/__control__/model_info")

        @self.app.route("/__control__/model_info", methods=["GET"])
        # Older name, kept so a dashboard can still talk to a server started
        # before the geometry fields were added.
        @self.app.route("/__control__/output_probe", methods=["GET"])
        def control_model_info():
            """Report the served model's geometry and output activation.

            Both halves exist so the dashboard does not have to build the model
            itself. It runs on whatever node launched the jobs -- often without
            a usable GPU -- and instantiating a model there just to read a shape
            is what made the finetune tab retry a full weight download and
            torch.export on every poll.

            The activation half is measured once during startup warmup (see
            Inferencer._warmup) and lets the dashboard propose a postprocessing
            chain, or flag one that contradicts the model -- e.g. a
            SigmoidPostprocessor on a model that already ends in a sigmoid.
            """
            inferencer = self.inferencer
            config = self.model_config.config

            # Geometry comes from the validated config rather than the warmup,
            # so it is reported even when the probe itself failed. Script-defined
            # models expose nothing to the dashboard through to_dict(), which
            # makes this the only place it can learn e.g. that a model has 3+
            # channels and might be predicting affinities.
            # Channel names, not just the count: the dashboard decides whether
            # a model predicts affinities by looking for "_aff" in them, and a
            # script model exposes nothing through to_dict(), so this is the
            # only way it can learn them without building the model.
            channels = (
                getattr(config, "channels", None)
                or getattr(config, "channels_names", None)
                or getattr(config, "classes", None)
            )

            def numbers(values):
                # Whole numbers as ints; int() truncated 5.24 nm to 5.
                return [
                    int(v) if float(v).is_integer() else float(v) for v in values
                ]

            info = {
                "output_channels": self.output_channels,
                "channels": [str(c) for c in channels] if channels else None,
                "write_shape": numbers(config.write_shape),
                "read_shape": numbers(config.read_shape),
                "output_voxel_size": numbers(config.output_voxel_size),
                "input_voxel_size": numbers(config.input_voxel_size),
            }

            output_class = getattr(inferencer, "output_class", None)
            if output_class is None:
                info.update(
                    {"available": False, "reason": "output probe did not run"}
                )
                return jsonify(info), HTTPStatus.OK

            output_range = getattr(inferencer, "output_range", None)
            info.update(
                {
                    "available": True,
                    "output_class": output_class,
                    "output_min": output_range[0] if output_range else None,
                    "output_max": output_range[1] if output_range else None,
                }
            )
            return jsonify(info), HTTPStatus.OK

        @self.app.route("/__control__/restart", methods=["POST"])
        def control_restart():
            if self.restart_callback is None:
                return jsonify({"success": False, "error": "Restart control not enabled"}), HTTPStatus.NOT_IMPLEMENTED
            # A restart can change what the job trains on, and this server
            # listens on every interface, so only the job manager that wrote
            # the job's token may trigger one.
            if not tokens_match(self.restart_token, request.headers.get(TOKEN_HEADER)):
                return jsonify({"success": False, "error": "unauthorized"}), HTTPStatus.UNAUTHORIZED
            try:
                payload = request.get_json(silent=True) or {}
                accepted = self.restart_callback(payload)
                if not accepted:
                    return jsonify({"success": False, "error": "Restart request rejected"}), HTTPStatus.CONFLICT
                return jsonify({"success": True}), HTTPStatus.OK
            except Exception as e:
                logger.error(f"Failed to process restart control request: {e}", exc_info=True)
                return jsonify({"success": False, "error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

        @self.app.route("/<path:dataset>/.zattrs", methods=["GET"])
        def top_level_attributes(dataset):
            self.refresh_dataset(dataset)
            return self._top_level_attributes_impl(dataset)

        @self.app.route("/<path:dataset>/s<int:scale>/.zarray", methods=["GET"])
        def attributes(dataset, scale):
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

    def _served_spatial_shape(self):
        """Output voxels from the grid origin to the end of the raw data."""
        raw_end = np.array(self.idi_raw.offset, dtype=float) + np.array(
            self.idi_raw.shape, dtype=float
        ) * np.array(self.idi_raw.voxel_size, dtype=float)
        output_voxel_size = np.array(self.output_voxel_size, dtype=float)
        return [int(v) for v in np.ceil((raw_end - self.origin) / output_voxel_size)]

    def _chain_for(self, dataset) -> ServedChain:
        """The chain the requested layer URL carries, built once per URL.

        A URL without an args block gets the process default (g's chain,
        empty in a server started from the CLI).
        """
        if not dataset or ARGS_KEY not in dataset:
            if not self._warned_no_chain:
                self._warned_no_chain = True
                if not (g.input_norms or g.postprocess):
                    get_process_dataset_url(dataset or "")  # logs the warning
            return ServedChain(None, None, None)

        parts = dataset.split(ARGS_KEY)
        key = parts[1] if len(parts) == 3 else dataset
        with self._chain_lock:
            chain = self._chains.get(key)
            if chain is not None:
                self._chains.move_to_end(key)
                return chain
            dashboard_url, input_norms, postprocess = get_process_dataset_url(dataset)
            chain = ServedChain(dashboard_url, list(input_norms), list(postprocess))
            self._chains[key] = chain
            while len(self._chains) > CHAIN_CACHE_SIZE:
                self._chains.popitem(last=False)
            return chain

    def refresh_dataset(self, dataset) -> ServedChain:
        """Resolve (and cache) the chain for ``dataset``. Changes no globals."""
        return self._chain_for(dataset)

    def _num_channels(self, chain: ServedChain) -> int:
        channels = self.output_channels
        for step in chain.effective_postprocess():
            if hasattr(step, "num_channels"):
                channels = step.num_channels
        return int(channels)

    def _zarr_geometry(self, chain: ServedChain):
        """(shape, chunks) of the served array under ``chain``."""
        shape = list(self._spatial_shape)
        chunks = list(self._spatial_block)
        if self.has_channel:
            channels = self._num_channels(chain)
            shape.append(channels)
            chunks.append(channels)
        return shape, chunks

    def _output_dtype(self, chain: ServedChain):
        return np.dtype(
            g.get_output_dtype(self.output_dtype, chain.effective_postprocess())
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
            # OME translation is the centre of voxel 0, so the grid's corner
            # plus half a voxel; Neuroglancer then draws voxel 0 at the origin.
            translation_values = zarr_v3.ome_translation(self.origin, scale_values)
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
        chain = self._chain_for(dataset)
        shape, chunks = self._zarr_geometry(chain)
        attr = {
            "chunks": chunks,
            "compressor": {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1},
            # The zarr v2 typestr of whatever the model or the chain declares
            # (a numpy class, an np.dtype or a string): "<f2", "|b1", "|i1"...
            "dtype": self._output_dtype(chain).str,
            "fill_value": 0,
            "filters": None,
            "order": "C",
            "shape": shape,
            "zarr_format": 2,
        }
        print(f"Array metadata (scale={scale}): {attr}", flush=True)
        return jsonify(attr), HTTPStatus.OK

    def _chunk_impl(self, dataset, scale, chunk_z, chunk_y, chunk_x):
        chain = self._chain_for(dataset)
        block = np.array(self._spatial_block)
        corner = block * np.array([chunk_z, chunk_y, chunk_x])
        box = np.array([corner, block]) * self.output_voxel_size
        roi = Roi(tuple(int(v) for v in self.origin + box[0]), tuple(int(v) for v in box[1]))
        try:
            chunk_data = self.inferencer.process_chunk(
                self.idi_raw,
                roi,
                input_norms=chain.input_norms,
                postprocess=chain.postprocess,
                cancelled=_client_gone_check(),
            )
        except ChunkCancelled:
            # Nobody is left to read it. 499 is nginx's "client closed request".
            logger.debug(f"Skipped chunk {chunk_z}.{chunk_y}.{chunk_x}: its client went away")
            return b"", 499

        # Reorder model output axes to Zarr-expected order
        if self.has_channel:
            chunk_data = self._reorder_to_zarr_axes(chunk_data)

        chunk_data = chunk_data.astype(self._output_dtype(chain))

        current_time = time.time()

        # assume only one has equivalences
        for postprocess in chain.effective_postprocess():
            if (
                # A chain encoded outside /api/process has no dashboard to
                # tell; concatenating None raised TypeError mid-chunk.
                chain.dashboard_url
                and hasattr(postprocess, "equivalences")
                and postprocess.equivalences is not None
                and (current_time - self.previous_refresh_time)
                > self.refresh_rate_seconds
            ):
                snapshot = getattr(postprocess, "equivalences_json", None)
                pairs = snapshot() if snapshot else postprocess.equivalences.to_json()
                equivalences = {
                    "dataset": dataset,
                    "equivalences": [
                        [int(item) for item in sublist] for sublist in pairs
                    ],
                }

                requests.post(
                    chain.dashboard_url.rstrip("/") + "/update/equivalences",
                    json=equivalences,
                )
                self.previous_refresh_time = current_time

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
        # The same news, for a launcher that asked for it as a file (see
        # jobs/ready.py): it then needs no bpeek. Written at the same moment
        # as the marker, so before the port below is bound.
        from cellmap_flow.jobs.ready import write_ready_file

        write_ready_file(address)

        self.app.run(
            host="0.0.0.0",
            port=port,
            debug=debug,
            use_reloader=debug,
            ssl_context=ssl_context,
        )
