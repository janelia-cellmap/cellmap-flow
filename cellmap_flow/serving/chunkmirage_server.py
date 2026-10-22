"""The inference server, built on chunkmirage.

``ChunkmirageServer`` serves one model's predictions over one dataset as
zarr v2, for Neuroglancer. chunkmirage does the serving: reading the raw
data around each chunk, caching the model's output, queueing chunks for the
device (finest level first, dropping those nobody waits for any more) and
the zarr format. cellmap-flow keeps the model, its chains
(``serving.chunkmirage_ops``) and the routes the dashboard reads.

A layer is ``zarr://<server>/<model><ARGS_KEY><blob><ARGS_KEY>/zarr``: a
name, then chunkmirage's format. The name is resolved into a pipeline on its
first request (``pipeline_for``), and the blob's chain is the layer's own, so
layers sharing a server never share a chain. The array has one level, s0,
with any channel axis first.

``/__control__/model_info`` (and its older name ``/output_probe``) report the
model's geometry and output activation, and ``/__control__/restart`` asks a
finetune job to train again. model_info says ``"engine": "chunkmirage"``: the
release before this one had two engines, and its dashboard reads the field
to tell their URLs apart.

``CELLMAP_FLOW_PREDICTION_CACHE_BYTES`` (default 16 GiB; 0 for none) bounds the
memory kept for computed chunks, the model's output and the raw data read for
it; ``CELLMAP_FLOW_RAW_CACHE_BYTES`` (default 1 GiB) tensorstore's decoded raw
chunks.
"""

import logging
import os
import secrets
import socket
from http import HTTPStatus

import numpy as np
from chunkmirage.cache import LRUCache
from chunkmirage.pipeline import Pipeline, select_axes
from chunkmirage.server import DatasetRegistry, create_app
from funlib.geometry.coordinate import Coordinate
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from cellmap_flow.inference.runner import DeviceSlots
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.io.raw import open_raw, selected_channel, spatial_axes, spatial_shape, stored_placement
from cellmap_flow.jobs.spec import IP_PATTERN
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.process_chain import process_chain
from cellmap_flow.serving.chunkmirage_ops import DevicePostprocessOp, InferenceOp, ServedModel, layer_ops, serve_model
from cellmap_flow.serving.protocol import ARGS_KEY, FORMAT, split_dataset_url
from cellmap_flow.serving.restart_token import TOKEN_HEADER, tokens_match
from cellmap_flow.utils.address import this_machine_ip

logger = logging.getLogger(__name__)

ENGINE = "chunkmirage"

PREDICTION_CACHE_BYTES_ENV = "CELLMAP_FLOW_PREDICTION_CACHE_BYTES"
# A Cellpose flows chunk (3 x 8 x 512 x 512 float32) is 25 MB, and a view of
# it asked for 80 chunks: 2 GiB dropped the first ones, under the cursor, and
# ran the model on them again. A server's job has 15 GB of memory a slot or
# more, and 4 slots.
PREDICTION_CACHE_BYTES_DEFAULT = 16 << 30
RAW_CACHE_BYTES_ENV = "CELLMAP_FLOW_RAW_CACHE_BYTES"
RAW_CACHE_BYTES_DEFAULT = 1 << 30
HALF_PRECISION_ENV = "CELLMAP_FLOW_HALF_PRECISION"
# Chunks read and normalized while the device runs another's forward: a
# 178^3 input took 0.03-0.3 s to normalize next to a 0.15 s forward on an A100.
INFERENCE_PREFETCH = 2


def _env_count(name, default):
    """A whole number from the environment (``1e9`` allowed), or ``default``."""
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None


def _env_flag(name):
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"{name} must be 1 or 0, got {os.environ[name]!r}")


def _whole(values, what):
    out = np.rint(np.asarray(values, dtype=float)).astype(int)
    if not np.allclose(values, out, atol=1e-6):
        raise ValueError(f"{what} {tuple(values)} is not whole voxels")
    return tuple(int(v) for v in out)


class ChunkmirageServer:
    """One model's predictions over ``dataset_name``, served by chunkmirage.

    ``resample`` reads a level resampled to the model's input voxel size when
    the data has none at it (``open_raw(..., resample=True)``, see
    ``cellmap_flow.io.raw``), instead of reading the nearest level as if it
    were at that size; ``cellmap_flow serve --resample``. ``restart_callback``
    enables POST /__control__/restart, which then only accepts requests
    carrying ``restart_token`` in the X-Restart-Token header.

    ``CELLMAP_FLOW_GPU_SLOTS`` (default 1) is how many chunks use the device
    at once (inference.runner.DeviceSlots); ``CELLMAP_FLOW_HALF_PRECISION=1``
    serves the model under half-precision autocast, as ``half_precision =
    True`` in its config does, if it agrees with fp32 at warmup (see
    Inferencer). The Inferencer is built before anything else, so a model
    whose output contradicts its declared shapes fails here, on the warmup
    forward, before the server announces itself.
    """

    def __init__(self, dataset_name, model_config, restart_callback=None, restart_token=None, resample=False):
        if restart_callback is not None and not restart_token:
            raise ValueError("restart_callback requires a restart_token")
        self.restart_callback = restart_callback
        self.restart_token = restart_token
        self.resample = resample

        # The runner's slots hold the device for the forward only.
        # chunkmirage's queue (its order, and dropping chunks
        # nobody waits for) admits INFERENCE_PREFETCH chunks more, so their
        # input normalization, which the InferenceOp does, overlaps the
        # forward: with the queue's slots the device's, normalizing and the
        # forward took turns, and the GPU sat idle half the time.
        device_slots = DeviceSlots.from_env()
        self.inferencer = Inferencer(
            model_config, device_slots=device_slots, half_precision=_env_flag(HALF_PRECISION_ENV)
        )
        InferenceOp.slots = device_slots.n + INFERENCE_PREFETCH
        DevicePostprocessOp.slots = device_slots.n
        self.geometry = model_config.geometry

        # The level the model reads, chosen and placed as open_raw does for
        # every reader (blockwise reads the same level, padded the same way:
        # its edge repeated, as BlockInput pads).
        self.raw = raw = open_raw(dataset_name, Coordinate(self.geometry.input_voxel_size), resample=resample)
        self.axes = list(spatial_axes(raw))
        self.has_channel = self.geometry.has_channel_axis
        input_voxel_size = np.asarray(Coordinate(self.geometry.input_voxel_size), dtype=float)
        output_voxel_size = Coordinate(self.geometry.output_voxel_size)
        block = tuple(int(v) for v in self.geometry.block_shape()[: len(self.axes)])
        declared = self.geometry.output_dtype
        stage_dtype = np.dtype(declared) if np.dtype(declared).kind in "iub" else np.dtype(np.float32)
        self.served = ServedModel(
            name=f"{getattr(model_config, 'name', None) or type(model_config).__name__}@{id(self):x}",
            runner=self.inferencer,
            translation=tuple(float(v) for v in raw.translation),
            level_shape=spatial_shape(raw),
            # Output voxel 0 starts at the corner of the level the model reads,
            # so every output voxel sits exactly on the input voxels it is
            # computed from; anchored at 0 it would be half a raw voxel off
            # Janelia data, whose corner is -4 nm. Whole nm: Roi is integral.
            origin=np.round(np.array(raw.corner, dtype=float)).astype(int),
            block=block,
            input_voxel_size=tuple(input_voxel_size),
            output_voxel_size=output_voxel_size,
            halo=_whole(np.asarray(self.geometry.context, dtype=float) / input_voxel_size, "the model's context"),
            ratio=tuple(np.asarray(output_voxel_size, dtype=float) / input_voxel_size),
            output_channels=int(self.geometry.output_channels),
            has_channel=self.has_channel,
            output_dtype=declared,
            stage_dtype=stage_dtype,
        )
        serve_model(self.served)

        self.registry = DatasetRegistry(
            cache=LRUCache(_env_count(PREDICTION_CACHE_BYTES_ENV, PREDICTION_CACHE_BYTES_DEFAULT)),
            source_cache_bytes=_env_count(RAW_CACHE_BYTES_ENV, RAW_CACHE_BYTES_DEFAULT),
            resolver=self.pipeline_for,
        )
        self.source = self._open_source()
        self._warned_no_chain = False
        # The datasets API stays shut (a random token nobody is told): a
        # dataset registered through it could read any file this user can.
        self.app = create_app(
            self.registry,
            extra_routes=self._routes(),
            allow_edit=False,
            token=secrets.token_urlsafe(32),
            route_plugins=False,
        )
        print(f"Host name: {socket.gethostname()}", flush=True)

    # --- what is served ------------------------------------------------------

    def _open_source(self):
        """The level the model reads, as a chunkmirage source placed where its
        data really lies: the level's own voxel size and voxel 0's centre, in
        nm, which for a relabelled level are not those ``raw`` reads at.
        chunkmirage then reads it at the model's voxel size as ``raw`` does
        (``pipeline_for``'s ``input_level``)."""
        from chunkmirage.sources import open_source

        raw = self.raw
        voxel_size, translation = stored_placement(raw)
        # Any channel axis comes first, in OME's order as in worldarray's for
        # a precomputed volume (whose only channel, if it has one, both drop:
        # chunkmirage opens the level through worldarray too); pipeline_for
        # selects from it.
        lead = len(raw.axes) - len(self.axes)
        # worldarray's path of the level, which chunkmirage opens as it is: a
        # precomputed volume's scale as ``<volume>/<scale key>``, a local path
        # absolute and without "..", which tensorstore refuses (`serve -d
        # ../data.zarr/raw` once failed to start here while the reader had
        # opened it).
        return open_source(
            raw.level_path,
            cache_bytes=self.registry.source_cache_bytes,
            cache=self.registry.cache,
            voxel_size=[1.0] * lead + list(voxel_size),
            translation=[0.0] * lead + list(translation),
            units=[""] * lead + ["nanometer"] * len(self.axes),
            axes=["c"] * lead + list(self.axes),
        )

    def _chain(self, name):
        """``(input_norm, postprocess, extras)`` steps the layer named ``name`` asks for.

        A name without an args block gets the process's chain, empty in a
        server started from the CLI; it then warns that the model is fed raw
        voxel values, which looks exactly like a model that trained badly.
        """
        blob = split_dataset_url(name)
        if blob is None:
            chain = process_chain()
            spec, extras = PipelineSpec.from_steps(chain.input_norms, chain.postprocess), {}
            if not spec.input_norm and not self._warned_no_chain:
                self._warned_no_chain = True
                logger.warning(
                    "Serving WITHOUT normalization or postprocessing: the layer URL "
                    f"has no {ARGS_KEY} block. Raw voxel values go to the model "
                    "unmodified. If the model expects normalized input (e.g. [-1, 1]) "
                    "its output will be meaningless. Re-add the layer from the "
                    "dashboard so the URL carries the current Input/Postprocess "
                    "configuration."
                )
        else:
            spec, extras = PipelineSpec.from_url_blob(blob)
            if not spec.input_norm:
                logger.warning(
                    f"{name}: the layer's chain has NO input normalizers. The model will see raw voxel values."
                )
        # Each value as its step's constructor parsed it: the dashboard's forms
        # send "0.0" where a YAML gave 0.0, and the ops' fields key the cache,
        # so submitting a postprocessing step ran the model again on every
        # chunk instead of reusing its output.
        spec = PipelineSpec.from_steps(*spec.build())
        return [dict(s) for s in spec.input_norm], [dict(s) for s in spec.postprocess], extras

    def pipeline_for(self, name):
        """The pipeline for layer ``name``: the raw data, the model on the
        layer's normalization, then the layer's postprocessing. Only this
        server's model and data; the name only chooses the chains."""
        input_norm, postprocess, extras = self._chain(name)
        ops = layer_ops(self.served.name, input_norm, postprocess, extras.get("dashboard_url"), f"{name}/{FORMAT}")
        logger.info(
            f"Serving {name.split(ARGS_KEY)[0] or name} with input normalizers "
            f"{[type(n).__name__ for n in ops[0]._norms]}, postprocessors "
            f"{[step['name'] for step in postprocess]}"
        )
        source = self.source
        if "c" in source.levels[0].info.axes:
            source = select_axes(source, {"c": selected_channel(ops[0]._norms)})
        return Pipeline(
            source,
            ops,
            cache=self.registry.cache,
            chunk_shape=list(self.served.block),
            padding="edge",
            input_level="resample" if self.resample else "nearest",
            # The level is the one cellmap-flow chose: read as it is only at
            # the model's voxel size.
            level_rtol=1e-6,
        )

    def weights_changed(self):
        """The model's weights changed in place: layers resolved from now on
        are computed anew (the finetune loop's new iteration, under a new name)."""
        self.served.weights_changed()

    # --- the dashboard's routes ----------------------------------------------

    def model_info(self) -> dict:
        """The served model's geometry and output activation, and the engine serving it.

        Both halves exist so the dashboard does not have to build the model
        itself: it runs on whatever node launched the jobs, often without a
        usable GPU. The geometry comes from the validated config, so it is
        reported even when the warmup probe failed; script-defined models
        expose nothing else to the dashboard, which makes this the only place
        it can learn their shapes and channel names. The activation half is
        measured once at warmup (Inferencer._warmup) and lets the dashboard
        propose a postprocessing chain, or flag one that contradicts the
        model. A resampled input really is at input_voxel_size; a relabelled
        one is at the level's, which moves the output (see to_model_info).
        """
        raw = self.raw
        stored, _ = stored_placement(raw)
        info = self.geometry.to_model_info(self.axes, raw.voxel_size if raw.resampled else stored)
        info["output_axes"] = (["c"] if self.has_channel else []) + list(self.axes)
        info["input_resampled_from"] = list(stored) if raw.resampled else None
        relabelled = not raw.resampled and not np.allclose(
            np.asarray(stored, dtype=float), np.asarray(self.geometry.input_voxel_size, dtype=float)
        )
        info["input_relabelled_from"] = list(stored) if relabelled else None
        info["display_channel"] = getattr(self.inferencer.model_config, "display_channel", None)
        info["engine"] = ENGINE
        output_class = getattr(self.inferencer, "output_class", None)
        if output_class is None:
            info.update({"available": False, "reason": "output probe did not run"})
            return info
        output_range = getattr(self.inferencer, "output_range", None)
        info.update({
            "available": True,
            "output_class": output_class,
            "output_min": output_range[0] if output_range else None,
            "output_max": output_range[1] if output_range else None,
        })
        return info

    def _routes(self):
        async def model_info(request):
            return JSONResponse(self.model_info())

        async def restart(request):
            if self.restart_callback is None:
                return JSONResponse({"success": False, "error": "Restart control not enabled"},
                                    HTTPStatus.NOT_IMPLEMENTED)
            # A restart can change what the job trains on, and this server
            # listens on every interface, so only the job manager that wrote
            # the job's token may trigger one.
            if not tokens_match(self.restart_token, request.headers.get(TOKEN_HEADER)):
                return JSONResponse({"success": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            try:
                payload = await request.json()
            except ValueError:
                payload = {}
            try:
                accepted = await run_in_threadpool(self.restart_callback, payload or {})
            except Exception as e:
                logger.error(f"Failed to process restart control request: {e}", exc_info=True)
                return JSONResponse({"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            if not accepted:
                return JSONResponse({"success": False, "error": "Restart request rejected"}, HTTPStatus.CONFLICT)
            return JSONResponse({"success": True})

        return [
            Route("/__control__/model_info", model_info, methods=["GET"]),
            # Older name, kept for dashboards from before the geometry fields.
            Route("/__control__/output_probe", model_info, methods=["GET"]),
            Route("/__control__/restart", restart, methods=["POST"]),
        ]

    # --- running -------------------------------------------------------------

    def read_chunk(self, index=(0, 0, 0), name="check"):
        """Chunk ``index`` of layer ``name`` (no chain: the process's), computed
        here without HTTP: what ``cellmap_flow infer --server-check`` runs."""
        pipeline = self.registry.resolve(name)
        lead = (0,) * (pipeline.info(0).ndim - len(index))
        return pipeline.chunk(0, lead + tuple(index))

    def run(self, port=None, certfile=None, keyfile=None):
        """Serve until stopped, on ``port`` (any free one for None or 0).

        The address is announced, as the IP_PATTERN marker and in the ready
        file a launcher asked for (jobs/ready.py), once the port is bound and
        the server is taking requests.
        """
        import chunkmirage

        from cellmap_flow.jobs.ready import write_ready_file

        tls = (certfile, keyfile) if certfile and keyfile else None

        def announce(server):
            self._server, self.port = server, server.port
            self.address = f"{'https' if tls else 'http'}://{this_machine_ip()}:{server.port}"
            output = f"{IP_PATTERN[0]}{self.address}{IP_PATTERN[1]}"
            logger.error(output)
            print(output, flush=True)
            write_ready_file(self.address)

        chunkmirage.serve(self.app, "0.0.0.0", int(port or 0), on_ready=announce, ssl=tls, block=True)

    def stop(self):
        """Ask a running ``run`` to return."""
        server = getattr(self, "_server", None)
        if server is not None:
            server.stop()
