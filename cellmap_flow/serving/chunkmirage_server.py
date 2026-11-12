"""The inference server, built on chunkmirage.

``ChunkmirageServer`` serves one model's predictions as zarr v2, for
Neuroglancer, over the datasets the dashboard opens on it. chunkmirage does
the serving: reading the raw data around each chunk, caching the model's
output, queueing chunks for the device (finest level first, dropping those
nobody waits for any more) and the zarr format. cellmap-flow keeps the
model, its chains (``serving.chunkmirage_ops``), the datasets and the routes
the dashboard reads.

A layer is ``zarr://<server>/<model><ARGS_KEY><blob><ARGS_KEY>/zarr``: a
name, then chunkmirage's format. The name is resolved into a pipeline on its
first request (``pipeline_for``), and the blob's chain is the layer's own, so
layers sharing a server never share a chain. The blob's ``DATA_KEY`` names
the dataset the layer shows; a layer naming none shows the server's current
one. The array has one level, s0, with any channel axis first.

A server starts on one dataset, its current one. Opening another in the
dashboard asks every running server to serve it (POST
``DATASET_CONTROL_PATH``, ``switch_dataset``) instead of starting new jobs:
the model stays loaded, its weights on the device once, and only the
dataset's metadata is read, in seconds. The new dataset becomes the current
one. The one current before it stays open, so its layers still load and
switching back is instant; any other is closed, and a layer still naming it
answers 400 until it is opened again.

``/__control__/model_info`` (and its older name ``/output_probe``) report the
model's geometry and output activation on the current dataset;
``/__control__/restart`` asks a finetune job to train again, and
``/__control__/stop`` to stop training after the epoch it is in. The three
POST routes take only requests carrying the server's control token in
``TOKEN_HEADER`` (``serving.restart_token``): the one its launcher wrote to
the file ``CONTROL_TOKEN_FILE_ENV`` names (``cellmap_flow serve`` reads it),
or a finetune job's. Without a token the dataset route is off (501), as
restart and stop are on any server but a finetune job's. model_info says
``"engine": "chunkmirage"``: the
release before this one had two engines, and its dashboard reads the field
to tell their URLs apart.

``CELLMAP_FLOW_PREDICTION_CACHE_BYTES`` (0 for none) bounds the memory kept
for computed chunks, the model's output and the raw data read for it: by
default 16 GiB, or for a model with an overlap enough for a view's
overlapping blocks (``prediction_cache_bytes``). Every open dataset shares
it: its keys include the source's path (chunkmirage's ``cache_key``) and
the model's name on that dataset, so two datasets never mix, and a closed
one's chunks are dropped as the cache fills.
``CELLMAP_FLOW_RAW_CACHE_BYTES`` (default 1 GiB) bounds tensorstore's
decoded raw chunks: one pool for the process, which every open dataset
draws on (chunkmirage opens every source in one tensorstore context per
budget). With at most two datasets open, a switch adds only their metadata
and pipelines to what the server holds.
"""

import itertools
import logging
import os
import secrets
import socket
import threading
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import numpy as np
from chunkmirage.cache import LRUCache
from chunkmirage.ops import Op
from chunkmirage.pipeline import Pipeline, select_axes
from chunkmirage.server import DatasetRegistry, create_app
from funlib.geometry.coordinate import Coordinate
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Route

from cellmap_flow.config.yaml import resolve_data_path
from cellmap_flow.inference.runner import DeviceSlots
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.io.raw import open_raw, selected_channel, spatial_axes, spatial_shape
from cellmap_flow.jobs.spec import IP_PATTERN
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.process_chain import process_chain
from cellmap_flow.serving.chunkmirage_ops import (
    DevicePostprocessOp,
    InferenceOp,
    ServedModel,
    layer_ops,
    serve_model,
    unserve_model,
)
from cellmap_flow.serving.protocol import ARGS_KEY, DATA_KEY, DATASET_CONTROL_PATH, FORMAT, split_dataset_url
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
# A model with an overlap (ModelGeometry.overlap) keeps its output blocks
# cached until every chunk they reach is blended; a block dropped before then
# is run again. A 2D slice of 6 x 6 chunks reaches (6 + 2)^2 x 3 = 192 blocks
# centred on chunks, (6 + 1)^2 x 2 = 98 on their corners, and Neuroglancer's
# default layout shows three slices. Blocks of 12 uint8 channels of 256^3 are
# 200 MB each: 103 GB for 512. A GPU queue's slot comes with 40 GB (15 GB on
# the L4 and T4 queues) and a server's job has 4, so the cache is held to
# OVERLAP_CACHE_SHARE of the memory the process may use (memory_limit_bytes).
OVERLAP_BLOCKS_CACHED = 512
OVERLAP_CACHE_SHARE = 0.5
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


def memory_limit_bytes():
    """The memory this process may use: its cgroup's limit (an LSF job's
    memory, under cgroup v2 or v1), else the machine's; None when neither can
    be read."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    candidates = []
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            candidates.append(f"/sys/fs/cgroup{path}/memory.max")
        elif "memory" in controllers.split(","):
            candidates.append(f"/sys/fs/cgroup/memory{path}/memory.limit_in_bytes")
    for candidate in candidates:
        try:
            with open(candidate) as f:
                value = f.read().strip()
        except OSError:
            continue
        # "max" (v2) or a huge number (v1) is no limit of its own.
        if value.isdigit() and int(value) < 1 << 60:
            return int(value)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def prediction_cache_bytes(geometry, stage_dtype) -> int:
    """The prediction cache's size when the environment sets none: 16 GiB, or
    for an overlapping model room for OVERLAP_BLOCKS_CACHED of its output
    blocks in ``stage_dtype`` (the dtype they are cached in) if that is more,
    held to OVERLAP_CACHE_SHARE of the memory the process may use."""
    if not any(geometry.overlap):
        return PREDICTION_CACHE_BYTES_DEFAULT
    block_bytes = int(np.prod(geometry.block_shape())) * np.dtype(stage_dtype).itemsize
    wanted = max(PREDICTION_CACHE_BYTES_DEFAULT, OVERLAP_BLOCKS_CACHED * block_bytes)
    limit = memory_limit_bytes()
    if limit is None:
        return wanted
    held = int(limit * OVERLAP_CACHE_SHARE)
    if held < wanted:
        logger.warning(
            f"The prediction cache holds {held / 1e9:.0f} GB of this model's {block_bytes / 1e6:.0f} MB blocks, "
            f"{held // block_bytes} of them, half the {limit / 1e9:.0f} GB this job may use: blocks dropped before "
            f"the chunks they reach are blended are run again. More memory, or a uint8 output_dtype, holds more."
        )
    return min(wanted, held)


def _chunkmirage_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("chunkmirage")
    except PackageNotFoundError:
        return "unknown version"


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


def _normalized(path) -> str:
    """A dataset's path as a server keys it: without surrounding blanks or a trailing "/"."""
    return str(path).strip().rstrip("/")


@dataclass
class _Dataset:
    """A dataset a server has open: ``path``, its key; ``raw``, the level the
    model reads (open_raw's); ``axes``, its spatial axes; ``served``, the
    model placed on it; ``source``, the level as the chunkmirage source its
    layers read."""

    path: str
    raw: Any
    axes: list
    served: ServedModel
    source: Any


class ChunkmirageServer:
    """One model's predictions over ``dataset_name``, and over the datasets
    the dashboard opens on it later (``switch_dataset``), served by chunkmirage.

    ``dataset_name`` is the level the model reads, the model's ``scale``
    already applied (the launcher applies it, ``config.yaml.resolve_data_path``);
    a dataset opened later is named as the dashboard has it, before its scale.
    ``resample`` reads a level resampled to the model's input voxel size when
    the data has none at it (``open_raw(..., resample=True)``, see
    ``cellmap_flow.io.raw``), instead of reading the nearest level as if it
    were at that size; ``cellmap_flow serve --resample``.
    ``control_token`` enables POST DATASET_CONTROL_PATH; ``restart_callback``
    POST /__control__/restart, and ``stop_callback`` POST /__control__/stop,
    and each needs the token. All three accept only requests carrying it in
    the TOKEN_HEADER header.

    ``CELLMAP_FLOW_GPU_SLOTS`` (default 1) is how many chunks use the device
    at once (inference.runner.DeviceSlots); ``CELLMAP_FLOW_HALF_PRECISION=1``
    serves the model under half-precision autocast, as ``half_precision =
    True`` in its config does, if it agrees with fp32 at warmup (see
    Inferencer). The Inferencer is built before anything else, so a model
    whose output contradicts its declared shapes fails here, on the warmup
    forward, before the server announces itself. It is built once: every
    dataset's ``ServedModel`` runs it.
    """

    def __init__(self, dataset_name, model_config, restart_callback=None, stop_callback=None, control_token=None,
                 resample=False):
        if (restart_callback is not None or stop_callback is not None) and not control_token:
            raise ValueError("restart_callback and stop_callback require a control_token")
        self.restart_callback = restart_callback
        self.stop_callback = stop_callback
        self.control_token = control_token
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
        self.has_channel = self.geometry.has_channel_axis
        # chunkmirage ignores an op's block_phase and blend before 0.1.0a7,
        # which has them: serving a model declaring others on an older one
        # would quietly lay and weight its blocks otherwise.
        if (self.geometry.block_phase, self.geometry.blend) != ("centred", "tent") and not hasattr(Op, "block_phase"):
            raise ValueError(
                f"{getattr(model_config, 'name', None) or 'The model'} declares block_phase "
                f"{self.geometry.block_phase!r} and blend {self.geometry.blend!r}, which need chunkmirage 0.1.0a7 "
                f"or later; this one ({_chunkmirage_version()}) lays every model's blocks centred, with the tent"
            )
        declared = self.geometry.output_dtype
        self._stage_dtype = np.dtype(declared) if np.dtype(declared).kind in "iub" else np.dtype(np.float32)
        self._model_name = getattr(model_config, "name", None) or type(model_config).__name__
        self._scale = getattr(model_config, "scale", None)

        self.registry = DatasetRegistry(
            cache=LRUCache(_env_count(PREDICTION_CACHE_BYTES_ENV,
                                      prediction_cache_bytes(self.geometry, self._stage_dtype))),
            source_cache_bytes=_env_count(RAW_CACHE_BYTES_ENV, RAW_CACHE_BYTES_DEFAULT),
            resolver=self.pipeline_for,
        )
        self._warned_no_chain = False

        # The open datasets by key (_key), the current one last and the one
        # current before it next to last; switch_dataset keeps no others.
        # _resolved is the dataset each registered layer was resolved on and
        # whether it named none (so follows the current one), or None once a
        # switch made it stale. _lock guards them and _weights_version;
        # _switch_lock lets one switch run at a time.
        self._lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._datasets: dict = {}
        self._resolved: dict = {}
        self._weights_version = 0
        self._keys: dict = {}
        self._served_names = itertools.count()
        dataset = self._open_dataset(_normalized(dataset_name))
        self._datasets[dataset.path] = dataset
        # A layer naming the very path the server started on shows it, the
        # model's scale applied or not (a finetune job's server is started
        # on its corrections' dataset as they name it).
        self._keys[str(dataset_name)] = dataset.path
        self._requested = str(dataset_name)
        self.registry.subscribe(self._drop_if_stale)

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

    # --- the datasets --------------------------------------------------------

    def _current(self) -> _Dataset:
        with self._lock:
            return next(reversed(self._datasets.values()))

    # The current dataset's, as a server had them when it served only one.
    @property
    def raw(self):
        return self._current().raw

    @property
    def axes(self):
        return self._current().axes

    @property
    def served(self) -> ServedModel:
        return self._current().served

    @property
    def source(self):
        return self._current().source

    def _key(self, path) -> str:
        """The key of dataset ``path``, named as the dashboard has it: the
        level this model reads there, its ``scale`` applied as the launcher
        applies it. Kept per path, since applying it inspects the node."""
        key = self._keys.get(path)
        if key is None:
            key = self._keys[path] = _normalized(resolve_data_path(path, self._scale))
        return key

    def _open_dataset(self, path) -> _Dataset:
        """The level at ``path`` the model reads, opened, with the model placed
        on it and registered under a name of its own, and one plain layer's
        pipeline built on it, so a dataset the model cannot read fails here.
        Blocking: it reads the dataset's metadata."""
        # The level the model reads, chosen and placed as open_raw does for
        # every reader (blockwise reads the same level, padded the same way:
        # its edge repeated, as BlockInput pads).
        raw = open_raw(path, Coordinate(self.geometry.input_voxel_size), resample=self.resample)
        axes = list(spatial_axes(raw))
        input_voxel_size = np.asarray(Coordinate(self.geometry.input_voxel_size), dtype=float)
        output_voxel_size = Coordinate(self.geometry.output_voxel_size)
        served = ServedModel(
            name=f"{self._model_name}@{id(self):x}.{next(self._served_names)}",
            runner=self.inferencer,
            translation=tuple(float(v) for v in raw.translation),
            level_shape=spatial_shape(raw),
            # Output voxel 0 starts at the corner of the level the model reads,
            # so every output voxel sits exactly on the input voxels it is
            # computed from; anchored at 0 it would be half a raw voxel off
            # Janelia data, whose corner is -4 nm. Whole nm: Roi is integral.
            origin=np.round(np.array(raw.corner, dtype=float)).astype(int),
            block=tuple(int(v) for v in self.geometry.served_block()[: len(axes)]),
            input_voxel_size=tuple(input_voxel_size),
            output_voxel_size=output_voxel_size,
            halo=_whole(np.asarray(self.geometry.context, dtype=float) / input_voxel_size, "the model's context"),
            ratio=tuple(np.asarray(output_voxel_size, dtype=float) / input_voxel_size),
            output_channels=int(self.geometry.output_channels),
            has_channel=self.has_channel,
            output_dtype=self.geometry.output_dtype,
            stage_dtype=self._stage_dtype,
            overlap=tuple(int(v) for v in self.geometry.overlap[: len(axes)]),
            block_phase=self.geometry.block_phase,
            blend=self.geometry.blend,
            weights_version=self._weights_version,
        )
        dataset = _Dataset(path, raw, axes, served, self._open_source(raw, axes))
        serve_model(served)
        try:
            self._pipeline(dataset, layer_ops(served.name, [], []))
        except Exception:
            unserve_model(served.name)
            raise
        return dataset

    def switch_dataset(self, path) -> dict:
        """Serve ``path`` (named as DATA_KEY names it) from now on, opening it
        unless it is open, and return model_info on it.

        The dataset current until now stays open; any other is closed: its
        layers leave the registry, so they are resolved again on their next
        request (and refused, naming a closed dataset), and its model and
        source are let go. Layers naming no dataset leave it too, to follow
        the current one. A dataset that cannot be opened raises, the server
        unchanged. Blocking: opening a dataset reads its metadata, seconds
        over NFS or the network; one switch runs at a time.
        """
        with self._switch_lock:
            key = self._key(path)
            with self._lock:
                dataset = self._datasets.get(key)
            if dataset is None:
                try:
                    dataset = self._open_dataset(key)
                except Exception:
                    self._keys.pop(path, None)  # it may yet appear, and be something else
                    raise
            with self._lock:
                self._requested = path
                if dataset is self._current():
                    return self.model_info()
                self._datasets.pop(key, None)
                # One opened after the weights changed starts at their version.
                dataset.served.weights_version = self._weights_version
                self._datasets[key] = dataset
                closed = [self._datasets.pop(k) for k in list(self._datasets)[:-2]]
                stale = [name for name, record in self._resolved.items() if record is not None and self._stale(record)]
                for name in stale:
                    self._resolved[name] = None
                info = self.model_info()
            for name in stale:
                self.registry.remove(name)
            for gone in closed:
                unserve_model(gone.served.name)
            logger.info(f"Serving {path} ({key}); open: {list(self._datasets)}"
                        + (f"; closed {[gone.path for gone in closed]}" if closed else ""))
            return info

    def _stale(self, record) -> bool:
        """Whether a layer resolved on ``record``'s dataset, naming one or not
        (its follows-current flag), may no longer be served from it."""
        dataset, follows_current = record
        return self._datasets.get(dataset.path) is not dataset or (follows_current and dataset is not self._current())

    def _drop_if_stale(self, name, pipeline):
        """The registry's subscriber: a layer a switch made stale while it was
        being resolved, registered once it was, leaves the registry again
        (its first request is answered from it, every later one anew)."""
        if pipeline is None:
            return
        with self._lock:
            stale = name in self._resolved and self._resolved[name] is None
            if stale:
                del self._resolved[name]
        if stale:
            self.registry.remove(name)

    # --- what is served ------------------------------------------------------

    def _open_source(self, raw, axes):
        """The level the model reads, as a chunkmirage source placed where its
        data really lies: the level's own voxel size and voxel 0's centre, in
        nm, which for a relabelled level are not those ``raw`` reads at.
        chunkmirage then reads it at the model's voxel size as ``raw`` does
        (``_pipeline``'s ``input_level``)."""
        from chunkmirage.sources import open_source

        voxel_size, translation = raw.source_voxel_size, raw.source_translation
        # Any channel axis comes first, in OME's order as in worldarray's for
        # a precomputed volume (whose only channel, if it has one, both drop:
        # chunkmirage opens the level through worldarray too); _pipeline
        # selects from it.
        lead = len(raw.axes) - len(axes)
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
            units=[""] * lead + ["nanometer"] * len(axes),
            axes=["c"] * lead + list(axes),
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
                    "dashboard so the URL carries the Processing tab's current "
                    "steps."
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
        """The pipeline for layer ``name``: the raw data of the dataset its
        blob's DATA_KEY names (the current one when it names none), the model
        on the layer's normalization, then the layer's postprocessing. Only
        this server's model and the datasets open on it: one that is not is
        refused with a ValueError, which chunkmirage answers with HTTP 400."""
        input_norm, postprocess, extras = self._chain(name)
        data = extras.get(DATA_KEY)
        key = None if data is None else self._key(data)
        # Built under the lock, so a switch cannot close the dataset (and let
        # its model go) before the ops have found it; they hold it from then.
        with self._lock:
            dataset = self._current() if key is None else self._datasets.get(key)
            if dataset is None:
                raise ValueError(
                    f"{data} is not open on this server, which serves {self._requested}: the dashboard opens a "
                    f"dataset there (POST {DATASET_CONTROL_PATH}) before showing it"
                )
            self._resolved[name] = (dataset, key is None)
            ops = layer_ops(dataset.served.name, input_norm, postprocess, extras.get("dashboard_url"),
                            f"{name}/{FORMAT}")
            pipeline = self._pipeline(dataset, ops)
        logger.info(
            f"Serving {name.split(ARGS_KEY)[0] or name} on {dataset.path} with input normalizers "
            f"{[type(n).__name__ for n in ops[0]._norms]}, postprocessors "
            f"{[step['name'] for step in postprocess]}"
        )
        return pipeline

    def _pipeline(self, dataset, ops):
        """``ops`` on ``dataset``'s level, on the model's chunk grid there."""
        source = dataset.source
        if "c" in source.levels[0].info.axes:
            source = select_axes(source, {"c": selected_channel(ops[0]._norms)})
        return Pipeline(
            source,
            ops,
            cache=self.registry.cache,
            chunk_shape=list(dataset.served.block),
            padding="edge",
            input_level="resample" if self.resample else "nearest",
            # The level is the one cellmap-flow chose: read as it is only at
            # the model's voxel size.
            level_rtol=1e-6,
        )

    def weights_changed(self):
        """The model's weights changed in place: layers resolved from now on,
        on every open dataset and on any opened later, are computed anew (the
        finetune loop's new iteration, under a new name)."""
        with self._lock:
            self._weights_version += 1
            for dataset in self._datasets.values():
                dataset.served.weights_version = self._weights_version

    # --- the dashboard's routes ----------------------------------------------

    def model_info(self) -> dict:
        """The served model's geometry and output activation, and the engine serving it.

        Both halves exist so the dashboard does not have to build the model
        itself: it runs on whatever node launched the jobs, often without a
        usable GPU. The geometry comes from the validated config, so it is
        reported even when the warmup probe failed; script-defined models
        expose nothing else to the dashboard, which makes this the only place
        it can learn their shapes and channel names. The activation half is
        read once, as the model is loaded (inference/runner.py), and lets the
        dashboard propose a postprocessing chain, or flag one that
        contradicts the model: ``output_class``, read off the network's graph
        (``output_class_from`` "graph", with ``output_op``, the op that ends
        it) or else from the warmup probe's range (``output_min``,
        ``output_max``) on noise ("probe", a guess); None, from neither,
        when the graph does not say and the probe's output was flat. A
        resampled input really is at input_voxel_size; a relabelled one is
        at the level's, which moves the output (see to_model_info). Those
        fields, and the axes, are the current dataset's, which ``dataset``
        names as it was last asked for.
        """
        with self._lock:
            current, requested = self._current(), self._requested
        raw, axes = current.raw, current.axes
        stored = raw.source_voxel_size
        info = self.geometry.to_model_info(axes, raw.voxel_size if raw.resampled else stored)
        info["dataset"] = requested
        info["output_axes"] = (["c"] if self.has_channel else []) + list(axes)
        info["input_resampled_from"] = list(stored) if raw.resampled else None
        relabelled = not raw.resampled and not np.allclose(
            np.asarray(stored, dtype=float), np.asarray(self.geometry.input_voxel_size, dtype=float)
        )
        info["input_relabelled_from"] = list(stored) if relabelled else None
        info["display_channel"] = getattr(self.inferencer.model_config, "display_channel", None)
        info["engine"] = ENGINE
        inferencer = self.inferencer
        output_class = getattr(inferencer, "output_class", None)
        output_range = getattr(inferencer, "output_range", None)
        if output_class is None and output_range is None:
            info.update({"available": False, "reason": "output probe did not run"})
            return info
        # Available once the graph or the probe has had its say, even when
        # neither knows (a flat probe): asking again would get the same.
        info.update({
            "available": True,
            "output_class": output_class,
            "output_class_from": getattr(inferencer, "output_class_from", None),
            "output_op": getattr(inferencer, "output_op", None),
            "output_min": output_range[0] if output_range else None,
            "output_max": output_range[1] if output_range else None,
        })
        return info

    def _routes(self):
        async def model_info(request):
            return JSONResponse(self.model_info())

        def control(action, callback):
            """POST /__control__/<action>: ``callback`` with the request's body."""

            async def handle(request):
                if callback is None:
                    return JSONResponse({"success": False, "error": f"{action.capitalize()} control not enabled"},
                                        HTTPStatus.NOT_IMPLEMENTED)
                # A restart can change what the job trains on, and a stop ends
                # its training, and this server listens on every interface, so
                # only the job manager that wrote the job's token may ask.
                if not tokens_match(self.control_token, request.headers.get(TOKEN_HEADER)):
                    return JSONResponse({"success": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
                try:
                    payload = await request.json()
                except ValueError:
                    payload = {}
                try:
                    accepted = await run_in_threadpool(callback, payload or {})
                except Exception as e:
                    logger.error(f"Failed to process {action} control request: {e}", exc_info=True)
                    return JSONResponse({"success": False, "error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                if not accepted:
                    return JSONResponse({"success": False, "error": f"{action.capitalize()} request rejected"},
                                        HTTPStatus.CONFLICT)
                return JSONResponse({"success": True})

            return handle

        async def dataset(request):
            """POST DATASET_CONTROL_PATH {"path"}: serve that dataset from now on
            (switch_dataset), answering as serving.protocol documents."""
            if not self.control_token:
                return JSONResponse({"success": False, "error": "Dataset control not enabled: this server was "
                                                                "started without a control token"},
                                    HTTPStatus.NOT_IMPLEMENTED)
            # Opening a dataset reads whatever path it is given, and this
            # server listens on every interface: only its launcher may ask.
            if not tokens_match(self.control_token, request.headers.get(TOKEN_HEADER)):
                return JSONResponse({"success": False, "error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            try:
                payload = await request.json()
            except ValueError:
                payload = None
            path = payload.get("path") if isinstance(payload, dict) else None
            if not isinstance(path, str) or not path.strip():
                return JSONResponse({"success": False, "error": 'no dataset given: the body is {"path": <its path>}'},
                                    HTTPStatus.BAD_REQUEST)
            try:
                info = await run_in_threadpool(self.switch_dataset, path)
            except Exception as e:
                logger.warning(f"Could not open {path}: {e}", exc_info=True)
                return JSONResponse({"success": False, "error": str(e) or type(e).__name__}, HTTPStatus.BAD_REQUEST)
            return JSONResponse({"success": True, "path": path, "model_info": info})

        return [
            Route("/__control__/model_info", model_info, methods=["GET"]),
            # Older name, kept for dashboards from before the geometry fields.
            Route("/__control__/output_probe", model_info, methods=["GET"]),
            Route(DATASET_CONTROL_PATH, dataset, methods=["POST"]),
            Route("/__control__/restart", control("restart", self.restart_callback), methods=["POST"]),
            Route("/__control__/stop", control("stop", self.stop_callback), methods=["POST"]),
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
