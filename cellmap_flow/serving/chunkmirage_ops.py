"""A served model and its chains as chunkmirage ops.

chunkmirage (``chunkmirage.ops.Op``) reads the raw data around each chunk,
runs the ops on it, caches what the cached ones make and serves the result.
A layer's pipeline is ``layer_ops``:

- ``InferenceOp``: the model, on its input voxel size, with the layer's input
  normalization. Its output is cached, so a layer revisiting a region, or
  another layer with the same normalization, does not run the model again.
- ``PostprocessOp``, one per step of the layer's postprocessing chain: the
  same ``PostProcessor`` steps the dashboard's Output tab builds, run as they
  always were. Changing them leaves the model's cached output valid.

The model itself, loaded and warmed up once per process, is a ``ServedModel``
registered by name (``serve_model``): ops are frozen pydantic models that
chunkmirage rebuilds whenever a pipeline is built, so they only name it.

Importing this module is cheap (no torch), since chunkmirage loads the ops
through the ``chunkmirage.ops`` entry point in any process that serves.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
from chunkmirage.core import ArrayInfo
from chunkmirage.ops import Op
from pydantic import PrivateAttr

from cellmap_flow.image_data_interface import apply_norms
from cellmap_flow.inference import timing
from cellmap_flow.inferencer import apply_postprocess
from cellmap_flow.io.geometry import Grid
from cellmap_flow.pipeline_spec import _output_info, chain_output_dtype
from cellmap_flow.serving import virtual_zarr

logger = logging.getLogger(__name__)

# How long a POST of merged ids to the dashboard may take, and how often one goes.
EQUIVALENCES_TIMEOUT_SECONDS = 10
EQUIVALENCES_EVERY_SECONDS = 5


@dataclass
class ServedModel:
    """A model a process serves, and the grid its chunks lie on.

    ``runner``: the warmed-up Inferencer (with no device slots: the
    InferenceOp's ``slots`` bound the device). ``grid``: the raw data's voxels
    as the model reads them, in nm (``ImageDataInterface._grid``: for a level
    read as if it were at the model's voxel size, its corner rescaled too).
    ``level_shape``: that level's voxels (its resampled ones when resampled).
    ``origin``: output voxel 0's lower corner in nm, where the chunk grid
    starts. ``block``: output voxels per chunk, spatial. ``halo``: input voxels
    the model reads beyond each side of what it writes. ``ratio``: input
    voxels per output voxel.
    """

    name: str
    runner: Any
    grid: Grid
    level_shape: tuple
    origin: np.ndarray
    block: tuple
    input_voxel_size: tuple
    output_voxel_size: tuple
    halo: tuple
    ratio: tuple
    output_channels: int
    has_channel: bool
    output_dtype: Any
    stage_dtype: np.dtype
    weights_version: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def weights_changed(self):
        """Say the model's weights changed in place (a finetune iteration): pipelines
        built from now on get new cache keys, so nothing computed before is served."""
        with self._lock:
            self.weights_version += 1


_SERVED: dict = {}


def serve_model(served: ServedModel):
    _SERVED[served.name] = served


def served_model(name: str) -> ServedModel:
    try:
        return _SERVED[name]
    except KeyError:
        raise KeyError(f"no model {name!r} is served in this process; serving {sorted(_SERVED)}") from None


class BlockInput:
    """What a model reads its input from: the block chunkmirage read for a chunk.

    It stands in for the ImageDataInterface a ModelRunner reads with. Every
    config's own ``process_chunk`` or ``predict``, like the default forward,
    only calls ``to_ndarray_ts(roi)`` with the chunk grown by its context.
    """

    def __init__(self, block, block_start, grid: Grid, level_shape, input_norms):
        self.block = block
        self.block_start = np.asarray(block_start, dtype=int)
        self.grid = grid
        self.level_shape = np.asarray(level_shape, dtype=int)
        self.input_norms = list(input_norms)

    def to_ndarray_ts(self, roi):
        """``roi`` (nm) through the input chain, padded with 0 past the data,
        after the chain, as ``ImageDataInterface.to_ndarray_ts`` reads it."""
        want = self.grid.world_to_box(roi)
        begin = np.asarray(want.begin, dtype=int)
        shape = np.asarray(want.shape, dtype=int)
        lo = np.clip(begin, 0, self.level_shape)
        hi = np.maximum(np.clip(begin + shape, 0, self.level_shape), lo)
        start, stop = lo - self.block_start, hi - self.block_start
        if np.any(start < 0) or np.any(stop > np.asarray(self.block.shape)):
            raise ValueError(
                f"the model read voxels {tuple(lo)} to {tuple(hi)}, outside the block read for it "
                f"({tuple(self.block_start)} + {self.block.shape})"
            )
        data = apply_norms(self.block[tuple(slice(a, b) for a, b in zip(start, stop))], self.input_norms)
        if np.array_equal(lo, begin) and np.array_equal(hi, begin + shape):
            return data
        out = np.zeros(tuple(shape), dtype=data.dtype)
        out[tuple(slice(a, b) for a, b in zip(lo - begin, hi - begin))] = data
        return out


def _whole(values, what):
    out = np.rint(values).astype(int)
    if not np.allclose(values, out, atol=1e-6):
        raise ValueError(f"{what} {tuple(values)} is not whole voxels")
    return out


class InferenceOp(Op):
    """A served model's raw output, ``(channels, z, y, x)``, from the layer's
    input normalization of the raw data.

    One level only (``input_voxel_size``): the model's. Cached, as running the
    model is the expensive part; ``slots`` (set by the server from
    ``CELLMAP_FLOW_GPU_SLOTS``) bounds how many chunks use the device at once.
    """

    name: ClassVar[str] = "cellmap_flow.inference"
    cache: ClassVar[bool] = True
    slots: ClassVar[Optional[int]] = 1

    model: str
    input_norm: list = []

    _norms: list = PrivateAttr(default_factory=list)
    _chunks: int = PrivateAttr(0)

    def model_post_init(self, __context):
        from cellmap_flow.pipeline_spec import PipelineSpec

        self._norms = PipelineSpec(self.input_norm, ()).build()[0]

    @property
    def served(self) -> ServedModel:
        return served_model(self.model)

    @property
    def halo(self):  # type: ignore[override]
        return tuple(int(h) for h in self.served.halo)

    def input_voxel_size(self):
        return tuple(float(v) for v in self.served.input_voxel_size)

    def cache_token(self):
        return f"weights {self.served.weights_version}"

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        served = self.served
        # Relative to what is read: a level read as it is ("nearest") moves
        # the output with it.
        spatial = tuple(float(v) * r for v, r in zip(info.voxel_size[-3:], served.ratio))
        out = info.rescaled(spatial)
        if not served.has_channel:
            return out.with_(dtype=served.stage_dtype, kind="image")
        c = int(served.output_channels)
        return ArrayInfo(
            shape=(c,) + out.shape[-3:],
            dtype=served.stage_dtype,
            chunk_shape=(c,) + out.chunk_shape[-3:],
            voxel_size=(1.0,) + out.voxel_size[-3:],
            units=("",) + out.units[-3:],
            axes=("c",) + out.axes[-3:],
            translation=(0.0,) + out.translation[-3:],
            kind="image",
        )

    def apply_at(self, block, box):
        served = self.served
        start = np.asarray(box.start[-3:])
        stop = np.asarray(box.stop[-3:])
        halo = np.asarray(self.halo)
        ratio = np.asarray(served.ratio, dtype=float)
        out_start = _whole((start + halo) / ratio, "an output chunk's start")
        out_shape = _whole((stop - start - 2 * halo) / ratio, "an output chunk's shape")
        index = out_start // np.asarray(served.block)
        # The whole chunk, as the model always computed it; one the data ends
        # in is cropped after.
        roi = virtual_zarr.chunk_roi(index, served.block, served.output_voxel_size, served.origin)
        reader = BlockInput(block, start, served.grid, served.level_shape, self._norms)
        timing.start()
        began = time.perf_counter()
        result = np.asarray(served.runner.predict(reader, roi))
        stages = ", ".join(f"{k} {v:.2f}" for k, v in timing.finish().items())
        self._chunks += 1
        logger.log(
            logging.INFO if self._chunks <= 20 else logging.DEBUG,
            f"Chunk {'.'.join(map(str, index))}: model {time.perf_counter() - began:.2f} s ({stages})",
        )
        if served.has_channel and result.ndim == 3:
            result = result[None]
        crop = tuple(slice(0, n) for n in out_shape)
        return np.asarray(result[(slice(None),) * (result.ndim - 3) + crop], dtype=served.stage_dtype)


def _post_equivalences(url, payload):
    """POST a layer's merged ids to the dashboard; a failure is only logged."""
    import requests

    try:
        requests.post(url, json=payload, timeout=EQUIVALENCES_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        logger.warning(f"Could not send equivalences to {url}: {e}")


class PostprocessOp(Op):
    """One step of the layer's postprocessing chain on the model's output.

    ``step`` is the step as the Output tab sends it (``{"name": ...,
    **params}``); the op runs that ``PostProcessor``, built once per layer so
    it keeps its state (the blockwise merger's equivalences, sent to the
    dashboard at ``_dashboard_url``). A chain is one of these per step, which
    chunkmirage runs back to back on a chunk. ``chunk_corner`` is the chunk's
    index on the served grid, as Morton and the affinity segmenter key their
    ids on.
    """

    name: ClassVar[str] = "cellmap_flow.postprocess"

    model: str
    step: dict

    _step: Any = PrivateAttr(None)
    _chunks: int = PrivateAttr(0)
    _dashboard_url: Optional[str] = PrivateAttr(None)
    _dataset: str = PrivateAttr("")
    _sent_at: float = PrivateAttr(0.0)
    _send_lock: Any = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context):
        from cellmap_flow.pipeline_spec import PipelineSpec

        self._step = PipelineSpec((), (self.step,)).build()[1][0]

    def for_dashboard(self, dashboard_url, dataset):
        """Send merged ids to ``dashboard_url`` as those of layer ``dataset``."""
        self._dashboard_url, self._dataset = dashboard_url, dataset
        return self

    @property
    def served(self) -> ServedModel:
        return served_model(self.model)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        has_channel = info.ndim == 4
        dtype, channels, is_segmentation = _output_info(self._step, info.dtype, info.shape[0] if has_channel else 1)
        kind = info.kind if is_segmentation is None else ("label" if is_segmentation else "image")
        info = info.with_(dtype=np.dtype(dtype), kind=kind)
        if has_channel:
            info = info.with_(shape=(int(channels),) + info.shape[1:], chunk_shape=(int(channels),) + info.chunk_shape[1:])
        return info

    def apply_at(self, block, box):
        block_shape = np.asarray(self.served.block)
        corner = tuple(int(v) for v in np.asarray(box.start[-3:]) // block_shape)
        began = time.perf_counter()
        with timing.stage("postprocess"):
            result = self._step(block, chunk_corner=corner, chunk_num_voxels=int(np.prod(block_shape)))
        self._chunks += 1
        logger.log(
            logging.INFO if self._chunks <= 20 else logging.DEBUG,
            f"Chunk {'.'.join(map(str, corner))}: {self.step.get('name')} {time.perf_counter() - began:.2f} s",
        )
        self._send_equivalences()
        return np.asarray(result)

    def _send_equivalences(self):
        """Send the dashboard the ids the step has merged, at most once every
        EQUIVALENCES_EVERY_SECONDS, from a thread of its own so a slow or
        unreachable dashboard neither holds up the chunk nor fails it."""
        if not self._dashboard_url or getattr(self._step, "equivalences", None) is None:
            return
        with self._send_lock:
            now = time.time()
            if now - self._sent_at <= EQUIVALENCES_EVERY_SECONDS:
                return
            self._sent_at = now
        snapshot = getattr(self._step, "equivalences_json", None)
        pairs = snapshot() if snapshot else self._step.equivalences.to_json()
        payload = {"dataset": self._dataset, "equivalences": [[int(i) for i in pair] for pair in pairs]}
        url = self._dashboard_url.rstrip("/") + "/update/equivalences"
        threading.Thread(target=_post_equivalences, args=(url, payload), daemon=True).start()


def layer_ops(model, input_norm, postprocess, dashboard_url=None, dataset=""):
    """The ops of a layer's pipeline: the model on ``input_norm``, then one op
    per step of ``postprocess``, then, if the chain leaves another dtype than
    the one served, chunkmirage's ``cast`` to it (as ``astype``)."""
    from chunkmirage.ops import Cast

    served = served_model(model)
    ops = [InferenceOp(model=model, input_norm=input_norm)]
    ops += [PostprocessOp(model=model, step=step).for_dashboard(dashboard_url, dataset) for step in postprocess]
    steps = [op._step for op in ops[1:]]
    served_dtype = np.dtype(chain_output_dtype(steps, served.output_dtype))
    if np.dtype(chain_output_dtype(steps, served.stage_dtype)) != served_dtype:
        ops.append(Cast(dtype=served_dtype.name, clip=False))
    return ops
