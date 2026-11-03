"""Probing the largest block size a model's forward pass fits on a GPU.

Used from the pipeline builder: before running a blockwise task, a user can
ask "what is the biggest block this model can process on queue X" instead of
guessing, or falling back to the model's own (often much smaller) native
output tile (the default blockwise has always used, ``ModelGeometry.
block_shape()``). The live dashboard path never does this -- its chunk size
is pinned to the model's declared block_shape, to match the OME-NGFF
multiscale grid neuroglancer already fetches from -- this is for the
persistence (blockwise) step alone, where the block size is a free choice.

A candidate block size is always an integer multiple of the model's native
output tile, scaled the same on every spatial axis: multiple=1 is the
model's own declared tile, multiple=2 is 2x per axis (8x the voxels), and so
on. That keeps every candidate a valid multiple of the grid blockwise
already schedules on (``CellMapFlowBlockwiseProcessor._blocks_roi``), and
turns the search into a single integer dimension the UI can show as a plain
list of options, rather than an open-ended search over 3 independent axis
sizes.

Fitting in memory is not enough. A GPU kernel handed a tensor past its
indexing range (16 feature maps of 570^3 voxels is 2^31 elements) runs and
returns a constant: on a B300 the setup64 ER model answered 1 everywhere at
a 570-voxel tile, 0.03 correlated with itself at 514, and right up to 458,
and a blockwise task at 570 wrote a solid volume for an hour. So every
candidate runs on the same smooth synthetic image (``runner.smooth_field``),
centred on the same point, and a multiple above 1 fits only if its answer
over the native tile's voxels is the native tile's (``runner.disagreement``,
``runner.MAX_DISAGREEMENT``); the search stops at the first that is not, as
it stops at the first that runs out of memory. ``ModelRunner`` makes the
same check at its warmup for a model that knows its reference tile
(``_check_tile_agreement``), so a tile typed by hand fails at the start
rather than writing garbage.

This must run in a process that actually has the GPU in question: torch
reports real CUDA memory only for the device it runs on. The dashboard
submits it as a short job to the requested queue (dashboard.routes.blockwise
computes_max_block_size), rather than running it in the dashboard's own
process, which usually has no GPU at all.
"""

import dataclasses
import gc
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# A model that fits even this many times its native tile is already well
# past where a bigger block helps much (blockwise's parallelism granularity
# gets coarser as blocks get bigger, for diminishing GPU-utilization gain);
# this is a safety cap on the search, not a claim that more could never fit.
DEFAULT_MAX_MULTIPLE = 8


def _is_oom(err: Exception) -> bool:
    message = str(err).lower()
    return "out of memory" in message or ("cuda" in message and "memory" in message)


def _explain(err: Exception, props) -> str:
    """What to record for a forward pass that failed other than by running
    out of memory: the error's first line (CUDA's come with three lines of
    debugging boilerplate), plus, for "no kernel image is available", what
    that actually means -- this torch build has no kernels for this GPU's
    architecture at all. That is what every Blackwell queue (gpu_b300's
    sm_103, gpu_rtx6000's sm_120) says to a torch built for CUDA 12.4 or
    older, and nothing about the block size: shown as a bare "does not fit"
    it reads as a memory limit and sends the user looking in the wrong place.
    """
    import torch

    first_line = next((line for line in str(err).strip().splitlines() if line.strip()), repr(err)).strip()
    if "no kernel image" not in first_line:
        return first_line
    capability = f"sm_{props.major}{props.minor}"
    supported = ", ".join(torch.cuda.get_arch_list()) or "none"
    return (
        f"{first_line}. Not a memory limit: this environment's torch {torch.__version__} "
        f"has no kernels for {props.name} ({capability}; it was built for {supported}), "
        f"so no block size fits on this queue. A torch built for a newer CUDA is needed "
        f"to use this GPU at all, or pick a queue whose GPU torch supports."
    )


class _SyntheticInput:
    """A synthetic raw input for the probe: a smooth image in [-1, 1] that is
    a function of world position (``runner.smooth_field``), so a voxel reads
    the same whatever tile asks for it, and a bigger tile's prediction can be
    held against the native tile's over the voxels they share.

    Only for the probe: the model's real read/normalize chain is never
    exercised, only its forward pass at a given output size.
    ``to_ndarray_ts`` is the one method ``ModelRunner.predict`` needs from
    an idi.
    """

    def __init__(self, voxel_size):
        self.voxel_size = np.asarray(voxel_size, dtype=float)

    def to_ndarray_ts(self, roi) -> np.ndarray:
        from cellmap_flow.inference.runner import smooth_field

        begin = np.asarray(roi.begin, dtype=float)
        shape = tuple(int(round(s)) for s in np.asarray(roi.shape, dtype=float) / self.voxel_size)
        return smooth_field([begin[i] + self.voxel_size[i] * (np.arange(shape[i]) + 0.5) for i in range(len(shape))])


@dataclasses.dataclass
class BlockSizeCandidate:
    """One multiple of the model's native output tile the probe tried.

    ``fits`` is whether the forward pass ran at this size and, for a
    multiple above 1, answered what the native tile answers over the voxels
    they share (``runner.disagreement``): a GPU kernel can run a tensor past
    its indexing range and return a constant, which fits in memory and is
    wrong everywhere. ``agreement`` is that comparison's correlation with
    the native tile's output (None for the native tile, for a model whose
    output is flat on the probe's synthetic image, and in results probed
    before this was checked).
    """

    multiple: int
    block_shape: Tuple[int, ...]  # (*spatial, channels), voxels
    fits: bool
    peak_bytes: Optional[int] = None
    error: Optional[str] = None
    agreement: Optional[float] = None

    def __post_init__(self):
        # Normalized once, so a candidate read back from the JSON cache
        # (where it round-tripped through a list) still equals a fresh one.
        self.block_shape = tuple(self.block_shape)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MaxBlockSizeResult:
    gpu_name: str
    total_memory_bytes: int
    # multiple=1, 2, 3, ... in order; stops at the first that does not fit
    # (or DEFAULT_MAX_MULTIPLE), so every entry before the last one fits.
    candidates: List[BlockSizeCandidate]

    @property
    def best(self) -> Optional[BlockSizeCandidate]:
        """The largest multiple that fit, or None if even the native tile did not."""
        fitting = [c for c in self.candidates if c.fits]
        return fitting[-1] if fitting else None

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "total_memory_bytes": self.total_memory_bytes,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaxBlockSizeResult":
        return cls(
            gpu_name=data["gpu_name"],
            total_memory_bytes=data["total_memory_bytes"],
            candidates=[BlockSizeCandidate(**c) for c in data["candidates"]],
        )


def probe_max_block_size(model_config, max_multiple: int = DEFAULT_MAX_MULTIPLE) -> MaxBlockSizeResult:
    """Try the model's native output tile scaled by 1, 2, 3, ... on this
    process's GPU, stopping at the first multiple that does not fit (an
    out-of-memory error, or a shape the network refuses) or ``max_multiple``.

    Raises:
        RuntimeError: no CUDA device is available in this process.
    """
    import daisy
    import torch

    from cellmap_flow.inference.runner import ModelRunner, disagreement

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; the probe must run on a GPU node")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    runner = ModelRunner(model_config)
    geometry = model_config.geometry
    base_shape = np.array(geometry.output_shape)  # voxels, spatial only
    channels = geometry.output_channels
    idi = _SyntheticInput(geometry.input_voxel_size)

    candidates: List[BlockSizeCandidate] = []
    # The native tile's answer, which every bigger tile is held against.
    reference = None
    for multiple in range(1, max_multiple + 1):
        spatial = tuple(int(v) for v in base_shape * multiple)
        block_shape = (*spatial, channels)
        write_shape_nm = tuple(float(v) for v in np.array(spatial) * np.array(geometry.output_voxel_size))
        # Every candidate centred on the same world point, so the native
        # tile's output is the centre of each bigger one's.
        out_roi = daisy.Roi(tuple(-v / 2 for v in write_shape_nm), write_shape_nm)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            output = runner.predict(idi, out_roi)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            if multiple == 1:
                reference = None if output is None else np.array(output, dtype=np.float32, copy=True)
                agreement, relative, wrong = None, 0.0, False
            else:
                agreement, relative, wrong = disagreement(reference, output)
            if wrong:
                # Ran, and fit -- and answered something else than the model
                # does at its own tile: a kernel past its indexing range (16
                # feature maps of 570^3 voxels is 2^31 elements) returns a
                # constant, which thresholds to a solid volume. Bigger tiles
                # only get further past it.
                error = (
                    f"runs, but answers differently from the model's own tile ({[int(v) for v in base_shape]}) over the "
                    f"same voxels: correlation {agreement if agreement is not None else 'none'}, difference "
                    f"{relative:.1f}x the native output's spread. A tensor this size is past what the GPU "
                    f"kernels index correctly; use a smaller multiple."
                )
                candidates.append(BlockSizeCandidate(multiple, block_shape, False, peak_bytes=peak, error=error,
                                                     agreement=agreement))
                logger.warning(f"Block size x{multiple} {block_shape}: {error}")
                break
            candidates.append(BlockSizeCandidate(multiple, block_shape, True, peak_bytes=peak, agreement=agreement))
            logger.info(
                f"Block size x{multiple} {block_shape}: fits, peak "
                f"{peak / (1024 ** 3):.2f} GiB ({100 * peak / props.total_memory:.1f}%)"
                + (f", agrees with the native tile ({agreement:.3f})" if agreement is not None else "")
            )
        except RuntimeError as e:
            oom = _is_oom(e)
            error = "out of memory" if oom else _explain(e, props)
            candidates.append(BlockSizeCandidate(multiple, block_shape, False, error=error))
            if oom:
                logger.info(f"Block size x{multiple} {block_shape}: does not fit (out of memory)")
            else:
                # Not a memory ceiling -- the network refused this shape, or
                # something else broke. Worth the full traceback: silently
                # stopping the search here with only the one-line message
                # once hid a real bug (and looked like an early, confusing
                # stop well under the GPU's capacity).
                logger.exception(
                    f"Block size x{multiple} {block_shape}: does not fit (not an out-of-memory error)"
                )
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return MaxBlockSizeResult(gpu_name=props.name, total_memory_bytes=props.total_memory, candidates=candidates)
