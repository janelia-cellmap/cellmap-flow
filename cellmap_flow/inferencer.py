"""Predicting a chunk and postprocessing it: the Inferencer.

The model itself (the device, the warmup, the forward and the device
slots) is ``inference.runner.ModelRunner``; an Inferencer is a ModelRunner
that also applies a normalization and postprocessing chain, the layer's own
or the process's (``process_chain()``).

``predict(read_roi, write_roi, config, **kwargs)`` and ``apply_postprocess``
are what model scripts are written against, so both stay importable from
here; ``predict`` itself is the runner's default forward.
"""

import logging

import numpy as np

from cellmap_flow.inference import timing
from cellmap_flow.inference.runner import ModelRunner
from cellmap_flow.inference.runner import predict  # noqa: F401  (the script contract; WRAPPERS.md)
from cellmap_flow.process_chain import process_chain

logger = logging.getLogger(__name__)


def apply_postprocess(data, postprocess=None, **kwargs):
    """Run ``data`` through a postprocessing chain.

    ``postprocess=None`` means the process's chain,
    ``process_chain().postprocess``; the server passes the chain of the layer
    being requested instead.
    """
    for pross in process_chain().postprocess if postprocess is None else postprocess:
        data = pross(data, **kwargs)
    return data


def _chunk_index(roi, grid_origin=None):
    """``roi``'s index on the grid of ``roi``-sized chunks starting at ``grid_origin`` (nm).

    The ops that give each chunk its own label ids (Morton, Affinity) key them
    on this index, and Morton keeps 10 bits of each axis, so it must count
    from the grid's first chunk: on data whose corner is -4 nm an index taken
    from 0 made the first chunk (-1, -1, -1), which Morton reads as 1023 on
    every axis.

    A ``roi`` off that grid (a blockwise box's block, laid from the box's
    corner) gets the index of the cell its corner is in, floored. Blocks of
    one size that do not overlap are a block apart on some axis, so their
    indices still differ, and neighbours' differ by 1.
    """
    begin = np.array(roi.get_begin()) - (0 if grid_origin is None else np.array(grid_origin))
    return tuple(int(v) for v in begin // np.array(roi.get_shape()))


class Inferencer(ModelRunner):
    """A ModelRunner that normalizes its input and postprocesses its output."""

    def process_chunk(self, idi, roi, input_norms=None, postprocess=None, grid_origin=None):
        """Predict ``roi``, read from ``idi``, and postprocess it.

        ``idi`` is what the model reads from: a ``RawInput``
        (``cellmap_flow.io.raw``), or the server's ``BlockInput``, which
        already reads through the layer's chain.

        ``input_norms`` / ``postprocess``: the chain to use for this chunk; a
        ``RawInput`` is read through ``input_norms`` (``with_input_norms``).
        ``None`` falls back to the process's chain (``process_chain()``), for
        callers (scripts) that set the chain process-wide.

        ``grid_origin``: where (nm) the chunk grid ``roi`` lies on starts, so
        the steps get ``roi``'s index on that grid as ``chunk_corner``; None
        means a grid starting at 0. The server and blockwise both pass the
        raw data's corner (see ``_chunk_index``), so a region gets the same
        unique label ids from either where their chunks are the same: the
        whole volume, or a box on that grid.
        """
        if input_norms is not None and hasattr(idi, "with_input_norms"):
            idi = idi.with_input_norms(input_norms)

        result = self.predict(idi, roi)

        with timing.stage("postprocess"):
            postprocessed = apply_postprocess(
                result,
                postprocess=postprocess,
                chunk_corner=_chunk_index(roi, grid_origin),
                chunk_num_voxels=self._output_voxels_in(roi, idi),
            )
        return postprocessed

    def _output_voxels_in(self, roi, idi):
        """How many output voxels ``roi`` holds, the spacing for unique label ids.

        This once used the raw reader's output voxel size, which the server
        and blockwise left at the input voxel size, so a model whose output is
        finer than its input spaced ids too closely and neighbouring chunks
        collided. It is the model's.
        """
        shape = np.array(roi.get_shape(), dtype=float) / np.array(
            self.model_config.geometry.output_voxel_size, dtype=float
        )
        return int(np.prod(np.ceil(shape)))
