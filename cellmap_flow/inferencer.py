"""Predicting a chunk and postprocessing it: the Inferencer.

The model itself (the device, the warmup, the forward and the device
slots) is ``inference.runner.ModelRunner``; an Inferencer is a ModelRunner
that also applies a normalization and postprocessing chain, the layer's own
or the process-wide one in ``g``.

``predict(read_roi, write_roi, config, **kwargs)`` and ``apply_postprocess``
are what model scripts are written against, so both stay importable from
here; ``predict`` itself is the runner's default forward.
"""

import logging

import numpy as np

from cellmap_flow.globals import g
from cellmap_flow.inference.runner import ModelRunner
from cellmap_flow.inference.runner import predict  # noqa: F401  (the script contract; WRAPPERS.md)

logger = logging.getLogger(__name__)


def apply_postprocess(data, postprocess=None, **kwargs):
    """Run ``data`` through a postprocessing chain.

    ``postprocess=None`` means the process-wide ``g.postprocess``; the server
    passes the chain of the layer being requested instead.
    """
    for pross in g.postprocess if postprocess is None else postprocess:
        data = pross(data, **kwargs)
    return data


class Inferencer(ModelRunner):
    """A ModelRunner that normalizes its input and postprocesses its output."""

    def process_chunk(self, idi, roi, input_norms=None, postprocess=None, cancelled=None):
        """Predict ``roi`` and postprocess it.

        ``input_norms`` / ``postprocess``: the chain to use for this chunk.
        ``None`` falls back to ``g.input_norms`` / ``g.postprocess``, for
        callers (scripts) that set the chain process-wide.

        ``cancelled``: asked while the chunk waits for a device slot; raises
        ChunkCancelled, without computing it, once that says yes.
        """
        if input_norms is not None and hasattr(idi, "with_input_norms"):
            idi = idi.with_input_norms(input_norms)

        result = self.predict(idi, roi, cancelled)

        postprocessed = apply_postprocess(
            result,
            postprocess=postprocess,
            chunk_corner=tuple(roi.get_begin() // roi.get_shape()),
            chunk_num_voxels=self._output_voxels_in(roi, idi),
        )
        return postprocessed

    def _output_voxels_in(self, roi, idi):
        """How many output voxels ``roi`` holds, the spacing for unique label ids.

        This used the IDI's output voxel size, which the server and blockwise
        leave at the input voxel size, so a model whose output is finer than
        its input spaced ids too closely and neighbouring chunks collided.
        """
        shape = np.array(roi.get_shape(), dtype=float) / np.array(
            self.model_config.geometry.output_voxel_size, dtype=float
        )
        return int(np.prod(np.ceil(shape)))
