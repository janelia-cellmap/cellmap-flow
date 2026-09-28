"""A script-defined model served through CellMapFlowServer's chunk path.

The fake model ignores its input and returns channel c filled with c + 1, in
the model convention (batch, channel, z, y, x). The server must hand back the
chunk in zarr order (z, y, x, channel), with the postprocessing chain applied.
"""

import os

import numpy as np

from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.post.postprocessors import PostProcessor
from cellmap_flow.server import CellMapFlowServer

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "fake_model_script.py")
RAW = os.path.join(HERE, "dummy.zarr/raw")
BLOCK = (10, 10, 10)
CHANNELS = 8


class TimesFive(PostProcessor):
    def _process(self, data):
        return (data * 5).astype(np.float16)

    @property
    def dtype(self):
        return np.float16


def _serve_one_chunk():
    server = CellMapFlowServer(RAW, ScriptModelConfig(script_path=SCRIPT))
    encoded, _status, _headers = server._chunk_impl("raw", 0, 2, 2, 2)
    dtype = np.dtype(g.get_output_dtype(server.output_dtype))
    decoded = np.frombuffer(server.chunk_encoder.decode(encoded), dtype=dtype)
    return decoded.reshape(*BLOCK, CHANNELS)


def test_chunk_comes_back_in_zarr_order():
    g.postprocess = []
    chunk = _serve_one_chunk()
    expected = np.arange(1, CHANNELS + 1)
    assert np.all(chunk == expected), "channel axis should be last, channel c == c + 1"


def test_postprocessing_is_applied_to_the_served_chunk():
    g.postprocess = [TimesFive()]
    chunk = _serve_one_chunk()
    assert chunk.dtype == np.float16
    assert np.all(chunk == 5 * np.arange(1, CHANNELS + 1))
