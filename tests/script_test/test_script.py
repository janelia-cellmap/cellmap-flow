import pytest
from cellmap_flow.post.postprocessors import DefaultPostprocessor, PostProcessor
from cellmap_flow.utils.data import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.globals import Flow
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DummyPostprocessor(PostProcessor):
    def _process(self, data):
        return (data * 5).astype(np.float16)

    @property
    def dtype(self):
        return np.float16


def test_fake_model_output():
    script_path = os.path.join(os.path.dirname(__file__), "fake_model_script.py")
    dummy_zarr = os.path.join(os.path.dirname(__file__), "dummy.zarr/raw")
    model_config = ScriptModelConfig(script_path=script_path)
    server = CellMapFlowServer(dummy_zarr, model_config)
    chunk_x = 2
    chunk_y = 2
    chunk_z = 2

    result = server._chunk_impl(None, None, chunk_x, chunk_y, chunk_z, None)
    encoder = server.chunk_encoder

    decoded_result = encoder.decode(result[0])
    assert np.all(decoded_result == 1), "Decoded result does not match expected output"

    expected_shape = np.array((60, 60, 60, 8))
    expected_shape = np.prod(expected_shape)
    assert (
        decoded_result.size == expected_shape
    ), f"Decoded result size {decoded_result.size} does not match expected size {expected_shape}"

    f1 = Flow()
    f2 = Flow()
    assert f1 is f2, "Flow should implement the singleton pattern"

    post = DefaultPostprocessor(0, 1, 0, 10)
    f1.postprocess = [post]

    assert (
        f1 is f2
    ), "Flow should implement the singleton pattern - should be the same after setting postprocess"

    result = server._chunk_impl(
        None, None, chunk_x, chunk_y, chunk_z, None, get_encoded=False
    )
    assert np.all(result == 10), "Simple result does not match expected output"
    f1.postprocess = [DummyPostprocessor()]
    server = CellMapFlowServer(dummy_zarr, model_config)
    encoder = server.chunk_encoder
    result2 = server._chunk_impl(
        None, None, chunk_x, chunk_y, chunk_z, None, get_encoded=True
    )
    decoded_result = encoder.decode(result2[0])
    assert np.all(decoded_result == 5), "Decoded result does not match expected output"
