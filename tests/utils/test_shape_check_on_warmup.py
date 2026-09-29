"""A served model's declared shapes are checked on its warmup forward, not by a forward of their own."""

import numpy as np
import pytest

from cellmap_flow.inferencer import Inferencer
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import write_raw, write_script

# An identity model that remembers what it was given; OUT is the declared output size.
RECORDING_MODEL = """
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(OUT, OUT, OUT) * output_voxel_size
output_channels = 1
block_shape = np.array((OUT, OUT, OUT, 1))


class Recording(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        return x


model = Recording()
"""


def _config(tmp_path, out):
    return ScriptModelConfig(script_path=write_script(tmp_path, RECORDING_MODEL.replace("OUT", str(out))))


def test_serving_runs_one_forward_the_warmup(tmp_path):
    config = _config(tmp_path, 4)
    inferencer = Inferencer(config)
    inputs = config.config.model.inputs
    assert len(inputs) == 1
    assert inputs[0].device == inferencer.device
    # The warmup's probe input, not the zeros the check at config load fed it.
    assert inputs[0].abs().max() > 0


def test_a_shape_mismatch_still_stops_the_server(tmp_path):
    raw = write_raw(tmp_path, np.zeros((8, 8, 8), dtype=np.uint8))
    config = _config(tmp_path, 2)
    with pytest.raises(ValueError, match="(?s)shape validation failed.*write_shape mismatch"):
        CellMapFlowServer(raw, config)
    (probe,) = config._config.model.inputs  # raised by the warmup's forward
    assert probe.abs().max() > 0


def test_a_config_built_without_an_inferencer_checks_on_its_own(tmp_path):
    config = _config(tmp_path, 2)
    with pytest.raises(ValueError, match="write_shape mismatch"):
        config.config
    assert len(config._config.model.inputs) == 1
