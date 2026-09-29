"""Opt-in half precision: off by default, on from the config or the server's env, checked against fp32.

On a CPU autocast means bfloat16, so these test the plumbing and the fallback, not GPU numerics.
"""

import logging

import numpy as np
import pytest
import torch
from funlib.geometry import Roi

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import IDENTITY_MODEL, write_raw, write_script

# The identity, shifted by SHIFT under autocast; it records whether each call ran under it.
PROBE = IDENTITY_MODEL.replace("model = Identity()", "") + """

class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.under_autocast = []

    def forward(self, x):
        on = torch.is_autocast_enabled(x.device.type)
        self.under_autocast.append(on)
        return x + (SHIFT if on else 0.0)


model = Probe()
"""


def _config(tmp_path, shift=0.0, in_script=False):
    body = PROBE.replace("SHIFT", str(shift)) + ("half_precision = True\n" if in_script else "")
    return ScriptModelConfig(script_path=write_script(tmp_path, body), name="probe")


def _chunk(tmp_path, inferencer):
    raw = write_raw(tmp_path, np.zeros((8, 8, 8), dtype=np.uint8))
    return inferencer.process_chunk(ImageDataInterface(raw, voxel_size=(8, 8, 8)), Roi((0, 0, 0), (32, 32, 32)))


def test_off_by_default(tmp_path):
    config = _config(tmp_path)
    inferencer = Inferencer(config)
    _chunk(tmp_path, inferencer)
    assert inferencer.autocast_dtype is None
    assert not any(config.config.model.under_autocast)


def test_a_script_turns_it_on_and_chunks_run_under_autocast(tmp_path):
    config = _config(tmp_path, in_script=True)
    inferencer = Inferencer(config)
    assert inferencer.autocast_dtype == torch.bfloat16  # on a CPU
    config.config.model.under_autocast.clear()
    assert _chunk(tmp_path, inferencer).dtype == np.float32
    assert config.config.model.under_autocast == [True]


def test_it_falls_back_to_fp32_when_it_disagrees(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        inferencer = Inferencer(_config(tmp_path, shift=0.5, in_script=True))
    assert inferencer.autocast_dtype is None
    assert "probe: torch.bfloat16 output differs from fp32 by up to 0.5" in caplog.text


@pytest.mark.parametrize("value, dtype", [("1", torch.bfloat16), ("0", None)])
def test_the_server_env_var_turns_it_on(tmp_path, monkeypatch, value, dtype):
    monkeypatch.setenv("CELLMAP_FLOW_HALF_PRECISION", value)
    raw = write_raw(tmp_path, np.zeros((8, 8, 8), dtype=np.uint8))
    assert CellMapFlowServer(raw, _config(tmp_path)).inferencer.autocast_dtype == dtype
