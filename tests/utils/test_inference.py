"""Running a model on a chunk: inference/runner and Inferencer.

Only the device part of a chunk waits for the GPU. A model's declared shapes
are checked on its warmup forward, half precision is opt-in and checked
against fp32, and the chain sees the chunk it is given.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from funlib.geometry import Roi

from cellmap_flow.inference.runner import DeviceSlots, predict
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.io.raw import RawInput, open_raw
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.post.postprocessors import PostProcessor
from cellmap_flow.process_chain import process_chain
from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
from tests.utils.serving_helpers import IDENTITY_MODEL

ROI = Roi((0, 0, 0), (32, 32, 32))


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _holding(slots):
    """A thread holding one of ``slots`` until the returned event is set."""
    release = threading.Event()

    def hold():
        with slots.hold():
            release.wait(5)

    threading.Thread(target=hold).start()
    return release


def test_a_chunk_is_read_while_the_device_is_busy():
    slots, read, forwarded = DeviceSlots(1), threading.Event(), threading.Event()
    idi = SimpleNamespace(to_ndarray_ts=lambda roi: read.set() or np.zeros((4, 4, 4)))
    model = SimpleNamespace(forward=lambda x: forwarded.set() or x)
    release = _holding(slots)
    _wait_until(lambda: slots._running == 1)
    call = threading.Thread(target=predict, args=(None, None, SimpleNamespace(model=model)),
                            kwargs=dict(idi=idi, device=torch.device("cpu"), device_slots=slots))
    call.start()
    assert read.wait(5) and not forwarded.wait(0.2)
    release.set()
    call.join(5)
    assert forwarded.is_set()


# The identity, shifted by $shift under autocast; it records whether each call ran under it.
PROBE = IDENTITY_MODEL.replace("model = Identity()", "") + """

class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.under_autocast = []

    def forward(self, x):
        on = torch.is_autocast_enabled(x.device.type)
        self.under_autocast.append(on)
        return x + ($shift if on else 0.0)


model = Probe()
"""


@pytest.mark.parametrize(
    "in_script, shift, env, expected",
    [
        pytest.param(False, 0.0, None, None, id="off-by-default"),
        pytest.param(True, 0.0, None, torch.bfloat16, id="a-script-turns-it-on"),  # bfloat16 on a CPU
        pytest.param(True, 0.5, None, None, id="fp32-when-the-two-disagree"),
        pytest.param(False, 0.0, "1", torch.bfloat16, id="the-servers-environment-turns-it-on"),
    ],
)
def test_half_precision_is_opt_in_and_checked_against_fp32(raw_zarr, model_script, monkeypatch, in_script, shift,
                                                            env, expected):
    monkeypatch.delenv("CELLMAP_FLOW_HALF_PRECISION", raising=False)
    if env:
        monkeypatch.setenv("CELLMAP_FLOW_HALF_PRECISION", env)
    script = model_script(PROBE + ("half_precision = True\n" if in_script else ""), shift=shift)
    config = ScriptModelConfig(script_path=script, name="probe")
    raw = raw_zarr(np.zeros((8, 8, 8), np.uint8))
    inferencer = ChunkmirageServer(raw, config).inferencer
    assert inferencer.autocast_dtype == expected
    config.config.model.under_autocast.clear()
    assert inferencer.process_chunk(RawInput(open_raw(raw, (8, 8, 8))), ROI).dtype == np.float32
    assert config.config.model.under_autocast == [expected is not None]


# An identity that keeps what it was given; $out is the output size it declares.
RECORDING = """
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate($out, $out, $out) * output_voxel_size
output_channels = 1
block_shape = np.array(($out, $out, $out, 1))


class Recording(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        return x


model = Recording()
"""


def test_a_declared_shape_the_model_does_not_produce_stops_the_server(raw_zarr, model_script):
    """The shapes are checked on the warmup forward, the only one serving
    runs before the first chunk: the check at config load ran a forward of
    its own, on zeros."""
    config = ScriptModelConfig(script_path=model_script(RECORDING, out=2))
    with pytest.raises(ValueError, match="write_shape"):
        ChunkmirageServer(raw_zarr(np.zeros((8, 8, 8), np.uint8)), config)
    (probe,) = config._config.model.inputs  # raised by the warmup's forward
    assert probe.abs().max() > 0


def test_postprocessors_space_their_ids_by_output_voxels(raw_zarr, model_script):
    """chunk_num_voxels counted input voxels, too few when the output is finer."""
    script = model_script("""
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(4, 4, 4)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(8, 8, 8) * output_voxel_size
output_channels = 1
block_shape = np.array((8, 8, 8, 1))
model = torch.nn.Upsample(scale_factor=2)
""")
    seen = []

    class Record(PostProcessor):
        def _process(self, data, chunk_num_voxels):
            seen.append(chunk_num_voxels)
            return data

    process_chain().postprocess = [Record()]
    raw = raw_zarr(np.zeros((8, 8, 8), np.uint8))
    Inferencer(ScriptModelConfig(script_path=script)).process_chunk(RawInput(open_raw(raw, (8, 8, 8))), ROI)
    assert seen == [8 * 8 * 8]
