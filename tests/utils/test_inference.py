"""Running a model on a chunk: inference/runner and Inferencer.

Only the device part of a chunk waits for the GPU, in arrival order, and a
chunk nobody waits for any more is not computed. A model's declared shapes are
checked on its warmup forward, half precision is opt-in and checked against
fp32, and the chain sees the chunk it is given.
"""

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch
from funlib.geometry import Roi
from werkzeug.serving import make_server

from cellmap_flow.globals import g
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inference.runner import DeviceSlots, ModelRunner, predict
from cellmap_flow.inferencer import Inferencer
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.post.postprocessors import PostProcessor, ThresholdPostprocessor
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import IDENTITY_MODEL, layer

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


def _enter(slots, entered, i):
    with slots.hold():
        entered.append(i)


def test_device_slots_are_first_come_first_served_and_bounded(monkeypatch):
    slots, entered = DeviceSlots(1), []
    release = _holding(slots)
    _wait_until(lambda: slots._running == 1)
    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=_enter, args=(slots, entered, i)))
        threads[-1].start()
        _wait_until(lambda: len(slots._waiting) == i + 1)
    release.set()
    [t.join(5) for t in threads]
    assert entered == [0, 1, 2, 3, 4]

    slots, lock, now, peak = DeviceSlots(2), threading.Lock(), [0], [0]

    def work():
        with slots.hold():
            with lock:
                now[0] += 1
                peak[0] = max(peak[0], now[0])
            time.sleep(0.01)
            with lock:
                now[0] -= 1

    threads = [threading.Thread(target=work) for _ in range(12)]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert peak[0] == 2

    for value in ("0", "two"):
        monkeypatch.setenv("CELLMAP_FLOW_GPU_SLOTS", value)
        with pytest.raises(ValueError, match="CELLMAP_FLOW_GPU_SLOTS"):
            DeviceSlots.from_env()


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


def test_a_chunk_whose_client_left_is_not_computed(raw_zarr, model_script):
    config = ScriptModelConfig(script_path=model_script())
    server = CellMapFlowServer(raw_zarr(np.zeros((8, 8, 8), np.uint8)), config)
    slots, release, computed = server.inferencer.device_slots, threading.Event(), []

    def forward(x):  # the first chunk keeps the only slot until released
        computed.append(x.shape)
        release.wait(5)
        return x

    config.config.model.forward = forward
    http = make_server("127.0.0.1", 0, server.app, threaded=True)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    path = f"/{layer()}/s0"
    url = f"http://127.0.0.1:{http.server_port}{path}"
    answers = []
    try:
        holder = threading.Thread(target=lambda: answers.append(requests.get(f"{url}/0.0.0.0")))
        holder.start()
        _wait_until(lambda: computed)
        gone = socket.create_connection(("127.0.0.1", http.server_port))
        gone.sendall(f"GET {path}/1.0.0.0 HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        _wait_until(lambda: len(slots._waiting) == 1)
        stays = threading.Thread(target=lambda: answers.append(requests.get(f"{url}/0.1.0.0")))
        stays.start()
        _wait_until(lambda: len(slots._waiting) == 2)
        gone.close()
        _wait_until(lambda: len(slots._waiting) == 1)  # the one that left gave up its place
        release.set()
        holder.join(10)
        stays.join(10)
    finally:
        release.set()
        http.shutdown()
    assert [a.status_code for a in answers] == [200, 200]
    assert len(computed) == 2  # the holder's and the one that stayed, not the one that left


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
        (False, 0.0, None, None),  # off by default
        (True, 0.0, None, torch.bfloat16),  # a script turns it on; bfloat16 on a CPU
        (True, 0.5, None, None),  # fp32 when the two disagree
        (False, 0.0, "1", torch.bfloat16),  # so does the server's environment
        (False, 0.0, "0", None),
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
    inferencer = CellMapFlowServer(raw, config).inferencer
    assert inferencer.autocast_dtype == expected
    config.config.model.under_autocast.clear()
    assert inferencer.process_chunk(ImageDataInterface(raw, voxel_size=(8, 8, 8)), ROI).dtype == np.float32
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


def test_declared_shapes_are_checked_on_the_warmup_forward(raw_zarr, model_script):
    config = ScriptModelConfig(script_path=model_script(RECORDING, out=4))
    inferencer = Inferencer(config)
    (probe,) = config.config.model.inputs  # one forward, the warmup's, on the device
    assert probe.device == inferencer.device and probe.abs().max() > 0, "the probe input, not zeros"

    # A shape the model does not produce stops the server, from that same forward.
    wrong = ScriptModelConfig(script_path=model_script(RECORDING, name="wrong.py", out=2))
    with pytest.raises(ValueError, match="(?s)shape validation failed.*write_shape mismatch"):
        CellMapFlowServer(raw_zarr(np.zeros((8, 8, 8), np.uint8)), wrong)
    (probe,) = wrong._config.model.inputs
    assert probe.abs().max() > 0
    # A config read without an inferencer checks on its own.
    alone = ScriptModelConfig(script_path=model_script(RECORDING, name="alone.py", out=2))
    with pytest.raises(ValueError, match="write_shape mismatch"):
        alone.config
    assert len(alone._config.model.inputs) == 1


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

    g.postprocess = [Record()]
    raw = raw_zarr(np.zeros((8, 8, 8), np.uint8))
    Inferencer(ScriptModelConfig(script_path=script)).process_chunk(ImageDataInterface(raw, voxel_size=(8, 8, 8)), ROI)
    assert seen == [8 * 8 * 8]


def test_a_model_runner_returns_the_models_own_output(raw_zarr, model_script):
    """No chain and nothing from g: the model's output for the region, read with its context."""
    data = (np.arange(512) % 251).astype(np.uint8).reshape(8, 8, 8)
    script = model_script(IDENTITY_MODEL.replace("read_shape = Coordinate(4, 4, 4)", "read_shape = Coordinate(6, 6, 6)")
                          + "\nmodel.forward = lambda x: x[:, :, 1:-1, 1:-1, 1:-1] * 2\n")
    g.postprocess = [ThresholdPostprocessor(threshold=0.5)]
    runner = ModelRunner(ScriptModelConfig(script_path=script))
    out = runner.predict(ImageDataInterface(raw_zarr(data), voxel_size=(8, 8, 8), input_norms=[]),
                         Roi((8, 8, 8), (32, 32, 32)))
    assert out.shape == (1, 4, 4, 4) and out.dtype == np.float32
    assert np.array_equal(out[0], 2.0 * data[1:5, 1:5, 1:5])
