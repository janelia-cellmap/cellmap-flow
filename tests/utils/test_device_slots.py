"""The inference server's device slots: first come first served, bounded, device part only."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cellmap_flow.inference.runner import DeviceSlots, predict


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_slots_admit_in_arrival_order():
    slots = DeviceSlots(1)
    release, entered = threading.Event(), []

    def gate():
        with slots.hold():
            release.wait(5)

    def worker(i):
        with slots.hold():
            entered.append(i)

    threads = [threading.Thread(target=gate)]
    threads[0].start()
    _wait_until(lambda: slots._running == 1)
    for i in range(5):
        threads.append(threading.Thread(target=worker, args=(i,)))
        threads[-1].start()
        _wait_until(lambda: len(slots._waiting) == i + 1)
    release.set()
    [t.join(5) for t in threads]
    assert entered == [0, 1, 2, 3, 4]


def test_never_more_than_n_hold_a_slot():
    slots, lock = DeviceSlots(2), threading.Lock()
    now, peak = [0], [0]

    def worker():
        with slots.hold():
            with lock:
                now[0] += 1
                peak[0] = max(peak[0], now[0])
            time.sleep(0.01)
            with lock:
                now[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(12)]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert peak[0] == 2


def test_predict_reads_while_the_slot_is_busy():
    """Only the device part waits for a slot; the read overlaps."""
    slots, release = DeviceSlots(1), threading.Event()
    read, forwarded = threading.Event(), threading.Event()
    idi = SimpleNamespace(to_ndarray_ts=lambda roi: read.set() or np.zeros((4, 4, 4)))
    model = SimpleNamespace(forward=lambda x: forwarded.set() or x)

    def gate():
        with slots.hold():
            release.wait(5)

    threading.Thread(target=gate).start()
    _wait_until(lambda: slots._running == 1)
    call = threading.Thread(
        target=predict,
        args=(None, None, SimpleNamespace(model=model)),
        kwargs=dict(idi=idi, device=torch.device("cpu"), device_slots=slots),
    )
    call.start()
    assert read.wait(5) and not forwarded.wait(0.2)
    release.set()
    call.join(5)
    assert forwarded.is_set()


@pytest.mark.parametrize("value", ["0", "two"])
def test_a_bad_slot_count_is_refused(monkeypatch, value):
    monkeypatch.setenv("CELLMAP_FLOW_GPU_SLOTS", value)
    with pytest.raises(ValueError, match="CELLMAP_FLOW_GPU_SLOTS"):
        DeviceSlots.from_env()
