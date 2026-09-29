"""A chunk whose client hung up before its turn on the device is not computed."""

import socket
import threading
import time

import numpy as np
import requests
from werkzeug.serving import make_server

from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import IDENTITY_MODEL, layer, write_raw, write_script


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_a_chunk_whose_client_left_is_skipped(tmp_path):
    raw = write_raw(tmp_path, np.zeros((8, 8, 8), dtype=np.uint8))
    config = ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL))
    server = CellMapFlowServer(raw, config)
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
