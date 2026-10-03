"""The chunkmirage engine (serving.chunkmirage_server) serves what the Flask server served.

Each case is checked against CellMapFlowServer on the same model and data:
the same voxels (channels first rather than last), placed at the same place.
"""

import json
import threading
import time

import numcodecs
import numpy as np
import pytest
import requests
from starlette.testclient import TestClient

from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import (
    LabelPostprocessor,
    MortonSegmentationRelabeling,
    ThresholdPostprocessor,
)
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.serving.chunkmirage_ops import InferenceOp
from cellmap_flow.serving.chunkmirage_server import INFERENCE_PREFETCH, ChunkmirageServer
from cellmap_flow.serving.engine import check_server, engine_name, make_server
from cellmap_flow.serving.protocol import ARGS_KEY
from cellmap_flow.serving.restart_token import TOKEN_HEADER
from tests.utils.serving_helpers import IDENTITY_MODEL, decode_chunk, get_json, layer, write_raw, write_script
from tests.utils.test_served_metadata_snapshot import CASES


def _flask_chunk(server, client, name, key):
    meta = get_json(client, f"/{name}/s0/.zarray")
    response = client.get(f"/{name}/s0/{key}")
    assert response.status_code == 200, response.data
    data = decode_chunk(server, response.data, meta["dtype"], meta["chunks"])
    return np.moveaxis(data, -1, 0) if data.ndim == 4 else data


def _chunk(client, name, index):
    """Chunk ``index`` (z, y, x) of layer ``name``, as chunkmirage serves it."""
    meta = client.get(f"/{name}/zarr/s0/.zarray").json()
    key = "/".join(map(str, ((0,) if len(meta["shape"]) == 4 else ()) + tuple(index)))
    response = client.get(f"/{name}/zarr/s0/{key}")
    assert response.status_code == 200, response.text
    raw = numcodecs.get_codec(meta["compressor"]).decode(response.content) if meta["compressor"] else response.content
    shape = [min(c, s - i * c) for c, s, i in zip(meta["chunks"][-3:], meta["shape"][-3:], index)]
    return np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["chunks"])[
        (slice(None),) * (len(meta["shape"]) - 3) + tuple(slice(0, n) for n in shape)
    ]


def _servers(path, script):
    config = lambda: ScriptModelConfig(script_path=script)  # noqa: E731
    flask = CellMapFlowServer(path, config())
    mirage = ChunkmirageServer(path, config())
    return flask, flask.app.test_client(), mirage, TestClient(mirage.app)


CHAINS = [(case, chain) for case, (_, _, chains, _) in CASES.items() for chain in ["plain", *chains]]


@pytest.mark.parametrize("case, chain", [pytest.param(case, chain, id=f"{case}-{chain}") for case, chain in CHAINS])
def test_each_model_and_chain_gives_the_voxels_the_flask_server_gave(tmp_path, case, chain):
    script, raw, chains, key = CASES[case]
    flask, old, mirage, new = _servers(raw(tmp_path), write_script(tmp_path, script))
    name = layer(posts=chains.get(chain, []))
    index = tuple(int(v) for v in key.split(".")[:3])

    expected = _flask_chunk(flask, old, name, key)
    got = _chunk(new, name, index)

    assert got.dtype == expected.dtype and got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("case", list(CASES))
def test_the_array_lies_where_the_flask_server_put_it(tmp_path, case):
    script, raw, _, _ = CASES[case]
    flask, old, mirage, new = _servers(raw(tmp_path), write_script(tmp_path, script))
    name = layer()

    def placed(attrs):
        multiscale = attrs["multiscales"][0]
        names = [a["name"] for a in multiscale["axes"]]
        transforms = {t["type"]: t for t in multiscale["datasets"][0]["coordinateTransformations"]}
        translation = transforms.get("translation", {}).get("translation", [0.0] * len(names))
        return {n: (transforms["scale"]["scale"][i], translation[i]) for i, n in enumerate(names) if n in "zyx"}

    info = new.get("/__control__/model_info").json()
    expected = placed(get_json(old, f"/{name}/.zattrs"))
    # A level read as if it were at the model's voxel size is served where it
    # really lies; the Flask server left that to the viewer's override.
    effective = info["effective_output_voxel_size"]
    expected = {a: (v, t / s * v) for (a, (s, t)), v in zip(expected.items(), effective)}
    assert placed(new.get(f"/{name}/zarr/.zattrs").json()) == expected


def test_model_info_says_what_the_flask_server_said_and_which_engine(tmp_path):
    script, raw, _, _ = CASES["8nm_to_16nm"]
    flask, old, mirage, new = _servers(raw(tmp_path), write_script(tmp_path, script))
    before, after = get_json(old, "/__control__/model_info"), new.get("/__control__/model_info").json()

    assert after.pop("engine") == "chunkmirage"
    assert after.pop("output_axes") == ["c", "z", "y", "x"]  # channels first, as served
    before.pop("output_axes")
    assert after.keys() == before.keys()
    assert {k: v for k, v in after.items() if k not in ("output_min", "output_max")} == {
        k: v for k, v in before.items() if k not in ("output_min", "output_max")
    }


def test_a_chunk_the_data_ends_in_is_computed_whole_and_cropped(tmp_path):
    """10 voxels in 4-voxel chunks: the last chunk holds 2, but the model
    still gets the 4 (and its context) it always did."""
    data = (np.arange(10**3) % 251).astype(np.uint8).reshape((10,) * 3)
    crop = CASES["corner_at_-4nm"][0]
    flask, old, mirage, new = _servers(write_raw(tmp_path, data), write_script(tmp_path, crop))
    name = layer()

    got = _chunk(new, name, (2, 2, 1))

    assert got.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(got, _flask_chunk(flask, old, name, "2.2.1.0")[:, :2, :2, :4])


OWN_PROCESS_CHUNK = """
input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(6, 6, 6) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 2
block_shape = np.array((4, 4, 4, 2))


class Crop(torch.nn.Module):  # the shapes the warmup checks; process_chunk is what serves
    def forward(self, x):
        return x[:, :, 1:-1, 1:-1, 1:-1].repeat(1, 2, 1, 1, 1)


model = Crop()


def process_chunk(idi, roi):
    context = (read_shape - write_shape) / 2
    data = idi.to_ndarray_ts(roi.grow(context, context)).astype(np.float32)
    inner = data[1:-1, 1:-1, 1:-1]
    # The neighbour on each side too, so a read in the wrong place shows.
    return np.stack([inner, data[2:, :-2, 1:-1] - data[:-2, 2:, 1:-1]])
"""


def test_a_config_reading_its_own_input_reads_the_same_voxels(tmp_path):
    """Cellpose, the BioImage zoo and model scripts read their input
    themselves, from the chunk grown by their context."""
    flask, old, mirage, new = _servers(write_raw(tmp_path, (np.arange(12**3) % 251).astype(np.uint8).reshape((12,) * 3)),
                                       write_script(tmp_path, OWN_PROCESS_CHUNK))
    name = layer()
    for key in ("0.0.0.0", "1.2.0.0", "2.2.2.0"):
        index = tuple(int(v) for v in key.split(".")[:3])
        np.testing.assert_array_equal(_chunk(new, name, index), _flask_chunk(flask, old, name, key))


def test_ids_that_depend_on_the_chunk_are_the_flask_servers(tmp_path):
    """Morton keys each chunk's ids on its index on the served grid."""
    data = np.zeros((16,) * 3, np.uint8)
    data[1:3, 1:3, 1:3] = data[9:12, 5:7, 13:15] = 200
    flask, old, mirage, new = _servers(write_raw(tmp_path, data), write_script(tmp_path, IDENTITY_MODEL))
    name = layer(posts=[ThresholdPostprocessor(threshold=100), LabelPostprocessor(), MortonSegmentationRelabeling()])
    for key in ("0.0.0.0", "2.1.3.0"):
        index = tuple(int(v) for v in key.split(".")[:3])
        got = _chunk(new, name, index)
        assert got.dtype == np.uint64 and got.max() > 0
        np.testing.assert_array_equal(got, _flask_chunk(flask, old, name, key))


COUNTING_MODEL = IDENTITY_MODEL.replace("""class Identity(torch.nn.Module):
    def forward(self, x):
        return x""", """class Identity(torch.nn.Module):
    calls = 0

    def forward(self, x):
        Identity.calls += 1
        return x""")


def _forwards(server):
    return type(server.inferencer.model_config.config.model).calls


def test_the_models_output_is_kept_for_revisits_and_postprocessing_changes(tmp_path):
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, COUNTING_MODEL)))
    client = TestClient(mirage.app)
    start = _forwards(mirage)

    _chunk(client, layer(), (0, 0, 0))
    _chunk(client, layer(), (0, 0, 0))  # panned back
    _chunk(client, layer(posts=[ThresholdPostprocessor(threshold=5)]), (0, 0, 0))  # Output tab changed

    assert _forwards(mirage) - start == 1


def test_a_chain_sent_again_with_its_values_as_text_reuses_the_models_output(tmp_path):
    """The dashboard's forms send "0.0" where the YAML gave 0.0: submitting a
    postprocessing step on the Output tab ran the model again on every chunk."""
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, COUNTING_MODEL)))
    client = TestClient(mirage.app)
    name = lambda norm, posts=(): f"m{ARGS_KEY}{PipelineSpec([norm], list(posts)).to_url_blob()}{ARGS_KEY}"  # noqa: E731
    start = _forwards(mirage)

    _chunk(client, name({"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}), (0, 0, 0))
    _chunk(client, name({"name": "MinMaxNormalizer", "min_value": "0.0", "max_value": "255.0", "invert": "False"},
                        [{"name": "ThresholdPostprocessor", "threshold": "0.5"}]), (0, 0, 0))

    assert _forwards(mirage) - start == 1


def test_the_next_chunks_are_normalized_while_the_device_runs_a_forward(tmp_path, monkeypatch):
    """The device's slots hold the forward only; chunkmirage's queue admits a
    few chunks more, whose normalization took turns with the forward when the
    queue's slots were the device's."""
    monkeypatch.setenv("CELLMAP_FLOW_GPU_SLOTS", "2")
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL)))

    assert mirage.inferencer.device_slots.n == 2
    assert InferenceOp.slots == 2 + INFERENCE_PREFETCH


def test_no_prediction_cache_runs_the_model_for_every_request(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLMAP_FLOW_PREDICTION_CACHE_BYTES", "0")
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, COUNTING_MODEL)))
    client = TestClient(mirage.app)
    start = _forwards(mirage)

    _chunk(client, layer(), (0, 0, 0))
    _chunk(client, layer(), (0, 0, 0))

    assert _forwards(mirage) - start == 2


def test_after_new_weights_a_new_layer_is_computed_anew(tmp_path):
    """A finetune iteration names its layer anew and updates the weights in
    place; nothing the old weights made is served under the new name."""
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, COUNTING_MODEL)))
    client = TestClient(mirage.app)
    _chunk(client, layer(model="model_iteration_1"), (0, 0, 0))
    start = _forwards(mirage)

    mirage.weights_changed()
    _chunk(client, layer(model="model_iteration_2"), (0, 0, 0))

    assert _forwards(mirage) - start == 1


def test_only_the_job_managers_token_restarts_training(tmp_path):
    asked = []
    path, script = write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL)
    closed = TestClient(ChunkmirageServer(path, ScriptModelConfig(script_path=script)).app)
    client = TestClient(ChunkmirageServer(path, ScriptModelConfig(script_path=script),
                                          restart_callback=lambda p: asked.append(p) or True,
                                          restart_token="secret").app)

    assert closed.post("/__control__/restart", json={}).status_code == 501
    assert client.post("/__control__/restart", json={"lr": 1}, headers={TOKEN_HEADER: "guess"}).status_code == 401
    assert client.post("/__control__/restart", json={"lr": 1}, headers={TOKEN_HEADER: "secret"}).status_code == 200
    assert asked == [{"lr": 1}]


def test_the_datasets_api_is_shut(tmp_path):
    """Anyone who can reach a node could otherwise serve any file this user can read."""
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL)))
    response = TestClient(mirage.app).post("/api/datasets", json={"name": "x", "spec": {"source": "/etc"}})
    assert response.status_code == 401


def test_the_address_is_announced_once_the_server_takes_requests(tmp_path, monkeypatch, capsys):
    ready = tmp_path / "server.ready"
    monkeypatch.setenv("CELLMAP_FLOW_READY_FILE", str(ready))
    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL)))
    thread = threading.Thread(target=mirage.run, kwargs={"port": 0}, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 30
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.05)
        url = json.loads(ready.read_text())["url"]
        # Announced, so it answers at once.
        info = requests.get(f"http://127.0.0.1:{mirage.port}/__control__/model_info", timeout=5).json()
        assert url.endswith(f":{mirage.port}") and info["engine"] == "chunkmirage"
        assert f"CELLMAP_FLOW_SERVER_IP({url})CELLMAP_FLOW_SERVER_IP" in capsys.readouterr().out
    finally:
        mirage.stop()
        thread.join(10)


def test_the_engine_is_chosen_by_the_environment(tmp_path, monkeypatch):
    path, script = write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL)
    assert engine_name() == "flask"
    monkeypatch.setenv("CELLMAP_FLOW_ENGINE", "chunkmirage")
    server = make_server(path, ScriptModelConfig(script_path=script))
    assert isinstance(server, ChunkmirageServer)
    assert check_server(server).shape == (1, 4, 4, 4)  # what `infer --server-check` computes
    monkeypatch.setenv("CELLMAP_FLOW_ENGINE", "fastest")
    with pytest.raises(ValueError, match="CELLMAP_FLOW_ENGINE"):
        engine_name()


def test_a_step_on_the_gpu_runs_one_chunk_at_a_time_and_the_rest_in_parallel(tmp_path):
    """Cellpose's masks follow the flows on the GPU: six at once each took 4 s
    instead of 0.8, and the chunks nearest the cursor came no sooner."""
    from cellmap_flow.serving.chunkmirage_ops import DevicePostprocessOp, PostprocessOp, layer_ops

    mirage = ChunkmirageServer(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)),
                               ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL)))
    ops = layer_ops(mirage.served.name, [], [{"name": "CellposeMasksPostprocessor"}, {"name": "ThresholdPostprocessor"}])
    assert [type(op) for op in ops[1:3]] == [DevicePostprocessOp, PostprocessOp]
    assert DevicePostprocessOp.slots == 1 and PostprocessOp.slots is None
