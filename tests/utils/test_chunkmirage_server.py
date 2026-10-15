"""The inference server (serving.chunkmirage_server): the rules behind what each layer URL is served.

chunkmirage reads the raw data around each chunk, caches, queues chunks for
the device and serves the zarr. cellmap-flow adds the model on the layer's
own chain, the chunk grid, the dashboard's routes and the announcement of
the address, which these test. A served chunk is checked against
``Inferencer.process_chunk`` on the chunk's region, the computation a
blockwise worker does for a block. What the server answers for five models
is pinned in test_served_metadata_snapshot.
"""

import json
import logging
import os
import threading
import time

import numpy as np
import pytest
import requests
from funlib.geometry import Roi
from starlette.testclient import TestClient

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.norm.input_normalize import LambdaNormalizer
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import (
    LabelPostprocessor,
    MortonSegmentationRelabeling,
    PostProcessor,
    SimpleBlockwiseMerger,
    ThresholdPostprocessor,
)
from cellmap_flow.process_chain import process_chain
from cellmap_flow.serving import chunkmirage_ops
from cellmap_flow.serving.chunkmirage_ops import InferenceOp, chunk_roi
from cellmap_flow.serving.chunkmirage_server import INFERENCE_PREFETCH, ChunkmirageServer
from cellmap_flow.serving.engine import check_server, make_server
from cellmap_flow.serving.protocol import ARGS_KEY, FORMAT
from cellmap_flow.serving.restart_token import TOKEN_HEADER
from tests.utils.serving_helpers import IDENTITY_MODEL, get_json, layer, read_chunk, write_raw, write_script
from tests.utils.test_served_metadata_snapshot import CASES, COARSER, _ome_raw, chunk_index


def _server(path, script, **kwargs):
    """A server for the script model over the data at ``path``, and a client on it."""
    server = ChunkmirageServer(path, ScriptModelConfig(script_path=script), **kwargs)
    return server, TestClient(server.app)


def _computed(server, index, posts=(), norms=()):
    """Chunk ``index`` computed outside chunkmirage: the server's Inferencer on
    the chunk's region, read from the level the server reads, through the
    chain -- what a blockwise worker writes for the block."""
    served = server.served
    roi = chunk_roi(index, served.block, served.output_voxel_size, served.origin)
    out = server.inferencer.process_chunk(
        server.idi_raw, roi, input_norms=list(norms), postprocess=list(posts), grid_origin=served.origin
    )
    return out[None] if served.has_channel and out.ndim == 3 else out


# --- the voxels ----------------------------------------------------------------

CHAINS = [(case, chain) for case, (_, _, chains, _) in CASES.items() for chain in ["plain", *chains]]


@pytest.mark.parametrize("case, chain", [pytest.param(case, chain, id=f"{case}-{chain}") for case, chain in CHAINS])
def test_each_model_and_chain_serves_the_models_output_for_the_chunks_region(tmp_path, case, chain):
    """The identity, a model without a channel axis, one writing coarser
    voxels than it reads, one with context, and a level read as if it were at
    the model's voxel size; each plain and through a chain."""
    script, raw, chains, key = CASES[case]
    server, client = _server(raw(tmp_path), write_script(tmp_path, script))
    posts = chains.get(chain, [])

    got = read_chunk(client, layer(posts=posts), chunk_index(key))
    expected = _computed(server, chunk_index(key), posts)

    assert got.dtype == expected.dtype and got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_a_scale_of_a_precomputed_volume_is_read_at_the_models_voxel_size(tmp_path, write_array):
    """The hemibrain on gs:// failed to start: cellmap-flow names a precomputed
    volume's scales ``…/s<N>``, which is no path in the volume."""
    data = np.arange(16**3, dtype=np.uint8).reshape((16,) * 3)
    # 4 nm voxels: the model's 8 nm are scale 1.
    path = write_array("precomputed", data, {"resolution": [4, 4, 4], "voxel_offset": [2, 4, 6]}, scales=2)
    server, client = _server(path, write_script(tmp_path, IDENTITY_MODEL))

    for index in [(0, 0, 0), (1, 1, 1)]:
        np.testing.assert_array_equal(read_chunk(client, layer(), index), _computed(server, index))
    assert server.model_info()["input_resampled_from"] is None


def test_a_chunk_the_data_ends_in_is_computed_whole_and_cropped(tmp_path):
    """10 voxels in 4-voxel chunks: the last chunk holds 2, but the model
    still gets the 4 (and its context) it always did."""
    data = (np.arange(10**3) % 251).astype(np.uint8).reshape((10,) * 3)
    crop = CASES["corner_at_-4nm"][0]
    server, client = _server(write_raw(tmp_path, data), write_script(tmp_path, crop))

    got = read_chunk(client, layer(), (2, 2, 1))

    assert got.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(got, _computed(server, (2, 2, 1))[:, :2, :2, :4])


def test_the_served_array_covers_the_data_rounded_up_and_its_chunks_start_at_the_raw_corner(tmp_path):
    """11 voxels of 8 nm from -4 nm (Janelia's corner, which OME metadata
    carries; a legacy offset is snapped onto the voxel grid) end at 84 nm:
    5.5 voxels of 16 nm, so 6. Chunk (1, 1, 1) of 4 output voxels starts
    64 nm from the corner and holds the 2 voxels inside the data."""
    data = (np.arange(11**3) % 251).astype(np.uint8).reshape((11,) * 3)
    server, client = _server(_ome_raw(tmp_path, data, 8.0, 0.0), write_script(tmp_path, COARSER))

    assert get_json(client, f"/{layer()}/{FORMAT}/s0/.zarray")["shape"] == [2, 6, 6, 6]
    assert chunk_roi((1, 1, 1), server.served.block, (16, 16, 16), server.served.origin) == Roi(
        (60,) * 3, (64,) * 3
    )
    assert read_chunk(client, layer(), (1, 1, 1)).shape == (2, 2, 2, 2)


def test_a_data_path_with_dot_dot_in_it_is_served(tmp_path):
    """``serve -d ../data.zarr/raw``: tensorstore refuses ".." in a path, which
    the reader resolves before handing it over; the server did not, and failed
    to start on a path the reader had just opened."""
    write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8))
    path = os.path.join(str(tmp_path), "raw.zarr", os.pardir, "raw.zarr", "raw")
    _, client = _server(path, write_script(tmp_path, IDENTITY_MODEL))
    assert np.all(read_chunk(client, layer(), (0, 0, 0)) == 10)


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
    themselves, from the chunk grown by their context: the block chunkmirage
    read for the chunk (chunkmirage_ops.BlockInput) gives them the voxels an
    ImageDataInterface does."""
    server, client = _server(write_raw(tmp_path, (np.arange(12**3) % 251).astype(np.uint8).reshape((12,) * 3)),
                             write_script(tmp_path, OWN_PROCESS_CHUNK))
    for index in [(0, 0, 0), (1, 2, 0), (2, 2, 2)]:
        np.testing.assert_array_equal(read_chunk(client, layer(), index), _computed(server, index))


def test_ids_that_depend_on_the_chunk_are_keyed_on_its_index_on_the_served_grid(tmp_path):
    """Morton keys each chunk's ids on its index on the served grid, which
    starts at the data's corner, as blockwise keys them (test_blockwise_processor)."""
    data = np.zeros((16,) * 3, np.uint8)
    data[1:3, 1:3, 1:3] = data[9:12, 5:7, 13:15] = 200
    server, client = _server(write_raw(tmp_path, data), write_script(tmp_path, IDENTITY_MODEL))
    posts = [ThresholdPostprocessor(threshold=100), LabelPostprocessor(), MortonSegmentationRelabeling()]
    for index in [(0, 0, 0), (2, 1, 3)]:
        got = read_chunk(client, layer(posts=posts), index)
        assert got.dtype == np.uint64 and got.max() > 0
        np.testing.assert_array_equal(got, _computed(server, index, posts))


# --- each layer's chain ---------------------------------------------------------


class Plus(PostProcessor):
    def _process(self, data):
        return data + 1


def test_each_layer_is_served_with_the_chain_in_its_own_url(tmp_path):
    """Two layers on one server (two tabs, or the old URL still loading after
    a Submit) each get their own normalization and dtype, and nothing of
    theirs leaks into the process's chain; a URL without a chain (in-process
    callers, --server-check) gets the process's."""
    chain = process_chain()
    chain.input_norms, chain.postprocess = [], [Plus()]  # the process's own chain
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    doubled, tripled = layer([LambdaNormalizer("x * 2")]), layer([LambdaNormalizer("x * 3")])
    thresholded = layer(posts=[ThresholdPostprocessor(threshold=0.5)])
    for name in ("plain", doubled, tripled, thresholded):  # each resolved in turn
        get_json(client, f"/{name}/{FORMAT}/.zattrs")

    assert np.all(read_chunk(client, doubled, (0, 0, 0)) == 20)
    assert np.all(read_chunk(client, tripled, (0, 0, 0)) == 30)
    assert get_json(client, f"/{thresholded}/{FORMAT}/s0/.zarray")["dtype"] == "|u1"
    assert np.all(read_chunk(client, thresholded, (0, 0, 0)) == 1)
    assert np.all(read_chunk(client, "plain", (0, 0, 0)) == 11)
    assert chain.input_norms == [] and [type(p) for p in chain.postprocess] == [Plus], "nothing leaks into it"


@pytest.mark.parametrize("dataset, warns", [
    pytest.param(layer([LambdaNormalizer("x * 2")]), False, id="with-an-input-chain"),
    pytest.param("plain", True, id="no-chain-in-the-url"),
    pytest.param(layer(), True, id="an-empty-input-chain"),
])
def test_a_layer_served_without_normalization_says_so(tmp_path, caplog, dataset, warns):
    """A model fed raw voxels looks like one that trained badly, so the
    server warns when a layer URL carries no input chain."""
    _, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    caplog.clear()  # building the model warns of other things (no GPU, no output_dtype declared)
    with caplog.at_level(logging.WARNING):
        read_chunk(client, dataset, (0, 0, 0))
    assert any(r.levelno == logging.WARNING for r in caplog.records) is warns


def test_a_stateful_step_keeps_its_state_across_metadata_requests(tmp_path):
    """A layer's steps are built once, when its name is resolved: the merger
    that keeps the ids it has merged is the same object for every request."""
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    merged = layer(posts=[SimpleBlockwiseMerger()])
    get_json(client, f"/{merged}/{FORMAT}/.zattrs")
    first = server.registry.resolve(merged).ops[1]._step
    read_chunk(client, merged, (0, 0, 0))
    get_json(client, f"/{merged}/{FORMAT}/.zattrs")
    assert server.registry.resolve(merged).ops[1]._step is first
    assert first.chunk_slice_position_to_coords_id_dict


def test_merged_ids_go_to_the_dashboard_without_holding_up_chunks(tmp_path, monkeypatch):
    """The POST ran inside the chunk request with no timeout, and the refresh
    time was set only once it returned: a slow dashboard held up the chunk,
    every request meanwhile posted too, and one that was gone failed it."""
    posted, release = [], threading.Event()

    def slow_post(url, json, timeout):
        posted.append((url, timeout))
        release.wait(5)

    monkeypatch.setattr(requests, "post", slow_post)
    _, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    blob = PipelineSpec.from_steps([], [SimpleBlockwiseMerger()]).to_url_blob(dashboard_url="http://dashboard/")
    merged = f"m{ARGS_KEY}{blob}{ARGS_KEY}"
    for index in [(0, 0, 0), (1, 0, 0)]:
        read_chunk(client, merged, index)
    release.set()
    deadline = time.monotonic() + 5
    while not posted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert posted == [("http://dashboard/update/equivalences", chunkmirage_ops.EQUIVALENCES_TIMEOUT_SECONDS)]


@pytest.mark.parametrize("declared, served", [
    pytest.param("np.float16", "<f2", id="numpy-type"),
    pytest.param('"float32"', "<f4", id="string"),
    pytest.param("np.uint8", "|u1", id="unsigned"),
])
def test_any_declared_output_dtype_is_served_as_a_zarr_dtype(tmp_path, declared, served):
    script = write_script(tmp_path, IDENTITY_MODEL + f"\noutput_dtype = {declared}\n")
    _, client = _server(write_raw(tmp_path, np.zeros((8,) * 3, np.uint8)), script)
    assert get_json(client, f"/plain/{FORMAT}/s0/.zarray")["dtype"] == served
    assert read_chunk(client, "plain", (0, 0, 0)).dtype == np.dtype(served)


# --- the model's output is kept ------------------------------------------------

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
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, COUNTING_MODEL))
    start = _forwards(server)

    read_chunk(client, layer(), (0, 0, 0))
    read_chunk(client, layer(), (0, 0, 0))  # panned back
    read_chunk(client, layer(posts=[ThresholdPostprocessor(threshold=5)]), (0, 0, 0))  # Output tab changed

    assert _forwards(server) - start == 1


def test_a_chain_sent_again_with_its_values_as_text_reuses_the_models_output(tmp_path):
    """The dashboard's forms send "0.0" where the YAML gave 0.0: submitting a
    postprocessing step on the Output tab ran the model again on every chunk."""
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, COUNTING_MODEL))
    name = lambda norm, posts=(): f"m{ARGS_KEY}{PipelineSpec([norm], list(posts)).to_url_blob()}{ARGS_KEY}"  # noqa: E731
    start = _forwards(server)

    read_chunk(client, name({"name": "MinMaxNormalizer", "min_value": 0.0, "max_value": 255.0, "invert": False}), (0, 0, 0))
    read_chunk(client, name({"name": "MinMaxNormalizer", "min_value": "0.0", "max_value": "255.0", "invert": "False"},
                            [{"name": "ThresholdPostprocessor", "threshold": "0.5"}]), (0, 0, 0))

    assert _forwards(server) - start == 1


def test_the_next_chunks_are_normalized_while_the_device_runs_a_forward(tmp_path, monkeypatch):
    """The device's slots hold the forward only; chunkmirage's queue admits a
    few chunks more, whose normalization took turns with the forward when the
    queue's slots were the device's."""
    monkeypatch.setenv("CELLMAP_FLOW_GPU_SLOTS", "2")
    server, _ = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))

    assert server.inferencer.device_slots.n == 2
    assert InferenceOp.slots == 2 + INFERENCE_PREFETCH


def test_no_prediction_cache_runs_the_model_for_every_request(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLMAP_FLOW_PREDICTION_CACHE_BYTES", "0")
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, COUNTING_MODEL))
    start = _forwards(server)

    read_chunk(client, layer(), (0, 0, 0))
    read_chunk(client, layer(), (0, 0, 0))

    assert _forwards(server) - start == 2


@pytest.mark.parametrize("env, cache_pool", [
    pytest.param(None, {"total_bytes_limit": 1 << 30}, id="1-GiB-by-default"),
    pytest.param("0", {}, id="none"),
    pytest.param("12345", {"total_bytes_limit": 12345}, id="set-by-the-environment"),
])
def test_the_raw_data_is_read_through_a_cache_the_environment_sizes(tmp_path, monkeypatch, env, cache_pool):
    """Neighbouring chunks' inputs overlap by two thirds: without a cache,
    each chunk read and decoded them all again."""
    monkeypatch.delenv("CELLMAP_FLOW_RAW_CACHE_BYTES", raising=False)
    if env is not None:
        monkeypatch.setenv("CELLMAP_FLOW_RAW_CACHE_BYTES", env)
    server, _ = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    context = server.source.levels[0].store.spec(retain_context=True).to_json()["context"]
    assert context["cache_pool"] == cache_pool


def test_after_new_weights_a_new_layer_is_computed_anew(tmp_path):
    """A finetune iteration names its layer anew and updates the weights in
    place; nothing the old weights made is served under the new name."""
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, COUNTING_MODEL))
    read_chunk(client, layer(model="model_iteration_1"), (0, 0, 0))
    start = _forwards(server)

    server.weights_changed()
    read_chunk(client, layer(model="model_iteration_2"), (0, 0, 0))

    assert _forwards(server) - start == 1


def test_each_chunk_logs_where_its_time_went(tmp_path, caplog):
    """So a slow first chunk on the cluster can be read off the job's log:
    the model's time by stage, then each postprocessing step's."""
    _, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    with caplog.at_level(logging.INFO, logger="cellmap_flow.serving.chunkmirage_ops"):
        read_chunk(client, layer(posts=[ThresholdPostprocessor(threshold=5)]), (0, 0, 0))
    model, step = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Chunk 0.0.0: ")]
    assert model.startswith("Chunk 0.0.0: model ") and step.startswith("Chunk 0.0.0: ThresholdPostprocessor ")
    stages = model.split("(", 1)[1].split(")", 1)[0]
    assert [s.rsplit(" ", 1)[0] for s in stages.split(", ")] == ["read", "gpu wait", "gpu"]


# --- the dashboard's routes, and running ----------------------------------------


def test_only_the_job_managers_token_restarts_training(tmp_path):
    asked = []
    path, script = write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL)
    _, closed = _server(path, script)
    _, client = _server(path, script, restart_callback=lambda p: asked.append(p) or True, restart_token="secret")

    assert closed.post("/__control__/restart", json={}).status_code == 501
    assert client.post("/__control__/restart", json={"lr": 1}, headers={TOKEN_HEADER: "guess"}).status_code == 401
    assert client.post("/__control__/restart", json={"lr": 1}, headers={TOKEN_HEADER: "secret"}).status_code == 200
    assert asked == [{"lr": 1}]


def test_the_datasets_api_is_shut(tmp_path):
    """Anyone who can reach a node could otherwise serve any file this user can read."""
    _, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    response = client.post("/api/datasets", json={"name": "x", "spec": {"source": "/etc"}})
    assert response.status_code == 401


def test_the_address_is_announced_once_the_server_takes_requests(tmp_path, monkeypatch, capsys):
    ready = tmp_path / "server.ready"
    monkeypatch.setenv("CELLMAP_FLOW_READY_FILE", str(ready))
    server, _ = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    thread = threading.Thread(target=server.run, kwargs={"port": 0}, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 30
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.05)
        url = json.loads(ready.read_text())["url"]
        # Announced, so it answers at once.
        info = requests.get(f"http://127.0.0.1:{server.port}/__control__/model_info", timeout=5).json()
        assert url.endswith(f":{server.port}") and info["engine"] == "chunkmirage"
        assert f"CELLMAP_FLOW_SERVER_IP({url})CELLMAP_FLOW_SERVER_IP" in capsys.readouterr().out
    finally:
        server.stop()
        thread.join(10)


# flag: (served scale and translation (z, y, x nm; the translation is voxel 0's
#        centre), served shape, model_info's effective_output_voxel_size,
#        input_resampled_from and input_relabelled_from)
RESAMPLE_OR_RELABEL = {
    # 16x4x4 nm data resampled to the model's 8 nm from its corner, (-8, -2, -2).
    "--resample": ([8.0, 8.0, 8.0], [-4.0, 2.0, 2.0], [1, 32, 8, 8], [8, 8, 8], [16, 4, 4], None),
    # With --no-resample, read voxel for voxel as if at 8 nm: the output really
    # is at 16x4x4, and is served where it really lies.
    "--no-resample": ([16.0, 4.0, 4.0], [0.0, 0.0, 0.0], [1, 16, 16, 16], [16, 4, 4], None, [16, 4, 4]),
}
RESAMPLE_OR_RELABEL[""] = RESAMPLE_OR_RELABEL["--resample"]  # the default


@pytest.mark.parametrize("flag", RESAMPLE_OR_RELABEL)
def test_cellmap_flow_serve_resample_serves_the_model_its_own_voxel_size(ome_pyramid, model_script, monkeypatch, flag):
    """From the command line a launcher runs to the chunks: a resampled input
    is at the model's voxel size and placed from the data's corner; a
    relabelled one is served where its level really lies, and model_info
    tells the two apart (the dashboard warns about the second)."""
    from click.testing import CliRunner

    from cellmap_flow.cli.main import cli

    path = ome_pyramid((((16, 4, 4), 0),))  # 16^3 voxels from (-8, -2, -2) nm, each its z index + 1
    served = []
    monkeypatch.setattr(ChunkmirageServer, "run", lambda self, **kwargs: served.append(self))
    entry = json.dumps({"type": "script", "script_path": model_script()})
    result = CliRunner().invoke(cli, ["serve", "--model", entry, "-d", path, *flag.split()])
    assert result.exit_code == 0, result.output + repr(result.exception)

    (server,) = served
    client = TestClient(server.app)
    scale, translation, shape, effective, resampled_from, relabelled_from = RESAMPLE_OR_RELABEL[flag]
    multiscale = get_json(client, f"/plain/{FORMAT}/.zattrs")["multiscales"][0]
    transforms = {t["type"]: t for t in multiscale["datasets"][0]["coordinateTransformations"]}
    assert transforms["scale"]["scale"][1:] == scale
    # A translation of 0 may be left out.
    assert transforms.get("translation", {}).get("translation", [0.0] * 4)[1:] == translation
    assert get_json(client, f"/plain/{FORMAT}/s0/.zarray")["shape"] == shape
    info = get_json(client, "/__control__/model_info")
    assert (info["effective_output_voxel_size"], info["input_resampled_from"], info["input_relabelled_from"]) == (
        effective, resampled_from, relabelled_from)
    # Each chunk is the model (the identity) on the input read the same way.
    reader = ImageDataInterface(path, voxel_size=(8, 8, 8), on_voxel_size_mismatch="relabel" if flag == "--no-resample" else "resample",
                                input_norms=[])
    for index in [(0, 0, 0), (0, 1, 1)]:
        corner = server.served.origin + 32 * np.array(index)
        expected = reader.to_ndarray_ts(Roi(tuple(corner), (32, 32, 32)))
        assert np.array_equal(read_chunk(client, "plain", index)[0], expected), index


def test_every_launcher_builds_this_server_and_the_check_computes_a_chunk(tmp_path):
    """`cellmap_flow serve`, `infer --server-check` and the finetune loop
    build their server with make_server."""
    path, script = write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL)
    server = make_server(path, ScriptModelConfig(script_path=script))
    assert isinstance(server, ChunkmirageServer)
    assert check_server(server).shape == (1, 4, 4, 4)  # what `infer --server-check` computes


def test_where_chunkmirage_is_missing_the_server_says_so(tmp_path, monkeypatch):
    """A model's own ``env`` (a pixi environment, an alias or a path) must
    have chunkmirage; one without it gets an error that says what to install
    rather than a bare import failure."""
    import sys

    monkeypatch.setitem(sys.modules, "cellmap_flow.serving.chunkmirage_server", None)
    path, script = write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL)

    with pytest.raises(ImportError, match="needs chunkmirage"):
        make_server(path, ScriptModelConfig(script_path=script))


def test_a_step_on_the_gpu_runs_one_chunk_at_a_time_and_the_rest_in_parallel(tmp_path):
    """Cellpose's masks follow the flows on the GPU: six at once each took 4 s
    instead of 0.8, and the chunks nearest the cursor came no sooner."""
    from cellmap_flow.serving.chunkmirage_ops import DevicePostprocessOp, PostprocessOp, layer_ops

    server, _ = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, IDENTITY_MODEL))
    ops = layer_ops(server.served.name, [], [{"name": "CellposeMasksPostprocessor"}, {"name": "ThresholdPostprocessor"}])
    assert [type(op) for op in ops[1:3]] == [DevicePostprocessOp, PostprocessOp]
    assert DevicePostprocessOp.slots == 1 and PostprocessOp.slots is None
