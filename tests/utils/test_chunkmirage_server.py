"""The inference server (serving.chunkmirage_server): the rules behind what each layer URL is served.

chunkmirage reads the raw data around each chunk, caches, queues chunks for
the device and serves the zarr. cellmap-flow adds the model on the layer's
own chain, the chunk grid and the dashboard's routes, which these test. A
served chunk is checked against ``Inferencer.process_chunk`` on the chunk's
region, the computation a blockwise worker does for a block.
"""

import json
import os
import threading
import time
from types import SimpleNamespace

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
from cellmap_flow.serving.chunkmirage_ops import chunk_roi
from cellmap_flow.serving.chunkmirage_server import ChunkmirageServer
from cellmap_flow.serving.protocol import ARGS_KEY, FORMAT
from cellmap_flow.serving.restart_token import TOKEN_HEADER
from tests.utils.serving_helpers import (
    COARSER_MODEL,
    IDENTITY_MODEL,
    SERVED_CASES,
    get_json,
    layer,
    read_chunk,
    write_raw,
    write_script,
)


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

CHAINS = [(case, chain) for case, (_, _, chains, _) in SERVED_CASES.items() for chain in ["plain", *chains]]


@pytest.mark.parametrize("case, chain", [pytest.param(case, chain, id=f"{case}-{chain}") for case, chain in CHAINS])
def test_each_model_and_chain_serves_the_models_output_for_the_chunks_region(tmp_path, raw_zarr, ome_pyramid,
                                                                             case, chain):
    """The identity, a model without a channel axis, one writing coarser
    voxels than it reads, one with context, and a level read as if it were at
    the model's voxel size; each plain and through a chain."""
    script, raw, chains, index = SERVED_CASES[case]
    server, client = _server(raw(SimpleNamespace(raw_zarr=raw_zarr, ome_pyramid=ome_pyramid)),
                             write_script(tmp_path, script))
    posts = chains.get(chain, [])

    got = read_chunk(client, layer(posts=posts), index)
    expected = _computed(server, index, posts)

    assert got.dtype == expected.dtype and got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_a_chunk_the_data_ends_in_is_computed_whole_and_cropped(tmp_path):
    """10 voxels in 4-voxel chunks: the last chunk holds 2, but the model
    still gets the 4 (and its context) it always did."""
    data = (np.arange(10**3) % 251).astype(np.uint8).reshape((10,) * 3)
    crop = SERVED_CASES["corner_at_-4nm"][0]
    server, client = _server(write_raw(tmp_path, data), write_script(tmp_path, crop))

    got = read_chunk(client, layer(), (2, 2, 1))

    assert got.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(got, _computed(server, (2, 2, 1))[:, :2, :2, :4])


def test_the_served_array_covers_the_data_rounded_up_and_its_chunks_start_at_the_raw_corner(tmp_path, ome_pyramid):
    """11 voxels of 8 nm from -4 nm (Janelia's corner, which OME metadata
    carries; a legacy offset is snapped onto the voxel grid) end at 84 nm:
    5.5 voxels of 16 nm, so 6. Chunk (1, 1, 1) of 4 output voxels starts
    64 nm from the corner and holds the 2 voxels inside the data."""
    server, client = _server(ome_pyramid(((8.0, 0.0),), shape=(11, 11, 11)), write_script(tmp_path, COARSER_MODEL))

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


def test_after_new_weights_a_new_layer_is_computed_anew(tmp_path):
    """A finetune iteration names its layer anew and updates the weights in
    place; nothing the old weights made is served under the new name."""
    server, client = _server(write_raw(tmp_path, np.full((8,) * 3, 10, np.uint8)), write_script(tmp_path, COUNTING_MODEL))
    read_chunk(client, layer(model="model_iteration_1"), (0, 0, 0))
    start = _forwards(server)

    server.weights_changed()
    read_chunk(client, layer(model="model_iteration_2"), (0, 0, 0))

    assert _forwards(server) - start == 1


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
    """The launcher hands the dashboard the address the ready file and the
    printed marker give, so both must come only once the port answers."""
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
