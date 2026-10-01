"""CellMapFlowServer: the rules behind what each layer URL is served.

What the server answers for five models is pinned in
test_served_metadata_snapshot; these are the rules it follows to get there.
"""

import logging
import threading
import time

import numpy as np
import pytest
from funlib.geometry import Roi

from cellmap_flow import server as server_module
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.norm.input_normalize import LambdaNormalizer
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import PostProcessor, SimpleBlockwiseMerger, ThresholdPostprocessor
from cellmap_flow.process_chain import process_chain
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.serving import virtual_zarr
from cellmap_flow.serving.protocol import ARGS_KEY
from tests.utils.serving_helpers import IDENTITY_MODEL, decode_chunk, get_json, layer


@pytest.fixture
def server(raw_zarr, model_script):
    """The identity model over 8^3 voxels of 10."""
    return CellMapFlowServer(raw_zarr(np.full((8, 8, 8), 10, np.uint8)), ScriptModelConfig(script_path=model_script()))


def _chunk(client, server, dataset, dtype=np.float32):
    response = client.get(f"/{dataset}/s0/0.0.0.0")
    assert response.status_code == 200, response.data
    return decode_chunk(server, response.data, dtype, (4, 4, 4, 1))


class Plus(PostProcessor):
    def _process(self, data):
        return data + 1


def test_each_layer_is_served_with_the_chain_in_its_own_url(server):
    """The server wrote every metadata request's chain into g, and chunks read
    whatever was there: two layers on one server (two tabs, or the old URL still
    loading after a Submit) got each other's normalization and dtype."""
    chain = process_chain()
    chain.input_norms, chain.postprocess = [], [Plus()]  # the process's own chain
    doubled, tripled = layer([LambdaNormalizer("x * 2")]), layer([LambdaNormalizer("x * 3")])
    thresholded = layer(posts=[ThresholdPostprocessor(threshold=0.5)])
    client = server.app.test_client()
    for dataset in ("plain", doubled, tripled, thresholded):  # each the "last writer" in turn
        get_json(client, f"/{dataset}/.zattrs")

    assert np.all(_chunk(client, server, doubled) == 20)
    assert np.all(_chunk(client, server, tripled) == 30)
    assert get_json(client, f"/{thresholded}/s0/.zarray")["dtype"] == "|u1"
    assert np.all(_chunk(client, server, thresholded, np.uint8) == 1)
    # A URL without a chain (in-process callers, --server-check) gets the process's.
    assert np.all(_chunk(client, server, "plain") == 11)
    assert chain.input_norms == [] and [type(p) for p in chain.postprocess] == [Plus], "nothing leaks into it"


def test_the_servers_address_leads_to_what_it_serves(server):
    """/ redirected to a Swagger page that documented nothing."""
    assert server.app.test_client().get("/").headers["Location"].endswith("/__control__/model_info")


@pytest.mark.parametrize("dataset, warns", [
    pytest.param(layer([LambdaNormalizer("x * 2")]), False, id="with-an-input-chain"),
    pytest.param("plain", True, id="no-chain-in-the-url"),
    pytest.param(layer(), True, id="an-empty-input-chain"),
])
def test_a_layer_served_without_normalization_says_so(server, caplog, dataset, warns):
    """A model fed raw voxels looks like one that trained badly, so the
    server warns when a layer URL carries no input chain."""
    with caplog.at_level(logging.WARNING):
        _chunk(server.app.test_client(), server, dataset)
    assert any(r.levelno == logging.WARNING for r in caplog.records) is warns


def test_a_stateful_step_keeps_its_state_across_metadata_requests(server):
    merged = layer(posts=[SimpleBlockwiseMerger()])
    client = server.app.test_client()
    get_json(client, f"/{merged}/.zattrs")
    first = server.chain_for(merged).postprocess[0]
    client.get(f"/{merged}/s0/0.0.0.0")
    get_json(client, f"/{merged}/.zattrs")
    assert server.chain_for(merged).postprocess[0] is first
    assert first.chunk_slice_position_to_coords_id_dict


def test_merged_ids_go_to_the_dashboard_without_holding_up_chunks(server, monkeypatch):
    """The POST ran inside the chunk request with no timeout, and the refresh
    time was set only once it returned: a slow dashboard held up the chunk,
    every request meanwhile posted too, and one that was gone failed it."""
    posted, release = [], threading.Event()

    def slow_post(url, json, timeout):
        posted.append((url, timeout))
        release.wait(5)

    monkeypatch.setattr(server_module.requests, "post", slow_post)
    blob = PipelineSpec.from_steps([], [SimpleBlockwiseMerger()]).to_url_blob(dashboard_url="http://dashboard/")
    merged = f"m{ARGS_KEY}{blob}{ARGS_KEY}"
    client = server.app.test_client()
    for index in ("0.0.0.0", "1.0.0.0"):
        assert client.get(f"/{merged}/s0/{index}").status_code == 200
    release.set()
    deadline = time.monotonic() + 5
    while not posted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert posted == [("http://dashboard/update/equivalences", server_module.EQUIVALENCES_TIMEOUT_SECONDS)]


@pytest.mark.parametrize("declared, served", [
    pytest.param("np.float16", "<f2", id="numpy-type"),
    pytest.param('"float32"', "<f4", id="string"),
    pytest.param("np.uint8", "|u1", id="unsigned"),
])
def test_any_declared_output_dtype_is_served_as_a_zarr_dtype(raw_zarr, model_script, declared, served):
    script = model_script(IDENTITY_MODEL + f"\noutput_dtype = {declared}\n")
    server = CellMapFlowServer(raw_zarr(np.zeros((8, 8, 8), np.uint8)), ScriptModelConfig(script_path=script))
    client = server.app.test_client()
    meta = get_json(client, "/plain/s0/.zarray")
    assert meta["dtype"] == served
    decode_chunk(server, client.get("/plain/s0/0.0.0.0").data, meta["dtype"], meta["chunks"])


def test_the_served_array_rounds_up_and_its_chunks_start_at_the_raw_corner():
    # 11 voxels of 8 nm from -4 nm end at 84 nm: 5.5 voxels of 16 nm, so 6.
    origin = np.array([-4] * 3)
    assert virtual_zarr.served_spatial_shape([-4] * 3, [11] * 3, [8] * 3, origin, [16] * 3) == [6] * 3
    assert virtual_zarr.chunk_roi((2, 2, 2), (4, 4, 4), (16, 16, 16), origin) == Roi((124,) * 3, (64,) * 3)


@pytest.mark.parametrize("shape, axes, expected", [
    pytest.param((2, 3, 4, 5), ("c", "z", "y", "x"), (3, 4, 5, 2), id="channel-first-moved-last"),
    pytest.param((3, 4, 5, 2), ("z", "y", "x", "c"), (3, 4, 5, 2), id="already-in-zarr-order"),
    pytest.param((3, 4, 5), ("c", "z", "y", "x"), (3, 4, 5), id="not-what-it-says-left-alone"),
])
def test_chunks_go_out_in_zarr_order(shape, axes, expected):
    data = np.arange(np.prod(shape)).reshape(shape)
    out = virtual_zarr.reorder_to_zarr_axes(data, axes, ("z", "y", "x"))
    assert out.shape == expected and out.flags.c_contiguous
    if len(shape) == 4:  # each voxel keeps its own channel values
        voxel = tuple(slice(None) if a == "c" else {"z": 1, "y": 2, "x": 3}[a] for a in axes)
        assert np.array_equal(out[1, 2, 3], data[voxel])


# flags: (served .zattrs translation, served shape, model_info's
#         effective_output_voxel_size and input_resampled_from)
RESAMPLE_OR_RELABEL = {
    # 16x4x4 nm data resampled to the model's 8 nm from its corner, (-8, -2, -2).
    "--resample": ([-4.0, 2.0, 2.0], [32, 8, 8, 1], [8, 8, 8], [16, 4, 4]),
    # Without it, read voxel for voxel as if at 8 nm: the output really is at 16x4x4.
    "": ([0.0, 0.0, 0.0], [16, 16, 16, 1], [16, 4, 4], None),
}


@pytest.mark.parametrize("flag", RESAMPLE_OR_RELABEL)
def test_cellmap_flow_serve_resample_serves_the_model_its_own_voxel_size(ome_pyramid, model_script, monkeypatch, flag):
    """From the command line a launcher runs to the chunks: a resampled input
    is at the model's voxel size and placed from the data's corner, and
    model_info tells it apart from a relabelled one, which the viewer then
    draws at the level's real voxel size."""
    import json

    from click.testing import CliRunner

    from cellmap_flow.cli.main import cli

    path = ome_pyramid((((16, 4, 4), 0),))  # 16^3 voxels from (-8, -2, -2) nm, each its z index + 1
    served = []
    monkeypatch.setattr(CellMapFlowServer, "run", lambda self, **kwargs: served.append(self))
    entry = json.dumps({"type": "script", "script_path": model_script()})
    result = CliRunner().invoke(cli, ["serve", "--model", entry, "-d", path, *flag.split()])
    assert result.exit_code == 0, result.output + repr(result.exception)

    (server,) = served
    client = server.app.test_client()
    translation, shape, effective, resampled_from = RESAMPLE_OR_RELABEL[flag]
    zattrs = get_json(client, "/plain/.zattrs")["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert (zattrs[1]["translation"][:3], get_json(client, "/plain/s0/.zarray")["shape"]) == (translation, shape)
    info = get_json(client, "/__control__/model_info")
    assert (info["effective_output_voxel_size"], info["input_resampled_from"]) == (effective, resampled_from)
    # Each chunk is the model (the identity) on the input read the same way.
    reader = ImageDataInterface(path, voxel_size=(8, 8, 8), on_voxel_size_mismatch="resample" if flag else "relabel",
                                input_norms=[])
    for index in ("0.0.0.0", "0.1.1.0"):
        corner = server.origin + 32 * np.array([int(i) for i in index.split(".")[:3]])
        expected = reader.to_ndarray_ts(Roi(tuple(corner), (32, 32, 32)))
        assert np.array_equal(_chunk_at(client, server, index), expected[..., None]), index


def _chunk_at(client, server, index):
    response = client.get(f"/plain/s0/{index}")
    assert response.status_code == 200, response.data
    return decode_chunk(server, response.data, np.float32, (4, 4, 4, 1))


def test_each_chunk_logs_where_its_time_went(server, caplog):
    """So a slow first chunk on the cluster can be read off the job's log."""
    client = server.app.test_client()
    with caplog.at_level(logging.INFO, logger="cellmap_flow.server"):
        _chunk(client, server, "plain")
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Chunk #")]
    assert line.startswith("Chunk #1 0.0.0: ") and line.endswith("0.00 s after the first chunk request, 1 in flight")
    stages = line.split("(", 1)[1].split(")", 1)[0]
    assert [s.rsplit(" ", 1)[0] for s in stages.split(", ")] == ["read", "gpu wait", "gpu", "postprocess", "encode"]
