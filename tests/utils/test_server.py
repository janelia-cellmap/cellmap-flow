"""CellMapFlowServer: the rules behind what each layer URL is served.

What the server answers for five models is pinned in
test_served_metadata_snapshot; these are the rules it follows to get there.
"""

import logging

import numpy as np
import pytest
from funlib.geometry import Roi

from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.norm.input_normalize import LambdaNormalizer
from cellmap_flow.post.postprocessors import PostProcessor, SimpleBlockwiseMerger, ThresholdPostprocessor
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.serving import virtual_zarr
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
    g.input_norms, g.postprocess = [], [Plus()]  # the process's own chain
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
    assert g.input_norms == [] and [type(p) for p in g.postprocess] == [Plus], "nothing leaks into g"


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
    first = server.refresh_dataset(merged).postprocess[0]
    client.get(f"/{merged}/s0/0.0.0.0")
    get_json(client, f"/{merged}/.zattrs")
    assert server.refresh_dataset(merged).postprocess[0] is first
    assert first.chunk_slice_position_to_coords_id_dict


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
