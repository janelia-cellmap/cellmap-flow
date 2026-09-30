"""Each layer URL is served with its own chain, not the last one fetched.

The server used to write the chain of every .zattrs/.zarray request into the
process-wide g.input_norms / g.postprocess, and chunk requests read whatever
was there. Two layers on one server (two tabs, two users, or the old URL still
loading after a Submit) got each other's normalization.
"""

import numpy as np
import pytest

from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.norm.input_normalize import LambdaNormalizer
from cellmap_flow.post.postprocessors import (
    PostProcessor,
    SimpleBlockwiseMerger,
    ThresholdPostprocessor,
)
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import (
    IDENTITY_MODEL,
    decode_chunk,
    get_json,
    layer,
    write_raw,
    write_script,
)


@pytest.fixture
def server(tmp_path):
    raw = write_raw(tmp_path, np.full((8, 8, 8), 10, dtype=np.uint8))
    script = write_script(tmp_path, IDENTITY_MODEL)
    return CellMapFlowServer(raw, ScriptModelConfig(script_path=script))


def _chunk(server, dataset):
    client = server.app.test_client()
    response = client.get(f"/{dataset}/s0/0.0.0.0")
    assert response.status_code == 200, response.data
    return decode_chunk(server, response.data, np.float32, (4, 4, 4, 1))


def test_chunks_use_their_own_layers_chain(server):
    g.input_norms = []
    g.postprocess = []
    doubled = layer([LambdaNormalizer("x * 2")])
    tripled = layer([LambdaNormalizer("x * 3")])
    client = server.app.test_client()

    get_json(client, f"/{doubled}/.zattrs")
    get_json(client, f"/{tripled}/.zattrs")  # the "last writer"

    assert np.all(_chunk(server, doubled) == 20)
    assert np.all(_chunk(server, tripled) == 30)
    # And nothing leaked into the process-wide chain.
    assert g.input_norms == [] and g.postprocess == []


def test_a_url_without_a_chain_does_not_reset_other_layers(server):
    g.input_norms = []
    g.postprocess = []
    doubled = layer([LambdaNormalizer("x * 2")])
    client = server.app.test_client()

    get_json(client, f"/{doubled}/.zattrs")
    get_json(client, "/plain/.zattrs")

    assert np.all(_chunk(server, doubled) == 20)
    assert np.all(_chunk(server, "plain") == 10)


def test_dtype_follows_the_requested_layer(server):
    thresholded = layer(posts=[ThresholdPostprocessor(threshold=0.5)])
    plain = layer()
    client = server.app.test_client()

    get_json(client, f"/{plain}/.zattrs")
    assert get_json(client, f"/{thresholded}/s0/.zarray")["dtype"] == "|u1"
    assert get_json(client, f"/{plain}/s0/.zarray")["dtype"] == "<f4"
    # The chunk is cast to its own layer's dtype, not the last one fetched.
    get_json(client, f"/{thresholded}/.zattrs")
    assert np.all(_chunk(server, plain) == 10)


def test_stateful_steps_keep_their_state_across_metadata_requests(server):
    merged = layer(posts=[SimpleBlockwiseMerger()])
    client = server.app.test_client()

    get_json(client, f"/{merged}/.zattrs")
    first = server.refresh_dataset(merged).postprocess[0]
    client.get(f"/{merged}/s0/0.0.0.0")
    get_json(client, f"/{merged}/.zattrs")

    assert server.refresh_dataset(merged).postprocess[0] is first
    assert first.chunk_slice_position_to_coords_id_dict


def test_process_default_chain_still_applies_without_a_url_chain(server):
    """In-process callers (tests, --server-check) set g and pass no blob."""

    class Plus(PostProcessor):
        def _process(self, data):
            return data + 1

    g.postprocess = [Plus()]
    server._chunk_impl(None, 0, 0, 0, 0)
    assert np.all(_chunk(server, "plain") == 11)


def test_unique_ids_are_spaced_by_output_voxels(tmp_path):
    """chunk_num_voxels counted input voxels, too few when the output is finer."""
    from cellmap_flow.inferencer import Inferencer
    from cellmap_flow.image_data_interface import ImageDataInterface
    from funlib.geometry import Roi

    raw = write_raw(tmp_path, np.zeros((8, 8, 8), dtype=np.uint8))
    script = write_script(
        tmp_path,
        """
        input_voxel_size = Coordinate(8, 8, 8)
        output_voxel_size = Coordinate(4, 4, 4)
        read_shape = Coordinate(4, 4, 4) * input_voxel_size
        write_shape = Coordinate(8, 8, 8) * output_voxel_size
        output_channels = 1
        block_shape = np.array((8, 8, 8, 1))
        model = torch.nn.Upsample(scale_factor=2)
        """,
    )
    seen = []

    class Record(PostProcessor):
        def _process(self, data, chunk_num_voxels):
            seen.append(chunk_num_voxels)
            return data

    config = ScriptModelConfig(script_path=script)
    inferencer = Inferencer(config)
    idi = ImageDataInterface(raw, voxel_size=(8, 8, 8))
    g.postprocess = [Record()]
    inferencer.process_chunk(idi, Roi((0, 0, 0), (32, 32, 32)))

    assert seen == [8 * 8 * 8]


def test_zarray_rank_matches_for_a_model_without_a_channel_axis(tmp_path):
    raw = write_raw(tmp_path, np.full((8, 8, 8), 3, dtype=np.uint8))
    script = write_script(
        tmp_path,
        IDENTITY_MODEL
        + """
chunk_output_axes = ("z", "y", "x")


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x[:, 0]


model = Squeeze()
""",
    )
    server = CellMapFlowServer(raw, ScriptModelConfig(script_path=script))
    client = server.app.test_client()

    meta = get_json(client, "/plain/s0/.zarray")
    assert len(meta["chunks"]) == len(meta["shape"]) == 3
    response = client.get("/plain/s0/0.0.0")
    chunk = decode_chunk(server, response.data, meta["dtype"], meta["chunks"])
    assert np.all(chunk == 3)


def test_a_model_runner_returns_the_models_own_output(tmp_path):
    """No chain and nothing from g: the model's output for the region, read with its context."""
    from cellmap_flow.inference.runner import ModelRunner
    from cellmap_flow.image_data_interface import ImageDataInterface
    from funlib.geometry import Roi

    data = (np.arange(512) % 251).astype(np.uint8).reshape(8, 8, 8)
    raw = write_raw(tmp_path, data)
    script = write_script(
        tmp_path,
        IDENTITY_MODEL.replace("read_shape = Coordinate(4, 4, 4)", "read_shape = Coordinate(6, 6, 6)")
        + "\nmodel.forward = lambda x: x[:, :, 1:-1, 1:-1, 1:-1] * 2\n",
    )
    g.postprocess = [ThresholdPostprocessor(threshold=0.5)]
    runner = ModelRunner(ScriptModelConfig(script_path=script))
    out = runner.predict(ImageDataInterface(raw, voxel_size=(8, 8, 8), input_norms=[]), Roi((8, 8, 8), (32, 32, 32)))

    assert out.shape == (1, 4, 4, 4) and out.dtype == np.float32
    assert np.array_equal(out[0], 2.0 * data[1:5, 1:5, 1:5])
