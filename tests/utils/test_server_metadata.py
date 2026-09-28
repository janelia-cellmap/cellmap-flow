"""The virtual zarr's .zarray describes what the server can actually serve."""

import numpy as np
import pytest

from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import (
    IDENTITY_MODEL,
    decode_chunk,
    get_json,
    write_raw,
    write_script,
)


def test_served_array_reaches_the_end_of_offset_raw_data(tmp_path):
    # 8^3 voxels at 8 nm, starting at 32 nm: the data ends at 96 nm.
    raw = write_raw(
        tmp_path, np.full((8, 8, 8), 5, dtype=np.uint8), offset=(32, 32, 32)
    )
    server = CellMapFlowServer(
        raw, ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL))
    )
    client = server.app.test_client()

    meta = get_json(client, "/plain/s0/.zarray")
    assert meta["shape"][:3] == [12, 12, 12]

    # The last chunk covers 64-96 nm, which is real data.
    response = client.get("/plain/s0/2.2.2.0")
    chunk = decode_chunk(server, response.data, meta["dtype"], meta["chunks"])
    assert np.all(chunk == 5)


@pytest.mark.parametrize(
    "declared, expected",
    [
        ("np.float16", "<f2"),
        ('np.dtype("float16")', "<f2"),
        ('"float32"', "<f4"),
        ("np.int8", "|i1"),
        ("np.uint8", "|u1"),
    ],
)
def test_any_declared_output_dtype_gives_a_valid_zarr_dtype(tmp_path, declared, expected):
    raw = write_raw(tmp_path, np.zeros((8, 8, 8), dtype=np.uint8))
    script = write_script(tmp_path, IDENTITY_MODEL + f"\noutput_dtype = {declared}\n")
    server = CellMapFlowServer(raw, ScriptModelConfig(script_path=script))
    client = server.app.test_client()

    meta = get_json(client, "/plain/s0/.zarray")
    assert meta["dtype"] == expected
    response = client.get("/plain/s0/0.0.0.0")
    assert response.status_code == 200
    decode_chunk(server, response.data, meta["dtype"], meta["chunks"])
