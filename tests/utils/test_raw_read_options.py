"""The inference server reads raw data in parallel and through a cache; other callers as before."""

import numpy as np
import pytest

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.server import CellMapFlowServer
from tests.utils.serving_helpers import IDENTITY_MODEL, write_raw, write_script

CACHE, CONCURRENCY = "CELLMAP_FLOW_RAW_CACHE_BYTES", "CELLMAP_FLOW_RAW_READ_CONCURRENCY"


def _context(idi):
    return idi._raw_ts().spec(retain_context=True).to_json()["context"]


@pytest.mark.parametrize(
    "env, cache_pool, limit",
    [
        ({}, {"total_bytes_limit": 1 << 30}, None),
        ({CACHE: "0", CONCURRENCY: "3"}, {}, 3),
    ],
)
def test_the_server_opens_its_raw_data_with_a_cache(
    tmp_path, monkeypatch, env, cache_pool, limit
):
    for name in (CACHE, CONCURRENCY):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    raw = write_raw(tmp_path, np.arange(512).reshape(8, 8, 8).astype(np.uint8))
    config = ScriptModelConfig(script_path=write_script(tmp_path, IDENTITY_MODEL))
    served = CellMapFlowServer(raw, config).idi_raw
    plain = ImageDataInterface(raw, voxel_size=(8, 8, 8))

    assert _context(served)["cache_pool"] == cache_pool
    assert _context(served)["data_copy_concurrency"].get("limit") == limit
    # Every other caller keeps one reader and no cache.
    assert _context(plain)["cache_pool"] == {}
    assert _context(plain)["data_copy_concurrency"] == {"limit": 1}
    for _ in range(2):  # the second read comes from the cache
        assert np.array_equal(served.to_ndarray_ts(served.roi), plain.to_ndarray_ts(plain.roi))
