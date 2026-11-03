"""max_block_size_cache's model+queue-keyed cache."""

import os
import types

import pytest

from cellmap_flow.blockwise import max_block_size_cache as cache
from cellmap_flow.blockwise.max_block_size import BlockSizeCandidate, MaxBlockSizeResult


@pytest.fixture(autouse=True)
def _isolated_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_PATH", str(tmp_path / "max_block_size_cache.json"))


def _result():
    return MaxBlockSizeResult(
        gpu_name="Fake H100",
        total_memory_bytes=80 * 1024 ** 3,
        candidates=[BlockSizeCandidate(1, (56, 56, 56, 2), True, peak_bytes=1024)],
    )


def test_a_script_model_is_keyed_by_path_and_mtime_and_the_queue(tmp_path):
    script = tmp_path / "model.py"
    script.write_text("model = None")
    model_config = types.SimpleNamespace(script_path=str(script))

    assert cache.load_cached(model_config, "gpu_h100") is None
    cache.store(model_config, "gpu_h100", _result())
    assert MaxBlockSizeResult.from_dict(cache.load_cached(model_config, "gpu_h100")) == _result()


def test_a_checkpoint_path_model_fly_is_also_cacheable(tmp_path):
    checkpoint = tmp_path / "model_checkpoint_60000"
    checkpoint.write_text("not a real checkpoint")
    model_config = types.SimpleNamespace(checkpoint_path=str(checkpoint))

    cache.store(model_config, "gpu_h100", _result())
    assert MaxBlockSizeResult.from_dict(cache.load_cached(model_config, "gpu_h100")) == _result()


def test_a_folder_path_model_cellmap_is_also_cacheable(tmp_path):
    folder = tmp_path / "exported_model"
    folder.mkdir()
    model_config = types.SimpleNamespace(folder_path=str(folder))

    cache.store(model_config, "gpu_h100", _result())
    assert MaxBlockSizeResult.from_dict(cache.load_cached(model_config, "gpu_h100")) == _result()


def test_a_huggingface_model_is_keyed_by_repo_and_revision():
    model_config = types.SimpleNamespace(repo="cellmap/some-model", revision="abc123")

    cache.store(model_config, "gpu_h100", _result())
    assert MaxBlockSizeResult.from_dict(cache.load_cached(model_config, "gpu_h100")) == _result()


def test_different_queues_are_different_cache_entries(tmp_path):
    checkpoint = tmp_path / "model_checkpoint_60000"
    checkpoint.write_text("not a real checkpoint")
    model_config = types.SimpleNamespace(checkpoint_path=str(checkpoint))

    cache.store(model_config, "gpu_h100", _result())
    assert cache.load_cached(model_config, "gpu_a100") is None
    assert cache.load_cached(model_config, "gpu_h100") is not None


def test_editing_the_checkpoint_misses_the_cache(tmp_path):
    checkpoint = tmp_path / "model_checkpoint_60000"
    checkpoint.write_text("v1")
    model_config = types.SimpleNamespace(checkpoint_path=str(checkpoint))
    cache.store(model_config, "gpu_h100", _result())

    checkpoint.write_text("v2")
    # Dated a second on, as an edit after a minutes-long probe is: a rewrite
    # this soon after the first write usually keeps its mtime, the
    # filesystem's clock ticking only every few milliseconds.
    stat = checkpoint.stat()
    os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert cache.load_cached(model_config, "gpu_h100") is None


def test_no_identifying_attribute_is_never_cached(tmp_path):
    model_config = types.SimpleNamespace(name="no path at all")
    cache.store(model_config, "gpu_h100", _result())  # must not raise
    assert cache.load_cached(model_config, "gpu_h100") is None


def test_no_queue_is_never_cached(tmp_path):
    checkpoint = tmp_path / "model_checkpoint_60000"
    checkpoint.write_text("v1")
    model_config = types.SimpleNamespace(checkpoint_path=str(checkpoint))
    assert cache.cache_key(model_config, None) is None
    assert cache.cache_key(model_config, "") is None
