"""``cellmap_flow max-block-size``, the probe job's command: writes its
result where --out says and, given --queue, into the max-block-size cache
too, so the dashboard finds the answer even after it stopped polling the
job. The probe itself is stubbed: test_max_block_size.py covers it."""

import json

import pytest
from click.testing import CliRunner

from cellmap_flow.blockwise import max_block_size as mbs
from cellmap_flow.blockwise import max_block_size_cache
from cellmap_flow.blockwise.max_block_size import BlockSizeCandidate, MaxBlockSizeResult
from cellmap_flow.blockwise.max_block_size_cli import max_block_size_cli
from cellmap_flow.models.models_config import ScriptModelConfig


@pytest.fixture(autouse=True)
def _isolated_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(max_block_size_cache, "CACHE_PATH", str(tmp_path / "max_block_size_cache.json"))


def _result():
    return MaxBlockSizeResult(
        gpu_name="Fake H100",
        total_memory_bytes=80 * 1024 ** 3,
        candidates=[
            BlockSizeCandidate(1, (56, 56, 56, 2), True, peak_bytes=1024),
            BlockSizeCandidate(2, (112, 112, 112, 2), False, error="out of memory"),
        ],
    )


@pytest.fixture
def probe_stub(monkeypatch):
    monkeypatch.setattr(mbs, "probe_max_block_size", lambda model_config, max_multiple=8: _result())


def _run(model_script, tmp_path, *extra):
    script = model_script()
    out = tmp_path / "result.json"
    entry = {"type": "script", "script_path": script, "name": "m"}
    answer = CliRunner().invoke(max_block_size_cli, ["--model", json.dumps(entry), "--out", str(out), *extra])
    return script, out, answer


def test_writes_the_result_and_caches_it_under_the_queue(model_script, tmp_path, probe_stub):
    script, out, answer = _run(model_script, tmp_path, "--queue", "gpu_h100")

    assert answer.exit_code == 0, answer.output
    assert "Largest block size on Fake H100: x1" in answer.output
    assert MaxBlockSizeResult.from_dict(json.loads(out.read_text())) == _result()
    cached = max_block_size_cache.load_cached(ScriptModelConfig(script, name="m"), "gpu_h100")
    assert MaxBlockSizeResult.from_dict(cached) == _result()
    assert max_block_size_cache.load_cached(ScriptModelConfig(script, name="m"), "gpu_h200") is None


def test_without_a_queue_only_the_file_is_written(model_script, tmp_path, probe_stub):
    script, out, answer = _run(model_script, tmp_path)

    assert answer.exit_code == 0, answer.output
    assert out.is_file()
    assert max_block_size_cache.load_cached(ScriptModelConfig(script, name="m"), "gpu_h100") is None


def test_the_probe_runs_the_model_without_the_entrys_sizes(tmp_path, monkeypatch):
    """A pick is written into the model node's input_size/output_size, so
    the entry may carry a block size already: the probe takes multiples of
    the model's own tile, which the type reads for itself without them."""
    probed = []

    def record(model_config, max_multiple=8):
        probed.append(model_config)
        return _result()

    monkeypatch.setattr(mbs, "probe_max_block_size", record)
    entry = {"type": "fly", "checkpoint_path": "/c.ts", "channels": ["mito"], "input_voxel_size": [8, 8, 8],
             "output_voxel_size": [8, 8, 8], "input_size": [514, 514, 514], "output_size": [392, 392, 392]}
    answer = CliRunner().invoke(max_block_size_cli, ["--model", json.dumps(entry), "--out", str(tmp_path / "r.json")])

    assert answer.exit_code == 0, answer.output
    (config,) = probed
    assert type(config).__name__ == "FlyModelConfig"
    assert (config.input_size, config.output_size) == (None, None)  # its own, read when built, not the entry's


def test_an_entry_without_sizes_is_probed_as_it_is(model_script, tmp_path, monkeypatch):
    probed = []

    def record(model_config, max_multiple=8):
        probed.append(model_config)
        return _result()

    monkeypatch.setattr(mbs, "probe_max_block_size", record)
    script, out, answer = _run(model_script, tmp_path)

    assert answer.exit_code == 0, answer.output
    (config,) = probed
    assert isinstance(config, ScriptModelConfig)


def test_a_probe_that_fails_exits_nonzero_without_a_result(model_script, tmp_path, monkeypatch):
    def boom(model_config, max_multiple=8):
        raise RuntimeError("No CUDA device available; the probe must run on a GPU node")

    monkeypatch.setattr(mbs, "probe_max_block_size", boom)
    script, out, answer = _run(model_script, tmp_path, "--queue", "gpu_h100")

    assert answer.exit_code == 1
    assert not out.exists()
    assert max_block_size_cache.load_cached(ScriptModelConfig(script, name="m"), "gpu_h100") is None
