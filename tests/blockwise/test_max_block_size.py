"""max_block_size.probe_max_block_size's integer-multiple search.

torch.cuda's surface is monkeypatched so this runs without a real GPU: the
search logic (stopping at the first multiple that does not fit, what each
candidate records) is what's under test, not torch itself.
"""

import pytest

import cellmap_flow.inference.runner as runner_module
from cellmap_flow.blockwise import max_block_size as mbs


class _FakeGeometry:
    output_shape = (56, 56, 56)
    output_channels = 2
    input_voxel_size = (8.0, 8.0, 8.0)
    output_voxel_size = (8.0, 8.0, 8.0)


class _FakeModelConfig:
    geometry = _FakeGeometry()


class _FakeDeviceProps:
    name = "Fake H100"
    total_memory = 80 * 1024 ** 3
    major, minor = 10, 3  # sm_103, a B300's -- beyond _patch_torch_cuda's arch list


class _FakeModelRunner:
    """Stands in for ModelRunner: records the out_roi each predict() got,
    and raises on the call index ``oom_at`` says (1-based), if any. Its
    "network" answers the probe's synthetic image itself on every channel
    (so a bigger tile's centre is the native tile's answer), except from the
    call ``garbage_at`` on, where it answers 1 everywhere, as a kernel past
    its indexing range does."""

    calls = []  # class-level: probe_max_block_size imports the class fresh each call
    garbage_at = None

    def __init__(self, model_config):
        self.model_config = model_config

    def predict(self, idi, out_roi):
        import numpy as np

        type(self).calls.append(out_roi)
        multiple = len(type(self).calls)
        if type(self).oom_at is not None and multiple >= type(self).oom_at:
            if type(self).oom_at_message == "oom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            raise RuntimeError(type(self).oom_at_message)
        image = idi.to_ndarray_ts(out_roi)
        channels = self.model_config.geometry.output_channels
        if type(self).garbage_at is not None and multiple >= type(self).garbage_at:
            return np.ones((channels, *image.shape), dtype=np.float32)
        return np.stack([image] * channels)


def _patch_torch_cuda(monkeypatch, peak_bytes=1024 ** 3):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: _FakeDeviceProps())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peak_bytes)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_90"])


def _patch_runner(monkeypatch, oom_at=None, oom_at_message="oom", garbage_at=None):
    _FakeModelRunner.calls = []
    _FakeModelRunner.oom_at = oom_at
    _FakeModelRunner.oom_at_message = oom_at_message
    _FakeModelRunner.garbage_at = garbage_at
    monkeypatch.setattr(runner_module, "ModelRunner", _FakeModelRunner)


def test_a_tile_the_gpu_answers_wrong_at_does_not_fit_even_though_it_runs(monkeypatch):
    """On a B300 the setup64 ER model ran at a 570-voxel tile (16 feature
    maps of 568^3 is past 2^31 elements) and answered 1 everywhere, and the
    probe, measuring memory alone, offered it: a blockwise task wrote a
    solid volume. Each tile's answer over the native tile's voxels is now
    held against the native tile's."""
    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, garbage_at=3)

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=8)

    assert [c.fits for c in result.candidates] == [True, True, False]
    assert result.best.multiple == 2
    assert result.candidates[0].agreement is None  # the native tile is the reference
    assert result.candidates[1].agreement == pytest.approx(1.0)
    wrong = result.candidates[2]
    assert wrong.peak_bytes is not None  # it ran
    assert wrong.agreement is None  # a constant correlates with nothing
    assert wrong.error.startswith("runs, but answers differently from the model's own tile ([56, 56, 56])")
    assert "use a smaller multiple" in wrong.error
    # Every candidate is centred on the same point, so the centres compare.
    assert [tuple(call.begin) for call in _FakeModelRunner.calls] == [
        (-224.0, -224.0, -224.0), (-448.0, -448.0, -448.0), (-672.0, -672.0, -672.0)]


def test_a_result_from_before_the_agreement_check_is_not_served_from_the_cache(tmp_path, monkeypatch):
    """The cache held x8 as fitting for the model that answered wrong there;
    it is probed again rather than offered."""
    from cellmap_flow.blockwise import max_block_size_cache as cache

    monkeypatch.setattr(cache, "CACHE_PATH", str(tmp_path / "cache.json"))
    config = _FakeModelConfig()
    monkeypatch.setattr(cache, "cache_key", lambda model_config, queue: f"k:{queue}")
    old = mbs.MaxBlockSizeResult("B300", 10, [
        mbs.BlockSizeCandidate(1, (56, 56, 56, 2), True, peak_bytes=1),
        mbs.BlockSizeCandidate(2, (112, 112, 112, 2), True, peak_bytes=2)])
    stale = old.to_dict()
    for candidate in stale["candidates"]:
        del candidate["agreement"]
    cache._write_cache({"k:gpu_b300": {**stale, "stored_at": 1.0}})
    assert cache.load_cached(config, "gpu_b300") is None

    cache.store(config, "gpu_b300", old)
    assert mbs.MaxBlockSizeResult.from_dict(cache.load_cached(config, "gpu_b300")) == old


def test_stops_at_the_first_multiple_that_ooms(monkeypatch):
    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, oom_at=3)

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=8)

    assert [c.multiple for c in result.candidates] == [1, 2, 3]
    assert [c.fits for c in result.candidates] == [True, True, False]
    assert result.candidates[-1].error == "out of memory"
    assert result.best.multiple == 2
    assert result.best.block_shape == (112, 112, 112, 2)
    assert result.gpu_name == "Fake H100"


def test_a_non_oom_runtime_error_also_stops_the_search(monkeypatch):
    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, oom_at=2, oom_at_message="size mismatch, network refuses this shape")

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=8)

    assert [c.fits for c in result.candidates] == [True, False]
    assert result.candidates[-1].error == "size mismatch, network refuses this shape"


NO_KERNEL_IMAGE = (
    "CUDA error: no kernel image is available for execution on the device\n"
    "CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\n"
    "For debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"
    "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
)


def test_a_gpu_this_torch_has_no_kernels_for_is_explained_not_called_too_small(monkeypatch):
    """A torch built for CUDA 12.4 says this to every Blackwell GPU (gpu_b300,
    gpu_rtx6000) on the first forward: recorded as a bare "does not fit" it
    read as a memory limit, and sent the user recomputing instead of
    changing environments."""
    import torch

    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, oom_at=1, oom_at_message=NO_KERNEL_IMAGE)

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=8)

    assert [c.fits for c in result.candidates] == [False]
    error = result.candidates[0].error
    assert error.startswith("CUDA error: no kernel image is available for execution on the device. Not a memory limit")
    assert "CUDA_LAUNCH_BLOCKING" not in error  # the boilerplate lines are dropped
    assert f"torch {torch.__version__}" in error
    assert "Fake H100 (sm_103; it was built for sm_80, sm_90)" in error


def test_every_multiple_fits_up_to_the_cap(monkeypatch):
    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, oom_at=None)

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=4)

    assert [c.multiple for c in result.candidates] == [1, 2, 3, 4]
    assert all(c.fits for c in result.candidates)
    assert result.best.multiple == 4


def test_even_the_native_tile_not_fitting_leaves_best_none(monkeypatch):
    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, oom_at=1)

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=8)

    assert [c.fits for c in result.candidates] == [False]
    assert result.best is None


def test_candidate_block_shapes_scale_each_axis_by_the_multiple(monkeypatch):
    _patch_torch_cuda(monkeypatch)
    _patch_runner(monkeypatch, oom_at=None)

    result = mbs.probe_max_block_size(_FakeModelConfig(), max_multiple=3)

    assert [c.block_shape for c in result.candidates] == [
        (56, 56, 56, 2), (112, 112, 112, 2), (168, 168, 168, 2),
    ]
    # Each out_roi handed to predict() is in nm, scaled the same way.
    sizes_nm = [tuple(call.shape) for call in _FakeModelRunner.calls]
    assert sizes_nm == [(448.0, 448.0, 448.0), (896.0, 896.0, 896.0), (1344.0, 1344.0, 1344.0)]


def test_raises_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CUDA device"):
        mbs.probe_max_block_size(_FakeModelConfig())


def test_result_round_trips_through_dict():
    from cellmap_flow.blockwise.max_block_size import BlockSizeCandidate, MaxBlockSizeResult

    result = MaxBlockSizeResult(
        gpu_name="Fake H100",
        total_memory_bytes=80 * 1024 ** 3,
        candidates=[
            BlockSizeCandidate(1, (56, 56, 56, 2), True, peak_bytes=1024),
            BlockSizeCandidate(2, (112, 112, 112, 2), False, error="out of memory"),
        ],
    )
    restored = MaxBlockSizeResult.from_dict(result.to_dict())
    assert restored == result
