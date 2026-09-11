"""Regression tests for LoRA gradient flow.

Background
----------
On 2026-04-28 a c-elegans script-path training run produced bit-for-bit
constant loss across many epochs. Diagnostic logging revealed
``mean|grad|=0.000e+00`` on the watched LoRA-B layer for every batch in
every epoch. Two compounding bugs were involved:

1. ``VirtualPatchDataset._worker_rng`` reseeded on every ``__getitem__``,
   so every patch was identical -- masked the symptom for a while.
2. The LoRA wrap on the script-path Sequential model produced trainable
   parameters that received zero gradient for some configurations
   (notably with distillation enabled, where the trainer toggles
   ``disable_adapter_layers()`` / ``enable_adapter_layers()`` around the
   teacher pass).

These tests assert at the wrap layer that:
  - Every ``lora_B`` weight gets a nonzero gradient after one forward +
    backward through the wrapped model.
  - The toggle dance leaves the model in a state where ``lora_B`` still
    receives gradient on the next forward.

Tiny synthetic UNet-style model (a few Conv3d blocks) is used so the
tests run in seconds on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest


def _tiny_sequential():
    """Mimic the script-path layout: nn.Sequential of 3D conv blocks."""
    return nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(4, 4, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(4, 1, kernel_size=1),
        nn.Sigmoid(),
    )


def _gradient_summary(model: nn.Module) -> dict[str, dict[str, float]]:
    """Return per-trainable-param mean|grad| (or None if grad is None)."""
    summary = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        summary[name] = {
            "mean_abs_grad": (
                None if p.grad is None else p.grad.detach().abs().mean().item()
            ),
            "numel": p.numel(),
        }
    return summary


def _assert_lora_b_grads_nonzero(model: nn.Module) -> None:
    """Assert every ``lora_B`` weight got a nonzero gradient."""
    summary = _gradient_summary(model)
    lora_b_grads = {
        name: info for name, info in summary.items() if "lora_B" in name
    }
    assert lora_b_grads, (
        "Expected at least one trainable lora_B weight after wrap; got none. "
        "wrap_model_with_lora may have failed to attach adapters."
    )
    zero = {
        name: info for name, info in lora_b_grads.items()
        if info["mean_abs_grad"] in (None, 0.0)
    }
    assert not zero, (
        "Some lora_B weights received no gradient after fwd+bwd; the LoRA "
        "branch isn't on the autograd path for these layers:\n"
        + "\n".join(f"  {name}: mean|grad|={info['mean_abs_grad']!r}"
                    for name, info in zero.items())
    )


def test_basic_lora_wrap_grad_flow():
    """One fwd+bwd through a tiny PEFT-wrapped Sequential should give every
    lora_B nonzero gradient."""
    from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

    base = _tiny_sequential()
    peft = wrap_model_with_lora(base, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    peft.train()

    x = torch.randn(1, 1, 8, 8, 8)
    y = peft(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    _assert_lora_b_grads_nonzero(peft)


def test_lora_wrap_grad_flow_after_disable_enable_toggle():
    """The trainer's distillation pass calls ``disable_adapter_layers()``
    before the teacher forward and ``enable_adapter_layers()`` after.

    If ``enable_adapter_layers()`` doesn't fully restore state, the
    student forward goes through the base model only and lora_B receives
    no gradient. This test exercises that exact dance."""
    from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

    base = _tiny_sequential()
    peft = wrap_model_with_lora(base, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    peft.train()

    x = torch.randn(1, 1, 8, 8, 8)

    # Teacher pass with adapters disabled (mirrors trainer).
    with torch.no_grad():
        peft.disable_adapter_layers()
        try:
            _teacher = peft(x)
        finally:
            peft.enable_adapter_layers()

    # Student pass — should activate LoRA branch and propagate gradient.
    y = peft(x)
    loss = y.float().pow(2).mean()
    loss.backward()

    _assert_lora_b_grads_nonzero(peft)


def test_lora_wrap_grad_flow_with_batch_loop_wrapper():
    """The trainer wraps UnflattenedModule with BatchLoopWrapper *before*
    PEFT, so PEFT sees ``BatchLoopWrapper(model)``. Verify gradient flows
    through BatchLoopWrapper at batch_size > 1 (where the loop actually
    runs) -- a stale issue we suspected when the loop returns
    ``torch.cat`` of N separate forward calls."""
    from cellmap_flow.finetune.lora_wrapper import (
        BatchLoopWrapper,
        wrap_model_with_lora,
    )

    base = BatchLoopWrapper(_tiny_sequential())
    peft = wrap_model_with_lora(base, lora_r=4, lora_alpha=8, lora_dropout=0.0)
    peft.train()

    # Batch size > 1 forces BatchLoopWrapper to actually iterate and cat.
    x = torch.randn(3, 1, 8, 8, 8)
    y = peft(x)
    assert y.shape[0] == 3
    loss = y.float().pow(2).mean()
    loss.backward()

    _assert_lora_b_grads_nonzero(peft)


def test_lambda_normalizer_is_picklable():
    """LambdaNormalizer used to store an eval()'d ``lambda`` on the
    instance. PyTorch DataLoader workers spawned via ``multiprocessing_context
    ='spawn'`` pickle the dataset (and therefore the normalizers) before
    starting -- lambdas can't be pickled, which crashed training before any
    batches ran. Regression test: a normalizer must round-trip through
    pickle and still produce the right output."""
    import pickle
    import numpy as np

    from cellmap_flow.norm.input_normalize import LambdaNormalizer

    n = LambdaNormalizer("x*2-1")
    pickled = pickle.dumps(n)
    n2 = pickle.loads(pickled)
    out = n2._process(np.array([0.5, 1.0]))
    assert out[0] == 0.0 and out[1] == 1.0, f"unexpected: {out}"


def test_virtual_patch_dataset_applies_input_norm():
    """Regression test for the train/inference normalization mismatch bug.

    The dashboard's inference path normalizes raw via ``g.input_norms``
    before feeding the model. The trainer is a separate LSF process where
    ``g.input_norms`` is empty -- so without an explicit per-dataset
    normalizer, the trainer trained the model on raw uint8 [0, 255] while
    inference fed it [-1, 1]. The trained model was nonsense at inference
    time. Asserts that VirtualPatchDataset, given an ``input_norm_config``
    matching the dashboard's typical config, returns raw patches in the
    expected normalized range -- not raw uint8.
    """
    import numpy as np
    import zarr
    import tempfile
    import os

    from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

    tmp = tempfile.mkdtemp()
    raw_path = os.path.join(tmp, "raw.zarr")
    g = zarr.open_group(raw_path, mode="w")
    g.create_dataset("s0", shape=(32, 32, 32), dtype="uint8", chunks=(16, 16, 16))
    g["s0"][:] = np.full((32, 32, 32), 128, dtype=np.uint8)  # constant
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [16.0, 16.0, 16.0]},
            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
        ]}],
    }]

    vol_path = os.path.join(tmp, "vol.zarr")
    v = zarr.open_group(vol_path, mode="w")
    v.create_group("annotation").create_dataset(
        "s0", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    arr = v["annotation"]["s0"][:]
    arr[4:28, 4:28, 4:28] = 2
    v["annotation"]["s0"][:] = arr
    v.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    v["annotation"].attrs["multiscales"] = g.attrs["multiscales"]

    common = dict(
        volume_zarr_path=vol_path,
        raw_dataset_path=raw_path,
        input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4),
        input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16),
        patches_per_epoch=4,
        seed=0,
    )

    # Without input_norm: raw is returned as native uint8 ~128.
    raw_unnormalized, _ = VirtualPatchDataset(input_norm_config=None, **common)[0]
    assert (
        110 < float(raw_unnormalized.min()) < 140
    ), (
        "Without input_norm, raw should pass through ~uint8 (~128); got "
        f"range [{raw_unnormalized.min()}, {raw_unnormalized.max()}]"
    )

    # With the dashboard's typical input_norm, raw should land in [-1, 1].
    # 128 / 255 * 2 - 1 = 0.0039.
    raw_normalized, _ = VirtualPatchDataset(
        input_norm_config={
            "MinMaxNormalizer": {"min_value": 0, "max_value": 255, "invert": False},
            "LambdaNormalizer": {"expression": "x*2-1"},
        },
        **common,
    )[0]
    rmin = float(raw_normalized.min())
    rmax = float(raw_normalized.max())
    assert -0.05 < rmin < 0.05 and -0.05 < rmax < 0.05, (
        "With input_norm, raw should be normalized to [-1, 1] range "
        f"(expect ~0.004); got [{rmin}, {rmax}]"
    )


def test_virtual_patch_dataset_rng_advances():
    """Regression test for a bug where ``VirtualPatchDataset._worker_rng``
    reseeded on every ``__getitem__`` call -- making every patch identical
    and silently breaking training. Two consecutive draws should yield
    different RNG samples."""
    import numpy as np
    import zarr
    import tempfile
    import os

    from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

    tmp = tempfile.mkdtemp()
    raw_path = os.path.join(tmp, "raw.zarr")
    g = zarr.open_group(raw_path, mode="w")
    g.create_dataset("s0", shape=(32, 32, 32), dtype="uint8", chunks=(16, 16, 16))
    g["s0"][:] = np.random.randint(0, 255, (32, 32, 32), dtype=np.uint8)
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [16.0, 16.0, 16.0]},
            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
        ]}],
    }]

    vol_path = os.path.join(tmp, "vol.zarr")
    v = zarr.open_group(vol_path, mode="w")
    v.create_group("annotation").create_dataset(
        "s0", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    arr = v["annotation"]["s0"][:]
    arr[4:28, 4:28, 4:28] = 2
    v["annotation"]["s0"][:] = arr
    v.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    v["annotation"].attrs["multiscales"] = g.attrs["multiscales"]

    ds = VirtualPatchDataset(
        volume_zarr_path=vol_path,
        raw_dataset_path=raw_path,
        input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4),
        input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16),
        patches_per_epoch=10,
        seed=0,
    )

    # Pull raw patches from several draws — they should not all be identical.
    raws = [ds[i][0].numpy().tobytes() for i in range(8)]
    assert len(set(raws)) > 1, (
        "All draws produced identical raw patches; the per-worker RNG is "
        "being re-seeded on every __getitem__ instead of advancing."
    )


def test_virtual_patch_dataset_stratified_sampling():
    """Regression test for two-pool stratified sampling. Without it, a
    session with a 600^3 imported crop (~40M FG voxels) and a small
    painted scribble (~hundreds of voxels) would draw 99.99% of patches
    from the dense crop and effectively ignore the scribble. Stratified
    sampling with default ratio=0.5 must give the sparse pool a real
    share of the patches.
    """
    import numpy as np
    import zarr
    import tempfile
    import os

    from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

    tmp = tempfile.mkdtemp()
    raw_path = os.path.join(tmp, "raw.zarr")
    g = zarr.open_group(raw_path, mode="w")
    g.create_dataset("s0", shape=(64, 64, 64), dtype="uint8", chunks=(16, 16, 16))
    g["s0"][:] = np.full((64, 64, 64), 128, dtype=np.uint8)
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [16.0, 16.0, 16.0]},
            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
        ]}],
    }]

    # Volume zarr: simulate one big imported crop (large dense FG region)
    # plus a tiny painted scribble outside its bbox.
    vol_path = os.path.join(tmp, "vol.zarr")
    v = zarr.open_group(vol_path, mode="w")
    v.create_group("annotation").create_dataset(
        "s0", shape=(64, 64, 64), chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    arr = v["annotation"]["s0"][:]
    # Dense imported crop: 32^3 region (~32K FG voxels)
    arr[0:32, 0:32, 0:32] = 2
    # Sparse scribble: 2^3 region outside the imported crop (~8 FG voxels)
    arr[40:42, 40:42, 40:42] = 2
    v["annotation"]["s0"][:] = arr
    v.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    v.attrs["imported_crops"] = [
        {
            "path": "/fake/crop.zarr",
            "name": None,
            "annotation_offset_voxels": [0, 0, 0],
            "annotation_shape_voxels": [32, 32, 32],
            "n_fg_voxels": 32 ** 3,
        }
    ]
    v["annotation"].attrs["multiscales"] = g.attrs["multiscales"]

    common = dict(
        volume_zarr_path=vol_path,
        raw_dataset_path=raw_path,
        input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4),
        input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16),
        seed=0,
    )

    # Default (auto): both pools exist → ratio resolves to 0.5. Roughly
    # half the patch anchors should land in the sparse region.
    ds = VirtualPatchDataset(**common, patches_per_epoch=200)
    assert abs(ds._effective_dense_ratio - 0.5) < 1e-9
    sparse_hits = 0
    dense_hits = 0
    for _ in range(200):
        rng = ds._worker_rng()
        use_dense = (
            ds._effective_dense_ratio >= 1.0
            or (ds._effective_dense_ratio > 0.0 and rng.random() < ds._effective_dense_ratio)
        )
        pool = ds._fg_index_dense if use_dense else ds._fg_index_sparse
        anchor = pool[rng.integers(0, pool.shape[0])]
        # Voxel in [40, 42)^3 came from the sparse scribble; rest from dense.
        if (anchor >= 40).all() and (anchor < 42).all():
            sparse_hits += 1
        else:
            dense_hits += 1
    # With ratio=0.5 over 200 draws we expect ~100 sparse hits. Allow a
    # wide band so the test isn't flaky; the failure mode we're guarding
    # against (no stratification) would give 0-1 sparse hits.
    assert sparse_hits > 50, (
        f"Stratified sampling gave only {sparse_hits}/200 sparse hits; "
        "expected ~100. Two-pool sampling is not active."
    )
    assert dense_hits > 50, (
        f"Stratified sampling gave only {dense_hits}/200 dense hits; "
        "expected ~100."
    )

    # Auto-degrade: explicit ratio=0.5 but only one pool populated.
    # Build a volume with NO imported_crops → all FG goes to sparse pool;
    # ratio should clamp to 0.0 so we don't try to sample an empty dense.
    vol_no_crops = os.path.join(tmp, "vol_no_crops.zarr")
    v2 = zarr.open_group(vol_no_crops, mode="w")
    v2.create_group("annotation").create_dataset(
        "s0", shape=(32, 32, 32), chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    a2 = v2["annotation"]["s0"][:]
    a2[8:24, 8:24, 8:24] = 2
    v2["annotation"]["s0"][:] = a2
    v2.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    v2["annotation"].attrs["multiscales"] = g.attrs["multiscales"]
    ds2 = VirtualPatchDataset(
        volume_zarr_path=vol_no_crops,
        raw_dataset_path=raw_path,
        input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4),
        input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16),
        patches_per_epoch=4,
        dense_to_sparse_ratio=0.5,  # explicit, but should clamp
        seed=0,
    )
    assert ds2._effective_dense_ratio == 0.0, (
        "With no imported_crops the dense pool is empty; ratio should "
        f"clamp to 0.0, got {ds2._effective_dense_ratio}"
    )
    assert ds2._fg_index_dense.shape[0] == 0
    assert ds2._fg_index_sparse.shape[0] > 0


def test_virtual_patch_dataset_default_patches_per_epoch():
    """Regression test: ``patches_per_epoch=None`` (the new default) means
    "cover every populated chunk roughly once per epoch" -- the dataset
    substitutes the populated-chunk count at index build time.
    """
    import numpy as np
    import zarr
    import tempfile
    import os

    from cellmap_flow.finetune.virtual_dataset import VirtualPatchDataset

    tmp = tempfile.mkdtemp()
    raw_path = os.path.join(tmp, "raw.zarr")
    g = zarr.open_group(raw_path, mode="w")
    g.create_dataset("s0", shape=(48, 48, 48), dtype="uint8", chunks=(16, 16, 16))
    g["s0"][:] = np.full((48, 48, 48), 128, dtype=np.uint8)
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
        "datasets": [{"path": "s0", "coordinateTransformations": [
            {"type": "scale", "scale": [16.0, 16.0, 16.0]},
            {"type": "translation", "translation": [0.0, 0.0, 0.0]},
        ]}],
    }]

    vol_path = os.path.join(tmp, "vol.zarr")
    v = zarr.open_group(vol_path, mode="w")
    # 48/16 = 3 chunks per dim → 27 total chunks, but we'll only populate 3.
    v.create_group("annotation").create_dataset(
        "s0", shape=(48, 48, 48), chunks=(16, 16, 16), dtype="uint8", fill_value=0
    )
    arr = v["annotation"]["s0"][:]
    # Three populated chunks (each chunk gets at least one FG voxel).
    arr[1, 1, 1] = 2
    arr[17, 17, 17] = 2
    arr[33, 33, 33] = 2
    v["annotation"]["s0"][:] = arr
    v.attrs["dataset_offset_nm"] = [0.0, 0.0, 0.0]
    v["annotation"].attrs["multiscales"] = g.attrs["multiscales"]

    ds = VirtualPatchDataset(
        volume_zarr_path=vol_path,
        raw_dataset_path=raw_path,
        input_size_voxels=(8, 8, 8),
        output_size_voxels=(4, 4, 4),
        input_voxel_size_nm=(16, 16, 16),
        output_voxel_size_nm=(16, 16, 16),
        patches_per_epoch=None,  # default → use populated chunk count
        seed=0,
    )
    assert ds.patches_per_epoch == 3, (
        f"Default patches_per_epoch should equal populated-chunk count "
        f"(3), got {ds.patches_per_epoch}"
    )
    assert len(ds) == 3
