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
