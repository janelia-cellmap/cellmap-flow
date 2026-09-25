"""Tests for the ``min_channels`` LoRA target filter.

Background
----------
PEFT builds the LoRA branch of a conv layer as ``lora_A = Conv(Cin, r, k)``
followed by ``lora_B = Conv(r, Cout, 1)``. On a narrow layer that is a lot
of adapter for very little layer: with r=64 on a 16-channel 3x3x3 conv,
``lora_A`` alone is 4x the FLOPs of the layer it adapts, and at full
resolution it runs bandwidth-bound. Measured on mito-aff-unet-setup-16
(2026-09-23), the seven layers narrower than 96 channels held ~1% of the
adapter's parameters and cost 41% of every training step.

``detect_adaptable_layers(min_channels=N)`` skips layers whose narrower
side is below N. These tests pin down that:
  - the filter skips exactly the narrow layers and nothing else,
  - 0 (the default) changes nothing,
  - ``wrap_model_with_lora`` honours it and the surviving adapter trains,
  - an explicit ``target_modules`` list bypasses the filter.

Tiny synthetic models so everything runs in seconds on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cellmap_flow.finetune.lora_wrapper import (
    detect_adaptable_layers,
    wrap_model_with_lora,
)


def _mixed_width_convs():
    """Narrow in/out layers around one wide one, like a UNet's top level."""
    return nn.Sequential(
        nn.Conv3d(1, 4, kernel_size=3, padding=1),      # "0": min 1
        nn.ReLU(),
        nn.Conv3d(4, 4, kernel_size=3, padding=1),      # "2": min 4
        nn.ReLU(),
        nn.Conv3d(4, 128, kernel_size=3, padding=1),    # "4": min 4
        nn.ReLU(),
        nn.Conv3d(128, 128, kernel_size=3, padding=1),  # "6": min 128 <- wide
        nn.ReLU(),
        nn.Conv3d(128, 4, kernel_size=1),               # "8": min 4
    )


def _lora_a_layers(model: nn.Module) -> list[str]:
    """Names of the adapted layers, as the base model knows them.

    PEFT prefixes every path with its own nesting (``base_model.model.``,
    one level deeper again for a plain nn.Module), so compare the layer's
    own name -- the Sequential index here -- not the full path.
    """
    return sorted(
        name.rsplit(".lora_A", 1)[0].rsplit(".", 1)[-1]
        for name, _ in model.named_modules()
        if name.endswith("lora_A.default")
    )


def test_min_channels_skips_exactly_the_narrow_layers():
    model = _mixed_width_convs()
    assert detect_adaptable_layers(model) == ["0", "2", "4", "6", "8"]
    assert detect_adaptable_layers(model, min_channels=64) == ["6"]
    # Threshold is on the narrower side, so 4->128 is out but 128->128 stays.
    assert detect_adaptable_layers(model, min_channels=5) == ["6"]
    assert detect_adaptable_layers(model, min_channels=4) == ["2", "4", "6", "8"]


def test_min_channels_zero_is_the_default():
    model = _mixed_width_convs()
    assert detect_adaptable_layers(model, min_channels=0) == detect_adaptable_layers(model)


def test_min_channels_applies_to_linear_layers_too():
    model = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256))
    assert detect_adaptable_layers(model) == ["0", "2"]
    assert detect_adaptable_layers(model, min_channels=16) == ["2"]


def test_wrap_honours_min_channels_and_the_survivor_trains():
    model = wrap_model_with_lora(
        _mixed_width_convs(), lora_r=4, lora_alpha=8, lora_dropout=0.0,
        lora_min_channels=64,
    )
    assert _lora_a_layers(model) == ["6"]

    # The one remaining adapter must still get gradient, or the filter has
    # quietly turned finetuning into a no-op.
    out = model(torch.randn(1, 1, 8, 8, 8))
    out.pow(2).mean().backward()
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert len(lora_b) == 1
    assert lora_b[0].grad is not None and lora_b[0].grad.abs().sum() > 0


def test_explicit_target_modules_bypass_the_filter():
    model = wrap_model_with_lora(
        _mixed_width_convs(), target_modules=["2"], lora_r=4, lora_alpha=8,
        lora_dropout=0.0, lora_min_channels=64,
    )
    # "2" is 4 channels wide, well under 64, and is adapted anyway because
    # it was named explicitly.
    assert _lora_a_layers(model) == ["2"]
