"""finetune.trainable: the blocks a model type hands its network to the trainer in,
and what a network can be finetuned with."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from cellmap_flow.finetune.trainable import (  # noqa: E402
    FULL,
    LORA,
    Crop,
    SliceWise,
    ZeroMeanUnitVariance,
    finetune_modes,
)


def test_a_2d_network_runs_on_each_z_slice_as_one_batch():
    seen = []

    class Net2d(nn.Module):
        def forward(self, x):
            seen.append(tuple(x.shape))
            return torch.cat([x, 2 * x], dim=1), "features"  # a network that returns more than its output

    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 1, 3, 4, 4)
    out = SliceWise(Net2d())(x)
    assert seen == [(6, 1, 4, 4)] and out.shape == (2, 2, 3, 4, 4)
    assert torch.equal(out[:, 0], x[:, 0]) and torch.equal(out[1, 1, 2], 2 * x[1, 0, 2])


def test_what_a_network_can_be_finetuned_with():
    conv = nn.Conv3d(1, 2, 3)
    assert finetune_modes(nn.Sequential(ZeroMeanUnitVariance(), conv)) == (LORA, FULL)
    # Compiled: every parameter trains, but there are no layers to attach adapters to.
    assert finetune_modes(torch.jit.script(conv)) == (FULL,)
    assert finetune_modes(nn.Sequential(Crop((1, 1, 1)))) == ()


def test_a_trainable_model_can_keep_lora_off_layers_it_uses_out_of_reach():
    """Cellpose reads patch_embed.proj.weight directly: an adapter there would never train."""
    pytest.importorskip("peft")
    from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

    def model():
        net = nn.Sequential()
        net.add_module("patch_embed", nn.Conv2d(1, 8, 3))
        net.add_module("block", nn.Conv2d(8, 8, 3))
        return net

    adapted = lambda m: sorted(n for n, _ in m.named_modules() if n.endswith("lora_A"))  # noqa: E731
    everything = adapted(wrap_model_with_lora(model(), lora_r=2, lora_alpha=4, lora_dropout=0.0))
    kept_off = model()
    kept_off.lora_exclude_patterns = ["patch_embed"]
    some = adapted(wrap_model_with_lora(kept_off, lora_r=2, lora_alpha=4, lora_dropout=0.0))
    assert any("patch_embed" in n for n in everything) and not any("patch_embed" in n for n in some)
    assert any("block" in n for n in some)
