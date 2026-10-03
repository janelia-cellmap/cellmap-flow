"""finetune.trainable: the blocks a model type hands its network to the trainer in,
and what a network can be finetuned with."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from cellmap_flow.finetune.trainable import (  # noqa: E402
    FULL,
    LORA,
    Clip,
    Crop,
    FixedZeroMeanUnitVariance,
    ScaleLinear,
    ScaleRange,
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


def test_a_crop_cuts_each_side_of_the_spatial_axes():
    x = torch.zeros(1, 2, 6, 8, 8)
    assert Crop((1, 2, 2))(x).shape == (1, 2, 4, 4, 4)
    assert Crop((0, 0, 0))(x) is x


def test_the_normalizations_do_what_bioimageio_and_cellpose_do():
    x = torch.rand(2, 1, 3, 8, 8) * 50 + 10
    z = ZeroMeanUnitVariance()(x)
    assert torch.allclose(z.mean(dim=(1, 2, 3, 4)), torch.zeros(2), atol=1e-5)
    assert torch.allclose(z.std(dim=(1, 2, 3, 4), unbiased=False), torch.ones(2), atol=1e-3)
    # Per slice (over y and x): Cellpose's 1-99% normalization of each image.
    s = ScaleRange(1, 99, dims=(3, 4))(x)
    flat = s.reshape(2, 1, 3, -1)
    assert torch.allclose(torch.quantile(flat, 0.01, dim=-1), torch.zeros(2, 1, 3), atol=1e-4)
    assert torch.allclose(torch.quantile(flat, 0.99, dim=-1), torch.ones(2, 1, 3), atol=1e-4)
    two = torch.ones(1, 2, 1, 1, 1)
    assert FixedZeroMeanUnitVariance([1.0, 3.0], [1.0, 2.0], eps=0)(two).flatten().tolist() == [0.0, -1.0]
    assert ScaleLinear([2.0, 3.0], 1.0)(two).flatten().tolist() == [3.0, 4.0]
    assert Clip(0, 0.5)(torch.tensor([-1.0, 0.25, 2.0])).tolist() == [0.0, 0.25, 0.5]


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


def test_lora_on_a_compiled_network_is_refused_with_how_to_finetune_it(run_cli, monkeypatch):
    from cellmap_flow.finetune import finetune_cli

    monkeypatch.setattr(finetune_cli, "load_trainable_model",
                        lambda config: torch.jit.script(nn.Conv3d(1, 1, 1)))
    with pytest.raises(SystemExit, match="only be fully finetuned.*LoRA rank to 0"):
        run_cli("--lora-r", "4")


def test_the_finetune_tab_is_told_what_each_model_can_be_finetuned_with():
    from types import SimpleNamespace

    from cellmap_flow.dashboard.routes.finetune.annotation_core import _finetune_modes
    from cellmap_flow.models.models_config import ScriptModelConfig

    def unreadable():
        raise OSError("its description could not be read")

    assert _finetune_modes(ScriptModelConfig(script_path="/s.py", name="m")) == [LORA, FULL]
    assert _finetune_modes(SimpleNamespace(name="ts", finetune_modes=lambda: (FULL,))) == [FULL]
    # Nothing to ask, or no answer: every option stays offered.
    assert _finetune_modes(None) is None and _finetune_modes(SimpleNamespace(name="job")) is None
    assert _finetune_modes(SimpleNamespace(name="m", finetune_modes=unreadable)) is None


def test_a_percentile_over_more_values_than_torch_quantile_takes_is_the_same(monkeypatch):
    from cellmap_flow.finetune import trainable

    x = torch.rand(1, 1, 4, 64, 64)
    expected = ScaleRange(1, 99)(x)
    monkeypatch.setattr(trainable, "_QUANTILE_MAX", 100)  # as if the patch were huge
    assert torch.allclose(ScaleRange(1, 99)(x), expected, atol=1e-6)
