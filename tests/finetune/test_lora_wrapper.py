"""wrap_model_with_lora: which layers get an adapter, that every adapter it adds
trains, and that a model which already carries one keeps it, in training and
when served."""

import tempfile

import pytest
import torch
import torch.nn as nn

from cellmap_flow.finetune.adaptation import strategy_for
from cellmap_flow.finetune.lora_wrapper import (
    BatchLoopWrapper,
    detect_adaptable_layers,
    load_lora_adapter,
    wrap_model_with_lora,
)


def _mixed_widths():
    """Narrow layers around a wide one, like a UNet's top level."""
    return nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(),  # "0": 1 channel at its narrowest
        nn.Conv3d(4, 4, 3, padding=1), nn.ReLU(),  # "2": 4
        nn.Conv3d(4, 128, 3, padding=1), nn.ReLU(),  # "4": 4
        nn.Conv3d(128, 128, 3, padding=1), nn.ReLU(),  # "6": 128
        nn.Conv3d(128, 4, 1),  # "8": 4
    )


def _small():
    return nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv3d(4, 4, 3, padding=1), nn.ReLU(),
                         nn.Conv3d(4, 1, 1), nn.Sigmoid())


class _UNetish(nn.Module):
    """Strided, dilated and grouped convs, and a transposed one, as torch.export unflattens them."""

    def __init__(self):
        super().__init__()
        self.down = nn.Conv3d(1, 4, 3, stride=2, padding=1)
        self.mid = nn.Conv3d(4, 4, 3, padding=2, dilation=2, groups=2)
        self.up = nn.ConvTranspose3d(4, 2, 2, stride=2)
        self.head = nn.Conv3d(2, 1, 1)

    def forward(self, x):
        return self.head(self.up(torch.relu(self.mid(self.down(x)))))


def _unflattened():
    from torch.export import export, unflatten

    return unflatten(export(_UNetish().eval(), (torch.rand(1, 1, 8, 8, 8),)))


@pytest.mark.parametrize("model, min_channels, layers", [
    pytest.param(_mixed_widths(), None, ["0", "2", "4", "6", "8"], id="every conv"),
    pytest.param(_mixed_widths(), 0, ["0", "2", "4", "6", "8"], id="min_channels 0 is the default"),
    # By the narrower side: 4 -> 128 is out, 128 -> 128 stays. At r=64 the seven
    # layers under 96 channels held 1% of the adapter and took 41% of each step.
    pytest.param(_mixed_widths(), 64, ["6"], id="min_channels 64"),
    pytest.param(_mixed_widths(), 5, ["6"], id="by the narrower side"),
    pytest.param(_mixed_widths(), 4, ["2", "4", "6", "8"], id="min_channels 4"),
    pytest.param(nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 256)), 16, ["2"], id="linear layers"),
    pytest.param(_UNetish(), None, ["down", "head"], id="not a transposed or a dilated conv"),  # PEFT adapts neither
])
def test_which_layers_get_an_adapter(model, min_channels, layers):
    kwargs = {} if min_channels is None else {"min_channels": min_channels}
    assert sorted(detect_adaptable_layers(model, **kwargs)) == layers


def _adapted(model):
    """The layers carrying an adapter, by their own names (peft prefixes its nesting)."""
    return sorted(name.rsplit(".lora_A", 1)[0].rsplit(".", 1)[-1]
                  for name, _ in model.named_modules() if name.endswith("lora_A.default"))


@pytest.mark.finetune
@pytest.mark.parametrize("make, settings, batch, layers", [
    pytest.param(_small, {}, 1, ["0", "2", "4"], id="plain"),
    # The trainer wraps BatchLoopWrapper(model): its loop and torch.cat keep the graph.
    pytest.param(lambda: BatchLoopWrapper(_small()), {}, 3, ["0", "2", "4"], id="batch loop"),
    pytest.param(_mixed_widths, dict(lora_min_channels=64), 1, ["6"], id="the one wide layer"),
    pytest.param(_mixed_widths, dict(target_modules=["2"], lora_min_channels=64), 1, ["2"],
                 id="named layers are not filtered"),
    # Unflattened convs were swapped for stride-1 unpadded ones, and a
    # ConvTranspose3d for a Conv3d with its channels swapped: the first forward died.
    pytest.param(_unflattened, {}, 1, ["down", "head"], id="unflattened"),
])
def test_every_adapter_starts_as_a_no_op_and_trains(make, settings, batch, layers):
    """A run's loss stayed constant for many epochs: every lora_B got no gradient,
    with distillation switching the adapter off and on around the teacher pass."""
    torch.manual_seed(0)
    base = make()
    x = torch.rand(batch, 1, 8, 8, 8)
    with torch.no_grad():
        expected = base.eval()(x)
    model = wrap_model_with_lora(base, lora_r=4, lora_alpha=8, lora_dropout=0.0, **settings)
    assert _adapted(model) == layers
    with torch.no_grad():
        assert torch.allclose(model(x), expected, atol=1e-6)  # lora_B starts at 0
    with torch.no_grad(), strategy_for(model).teacher(model) as teacher:
        teacher(x)
    model.train()
    model(x).float().pow(2).mean().backward()
    grads = [p.grad for name, p in model.named_parameters() if "lora_B" in name]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)


class _Head(nn.Module):
    """A body conv and a 1x1x1 three-channel head: peft merges a 1x1x1 Conv3d
    through its conv2d shortcut, which fails; a net without one never takes the
    path the real model does."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 3, padding=1)
        self.final_conv = nn.Conv3d(2, 3, 1)

    def forward(self, x):
        return self.final_conv(self.conv(x))


def _with_adapter(base, adapter_dir, std):
    """``base`` in a PeftModel whose adapter is ``adapter_dir``'s (or a new r=64
    one, saved there, with lora_B ~ N(0, std)), as a script model loads a finetune."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if std:
        model = get_peft_model(_Head(), LoraConfig(r=64, lora_alpha=128, target_modules=["conv", "final_conv"]))
        for name, p in model.named_parameters():
            if "lora_B" in name:
                nn.init.normal_(p, std=std)
        model.save_pretrained(adapter_dir)
    model = PeftModel.from_pretrained(_Head().eval(), adapter_dir, is_trainable=False)
    for layer in ("conv", "final_conv"):
        getattr(model.base_model.model, layer).base_layer.load_state_dict(getattr(base, layer).state_dict())
    return model


@pytest.mark.finetune
def test_finetuning_an_adapted_model_keeps_its_adapter():
    """get_peft_model() on a PeftModel replaces its "default" adapter with the new
    one: every "continue from my last run" trained from the untuned base, and was
    judged against the adapter it had just thrown away."""
    torch.manual_seed(0)
    x, base = torch.rand(1, 1, 4, 4, 4), _Head()
    adapted = _with_adapter(base, tempfile.mkdtemp(), std=0.5)
    with torch.no_grad():
        before, untuned = adapted(x), base(x)
    assert not torch.allclose(before, untuned, atol=1e-5)

    model = wrap_model_with_lora(adapted, target_modules=["conv", "final_conv"], lora_r=8, lora_alpha=16)
    with torch.no_grad():
        assert torch.allclose(model(x), before, atol=1e-5)
        # Good regions anchor to the teacher, so it must be the model the user judged good.
        with strategy_for(model).teacher(model) as teacher:
            assert torch.allclose(teacher(x), before, atol=1e-5)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)


@pytest.mark.finetune
def test_a_finetune_of_an_adapted_model_serves_what_it_trained():
    """Loading the adapter onto the script's model, itself a PeftModel, nested
    every name twice, so no saved key matched and the untouched base was served."""
    torch.manual_seed(0)
    x, base, first = torch.rand(1, 1, 4, 4, 4), _Head(), tempfile.mkdtemp()
    trained = wrap_model_with_lora(_with_adapter(base, first, std=0.5), target_modules=["conv", "final_conv"],
                                   lora_r=8, lora_alpha=16)
    for name, p in trained.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(p, std=0.3)  # stands in for a training run
    trained.eval()  # or lora_dropout makes both sides random
    second = tempfile.mkdtemp()
    trained.save_pretrained(second)

    served = load_lora_adapter(_with_adapter(base, first, std=0), second, is_trainable=False).eval()
    with torch.no_grad():
        assert torch.allclose(served(x), trained(x), atol=1e-5)
        assert not torch.allclose(served(x), base(x), atol=1e-5)
