"""Only unflattened conv/linear leaves are swapped for real layers, faithfully.

_replace_interpreter_modules replaced any non-Conv/Linear module that had a
weight with a stride-1, unpadded Conv built from the weight's shape: a real
nn.ConvTranspose3d became a Conv3d with its channel counts swapped, and an
unflattened strided or padded conv lost its stride and padding. LoRA on a
UNet with transposed-conv upsampling died at the first forward pass.
"""

import pytest
import torch
import torch.nn as nn

from cellmap_flow.finetune.lora_wrapper import (
    _replace_interpreter_modules,
    detect_adaptable_layers,
)


class _UNetish(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv3d(1, 4, 3, stride=2, padding=1)
        self.mid = nn.Conv3d(4, 4, 3, padding=2, dilation=2, groups=2)
        self.up = nn.ConvTranspose3d(4, 2, 2, stride=2)
        self.head = nn.Conv3d(2, 1, 1)

    def forward(self, x):
        return self.head(self.up(torch.relu(self.mid(self.down(x)))))


def _unflattened(model, x):
    from torch.export import export, unflatten

    return unflatten(export(model.eval(), (x,)))


def test_unflattened_convs_keep_their_stride_padding_dilation_and_groups():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 8, 8, 8)
    model = _UNetish()
    expected = model.eval()(x)
    unflat = _unflattened(model, x)

    replaced = _replace_interpreter_modules(unflat)

    assert replaced == 3, "down, mid and head; not the transposed conv"
    assert isinstance(unflat.down, nn.Conv3d) and unflat.down.stride == (2, 2, 2)
    assert unflat.mid.groups == 2 and unflat.mid.dilation == (2, 2, 2)
    assert type(unflat.up).__name__ == "InterpreterModule"
    assert torch.allclose(unflat(x), expected, atol=1e-6)


def test_a_real_transposed_conv_is_left_alone():
    torch.manual_seed(0)
    model = _UNetish()
    x = torch.rand(1, 1, 8, 8, 8)
    expected = model.eval()(x)

    assert _replace_interpreter_modules(model) == 0
    assert isinstance(model.up, nn.ConvTranspose3d)
    assert torch.allclose(model(x), expected)
    assert "up" not in detect_adaptable_layers(model)


@pytest.mark.finetune
def test_lora_wraps_an_unflattened_unet_with_transposed_convs():
    from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

    torch.manual_seed(0)
    x = torch.rand(1, 1, 8, 8, 8)
    model = _UNetish()
    expected = model.eval()(x)
    lora = wrap_model_with_lora(_unflattened(model, x), lora_r=2, lora_alpha=4, lora_dropout=0.0)
    with torch.no_grad():
        # lora_B starts at zero, so the wrapped model is the base model.
        assert torch.allclose(lora(x), expected, atol=1e-6)
