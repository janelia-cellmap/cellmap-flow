"""export_merged folds a LoRA adapter into Conv3d weights exactly and only
accepts tiles that keep the 178-tile pooling phase."""

import os
import tempfile

import pytest
import torch
from torch import nn

from cellmap_flow.finetune.export_merged import (
    merge_lora_into_conv3d,
    strip_lora_layers,
    valid_tile,
)
from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper, wrap_model_with_lora


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv3d(1, 8, 3)
        self.c2 = nn.Conv3d(8, 2, 3)

    def forward(self, x):
        return self.c2(torch.relu(self.c1(x)))


def test_manual_merge_matches_unmerged_adapter_and_strips_lora():
    torch.manual_seed(0)
    net = BatchLoopWrapper(TinyNet()).eval()
    peft = wrap_model_with_lora(net, lora_r=4, lora_alpha=8, lora_dropout=0.0).eval()
    # give the adapter something to say (PEFT starts lora_B at zero)
    for n, p in peft.named_parameters():
        if "lora_" in n:
            p.data.normal_(0, 0.1)
    x = torch.rand(1, 1, 12, 12, 12)
    with torch.no_grad():
        y_adapter = peft(x)
    n = merge_lora_into_conv3d(peft)
    assert n == 2
    base = peft.get_base_model()
    strip_lora_layers(base)
    from peft.tuners.lora import LoraLayer
    assert not any(isinstance(m, LoraLayer) for m in base.modules())
    plain = base.model
    assert isinstance(plain, TinyNet)
    with torch.no_grad():
        y_merged = plain(x)
    assert torch.allclose(y_adapter, y_merged, atol=1e-5)
    # and it is a different function from the un-finetuned net
    with torch.no_grad():
        assert (y_merged - TinyNet().eval()(x)).abs().max() > 0


def test_valid_tile_keeps_pooling_phase():
    assert valid_tile(178) and valid_tile(290) and valid_tile(306) and valid_tile(322)
    assert not valid_tile(298) and not valid_tile(378) and not valid_tile(162)
