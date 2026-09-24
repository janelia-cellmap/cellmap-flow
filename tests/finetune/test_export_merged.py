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


class Scaled(nn.Module):
    """A net whose output magnitude is set by ``k`` -- these models emit unbounded
    logits, and the observed scale varied 250x between two mito finetunes."""

    def __init__(self, net, k):
        super().__init__()
        self.net = net
        self.k = k

    def forward(self, x):
        return self.net(x) * self.k


def _lean_net(seed=0, nudge=0.0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv3d(1, 2, 3), nn.ReLU(), nn.Conv3d(2, 1, 3)).eval()
    if nudge:
        net[0].weight.data *= 1.0 + nudge
    return net


def test_equivalence_is_judged_relative_to_output_scale():
    """A given *relative* disagreement is judged the same way at any output scale.

    Regression test: the gate used to be an absolute 1e-3 on the raw output, so a
    finetune whose logits happened to live near 1400 was refused over a one-ULP
    float32 difference, while an identical-quality run near 5.5 passed.
    """
    from cellmap_flow.finetune.export_merged import REL_TOL, check_equivalence

    small = check_equivalence(Scaled(_lean_net(), 1.0), Scaled(_lean_net(0, 0.01), 1.0), 194)
    big = check_equivalence(Scaled(_lean_net(), 1000.0), Scaled(_lean_net(0, 0.01), 1000.0), 194)
    d_small, r_small, s_small = small[0], small[1], small[2]
    d_big, r_big, s_big = big[0], big[1], big[2]

    # the absolute difference and the scale both grow 1000x ...
    assert d_big == pytest.approx(1000 * d_small, rel=1e-3)
    assert s_big == pytest.approx(1000 * s_small, rel=1e-3)
    # ... while the relative difference -- what the gate now judges -- does not
    assert r_big == pytest.approx(r_small, rel=1e-3)
    # and a 1% weight perturbation is rejected at either scale
    assert r_small > REL_TOL and r_big > REL_TOL


def test_equivalence_passes_for_an_exact_merge_at_a_large_output_scale():
    """The axolotl-heart mito_daughter case: huge logits, exact agreement."""
    from cellmap_flow.finetune.export_merged import REL_TOL, check_equivalence

    d_ref, r_ref, scale, d_tile, r_tile, out = check_equivalence(
        Scaled(_lean_net(), 1400.0), Scaled(_lean_net(), 1400.0), 194
    )
    assert d_ref == 0.0 and r_ref == 0.0
    assert scale > 100  # the regime that tripped the old absolute gate
    assert r_tile <= REL_TOL
    assert out == (190, 190, 190)
