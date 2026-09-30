"""export_merged folds a finetune into the weights, to within float rounding,
judges the re-export relative to the output's scale, and only accepts tiles
that keep the 178-tile pooling phase."""

import pytest
import torch
from torch import nn

from cellmap_flow.finetune.export_merged import REL_TOL, apply_finetune, check_equivalence, valid_tile
from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper, wrap_model_with_lora


class Mixed(nn.Module):
    """Conv3d layers, a 1x1x1 head among them, and a Linear: LoRA adapts all three."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c1 = nn.Conv3d(1, 8, 3)
        self.c2 = nn.Conv3d(8, 2, 1)
        self.fc = nn.Linear(6, 6)  # over the last axis

    def forward(self, x):
        return self.fc(self.c2(torch.relu(self.c1(x))))


@pytest.mark.finetune
def test_apply_finetune_folds_in_every_adapted_layer(tmp_path):
    """Its own Conv3d-only merge silently dropped the Linear's adapter."""
    peft = wrap_model_with_lora(BatchLoopWrapper(Mixed()), lora_r=4, lora_alpha=8, lora_dropout=0.0)
    for n, p in peft.named_parameters():
        if "lora_" in n:
            p.data.normal_(0, 0.1)
    peft.save_pretrained(str(tmp_path / "adapter"))
    x = torch.rand(1, 1, 8, 8, 8)
    with torch.no_grad():
        y_adapter = peft.eval()(x)
        y_merged = apply_finetune(Mixed().eval(), lora_adapter_path=str(tmp_path / "adapter"))(x)
    assert (y_merged - y_adapter).abs().max() <= 1e-6 * y_adapter.abs().max()


def test_valid_tile_keeps_pooling_phase():
    assert valid_tile(178) and valid_tile(290) and valid_tile(306) and valid_tile(322)
    assert not valid_tile(298) and not valid_tile(378) and not valid_tile(162)


class Scaled(nn.Module):
    """A net whose output magnitude is set by ``k``: these models emit unbounded
    logits, and the scale varied 250x between two mito finetunes."""

    def __init__(self, k, nudge=0.0):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Conv3d(1, 2, 3), nn.ReLU(), nn.Conv3d(2, 1, 3)).eval()
        self.net[0].weight.data *= 1.0 + nudge
        self.k = k

    def forward(self, x):
        return self.net(x) * self.k


def test_equivalence_is_judged_relative_to_the_output_scale():
    """The gate was an absolute 1e-3 on the raw output: a finetune whose logits
    lived near 1400 was refused over a one-ULP float32 difference, while one of
    the same quality near 5.5 passed."""
    small = check_equivalence(Scaled(1.0), Scaled(1.0, nudge=0.01), 194)
    big = check_equivalence(Scaled(1000.0), Scaled(1000.0, nudge=0.01), 194)
    # The absolute difference and the scale grow 1000x; the relative difference does not ...
    assert big[0] == pytest.approx(1000 * small[0], rel=1e-3) and big[2] == pytest.approx(1000 * small[2], rel=1e-3)
    assert big[1] == pytest.approx(small[1], rel=1e-3)
    # ... and a 1% weight change is refused at either scale, while an exact one
    # passes at the axolotl-heart mito_daughter's huge logits.
    assert small[1] > REL_TOL and big[1] > REL_TOL
    d_ref, r_ref, scale, _, r_tile, out = check_equivalence(Scaled(1400.0), Scaled(1400.0), 194)
    assert (d_ref, r_ref, out) == (0.0, 0.0, (190, 190, 190)) and scale > 100 and r_tile <= REL_TOL
