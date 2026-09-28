"""--loss-type mse compares probabilities with the target, not raw logits.

On a model that outputs logits, MSE trained the logits themselves toward the
0/1 target; the served sigmoid then turns those into 0.5 and 0.73, which
wrecks any threshold. Dice and margin already applied the sigmoid first.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _supervised_mse(tmp_path, model):
    ann = torch.full((1, 1, 4, 4, 4), 2.0)  # all foreground: target 1
    loader = DataLoader(TensorDataset(torch.rand(1, 1, 4, 4, 4), ann), batch_size=1)
    trainer = LoRAFinetuner(
        model, loader, output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="mse", learning_rate=0.0,
        target_transform=BinaryTargetTransform(), tensorboard=False,
    )
    trainer.train()
    return trainer.last_supervised_loss


def _constant(value, sigmoid=False):
    conv = torch.nn.Conv3d(1, 1, 1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.fill_(value)
    return torch.nn.Sequential(conv, torch.nn.Sigmoid()) if sigmoid else conv


def test_mse_on_a_logit_model_uses_the_sigmoid(tmp_path):
    loss = _supervised_mse(tmp_path, _constant(3.0))
    p = 1 / (1 + math.exp(-3.0))
    assert loss == pytest.approx((p - 1) ** 2, rel=1e-4)  # not (3 - 1)^2 = 4


def test_mse_on_a_sigmoid_model_is_unchanged(tmp_path):
    loss = _supervised_mse(tmp_path, _constant(3.0, sigmoid=True))
    p = 1 / (1 + math.exp(-3.0))
    assert loss == pytest.approx((p - 1) ** 2, rel=1e-4)
