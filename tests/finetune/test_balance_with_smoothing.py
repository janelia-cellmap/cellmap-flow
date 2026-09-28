"""--balance-classes weighs the classes as annotated, whatever the smoothing.

The fg/bg split was made from the smoothed target, so every background voxel
carried s/2 of foreground weight. With little foreground that swamped it: at
100 fg voxels against 100k bg and s = 0.1, 98% of the "foreground" half of
the loss was background.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _bce(p, t):
    return -(t * math.log(p) + (1 - t) * math.log(1 - p))


def test_balanced_bce_splits_classes_by_the_unsmoothed_target(tmp_path):
    model = torch.nn.Conv3d(1, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(2.0)  # every voxel predicts sigmoid(2)
    ann = torch.ones(1, 1, 10, 10, 10)
    ann[0, 0, 0, 0, 0] = 2  # one foreground voxel among 999 background
    loader = DataLoader(TensorDataset(torch.rand(1, 1, 10, 10, 10), ann), batch_size=1)
    trainer = LoRAFinetuner(
        model, loader, output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", balance_classes=True,
        label_smoothing=0.1, learning_rate=0.0, target_transform=BinaryTargetTransform(),
        tensorboard=False,
    )
    trainer.train()

    p = 1 / (1 + math.exp(-2.0))
    expected = (_bce(p, 0.95) + _bce(p, 0.05)) / 2  # each class's own mean, halved
    assert trainer.last_supervised_loss == pytest.approx(expected, rel=1e-4)
