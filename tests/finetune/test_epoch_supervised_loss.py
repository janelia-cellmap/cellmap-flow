"""The epoch's supervised loss, which picks the best checkpoint, skips empty batches.

It averaged every batch's masked mean, and a batch with no supervised voxels
-- only rehearsal patches, common at batch size 1-2 once good regions exist
-- contributes exactly 0. So "best epoch" partly tracked how many rehearsal
draws an epoch happened to get.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def test_a_batch_with_nothing_supervised_does_not_lower_the_epoch_loss(tmp_path):
    model = torch.nn.Conv3d(1, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        # Logit 2 everywhere (outside [0, 1], so the trainer's sigmoid probe
        # sees a logit model): BCE against target 1 is -ln(sigmoid(2)).
        model.bias.fill_(2.0)
    ann = torch.zeros(2, 1, 4, 4, 4)
    ann[0] = 2  # the first sample is annotated, the second not at all
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 4, 4, 4), ann), batch_size=1)
    trainer = LoRAFinetuner(
        model, loader, output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", learning_rate=0.0,
        target_transform=BinaryTargetTransform(), tensorboard=False,
    )
    trainer.train()
    expected = -math.log(1 / (1 + math.exp(-2.0)))
    assert trainer.last_supervised_loss == pytest.approx(expected, rel=1e-5)


def test_an_epoch_with_nothing_supervised_has_no_supervised_loss(tmp_path):
    loader = DataLoader(TensorDataset(torch.rand(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4)), batch_size=1)
    trainer = LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1), loader, output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", target_transform=BinaryTargetTransform(),
        tensorboard=False,
    )
    trainer._train_epoch()
    assert math.isnan(trainer.last_supervised_loss)
