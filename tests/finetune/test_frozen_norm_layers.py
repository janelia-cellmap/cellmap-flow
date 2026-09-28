"""Under LoRA the frozen base stays frozen, norm layers included; the teacher is deterministic.

model.train() put the base's BatchNorm layers in train mode, so they
normalized by each tiny batch and kept updating running statistics that are
not part of the adapter: the model served in-process drifted from adapter +
fresh base. The distillation teacher ran in train mode too, so the base's
dropout made its targets noisy.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _base():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv3d(1, 4, 3, padding=1), nn.BatchNorm3d(4), nn.ReLU(),
        nn.Dropout(0.5), nn.Conv3d(4, 1, 1),
    )


def _lora_trainer(tmp_path, **kw):
    from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

    model = wrap_model_with_lora(_base(), lora_r=2, lora_alpha=4, lora_dropout=0.0)
    ann = torch.ones(2, 1, 6, 6, 6)
    ann[:, :, :3] = 2
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 6, 6, 6) * 5, ann), batch_size=2)
    return LoRAFinetuner(
        model, loader, output_dir=str(tmp_path), num_epochs=2, device="cpu",
        use_mixed_precision=False, loss_type="bce", target_transform=BinaryTargetTransform(),
        tensorboard=False, **kw,
    )


def _batchnorm(model):
    return next(m for m in model.modules() if isinstance(m, nn.BatchNorm3d))


@pytest.mark.finetune
def test_lora_leaves_the_bases_batchnorm_statistics_alone(tmp_path):
    trainer = _lora_trainer(tmp_path)
    bn = _batchnorm(trainer.model)
    before = (bn.running_mean.clone(), bn.running_var.clone())
    trainer.train()
    assert torch.equal(bn.running_mean, before[0])
    assert torch.equal(bn.running_var, before[1])


@pytest.mark.finetune
def test_the_lora_teacher_is_deterministic(tmp_path):
    trainer = _lora_trainer(tmp_path, distillation_lambda=0.5)
    trainer._set_train_mode()
    x = torch.rand(1, 1, 6, 6, 6)
    first, second = trainer._teacher_forward(x), trainer._teacher_forward(x)
    assert torch.equal(first, second)
    assert trainer.model.training, "back in train mode after the teacher pass"


def test_a_full_finetune_still_trains_its_batchnorm(tmp_path):
    ann = torch.ones(2, 1, 6, 6, 6)
    ann[:, :, :3] = 2
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 6, 6, 6) * 5, ann), batch_size=2)
    model = _base()
    trainer = LoRAFinetuner(
        model, loader, output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", target_transform=BinaryTargetTransform(),
        tensorboard=False,
    )
    before = _batchnorm(model).running_mean.clone()
    trainer.train()
    assert not torch.equal(_batchnorm(model).running_mean, before)
