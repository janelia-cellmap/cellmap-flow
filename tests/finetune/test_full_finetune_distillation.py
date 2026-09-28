"""Distillation on a full finetune (--lora-r 0).

The teacher used to be the model with its LoRA adapters switched off. A full
finetune has no adapters, so the first batch died with
``AttributeError: ... has no attribute 'disable_adapter_layers'`` -- and the
dashboard's default distillation weight is 0.01, marking a good region forces
it to 1.0, so "rank 0" from the dashboard could not train at all. The teacher
is now a frozen copy of the starting weights.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _data():
    torch.manual_seed(0)
    raw = torch.rand(4, 1, 6, 6, 6)
    ann = torch.zeros(4, 1, 6, 6, 6)
    ann[:, :, :3] = 1  # background on half the patch, the rest unannotated
    ann[:, :, :2, :2] = 2
    return DataLoader(TensorDataset(raw, ann), batch_size=2)


def _model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv3d(4, 1, 1))


def test_full_finetune_trains_with_distillation(tmp_path):
    model = _model()
    start = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = LoRAFinetuner(
        model, _data(), output_dir=str(tmp_path), num_epochs=3, device="cpu",
        use_mixed_precision=False, loss_type="bce", distillation_lambda=0.5,
        target_transform=BinaryTargetTransform(), tensorboard=False,
        learning_rate=1e-2,
    )
    stats = trainer.train()

    assert not stats.get("diverged")
    teacher = trainer.teacher_model
    assert teacher is not None and teacher is not model
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    # The teacher is the model as it started; the student moved away from it.
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, start[k])
    assert any(not torch.equal(v, start[k]) for k, v in model.state_dict().items())


def test_a_given_teacher_is_reused(tmp_path):
    from cellmap_flow.finetune.lora_trainer import frozen_teacher_copy

    model = _model()
    teacher = frozen_teacher_copy(model)
    trainer = LoRAFinetuner(
        model, _data(), output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", distillation_lambda=0.5,
        target_transform=BinaryTargetTransform(), tensorboard=False,
        teacher_model=teacher,
    )
    assert trainer.teacher_model is teacher


def test_no_teacher_is_made_without_distillation(tmp_path):
    trainer = LoRAFinetuner(
        _model(), _data(), output_dir=str(tmp_path), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", distillation_lambda=0.0,
        target_transform=BinaryTargetTransform(), tensorboard=False,
    )
    assert trainer.teacher_model is None
