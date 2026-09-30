"""A non-finite loss never reaches the weights, and a full finetune can be reset.

The NaN check ran after scaler.step(). With the scaler off (bf16, fp32) AdamW
had already written NaN into every trainable weight by then. LoRA recovered on
restart by re-making its adapter; a full finetune kept the NaN weights, served
them, and restarted from them, since nothing ever reset its weights.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import BinaryTargetTransform


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.ReLU(), nn.Conv3d(2, 1, 1))


def _loader(nan_in_second_batch):
    torch.manual_seed(1)
    raw = torch.rand(4, 1, 5, 5, 5)
    if nan_in_second_batch:
        raw[2:] = float("nan")
    ann = torch.ones(4, 1, 5, 5, 5)
    ann[:, :, :2] = 2
    return DataLoader(TensorDataset(raw, ann), batch_size=2)


def _trainer(tmp_path, loader, **kw):
    return LoRAFinetuner(
        _model(), loader, output_dir=str(tmp_path), num_epochs=2, device="cpu",
        use_mixed_precision=False, loss_type="bce", tensorboard=False,
        target_transform=BinaryTargetTransform(), **kw,
    )


def test_a_nan_batch_leaves_the_weights_finite(tmp_path):
    trainer = _trainer(tmp_path, _loader(nan_in_second_batch=True))
    stats = trainer.train()
    assert stats["diverged"]
    for name, p in trainer.model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} was updated with a NaN step"


def test_gradient_accumulation_does_not_apply_a_nan_either(tmp_path):
    trainer = _trainer(
        tmp_path, _loader(nan_in_second_batch=True), gradient_accumulation_steps=2
    )
    trainer.train()
    for p in trainer.model.parameters():
        assert torch.isfinite(p).all()


def test_a_full_finetune_resets_to_its_starting_weights(tmp_path):
    trainer = _trainer(tmp_path, _loader(nan_in_second_batch=False), learning_rate=1e-2)
    start = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.train()
    assert any(not torch.equal(v, start[k]) for k, v in trainer.model.state_dict().items())

    trainer._reset_training_state()
    for k, v in trainer.model.state_dict().items():
        assert torch.equal(v, start[k])


def test_a_restart_resets_a_full_finetune(tmp_path):
    from cellmap_flow.finetune.finetune_cli import _reset_for_restart
    from cellmap_flow.finetune.adaptation import cpu_state_copy

    model = _model()
    initial = cpu_state_copy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    args = argparse.Namespace(lora_r=0, lora_alpha=0, lora_dropout=0.1, lora_min_channels=0)

    restarted = _reset_for_restart(model, args, initial)

    assert restarted is model, "the served model object must stay the same"
    assert restarted.training
    for k, v in restarted.state_dict().items():
        assert torch.equal(v, initial[k])
