"""The masked means every finetune loss is made of, and the losses built on them."""

import pytest
import torch

from cellmap_flow.finetune import losses, lora_trainer
from cellmap_flow.finetune.losses import (
    CombinedLoss,
    DiceLoss,
    MarginLoss,
    balanced_mean,
    distillation_loss,
    masked_mean,
)

X = torch.tensor([1.0, 2.0, 3.0, 10.0])
TARGET = torch.tensor([1.0, 0.0, 0.0, 1.0])
MASK = torch.tensor([1.0, 1.0, 1.0, 0.0])  # the last voxel is unannotated


@pytest.mark.parametrize("value, expected", [
    (masked_mean(X, MASK), 2.0),
    (masked_mean(X, torch.zeros(4)), 0.0),  # nothing supervised: 0, not NaN
    (balanced_mean(X, TARGET, MASK), (1.0 + 2.5) / 2),  # one fg voxel weighs as much as two bg
    (distillation_loss(X, torch.zeros(4), "all"), (1 + 4 + 9 + 100) / 4),
    (distillation_loss(X, torch.zeros(4), "unlabeled", unlabeled_mask=1.0 - MASK), 100.0),
    (distillation_loss(X, torch.zeros(4), "anchor", anchor_mask=TARGET), (1 + 100) / 2),
])
def test_the_means(value, expected):
    assert value.item() == pytest.approx(expected)


def test_an_anchor_mask_is_broadcast_over_channels():
    student = torch.arange(8.0).reshape(1, 2, 4)  # two channels
    anchor = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # one
    assert distillation_loss(student, torch.zeros_like(student), "anchor", anchor_mask=anchor).item() == 8.0


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="scope"):
        distillation_loss(X, X, "everywhere")


@pytest.mark.parametrize("loss", [DiceLoss(), CombinedLoss(), MarginLoss(), MarginLoss(balance_classes=True)])
def test_an_unannotated_voxel_does_not_count(loss):
    shape = (1, 1, 4)
    target, mask = TARGET.reshape(shape), MASK.reshape(shape)
    pred = torch.tensor([2.0, -1.0, 0.5, 3.0]).reshape(shape)
    moved = pred.clone()
    moved[..., 3] = -30.0
    assert torch.equal(loss(pred, target, mask), loss(moved, target, mask))
    assert not torch.equal(loss(pred, target), loss(moved, target))


def test_the_trainer_still_exports_the_losses():
    for name in losses.__all__:
        assert getattr(lora_trainer, name) is getattr(losses, name)
