"""The masked means every finetune loss is made of, and the losses built on them."""

import math

import pytest
import torch

from cellmap_flow.finetune.losses import (
    CombinedLoss,
    DiceLoss,
    MarginLoss,
    as_probabilities,
    balanced_mean,
    distillation_loss,
    masked_mean,
    soft_target_entropy,
)

X = torch.tensor([1.0, 2.0, 3.0, 10.0])
TARGET = torch.tensor([1.0, 0.0, 0.0, 1.0])
MASK = torch.tensor([1.0, 1.0, 1.0, 0.0])  # the last voxel is unannotated
TWO_CHANNELS = torch.arange(8.0).reshape(1, 2, 4)
ONE_CHANNEL = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])


@pytest.mark.parametrize("value, expected", [
    (masked_mean(X, MASK), 2.0),
    (masked_mean(X, torch.zeros(4)), 0.0),  # nothing supervised: 0, not NaN
    (balanced_mean(X, TARGET, MASK), (1.0 + 2.5) / 2),  # one fg voxel weighs as much as two bg
    (distillation_loss(X, torch.zeros(4), "all"), (1 + 4 + 9 + 100) / 4),
    (distillation_loss(X, torch.zeros(4), "unlabeled", unlabeled_mask=1.0 - MASK), 100.0),
    (distillation_loss(X, torch.zeros(4), "anchor", anchor_mask=TARGET), (1 + 100) / 2),
    # a good region is about location: one channel, broadcast over the output's
    (distillation_loss(TWO_CHANNELS, torch.zeros(1, 2, 4), "anchor", anchor_mask=ONE_CHANNEL), (0 + 16) / 2),
    # BCE on a soft target cannot go below the target's entropy: log 2 at 0.5, 0 when hard.
    (soft_target_entropy(torch.tensor(0.5)), math.log(2)),
    (soft_target_entropy(torch.tensor(1.0)), 0.0),
    (soft_target_entropy(torch.tensor(0.1)), -(0.1 * math.log(0.1) + 0.9 * math.log(0.9))),  # BCE(t, t)
    # A model that ends in a sigmoid is not squashed again (into [0.5, 0.73]).
    (as_probabilities(torch.tensor(0.3), model_has_sigmoid=True), 0.3),
    (as_probabilities(torch.tensor(0.0), model_has_sigmoid=False), 0.5),
])
def test_the_means(value, expected):
    assert value.item() == pytest.approx(expected, abs=1e-5)


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
