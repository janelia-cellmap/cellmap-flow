"""The masked means every finetune loss is made of, and the losses built on them."""

import math

import pytest
import torch

from cellmap_flow.finetune.losses import (
    CombinedLoss,
    DiceLoss,
    IntervalLoss,
    MarginLoss,
    as_logits,
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
    pytest.param(masked_mean(X, MASK), 2.0, id="masked mean"),
    pytest.param(masked_mean(X, torch.zeros(4)), 0.0, id="nothing supervised is 0, not NaN"),
    pytest.param(balanced_mean(X, TARGET, MASK), (1.0 + 2.5) / 2, id="one fg voxel weighs as much as two bg"),
    pytest.param(distillation_loss(X, torch.zeros(4), "all"), (1 + 4 + 9 + 100) / 4, id="distil everywhere"),
    pytest.param(distillation_loss(X, torch.zeros(4), "unlabeled", unlabeled_mask=1.0 - MASK), 100.0,
                 id="distil the unlabeled voxels"),
    pytest.param(distillation_loss(X, torch.zeros(4), "anchor", anchor_mask=TARGET), (1 + 100) / 2,
                 id="distil the anchored voxels"),
    # A good region is about location: one channel, broadcast over the output's.
    pytest.param(distillation_loss(TWO_CHANNELS, torch.zeros(1, 2, 4), "anchor", anchor_mask=ONE_CHANNEL),
                 (0 + 16) / 2, id="a one-channel anchor over two channels"),
])
def test_the_means(value, expected):
    assert value.item() == pytest.approx(expected)


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="scope"):
        distillation_loss(X, X, "everywhere")


@pytest.mark.parametrize("loss", [
    pytest.param(DiceLoss(), id="dice"),
    pytest.param(CombinedLoss(), id="dice and bce"),
    pytest.param(MarginLoss(), id="margin"),
    pytest.param(MarginLoss(balance_classes=True), id="balanced margin"),
])
def test_an_unannotated_voxel_does_not_count(loss):
    shape = (1, 1, 4)
    target, mask = TARGET.reshape(shape), MASK.reshape(shape)
    pred = torch.tensor([2.0, -1.0, 0.5, 3.0]).reshape(shape)
    moved = pred.clone()
    moved[..., 3] = -30.0
    assert torch.equal(loss(pred, target, mask), loss(moved, target, mask))
    assert not torch.equal(loss(pred, target), loss(moved, target))


@pytest.mark.parametrize("target, floor", [
    pytest.param(0.5, math.log(2), id="0.5 pays log 2"),
    pytest.param(1.0, 0.0, id="a hard target pays nothing"),
    pytest.param(0.1, -(0.1 * math.log(0.1) + 0.9 * math.log(0.9)), id="BCE of the target against itself"),
])
def test_bce_on_a_soft_target_cannot_go_below_its_entropy(target, floor):
    """The trainer reports BCE above this floor: no prediction, however well
    calibrated, gets lower on a soft target."""
    assert soft_target_entropy(torch.tensor(target)).item() == pytest.approx(floor, abs=1e-5)


def test_a_model_ending_in_a_sigmoid_is_not_squashed_again():
    """A sigmoid of a probability lands in [0.5, 0.73], which wrecks any threshold."""
    logits = torch.tensor([-3.0, 0.0, 3.0])
    probabilities = torch.sigmoid(logits)
    assert torch.equal(as_probabilities(probabilities, model_has_sigmoid=True), probabilities)
    assert torch.allclose(as_probabilities(logits, model_has_sigmoid=False), probabilities)


def test_a_model_ending_in_a_sigmoid_is_taken_back_to_logits():
    logits = torch.tensor([-6.0, 0.0, 6.0])
    assert torch.allclose(as_logits(torch.sigmoid(logits), model_has_sigmoid=True), logits, atol=1e-4)
    assert torch.equal(as_logits(logits, model_has_sigmoid=False), logits)


# sigma 6 voxels: a slack of one voxel is a logit of 1/3.
INTERVAL = IntervalLoss(sigma_nm=6.0, voxel_size_nm=(1, 1, 1), slope_weight=0.0)
FOREGROUND_BOUNDS = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1, 1)  # 3 to 6 voxels inside


@pytest.mark.parametrize("z, expected", [
    pytest.param(1.5, 0.0, id="inside the bounds"),
    pytest.param(0.8, 0.0, id="within the slack of a bound"),
    pytest.param(0.5, 1.0 - 1 / 3 - 0.5, id="below the lower bound"),
    pytest.param(3.0, 3.0 - 2.0 - 1 / 3, id="above the upper bound"),
    # Past the slack on the wrong side, each logit further costs five more.
    pytest.param(0.0, 1.0 - 1 / 3, id="on the boundary"),
    pytest.param(-1.0, (1.0 - 1 / 3 + 1.0) + 5 * (1.0 - 1 / 3), id="on the wrong side"),
])
def test_the_interval_loss_is_zero_inside_the_bounds(z, expected):
    assert INTERVAL(torch.full((1, 1, 1, 1, 1), z), FOREGROUND_BOUNDS, torch.ones(1, 1, 1, 1, 1)).item() == \
        pytest.approx(expected, abs=1e-6)


def test_background_bounds_are_negative_and_an_unpainted_voxel_is_free():
    bounds = torch.tensor([[-2.0, 1.0], [-1.0, 2.0]]).reshape(1, 2, 1, 1, 2)  # background, then unpainted
    z = torch.tensor([-1.5, 50.0]).reshape(1, 1, 1, 1, 2)
    assert INTERVAL(z, bounds, torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2)).item() == 0.0
    assert INTERVAL(-z, bounds, torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2)).item() > 0.0


@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (4, 2, 2)])
def test_the_slope_limit_holds_back_a_field_steeper_than_a_distance_field(voxel_size):
    """A sphere's signed distance in logits, 2d/sigma, passes; twice as steep does not."""
    sigma_nm = 8.0 * min(voxel_size)
    grid = torch.meshgrid(*[torch.arange(24.0) * v for v in voxel_size], indexing="ij")
    d = 20.0 - torch.sqrt(sum((g - 12 * v) ** 2 for g, v in zip(grid, voxel_size)))
    z = (2 * d / sigma_nm)[None, None]
    slope = IntervalLoss(sigma_nm, voxel_size).slope_term
    assert slope(z).item() == 0.0
    assert slope(2 * z).item() > 0.0
