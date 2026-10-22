"""The interval loss a distance model trains on sparse paint with: free inside
its bounds, and held back where its field is steeper than a distance field."""

import pytest
import torch

from cellmap_flow.finetune.losses import IntervalLoss

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
