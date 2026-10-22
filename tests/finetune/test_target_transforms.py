"""Target transforms: an annotation (0 unannotated, 1 background, 2+ objects) as
the target and the mask of voxels a loss is computed on. The binary and
affinity targets are checked against the browser's by web/test/parity/targets."""

import math

import numpy as np
import pytest
import torch

from cellmap_flow.finetune.target_transforms import DistanceTargetTransform, IntervalTargetTransform


def _soft(d, sigma):
    return (math.tanh(d / sigma) + 1.0) / 2.0


def _distance(ann, sigma):
    target, mask = DistanceTargetTransform(sigma)(torch.from_numpy(ann))
    return target[0, 0].numpy(), mask[0, 0].numpy()


def _slab():
    """A 3x3x21 patch of background whose foreground starts at x = 10."""
    ann = np.ones((1, 1, 3, 3, 21), np.float32)
    ann[..., 10:] = 2
    return ann


def test_the_distance_target_is_the_tanh_of_the_signed_distance():
    """0.5 on the boundary, as the fly-organelles models were trained: voxel 10
    is 1 inside the object, voxel 9 is 1 outside it."""
    target, _ = _distance(_slab(), 2.0)
    line = target[1, 1]
    assert [line[x] for x in (10, 9, 14, 5)] == pytest.approx([_soft(d, 2.0) for d in (1, -1, 5, -5)], abs=1e-6)
    assert np.all(np.diff(line) >= 0) and 0.0 <= line.min() and line.max() <= 1.0


def test_the_foreground_of_a_cut_object_is_measured_to_annotated_background():
    """A dense crop cutting through an object: its foreground runs to the crop's
    edge, with unannotated voxels beyond. Counting those as background taught a
    boundary that is not there; measured to real background, 8+ voxels away,
    voxels near the unknown are left out instead."""
    ann = np.ones((1, 1, 3, 3, 21), np.float32)
    ann[..., 5:15] = 2
    ann[..., 15:] = 0
    target, mask = _distance(ann, 6.0)
    assert mask[1, 1, 13] == 0  # 2 from the unknown, 9 from real background
    assert mask[1, 1, 5] == 1 and target[1, 1, 5] == pytest.approx(_soft(1.0, 6.0), abs=1e-6)


def _bounds(ann, sigma, voxel_size=None):
    """The (lower, upper) logit bounds and mask of a (Z, Y, X) annotation."""
    bounds, mask = IntervalTargetTransform(sigma, voxel_size)(torch.from_numpy(ann[None, None].astype(np.float32)))
    return bounds[0, 0].numpy(), bounds[0, 1].numpy(), mask[0, 0].numpy()


def test_dense_paint_bounds_the_distance_exactly():
    """Where the other class is nearer than anything unknown the two bounds meet,
    at the distance target's own logit 2d/sigma; nearer the patch edge than the
    other class, the edge is the lower bound."""
    ann = np.ones((15, 15, 15))
    ann[..., 8:] = 2
    lower, upper, _ = _bounds(ann, 6.0)
    for x, d in [(5, -3), (7, -1), (8, 1), (9, 2)]:
        assert lower[7, 7, x] == upper[7, 7, x] == pytest.approx(2 * d / 6.0)
    assert (lower[7, 7, 13], upper[7, 7, 13]) == pytest.approx((2 * 2 / 6.0, 2 * 6 / 6.0))


def test_past_three_sigma_a_bound_saturates():
    lower, upper, _ = _bounds(np.full((21, 21, 21), 2), 1.0)
    assert (lower[10, 10, 10], upper[10, 10, 10]) == (6.0, np.inf)  # 11 voxels from the edge, no background


@pytest.mark.parametrize("voxel_size", [(1, 1, 1), (3, 1, 1)])
def test_scribbles_bracket_the_true_distance(voxel_size):
    """Two strokes through a sphere, in a patch that cuts it: every painted voxel's
    true distance, measured on the whole labelled volume, lies within its bounds."""
    from scipy.ndimage import distance_transform_edt as edt

    grid = np.meshgrid(*[np.arange(32) * v for v in voxel_size], indexing="ij")
    fg = sum((g - 16 * v) ** 2 for g, v in zip(grid, voxel_size)) <= 10 ** 2
    truth = np.where(fg, edt(fg, sampling=voxel_size), -edt(~fg, sampling=voxel_size))
    painted = np.zeros(fg.shape, bool)
    painted[16], painted[:, 12] = True, True
    ann = np.where(painted, np.where(fg, 2, 1), 0)
    patch = np.s_[4:28, 10:30, 3:20]
    lower, upper, mask = _bounds(ann[patch], 4.0, voxel_size)
    logit = 2 * truth[patch] / (4.0 * min(voxel_size))
    on = mask > 0
    assert np.all((lower[on] <= logit[on] + 1e-5) & (logit[on] <= upper[on] + 1e-5))
    assert (lower[on] < upper[on]).mean() > 0.5  # a stroke brackets, it does not pin down
    assert not lower[~on].any() and not upper[~on].any()
