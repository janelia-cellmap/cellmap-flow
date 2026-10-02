"""Target transforms: an annotation (0 unannotated, 1 background, 2+ objects) as
the target and the mask of voxels a loss is computed on."""

import math

import numpy as np
import pytest
import torch

from cellmap_flow.finetune.target_transforms import (
    AffinityTargetTransform,
    BinaryTargetTransform,
    BroadcastBinaryTargetTransform,
    DistanceTargetTransform,
    IntervalTargetTransform,
    read_offsets_from_script,
)


def _line(*values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)


X_NEXT, X_PREVIOUS = [[0, 0, 1]], [[0, 0, -1]]


@pytest.mark.parametrize("transform, ann, target, mask", [
    pytest.param(BinaryTargetTransform(), [0, 1, 2, 0, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 1],
                 id="binary: only what was annotated counts"),
    pytest.param(BinaryTargetTransform(), [1, 2, 3], [0, 1, 1], [1, 1, 1], id="binary: every object is foreground"),
    # Along x: 1 inside an object, 0 across a boundary or between two objects;
    # masked where either voxel is unannotated or has no neighbour.
    pytest.param(AffinityTargetTransform(X_NEXT), [1, 2, 2, 1], [0, 1, 0, 0], [1, 1, 1, 0],
                 id="affinity: inside an object"),
    pytest.param(AffinityTargetTransform(X_NEXT), [2, 3, 2], [0, 0, 0], [1, 1, 0],
                 id="affinity: between two objects"),
    pytest.param(AffinityTargetTransform(X_NEXT), [2, 0, 2, 1], [0, 0, 0, 0], [0, 0, 1, 0],
                 id="affinity: next to an unannotated voxel"),
    pytest.param(AffinityTargetTransform(X_PREVIOUS), [1, 2, 2, 1], [0, 0, 1, 0], [0, 1, 1, 1],
                 id="affinity: a negative offset"),
])
def test_targets_along_a_line(transform, ann, target, mask):
    t, m = transform(_line(*ann))
    assert (t[0, 0, 0, 0].tolist(), m[0, 0, 0, 0].tolist()) == (target, mask)


def test_affinities_along_each_axis():
    ann = torch.full((1, 1, 3, 3, 3), 2.0)
    ann[0, 0, 0, 0, 0] = 1  # a background corner
    target, mask = AffinityTargetTransform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])(ann)
    assert target.shape == mask.shape == (1, 3, 3, 3, 3)
    assert mask[0, 0, 2].sum() == 0 and mask[0, 0, :2].sum() == 18  # z = 2 has no z + 1
    assert (target[0, 0, 0, 0, 0], target[0, 0, 1, 0, 0]) == (0, 1)


@pytest.mark.parametrize("transform, channels, supervised", [
    pytest.param(BroadcastBinaryTargetTransform(num_channels=3), 3, [True] * 3, id="binary, broadcast"),
    # Channels past the offsets (LSDs, say) are not supervised.
    pytest.param(AffinityTargetTransform(X_NEXT, num_channels=4), 4, [True, False, False, False],
                 id="affinities and extra channels"),
    pytest.param(DistanceTargetTransform(2.0, num_channels=3), 3, [True] * 3, id="distance, broadcast"),
])
def test_every_model_channel_gets_a_target(transform, channels, supervised):
    ann = torch.ones((2, 1, 7, 7, 7))
    ann[..., 3:] = 2
    target, mask = transform(ann)
    assert target.shape == mask.shape == (2, channels, 7, 7, 7) and target.device == ann.device
    assert [bool(mask[:, c].any()) for c in range(channels)] == supervised
    assert all(torch.equal(target[:, c], target[:, 0]) for c in range(channels) if supervised[c])


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


def test_a_voxel_whose_boundary_may_lie_beyond_the_patch_is_not_trusted():
    """Along the line |d| grows away from the boundary while the distance to the
    patch's edge shrinks; once |d| exceeds it, a closer boundary could sit
    beyond the edge, and the voxel is not supervised."""
    _, mask = _distance(_slab(), 100.0)  # never saturates in this patch
    line = mask[1, 1]
    assert line[10] == 1  # |d| = 1, and the y/z faces are 2 away
    assert line[12] == 0  # |d| = 3
    assert line[0] == 0 and line[20] == 0


def test_a_saturated_voxel_is_trusted_whatever_lies_beyond_the_patch():
    """Within 3 sigma of trust the value is saturated anyway: a boundary further off changes nothing."""
    cube = np.ones((1, 1, 15, 15, 15), np.float32)
    cube[..., 8:] = 2
    _, mask = _distance(cube, 2.0)
    assert mask[7, 7, 7] == 1  # |d| = 1, trust 8
    assert mask[7, 7, 3] == 0  # |d| = 5 over its trust of 4, which is under 3 sigma
    target, mask = _distance(cube, 0.5)
    assert mask[7, 7, 3] == 1 and target[7, 7, 3] < 1e-3  # trust 4 is 3 sigma: saturated, and kept


def test_a_patch_without_a_boundary_is_trusted_only_far_from_its_edges():
    target, mask = _distance(np.full((1, 1, 9, 9, 9), 2, np.float32), 1.0)
    assert np.all(target == 1.0)
    assert [mask[z, 4, 4] for z in (0, 1, 2, 4)] == [0, 0, 1, 1]  # 3 sigma from the padded edge


def test_unannotated_voxels_are_unknown_not_background():
    """Zeros get no target weight, and they shorten their annotated neighbours' trust."""
    ann = np.ones((1, 1, 9, 9, 9), np.float32)
    ann[..., 4:] = 2
    ann[0, 0, 4, 4, 0:2] = 0  # an unannotated pocket in the background
    _, mask = _distance(ann, 1.0)
    assert mask[4, 4, 0] == 0 and mask[4, 4, 1] == 0
    assert mask[4, 4, 2] == 0  # |d| = 2 from the object, but the pocket is 1 away
    assert mask[2, 4, 2] == 1  # the same column without a pocket


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


def test_a_distance_target_needs_a_positive_sigma():
    with pytest.raises(ValueError):
        DistanceTargetTransform(0.0)


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


@pytest.mark.parametrize("script, expected", [
    pytest.param("offsets = [[1, 0, 0], [0, 1, 0]]\nmodel = None\n", [[1, 0, 0], [0, 1, 0]], id="offsets"),
    pytest.param("model = None\n", None, id="no offsets"),
    pytest.param("offsets = [[1, 0, 0]\n", None, id="a script that does not parse"),
])
def test_read_offsets_from_script(tmp_path, script, expected):
    path = tmp_path / "model.py"
    path.write_text(script)
    assert read_offsets_from_script(path) == expected
