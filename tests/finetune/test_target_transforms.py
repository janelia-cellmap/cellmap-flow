"""Target transforms: an annotation (0 unannotated, 1 background, 2+ objects) as
the target and the mask of voxels a loss is computed on. The maths is
cellmap_flow/portable/targets.py, which the browser runs too, through
portable/targets_calls.py; the server's classes wrap it."""

import json
import math

import numpy as np
import pytest
import torch

from cellmap_flow.finetune.target_transforms import (
    AffinityTargetTransform,
    BroadcastBinaryTargetTransform,
    DistanceTargetTransform,
    IntervalTargetTransform,
)
from cellmap_flow.portable import targets


def test_the_binary_target_is_the_objects_masked_where_painted_on_every_channel():
    ann = torch.tensor([0, 1, 2, 3], dtype=torch.float32).reshape(1, 1, 1, 1, 4)
    target, mask = BroadcastBinaryTargetTransform(2)(ann)
    assert target.shape == mask.shape == (1, 2, 1, 1, 4) and target.dtype == torch.float32
    assert target[0, :, 0, 0].tolist() == [[0, 0, 1, 1]] * 2
    assert mask[0, :, 0, 0].tolist() == [[0, 1, 1, 1]] * 2


def test_an_affinity_is_one_within_an_object_and_counts_where_both_voxels_are_painted():
    """Along x: 2|2 the same object, 2|3 two objects, 3|1 object and background,
    1|0 a pair half unpainted, and the last voxel's neighbour past the patch.
    A channel past the offsets (an LSD, say) and one whose offset is None
    train nothing."""
    ann = torch.tensor([2, 2, 3, 1, 0], dtype=torch.float32).reshape(1, 1, 1, 1, 5)
    target, mask = AffinityTargetTransform([[0, 0, 1], None], num_channels=3)(ann)
    assert target[0, 0, 0, 0].tolist() == [1, 0, 0, 0, 0]
    assert mask[0, 0, 0, 0].tolist() == [1, 1, 1, 0, 0]
    assert not mask[0, 1:].any() and not target[0, 1:].any()


@pytest.mark.parametrize("dz", [3, 4, 7])
def test_an_offset_no_shorter_than_the_patch_pairs_nothing(dz):
    """A one-slice patch's z channel (a 2D student's head) has an empty mask,
    and so has a long-range offset past a short patch, either way along the
    axis. Between one and two patch lengths that used to raise, as slicing
    counted the stop from the far end."""
    ann = np.full((3, 4, 4), 2, np.uint8)
    _, one_slice = targets.affinity(ann[:1], [[1, 0, 0], [0, 1, 0]])
    assert not one_slice[0].any() and one_slice[1].any()
    target, mask = targets.affinity(ann, [[dz, 0, 0], [-dz, 0, 0], [2, 0, 0]])
    assert not mask[:2].any() and not target[:2].any()
    assert mask[2, :1].all() and not mask[2, 1:].any()


def test_the_browser_call_answers_a_chunks_targets_as_the_server_makes_them():
    """targets_calls.targets marshalled as chunkmirage's worker marshals it: an
    instance volume's uint16 ids in, float32 (channels, z, y, x) back."""
    from chunkmirage import calls

    rng = np.random.default_rng(0)
    labels = rng.choice(np.array([0, 1, 2, 300, 40000], np.uint16), size=(2, 3, 4))
    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1], None]
    args = {"shape": [2, 3, 4], "labels_dtype": "uint16", "channels": 5}
    for given, server in [(None, BroadcastBinaryTargetTransform(5)), (offsets, AffinityTargetTransform(offsets, 5))]:
        value, (target, mask) = calls.call("cellmap_flow.portable.targets_calls:targets", json.dumps({**args, "offsets": given}), [labels.tobytes()])
        want = [t[0].numpy() for t in server(torch.from_numpy(labels.astype(np.float32))[None, None])]
        assert json.loads(value) == {"channels": 5}
        assert [np.frombuffer(b, np.float32).reshape(5, 2, 3, 4).tolist() for b in (target, mask)] == [w.tolist() for w in want]


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
