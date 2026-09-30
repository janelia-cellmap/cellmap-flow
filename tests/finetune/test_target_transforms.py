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
    read_offsets_from_script,
)


def _line(*values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, -1)


@pytest.mark.parametrize("transform, ann, target, mask", [
    (BinaryTargetTransform(), [0, 1, 2, 0, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 1]),
    (BinaryTargetTransform(), [1, 2, 3], [0, 1, 1], [1, 1, 1]),  # every object is foreground
    # Along x: 1 inside an object, 0 across a boundary or between two objects;
    # masked where either voxel is unannotated or has no neighbour.
    (AffinityTargetTransform([[0, 0, 1]]), [1, 2, 2, 1], [0, 1, 0, 0], [1, 1, 1, 0]),
    (AffinityTargetTransform([[0, 0, 1]]), [2, 3, 2], [0, 0, 0], [1, 1, 0]),
    (AffinityTargetTransform([[0, 0, 1]]), [2, 0, 2, 1], [0, 0, 0, 0], [0, 0, 1, 0]),
    (AffinityTargetTransform([[0, 0, -1]]), [1, 2, 2, 1], [0, 0, 1, 0], [0, 1, 1, 1]),
])
def test_along_a_line(transform, ann, target, mask):
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
    (BroadcastBinaryTargetTransform(num_channels=3), 3, [True] * 3),
    (AffinityTargetTransform([[0, 0, 1]], num_channels=4), 4, [True, False, False, False]),  # LSDs: unsupervised
    (DistanceTargetTransform(2.0, num_channels=3), 3, [True] * 3),
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


def _ann(shape, fill, *boxes):
    ann = np.full((1, 1, *shape), fill, dtype=np.float32)
    for value, where in boxes:
        ann[(0, 0, *where)] = value
    return torch.from_numpy(ann)


SLAB = _ann((3, 3, 21), 1, (2, np.s_[..., 10:]))  # background, then foreground from x = 10
X = lambda x: (1, 1, x)  # noqa: E731  a voxel on the slab's middle line


@pytest.mark.parametrize("ann, sigma, voxels", [
    # tanh of the signed distance, 0.5 on the boundary: voxel 10 is 1 inside, 9 one outside.
    (SLAB, 2.0, {X(10): (_soft(1, 2), 1), X(9): (_soft(-1, 2), 1), X(14): (_soft(5, 2), None),
                 X(5): (_soft(-5, 2), None)}),
    # A voxel further from the boundary than from the patch's edge may have a
    # closer boundary beyond the edge: it is not supervised.
    (SLAB, 100.0, {X(10): (None, 1), X(12): (None, 0), X(0): (None, 0), X(20): (None, 0)}),
    # ... unless its value is saturated anyway (3 sigma of trust).
    (_ann((15, 15, 15), 1, (2, np.s_[..., 8:])), 2.0, {(7, 7, 7): (None, 1), (7, 7, 3): (None, 0)}),
    (_ann((15, 15, 15), 1, (2, np.s_[..., 8:])), 0.5, {(7, 7, 3): (_soft(-5, 0.5), 1)}),
    # Unannotated is unknown, not background: no target, and it shortens its neighbours' trust.
    (_ann((9, 9, 9), 1, (2, np.s_[..., 4:]), (0, np.s_[4, 4, 0:2])), 1.0,
     {(4, 4, 0): (None, 0), (4, 4, 2): (None, 0), (2, 4, 2): (None, 1)}),
    # A crop cutting through an object: its depth is measured to annotated
    # background, not to the unannotated voxels past the crop's edge.
    (_ann((3, 3, 21), 1, (2, np.s_[..., 5:15]), (0, np.s_[..., 15:])), 6.0,
     {X(13): (None, 0), X(5): (_soft(1, 6), 1)}),
    # No boundary at all: saturated, and trusted only 3 sigma from every edge.
    (_ann((9, 9, 9), 2), 1.0, {(4, 4, 4): (1.0, 1), (0, 4, 4): (1.0, 0), (1, 4, 4): (None, 0), (2, 4, 4): (None, 1)}),
    (SLAB, 0.0, None),
], ids=["profile", "trust", "saturated", "saturated at a small sigma", "unannotated", "cut by the crop",
        "all foreground", "sigma 0"])
def test_a_distance_target_and_where_it_is_trusted(ann, sigma, voxels):
    if voxels is None:
        with pytest.raises(ValueError):
            DistanceTargetTransform(sigma)
        return
    target, mask = DistanceTargetTransform(sigma)(ann)
    assert 0.0 <= target.min() and target.max() <= 1.0
    for voxel, (value, trusted) in voxels.items():
        if value is not None:
            assert target[(0, 0, *voxel)].item() == pytest.approx(value, abs=1e-6), voxel
        if trusted is not None:
            assert mask[(0, 0, *voxel)].item() == trusted, voxel


@pytest.mark.parametrize("script, expected", [
    ("offsets = [[1, 0, 0], [0, 1, 0]]\nmodel = None\n", [[1, 0, 0], [0, 1, 0]]),
    ("model = None\n", None),
    ("offsets = [[1, 0, 0]\n", None),  # does not parse
])
def test_read_offsets_from_script(tmp_path, script, expected):
    path = tmp_path / "model.py"
    path.write_text(script)
    assert read_offsets_from_script(path) == expected
