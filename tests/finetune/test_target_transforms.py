"""Tests for target transforms."""

import torch
from cellmap_flow.finetune.target_transforms import (
    BinaryTargetTransform,
    BroadcastBinaryTargetTransform,
    AffinityTargetTransform,
    _offset_slices,
)


def test_binary_transform_basic():
    """Test that BinaryTargetTransform produces correct targets and masks."""
    # annotation: 0=unannotated, 1=bg, 2=fg
    annotation = torch.tensor([[[[[0, 1, 2, 0, 1]]]]]).float()  # (1, 1, 1, 1, 5)
    transform = BinaryTargetTransform()
    target, mask = transform(annotation)

    # mask: 1 where annotated (>0)
    assert mask.tolist() == [[[[[0, 1, 1, 0, 1]]]]]
    # target: 0 for bg (was 1), 1 for fg (was 2), 0 for unannotated
    assert target.tolist() == [[[[[0, 0, 1, 0, 0]]]]]


def test_binary_transform_multi_object():
    """Labels 2 and 3 both become foreground (1)."""
    annotation = torch.tensor([[[[[1, 2, 3]]]]]).float()
    transform = BinaryTargetTransform()
    target, mask = transform(annotation)

    assert target.tolist() == [[[[[0, 1, 1]]]]]
    assert mask.tolist() == [[[[[1, 1, 1]]]]]


def test_broadcast_transform():
    """Test broadcasting to multiple channels."""
    annotation = torch.tensor([[[[[0, 1, 2]]]]]).float()  # (1, 1, 1, 1, 3)
    transform = BroadcastBinaryTargetTransform(num_channels=3)
    target, mask = transform(annotation)

    assert target.shape == (1, 3, 1, 1, 3)
    assert mask.shape == (1, 3, 1, 1, 3)
    # All channels should be identical
    for c in range(3):
        assert target[0, c].tolist() == [[[0, 0, 1]]]
        assert mask[0, c].tolist() == [[[0, 1, 1]]]


def test_affinity_transform_same_object():
    """Two adjacent voxels of the same object should have affinity=1."""
    # 1D-like: [bg, obj2, obj2, bg] along X
    annotation = torch.zeros(1, 1, 1, 1, 4)
    annotation[0, 0, 0, 0, :] = torch.tensor([1, 2, 2, 1]).float()

    offsets = [[0, 0, 1]]  # X offset
    transform = AffinityTargetTransform(offsets)
    target, mask = transform(annotation)

    # target shape: (1, 1, 1, 1, 4)
    assert target.shape == (1, 1, 1, 1, 4)

    # Pairs (along X, offset +1):
    # (0,1): bg-obj2 -> 0, both annotated -> mask=1
    # (1,2): obj2-obj2 -> 1, both annotated -> mask=1
    # (2,3): obj2-bg -> 0, both annotated -> mask=1
    # Position 3 has no pair (boundary) -> target=0, mask=0
    assert target[0, 0, 0, 0, :3].tolist() == [0, 1, 0]
    assert mask[0, 0, 0, 0, :3].tolist() == [1, 1, 1]
    assert mask[0, 0, 0, 0, 3].item() == 0  # no pair for last voxel


def test_affinity_transform_different_objects():
    """Adjacent voxels of different objects should have affinity=0."""
    annotation = torch.zeros(1, 1, 1, 1, 3)
    annotation[0, 0, 0, 0, :] = torch.tensor([2, 3, 2]).float()

    offsets = [[0, 0, 1]]
    transform = AffinityTargetTransform(offsets)
    target, mask = transform(annotation)

    # (0,1): obj2-obj3 -> 0
    # (1,2): obj3-obj2 -> 0
    assert target[0, 0, 0, 0, :2].tolist() == [0, 0]
    assert mask[0, 0, 0, 0, :2].tolist() == [1, 1]


def test_affinity_transform_unannotated_masking():
    """Unannotated voxels should produce mask=0."""
    annotation = torch.zeros(1, 1, 1, 1, 4)
    annotation[0, 0, 0, 0, :] = torch.tensor([2, 0, 2, 1]).float()

    offsets = [[0, 0, 1]]
    transform = AffinityTargetTransform(offsets)
    target, mask = transform(annotation)

    # (0,1): obj2-unannotated -> mask=0
    # (1,2): unannotated-obj2 -> mask=0
    # (2,3): obj2-bg -> mask=1, target=0
    assert mask[0, 0, 0, 0, 0].item() == 0
    assert mask[0, 0, 0, 0, 1].item() == 0
    assert mask[0, 0, 0, 0, 2].item() == 1
    assert target[0, 0, 0, 0, 2].item() == 0


def test_affinity_transform_multiple_offsets():
    """Test with Z, Y, X offsets."""
    annotation = torch.zeros(1, 1, 3, 3, 3)
    # Fill with same object
    annotation[:] = 2
    # Set corners to background
    annotation[0, 0, 0, 0, 0] = 1

    offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    transform = AffinityTargetTransform(offsets)
    target, mask = transform(annotation)

    assert target.shape == (1, 3, 3, 3, 3)
    assert mask.shape == (1, 3, 3, 3, 3)

    # All annotated (>0), so mask should be 1 everywhere there's a valid pair
    # Z offset channel: mask=1 for z=0,1 (pairs with z+1 exist), mask=0 for z=2
    assert mask[0, 0, 2, :, :].sum().item() == 0  # no z+1 for z=2
    assert mask[0, 0, 0, :, :].sum().item() == 9  # all y,x pairs valid
    assert mask[0, 0, 1, :, :].sum().item() == 9

    # Corner (0,0,0) is bg, (1,0,0) is fg -> Z-offset affinity at (0,0,0) = 0
    assert target[0, 0, 0, 0, 0].item() == 0
    # (1,0,0) and (2,0,0) both fg -> Z-offset affinity at (1,0,0) = 1
    assert target[0, 0, 1, 0, 0].item() == 1


def test_affinity_transform_negative_offset():
    """Test that negative offsets work correctly."""
    annotation = torch.zeros(1, 1, 1, 1, 4)
    annotation[0, 0, 0, 0, :] = torch.tensor([1, 2, 2, 1]).float()

    offsets = [[0, 0, -1]]  # Negative X offset
    transform = AffinityTargetTransform(offsets)
    target, mask = transform(annotation)

    # With offset -1, source starts at index 1, dest starts at index 0
    # Pair (1,0): obj2-bg -> 0, both annotated -> mask=1
    # Pair (2,1): obj2-obj2 -> 1, both annotated -> mask=1
    # Pair (3,2): bg-obj2 -> 0, both annotated -> mask=1
    assert target[0, 0, 0, 0, 1].item() == 0
    assert target[0, 0, 0, 0, 2].item() == 1
    assert target[0, 0, 0, 0, 3].item() == 0
    assert mask[0, 0, 0, 0, 0].item() == 0  # no pair for index 0


def test_offset_slices():
    """Test _offset_slices helper."""
    # Positive offset
    src, dst = _offset_slices(10, 10, 10, 1, 0, 0)
    assert src == (slice(None, 9), slice(None), slice(None))
    assert dst == (slice(1, None), slice(None), slice(None))

    # Negative offset
    src, dst = _offset_slices(10, 10, 10, 0, 0, -2)
    assert src == (slice(None), slice(None), slice(2, None))
    assert dst == (slice(None), slice(None), slice(None, 8))

    # Zero offset
    src, dst = _offset_slices(10, 10, 10, 0, 0, 0)
    assert src == (slice(None), slice(None), slice(None))
    assert dst == (slice(None), slice(None), slice(None))


def test_affinity_transform_extra_channels_masked():
    """Extra channels (e.g. LSDs) should have mask=0."""
    annotation = torch.zeros(1, 1, 1, 1, 4)
    annotation[0, 0, 0, 0, :] = torch.tensor([1, 2, 2, 1]).float()

    offsets = [[0, 0, 1]]  # 1 affinity channel
    transform = AffinityTargetTransform(offsets, num_channels=4)  # 1 aff + 3 extra
    target, mask = transform(annotation)

    assert target.shape == (1, 4, 1, 1, 4)
    assert mask.shape == (1, 4, 1, 1, 4)

    # Channel 0 (affinity) should have valid mask
    assert mask[0, 0, 0, 0, :3].sum().item() == 3
    # Channels 1-3 (extra, e.g. LSDs) should be fully masked out
    assert mask[0, 1, :, :, :].sum().item() == 0
    assert mask[0, 2, :, :, :].sum().item() == 0
    assert mask[0, 3, :, :, :].sum().item() == 0


if __name__ == "__main__":
    test_binary_transform_basic()
    test_binary_transform_multi_object()
    test_broadcast_transform()
    test_affinity_transform_same_object()
    test_affinity_transform_different_objects()
    test_affinity_transform_unannotated_masking()
    test_affinity_transform_multiple_offsets()
    test_affinity_transform_negative_offset()
    test_offset_slices()
    test_affinity_transform_extra_channels_masked()
    print("All tests passed!")


# ---------------------------------------------------------------------------
# DistanceTargetTransform
# ---------------------------------------------------------------------------

import math

import numpy as np
import pytest

from cellmap_flow.finetune.target_transforms import DistanceTargetTransform


def _soft(d, sigma):
    return (math.tanh(d / sigma) + 1.0) / 2.0


def test_distance_target_matches_fly_organelles_formula():
    """A slab of fg in a dense bg patch: 0.5 at the boundary, tanh profile away from it."""
    sigma = 2.0
    ann = np.ones((1, 1, 3, 3, 21), dtype=np.float32)      # all annotated bg
    ann[..., 10:] = 2                                        # fg from x=10 on
    target, mask = DistanceTargetTransform(sigma)(torch.from_numpy(ann))
    line = target[0, 0, 1, 1].numpy()
    # fg voxel at x=10 sits 1 voxel inside (nearest bg at x=9); bg at x=9 is 1 outside.
    assert line[10] == pytest.approx(_soft(1.0, sigma), abs=1e-6)
    assert line[9] == pytest.approx(_soft(-1.0, sigma), abs=1e-6)
    assert line[14] == pytest.approx(_soft(5.0, sigma), abs=1e-6)
    assert line[5] == pytest.approx(_soft(-5.0, sigma), abs=1e-6)
    # monotone across the boundary, bounded in [0, 1]
    assert np.all(np.diff(line) >= 0)
    assert 0.0 <= line.min() and line.max() <= 1.0


def test_distance_mask_drops_voxels_whose_boundary_may_lie_outside_the_patch():
    """Along the line, |d| grows away from the boundary while the distance to
    the patch edge shrinks; once |d| exceeds it the voxel is unsupervised."""
    sigma = 100.0  # never saturates inside this patch
    ann = np.ones((1, 1, 3, 3, 21), dtype=np.float32)
    ann[..., 10:] = 2
    _, mask = DistanceTargetTransform(sigma)(torch.from_numpy(ann))
    line = mask[0, 0, 1, 1].numpy()
    # x=10: |d|=1, distance to nearest edge (y/z faces are 2 voxels away) = 2 -> kept
    assert line[10] == 1.0
    # x=12: |d|=3 > trust 2 -> a closer boundary could sit beyond the y/z faces
    assert line[12] == 0.0
    # in a thin patch nothing far from the boundary is trusted
    assert line[0] == 0.0 and line[20] == 0.0


def test_distance_mask_keeps_saturated_voxels_far_from_any_edge():
    """Deep inside a big patch trust >= 3 sigma, so the value is saturated and kept."""
    sigma = 2.0
    ann = np.ones((1, 1, 15, 15, 15), dtype=np.float32)
    ann[..., 8:] = 2
    target, mask = DistanceTargetTransform(sigma)(torch.from_numpy(ann))
    # centre voxel (7,7,7): bg, |d|=1, trust=8 -> reliable
    assert mask[0, 0, 7, 7, 7] == 1.0
    # (7,7,3): bg, |d|=5 > trust 4 (index 3 is 4 voxels from the padded
    # edge), and trust < 3 sigma = 6 -> masked
    assert mask[0, 0, 7, 7, 3] == 0.0
    # sigma small enough that trust 4 >= 3 sigma: saturated, kept, ~0
    target2, mask2 = DistanceTargetTransform(0.5)(torch.from_numpy(ann))
    assert mask2[0, 0, 7, 7, 3] == 1.0
    assert target2[0, 0, 7, 7, 3] < 1e-3


def test_distance_unannotated_voxels_are_unknown_not_background():
    """Zeros are neither fg nor bg: they get no target weight and shrink the
    trust radius of their annotated neighbours."""
    ann = np.ones((1, 1, 9, 9, 9), dtype=np.float32)
    ann[..., 4:] = 2
    ann[0, 0, 4, 4, 0:2] = 0                 # an unannotated pocket in the bg
    _, mask = DistanceTargetTransform(1.0)(torch.from_numpy(ann))
    assert mask[0, 0, 4, 4, 0] == 0.0 and mask[0, 0, 4, 4, 1] == 0.0
    # bg voxel at x=2: |d|=2 (fg starts at 4) but the pocket is 1 away -> masked
    assert mask[0, 0, 4, 4, 2] == 0.0
    # the same column in a pocket-free row is fine
    assert mask[0, 0, 2, 4, 2] == 1.0


def test_distance_all_foreground_patch_is_saturated_only_when_deep():
    ann = np.full((1, 1, 9, 9, 9), 2, dtype=np.float32)
    target, mask = DistanceTargetTransform(1.0)(torch.from_numpy(ann))
    # no boundary anywhere: target is 1 everywhere, but only voxels whose
    # distance to the patch edge is >= 3 sigma (=3) are trusted
    assert torch.all(target == 1.0)
    assert mask[0, 0, 4, 4, 4] == 1.0
    assert mask[0, 0, 0, 4, 4] == 0.0
    # index 1 is 2 voxels from the padded edge (< 3), index 2 is 3 (>= 3)
    assert mask[0, 0, 1, 4, 4] == 0.0 and mask[0, 0, 2, 4, 4] == 1.0


def test_distance_broadcasts_to_model_channels_and_keeps_device():
    ann = torch.ones((2, 1, 5, 5, 5))
    ann[..., 2:] = 2
    target, mask = DistanceTargetTransform(2.0, num_channels=3)(ann)
    assert target.shape == (2, 3, 5, 5, 5) and mask.shape == (2, 3, 5, 5, 5)
    assert torch.equal(target[:, 0], target[:, 2])
    assert target.device == ann.device


def test_distance_rejects_bad_sigma():
    with pytest.raises(ValueError):
        DistanceTargetTransform(0.0)


# ---------------------------------------------------------------------------
# BCE-on-soft-targets helpers (lora_trainer)
# ---------------------------------------------------------------------------

from cellmap_flow.finetune.lora_trainer import as_probabilities, soft_target_entropy


def test_soft_target_entropy_is_the_bce_floor():
    t = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    h = soft_target_entropy(t)
    # hard targets pay nothing, t=0.5 pays log 2, symmetric
    assert h[0] == pytest.approx(0.0, abs=1e-5) and h[4] == pytest.approx(0.0, abs=1e-5)
    assert h[2] == pytest.approx(math.log(2), abs=1e-6)
    assert h[1] == pytest.approx(h[3], abs=1e-6)
    # equals BCE(t, t): a perfectly calibrated prediction cannot go lower
    bce = torch.nn.functional.binary_cross_entropy(t.clamp(1e-7, 1 - 1e-7), t, reduction="none")
    assert torch.allclose(h, bce, atol=1e-5)


def test_as_probabilities_does_not_double_sigmoid():
    logits = torch.tensor([-3.0, 0.0, 3.0])
    probs = torch.sigmoid(logits)
    assert torch.equal(as_probabilities(probs, model_has_sigmoid=True), probs)
    assert torch.allclose(as_probabilities(logits, model_has_sigmoid=False), probs)
    # the failure mode this guards: sigmoid of a probability lands in [0.5, 0.73]
    squashed = torch.sigmoid(probs)
    assert squashed.min() >= 0.5 and squashed.max() <= 0.732
