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
