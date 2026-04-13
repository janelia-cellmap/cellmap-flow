"""
Target transforms for converting user annotations to training targets.

Each transform takes a raw annotation tensor (B, 1, Z, Y, X) with values:
  0 = unannotated (ignored in loss)
  1 = background
  2 = first foreground object
  3 = second foreground object, etc.

And produces:
  target: (B, C, Z, Y, X) — training target matching model output channels
  mask: (B, C, Z, Y, X) or (B, 1, Z, Y, X) — valid loss mask
"""

from typing import List, Tuple

import torch
from torch import Tensor


class TargetTransform:
    """Base class for target transforms."""

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert annotation to (target, mask) pair."""
        raise NotImplementedError


class BinaryTargetTransform(TargetTransform):
    """Standard binary segmentation transform (current default behavior).

    Produces single-channel binary target: bg=0, fg=1.
    Mask marks annotated regions.
    """

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        mask = (annotation > 0).float()
        target = torch.clamp(annotation - 1, min=0)
        target = (target > 0).float()
        return target, mask


class BroadcastBinaryTargetTransform(TargetTransform):
    """Binary target broadcast to N channels.

    All output channels receive the same fg/bg target.
    Useful for treating multi-channel models (affinities, distances)
    as simple binary segmentation.
    """

    def __init__(self, num_channels: int):
        self.num_channels = num_channels

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        mask = (annotation > 0).float()
        target = (torch.clamp(annotation - 1, min=0) > 0).float()
        # expand is lazy (no memory copy), contiguous() ensures safe downstream use
        target = target.expand(-1, self.num_channels, -1, -1, -1).contiguous()
        mask = mask.expand(-1, self.num_channels, -1, -1, -1).contiguous()
        return target, mask


class AffinityTargetTransform(TargetTransform):
    """Compute affinity targets from instance labels.

    For each offset, affinity is:
      1 if both voxels belong to the same foreground object (same label > 1)
      0 if different objects, or either is background

    The loss mask requires both voxels in each pair to be annotated (label > 0),
    producing a per-channel mask since each offset shifts differently.

    Args:
        offsets: List of [dz, dy, dx] offset tuples defining neighbor relationships.
        num_channels: Total number of model output channels. If greater than
                      len(offsets), extra channels (e.g. LSDs) are masked out
                      (mask=0) so they receive no gradient. If None, defaults
                      to len(offsets).
    """

    def __init__(self, offsets: List[List[int]], num_channels: int = None):
        self.offsets = offsets
        self.num_channels = num_channels if num_channels is not None else len(offsets)

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        B, _C, Z, Y, X = annotation.shape
        # Allocate for all output channels; non-affinity channels stay zero (masked out)
        target = torch.zeros(B, self.num_channels, Z, Y, X, device=annotation.device)
        mask = torch.zeros(B, self.num_channels, Z, Y, X, device=annotation.device)

        labels = annotation[:, 0]  # (B, Z, Y, X)
        annotated = labels > 0  # bool

        for i, offset in enumerate(self.offsets):
            dz, dy, dx = offset
            src_slices, dst_slices = _offset_slices(Z, Y, X, dz, dy, dx)

            src_labels = labels[(slice(None), *src_slices)]
            dst_labels = labels[(slice(None), *dst_slices)]
            src_ann = annotated[(slice(None), *src_slices)]
            dst_ann = annotated[(slice(None), *dst_slices)]

            # Affinity = 1 iff same foreground object
            same_fg = (src_labels == dst_labels) & (src_labels > 1)
            both_annotated = src_ann & dst_ann

            target[(slice(None), i, *src_slices)] = same_fg.float()
            mask[(slice(None), i, *src_slices)] = both_annotated.float()

        return target, mask


def _offset_slices(Z, Y, X, dz, dy, dx):
    """Compute source and destination slices for an offset.

    For a volume of shape (Z, Y, X) and offset (dz, dy, dx),
    returns slices such that:
      volume[src_slices] and volume[dst_slices]
    are aligned views offset by (dz, dy, dx).
    """

    def _dim_slices(size, d):
        if d > 0:
            return slice(None, size - d), slice(d, None)
        elif d < 0:
            return slice(-d, None), slice(None, size + d)
        else:
            return slice(None), slice(None)

    sz, dz_s = _dim_slices(Z, dz)
    sy, dy_s = _dim_slices(Y, dy)
    sx, dx_s = _dim_slices(X, dx)

    return (sz, sy, sx), (dz_s, dy_s, dx_s)
