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

except IntervalTargetTransform, whose "target" is a lower and an upper bound
on a distance model's output, for losses.IntervalLoss.

The maths is ``cellmap_flow/portable/targets.py``, numpy functions the
browser runs too (in chunkmirage's Pyodide worker, on one painted chunk);
the classes here hold a transform's settings and move the trainer's tensors
to and from them, so both modes make the same targets from the same paint.

A flow model's target (Cellpose's: flows towards each instance's centre and
a foreground channel, with a mask per channel) is made by
``instance_flows.FlowTargetTransform``, beside the loss it goes with.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from cellmap_flow.portable import targets
from cellmap_flow.portable.targets import SATURATION_SIGMAS  # noqa: F401  (losses.py reads it here)

logger = logging.getLogger(__name__)


def read_offsets_from_script(script_path) -> Optional[list]:
    """The ``offsets`` a model script assigns, read by parsing it (not running it); None if none.

    An affinity model's script names its neighbour offsets, which the
    affinity target needs. The dashboard reads them too, to tell an affinity
    model when a session is set up.
    """
    import ast

    try:
        with open(script_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "offsets":
                        return ast.literal_eval(node.value)
    except Exception as e:
        logger.debug(f"Could not read offsets from {script_path}: {e}")

    return None


def _labels(annotation: Tensor) -> np.ndarray:
    """The annotation's labels as numpy, (B, Z, Y, X): the batch with its one channel taken out."""
    ann = annotation.detach().cpu().numpy()
    if ann.ndim != 5 or ann.shape[1] != 1:
        raise ValueError(f"Expected annotation of shape (B, 1, Z, Y, X), got {tuple(ann.shape)}")
    return ann[:, 0]


def _tensors(annotation: Tensor, *arrays: np.ndarray) -> Tuple[Tensor, ...]:
    """The portable function's arrays as tensors on the annotation's device."""
    return tuple(torch.from_numpy(a).to(annotation.device) for a in arrays)


class TargetTransform:
    """Base class for target transforms."""

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert annotation to (target, mask) pair."""
        raise NotImplementedError


class BinaryTargetTransform(TargetTransform):
    """Standard binary segmentation transform (current default behavior).

    Produces single-channel binary target: bg=0, fg=1.
    Mask marks annotated regions (``targets.binary``).
    """

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        return _tensors(annotation, *targets.binary(_labels(annotation)))


class BroadcastBinaryTargetTransform(TargetTransform):
    """Binary target broadcast to N channels.

    All output channels receive the same fg/bg target.
    Useful for treating multi-channel models (affinities, distances)
    as simple binary segmentation (``targets.binary`` on N channels).
    """

    def __init__(self, num_channels: int):
        self.num_channels = num_channels

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        return _tensors(annotation, *targets.binary(_labels(annotation), self.num_channels))


class AffinityTargetTransform(TargetTransform):
    """Compute affinity targets from instance labels (``targets.affinity``).

    For each offset, affinity is:
      1 if both voxels belong to the same foreground object (same label > 1)
      0 if different objects, or either is background

    The loss mask requires both voxels in each pair to be annotated (label > 0),
    producing a per-channel mask since each offset shifts differently. An
    offset as long as the patch along its axis or longer pairs no voxel, and
    its channel's mask is empty.

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
        return _tensors(annotation, *targets.affinity(_labels(annotation), self.offsets, self.num_channels))


class DistanceTargetTransform(TargetTransform):
    """Soft signed-distance target for models trained the fly_organelles way
    (``targets.distance``, which says how it is made and masked).

    The target is ``(tanh((edt(fg) - edt(not fg)) / sigma) + 1) / 2`` with
    distances in voxels, 0.5 on the object boundary, so sigmoid(model output)
    is comparable to it voxel for voxel. Use it with a BCE-with-logits loss,
    which accepts soft targets; margin and dice do not. A voxel is
    supervised only where no boundary unseen in the patch can be nearer than
    the one it measures, or where tanh has saturated anyway.

    Args:
        sigma_voxels: tanh scale in output voxels (6 for the cellmap
            distance models).
        num_channels: model output channels; the target is broadcast to all
            of them, as BroadcastBinaryTargetTransform does.
    """

    def __init__(self, sigma_voxels: float = 6.0, num_channels: int = 1):
        if sigma_voxels <= 0:
            raise ValueError(f"sigma_voxels must be positive, got {sigma_voxels}")
        self.sigma = float(sigma_voxels)
        self.num_channels = int(num_channels)

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        return _tensors(annotation, *targets.distance(_labels(annotation), self.sigma, self.num_channels))


class IntervalTargetTransform(TargetTransform):
    """Bounds on a distance model's output at each painted voxel, for scribbles
    (``targets.interval``, with ``targets.distance_bounds``, says how).

    A stroke does not say where the object boundary is, but it bounds it:
    no further than the nearest voxel painted as the other class, no nearer
    than the nearest voxel not painted as the voxel's own. Where the paint is
    dense the two meet at the distance DistanceTargetTransform would give.
    Distances are Euclidean in nm (``voxel_size_nm``), sigma is in the finest
    axis's output voxels, and the bounds are in the model's logit space.

    Returns ``(bounds, mask)``: bounds (B, 2, Z, Y, X), the lower and upper
    bound on the logit (0 off the paint), and mask (B, 1, Z, Y, X), the
    painted voxels. Both have one channel per bound whatever the model's
    channels: losses.IntervalLoss broadcasts them over its output.
    """

    def __init__(self, sigma_voxels: float = 6.0, voxel_size_nm=None):
        if sigma_voxels <= 0:
            raise ValueError(f"sigma_voxels must be positive, got {sigma_voxels}")
        voxel_size = (1.0, 1.0, 1.0) if voxel_size_nm is None else voxel_size_nm
        self.voxel_size_nm = tuple(float(v) for v in voxel_size)
        self.sigma_voxels = float(sigma_voxels)
        self.sigma_nm = targets.sigma_nm(sigma_voxels, self.voxel_size_nm)

    def distance_bounds(self, ann):
        """(L, U) in nm for each voxel of a (Z, Y, X) annotation; 0 off the paint, U inf with no other class."""
        return targets.distance_bounds(ann, self.voxel_size_nm)

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        return _tensors(annotation, *targets.interval(_labels(annotation), self.sigma_voxels, self.voxel_size_nm))
