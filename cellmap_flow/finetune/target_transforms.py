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

A flow model's target (Cellpose's: flows towards each instance's centre and
a foreground channel, with a mask per channel) is made by
``instance_flows.FlowTargetTransform``, beside the loss it goes with.
"""

import logging
from typing import List, Optional, Tuple

import torch
from torch import Tensor

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


class DistanceTargetTransform(TargetTransform):
    """Soft signed-distance target for models trained the fly_organelles way.

    The target is ``(tanh((edt(fg) - edt(not fg)) / sigma) + 1) / 2`` with
    distances in voxels: 0.5 on the object boundary, rising to 1 inside and
    falling to 0 outside, saturating a few sigma away. That is exactly what
    ``fly_organelles.utils.Distance`` fed the cellmap distance models (the
    nuc/mito "*_distance_*" HuggingFace repos use sigma=6), so sigmoid(model
    output) is comparable to this target voxel for voxel. Use it with a
    BCE-with-logits loss, which accepts soft targets; margin and dice do not.

    Masking: an EDT computed inside a patch only sees boundaries inside the
    patch. For an annotated voxel the computed |d| is an upper bound on the
    truth -- the real nearest boundary may sit just past the patch edge or
    inside an unannotated pocket. A voxel is supervised only when its
    computed |d| is no larger than its distance to the nearest unannotated
    voxel or patch edge (so no unseen boundary can be closer), or when that
    distance is already past 3*sigma, where tanh has saturated and the exact
    value no longer matters.

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

    def _one(self, ann):
        """(Z, Y, X) uint annotation -> (target, mask) float32 numpy arrays."""
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt

        annotated = ann > 0
        fg = ann >= 2
        bg = ann == 1
        target = np.zeros(ann.shape, dtype=np.float32)
        mask = np.zeros(ann.shape, dtype=np.float32)
        if not annotated.any():
            return target, mask

        # edt(x) is the distance from each nonzero voxel of x to the nearest
        # zero. With no zero anywhere scipy returns a large finite number for
        # every voxel; treat that as "no boundary in this patch" explicitly.
        #
        # Both sides measure to the nearest *known* voxel of the other class:
        # a background voxel's distance to annotated foreground, and a
        # foreground voxel's distance to annotated background. d_in used to
        # be edt(fg), which counts unannotated voxels as background too, so a
        # foreground voxel next to an unannotated one -- at the edge of a
        # dense crop that cuts through an object, say -- got a small d_in that
        # always passed the trust test below, and was supervised toward 0.58
        # to 0.88 as though the object ended at the crop edge.
        d_in = edt(~bg) if bg.any() else np.full(ann.shape, np.inf)
        d_out = edt(~fg) if fg.any() else np.full(ann.shape, np.inf)
        signed = np.where(fg, d_in, -d_out)

        # Distance from each annotated voxel to the nearest unannotated voxel
        # or to just outside the patch (the one-voxel pad of False).
        known = np.pad(annotated, 1, constant_values=False)
        trust = edt(known)[1:-1, 1:-1, 1:-1]

        reliable = np.abs(signed) <= trust
        saturated = trust >= 3.0 * self.sigma
        mask[annotated & (reliable | saturated)] = 1.0
        target[:] = (np.tanh(signed / self.sigma) + 1.0) / 2.0
        return target, mask

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        import numpy as np

        ann = annotation.detach().cpu().numpy()
        if ann.ndim != 5 or ann.shape[1] != 1:
            raise ValueError(
                f"Expected annotation of shape (B, 1, Z, Y, X), got {tuple(ann.shape)}"
            )
        targets = np.empty(ann.shape, dtype=np.float32)
        masks = np.empty(ann.shape, dtype=np.float32)
        for b in range(ann.shape[0]):
            targets[b, 0], masks[b, 0] = self._one(ann[b, 0])
        target = torch.from_numpy(targets).to(annotation.device)
        mask = torch.from_numpy(masks).to(annotation.device)
        if self.num_channels > 1:
            target = target.expand(-1, self.num_channels, -1, -1, -1).contiguous()
            mask = mask.expand(-1, self.num_channels, -1, -1, -1).contiguous()
        return target, mask


# How far out, in sigma, a distance bound still says anything: tanh(3) is
# 0.995, so a voxel known to be at least 3 sigma inside is as good as at 3
# sigma, and an upper bound beyond it constrains nothing a threshold sees.
SATURATION_SIGMAS = 3.0


class IntervalTargetTransform(TargetTransform):
    """Bounds on a distance model's output at each painted voxel, for scribbles.

    A distance target needs every voxel's distance to the object boundary,
    and a stroke does not say where the boundary is. It does bound it:

    - upper, U: the nearest voxel painted as the other class. The boundary
      lies between the two, so |d| <= U.
    - lower, L: the nearest voxel not painted as the voxel's own class --
      the other class, unpainted (it may be either), or past the patch edge.
      Every voxel closer is known to be its own class, so |d| >= L.

    Where the paint is dense, the nearest voxel not of its own class is of
    the other, and L == U: the exact distance DistanceTargetTransform would
    give, with the same convention (edt to the nearest voxel of the other
    class, so 1 voxel on either side of a boundary). On a stroke the two
    bracket the truth. Both are computed in the patch the trainer gets, which
    keeps them valid: a closer other-class voxel outside the patch would only
    lower the true |d|, which U already exceeds or meets, and the patch edge
    counts as "not its own class", which can only lower L. Reading context
    around the patch would tighten bounds near its edges, but the dataset
    hands the trainer the patch alone (augmented with it), so they are left
    as loose as the patch makes them.

    Distances are Euclidean in nm (``voxel_size_nm``, so an anisotropic
    grid is measured as it is), and sigma, which is in output voxels as for
    the distance target, is taken in the finest axis's voxels. The bounds
    are returned in the model's logit space, where the distance target
    (tanh(d/sigma) + 1)/2 = sigmoid(2d/sigma) is the line z = 2d/sigma, with
    the sign of the painted class. Past SATURATION_SIGMAS a lower bound is
    capped there and an upper bound dropped (+-inf).

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
        self.sigma_nm = float(sigma_voxels) * min(self.voxel_size_nm)

    def distance_bounds(self, ann):
        """(L, U) in nm for each voxel of a (Z, Y, X) annotation; 0 off the paint, U inf with no other class."""
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt

        sampling = self.voxel_size_nm
        fg, bg = ann >= 2, ann == 1
        lower = np.zeros(ann.shape)
        upper = np.zeros(ann.shape)
        for own, other in ((fg, bg), (bg, fg)):
            if not own.any():
                continue
            # edt(x) is the distance from each nonzero voxel of x to the
            # nearest zero; with no zero at all scipy returns garbage, so a
            # patch without the other class has no upper bound, explicitly.
            to_other = edt(~other, sampling=sampling) if other.any() else np.inf
            # Padded with a ring of "not own class": the patch's surroundings.
            to_not_own = edt(np.pad(own, 1, constant_values=False), sampling=sampling)[1:-1, 1:-1, 1:-1]
            upper[own] = np.broadcast_to(to_other, ann.shape)[own]
            lower[own] = to_not_own[own]
        return lower, upper

    def _one(self, ann):
        """(Z, Y, X) annotation -> (lower, upper, mask) float32 numpy arrays, the bounds in logits."""
        import numpy as np

        lower, upper = self.distance_bounds(ann)
        logit_per_nm = 2.0 / self.sigma_nm
        saturation = SATURATION_SIGMAS * self.sigma_nm
        near = logit_per_nm * np.minimum(lower, saturation)
        far = np.where(upper < saturation, logit_per_nm * upper, np.inf)
        fg, bg = ann >= 2, ann == 1
        # Background lies on the negative side: its bounds swap and change sign.
        lo = np.where(fg, near, np.where(bg, -far, 0.0))
        hi = np.where(fg, far, np.where(bg, -near, 0.0))
        return lo.astype(np.float32), hi.astype(np.float32), (ann > 0).astype(np.float32)

    def __call__(self, annotation: Tensor) -> Tuple[Tensor, Tensor]:
        import numpy as np

        ann = annotation.detach().cpu().numpy()
        if ann.ndim != 5 or ann.shape[1] != 1:
            raise ValueError(
                f"Expected annotation of shape (B, 1, Z, Y, X), got {tuple(ann.shape)}"
            )
        bounds = np.empty((ann.shape[0], 2, *ann.shape[2:]), dtype=np.float32)
        masks = np.empty(ann.shape, dtype=np.float32)
        for b in range(ann.shape[0]):
            bounds[b, 0], bounds[b, 1], masks[b, 0] = self._one(ann[b, 0])
        return torch.from_numpy(bounds).to(annotation.device), torch.from_numpy(masks).to(annotation.device)
