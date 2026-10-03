"""The finetune losses, and the masked means they are all made of.

Every loss here is averaged over the voxels a mask supervises, not over the
patch: a correction labels a few voxels of it, and the rest must not count.
``masked_mean`` is that average, ``balanced_mean`` gives the foreground and
the background half the weight each, and the losses and the distillation
term below are built from the two. ``IntervalLoss``, a distance model's loss
on scribbles, works in logits instead of probabilities (``as_logits``). A
flow model's loss, ``instance_flows.FlowLoss``, is built from
``masked_mean`` too, and lives beside the target it compares with.

The float operations are kept in the order the trainer always used, so a
run computes the same numbers bit for bit
(tests/finetune/test_training_snapshot.py).
"""

from typing import Optional

import torch
import torch.nn as nn

from cellmap_flow.finetune.target_transforms import SATURATION_SIGMAS

__all__ = [
    "masked_mean",
    "balanced_mean",
    "as_probabilities",
    "as_logits",
    "soft_target_entropy",
    "distillation_loss",
    "DiceLoss",
    "CombinedLoss",
    "MarginLoss",
    "IntervalLoss",
]


def masked_mean(x, weight):
    """Mean of ``x`` weighted by ``weight`` (a 0/1 mask, usually); 0 when nothing is weighted."""
    return (x * weight).sum() / weight.sum().clamp(min=1)


def balanced_mean(x, hard_target, mask):
    """The foreground's and the background's masked means, averaged.

    Each class counts the same however many voxels it has. The split is by
    the target as annotated (0/1), before any label smoothing: split by the
    smoothed target, every background voxel carried s/2 of foreground weight.
    """
    fg = masked_mean(x, hard_target * mask)
    bg = masked_mean(x, (1.0 - hard_target) * mask)
    return (fg + bg) / 2.0


def as_probabilities(pred, model_has_sigmoid):
    """The model's output as probabilities.

    Left alone when the model already ends in a sigmoid (the cellmap
    *_distance_* UNets do); a second sigmoid would squash [0, 1] into
    [0.5, 0.73] and make a well-fitting prediction look like a constant.
    """
    return pred if model_has_sigmoid else torch.sigmoid(pred)


def as_logits(pred, model_has_sigmoid):
    """The model's output as logits: ``as_probabilities``' inverse.

    A model ending in a sigmoid is taken back through it. The clamp keeps
    log(0) out; it costs no gradient short of |logit| ~16, far past where any
    distance bound sits (SATURATION_SIGMAS is a logit of 6).
    """
    return torch.logit(pred, eps=1e-7) if model_has_sigmoid else pred


def soft_target_entropy(target, eps=1e-7):
    """Per-voxel BCE that a perfectly calibrated prediction still pays.

    -(t log t + (1-t) log(1-t)): zero for hard 0/1 targets, log 2 at
    t = 0.5. On soft targets (distance, smoothed labels) this is the floor
    of the BCE curve, and it moves with the batch, so a "flat" BCE can be a
    model sitting on its floor. Report the loss minus this instead.
    """
    t = target.clamp(eps, 1 - eps)
    return -(t * torch.log(t) + (1 - t) * torch.log(1 - t))


def distillation_loss(student, teacher, scope, anchor_mask=None, unlabeled_mask=None):
    """Mean squared distance of the student's output from the teacher's, over ``scope``.

    - ``"anchor"``: the good regions the user vouched for, ``anchor_mask``.
      It is about location, so one channel, broadcast over the output's.
    - ``"unlabeled"``: the voxels the supervised loss leaves out,
      ``unlabeled_mask`` (1 - the supervised mask).
    - ``"all"``: every voxel.

    The two are compared as the model emits them: logits, or probabilities
    for a model that ends in a sigmoid.
    """
    per_voxel = (student - teacher) ** 2
    if scope == "anchor":
        return masked_mean(per_voxel.float(), anchor_mask.float().expand_as(per_voxel))
    if scope == "unlabeled":
        # float32 before the sum, or fp16 overflows over many voxels and
        # channels (13-channel models).
        return masked_mean(per_voxel.float(), unlabeled_mask.float())
    if scope == "all":
        return per_voxel.mean()
    raise ValueError(f"Unknown distillation scope: {scope!r}")


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    Dice loss is effective for imbalanced datasets where the target class
    may be sparse (e.g., mitochondria in EM images).

    Formula: 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth  # keeps an empty class from dividing by zero
        self.apply_sigmoid = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dice loss of (B, C, Z, Y, X) logits (or probabilities) against 0/1 targets, over ``mask``."""
        # Flatten spatial dimensions
        pred = pred.reshape(pred.size(0), pred.size(1), -1)  # (B, C, N)
        target = target.reshape(target.size(0), target.size(1), -1)  # (B, C, N)

        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        # Apply mask if provided. Mask may be (B, 1, ...) for a shared mask
        # or (B, C, ...) for a per-channel mask (e.g. AffinityTargetTransform
        # produces one mask per affinity offset).
        if mask is not None:
            mask = mask.reshape(mask.size(0), mask.size(1), -1)  # (B, Cmask, N)
            pred = pred * mask
            target = target * mask

        # Compute intersection and union
        intersection = (pred * target).sum(dim=2)  # (B, C)
        union = pred.sum(dim=2) + target.sum(dim=2)  # (B, C)

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss (1 - dice)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss for better convergence.

    Uses both Dice loss (for overlap) and BCE loss (for pixel-wise accuracy).
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Weighted Dice + BCE of (B, C, Z, Y, X) logits against 0/1 targets, over ``mask``."""
        dice = self.dice_loss(pred, target, mask)

        bce = self.bce_loss(pred, target)
        bce = masked_mean(bce, mask) if mask is not None else bce.mean()

        return self.dice_weight * dice + self.bce_weight * bce


class MarginLoss(nn.Module):
    """
    Margin-based loss for sparse/scribble annotations.

    Only penalizes predictions on the wrong side of a margin threshold.
    For post-sigmoid outputs in [0, 1]:
    - Foreground (target=1): loss = relu(threshold - pred)^2, threshold = 1 - margin
    - Background (target=0): loss = relu(pred - margin)^2
    - No loss when prediction is already correct with sufficient confidence.
    """

    def __init__(self, margin: float = 0.3, balance_classes: bool = False):
        super().__init__()
        self.margin = margin
        self.balance_classes = balance_classes
        self.apply_sigmoid = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.apply_sigmoid:
            pred = torch.sigmoid(pred)

        threshold_high = 1.0 - self.margin  # e.g., 0.7
        threshold_low = self.margin          # e.g., 0.3

        # Foreground loss: penalize if pred < threshold_high
        fg_loss = torch.relu(threshold_high - pred) ** 2
        # Background loss: penalize if pred > threshold_low
        bg_loss = torch.relu(pred - threshold_low) ** 2

        if self.balance_classes and mask is not None:
            # Average each class separately so fg/bg contribute equally
            # regardless of how many scribble voxels each has. Each class has
            # its own per-voxel loss, so this is balanced_mean by hand.
            fg_contrib = masked_mean(fg_loss, target * mask)
            bg_contrib = masked_mean(bg_loss, (1.0 - target) * mask)
            return (fg_contrib + bg_contrib) / 2.0

        # Blend by target: target=1 -> fg_loss, target=0 -> bg_loss
        loss = target * fg_loss + (1.0 - target) * bg_loss

        if mask is not None:
            return masked_mean(loss, mask)
        return loss.mean()


# How much steeper than a distance field the predicted field may get before
# IntervalLoss's slope term objects. A true distance field changes by one
# voxel per voxel, but the targets the distance models learnt (edt to the
# nearest voxel of the other class) step from -1 to +1 voxel across a
# boundary: a central difference of 1.5 across a flat one, and up to ~2
# across an oblique one, which pays a little. Held to 1, every boundary the
# model draws would be penalized; a collapse into a step is far steeper
# than either.
SLOPE_TOLERANCE = 1.5


class IntervalLoss(nn.Module):
    """A distance model's loss on scribbles: stay within the bounds the paint implies.

    After iSDF (Ortiz et al., RSS 2022). IntervalTargetTransform gives each
    painted voxel a lower and an upper bound on its logit z (the distance
    target is z = 2d/sigma). Per painted voxel the loss is

        relu(lower - slack - z) + relu(z - upper - slack)
            + sign_weight * relu(-s * z - slack)

    with s the painted side (+1 foreground, -1 background): zero inside the
    bounds, linear outside them, and much steeper on the wrong side of the
    boundary. The slack, a voxel of distance by default, forgives a stroke
    that strays a voxel over an edge. Unpainted voxels get none of it:
    distillation and the anchor patches hold them.

    Bounds alone let the field collapse: a step at the boundary meets every
    lower bound and most upper ones, and is what margin loss made of a
    distance model. So a second term limits the slope, everywhere in the
    patch: the mean of relu(|grad z| / max_slope - 1), from central
    differences scaled by the voxel size in nm, with max_slope =
    SLOPE_TOLERANCE * 2/sigma per nm, the steepest a distance field gets in
    logits. One-sided: the field may be flatter. z is clamped to the
    saturation (SATURATION_SIGMAS) first, so a model confident far from any
    boundary is not asked to bring its far field in.

    Args:
        sigma_nm, voxel_size_nm: from the IntervalTargetTransform that made
            the bounds.
        slope_weight: the slope term's weight against the bounds' (default 1).
        balance_classes: average foreground and background voxels separately.
        slack_voxels: the slack, in voxels of the finest axis (default 1).
        sign_weight: the extra slope of the wrong-side penalty (default 5).
    """

    def __init__(self, sigma_nm, voxel_size_nm, slope_weight=1.0, balance_classes=False,
                 slack_voxels=1.0, sign_weight=5.0):
        super().__init__()
        logit_per_nm = 2.0 / sigma_nm
        self.voxel_size_nm = tuple(float(v) for v in voxel_size_nm)
        self.slack = logit_per_nm * slack_voxels * min(self.voxel_size_nm)
        self.max_slope = SLOPE_TOLERANCE * logit_per_nm
        self.saturation = 2.0 * SATURATION_SIGMAS
        self.slope_weight = slope_weight
        self.balance_classes = balance_classes
        self.sign_weight = sign_weight

    def bounds_term(self, z, bounds, mask):
        """The mean penalty for leaving the bounds, over the painted voxels."""
        lower, upper = bounds[:, :1], bounds[:, 1:2]
        # Foreground bounds are positive, background's negative (0 off the paint, masked out).
        side = torch.where(lower > 0, 1.0, -1.0)
        per_voxel = (
            torch.relu(lower - self.slack - z)
            + torch.relu(z - upper - self.slack)
            + self.sign_weight * torch.relu(-side * z - self.slack)
        )
        weight = mask.expand_as(per_voxel)
        if self.balance_classes:
            return balanced_mean(per_voxel, (side > 0).float().expand_as(per_voxel), weight)
        return masked_mean(per_voxel, weight)

    def slope_term(self, z):
        """The mean excess of |grad z| over max_slope, as a fraction of it, over the patch's interior."""
        z = z.clamp(-self.saturation, self.saturation)
        spatial = range(z.dim() - 3, z.dim())
        axes = [(a, h) for a, h in zip(spatial, self.voxel_size_nm) if z.shape[a] >= 3]
        if not axes:
            return z.new_zeros(())
        interior = [slice(None)] * z.dim()
        for a, _ in axes:
            interior[a] = slice(1, -1)
        squared = 0.0
        for a, h in axes:
            ahead, behind = list(interior), list(interior)
            ahead[a], behind[a] = slice(2, None), slice(None, -2)
            squared = squared + ((z[tuple(ahead)] - z[tuple(behind)]) / (2.0 * h)) ** 2
        # The epsilon keeps sqrt's derivative finite on a flat field.
        slope = torch.sqrt(squared + 1e-12)
        return torch.relu(slope / self.max_slope - 1.0).mean()

    def forward(self, z: torch.Tensor, bounds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The loss of (B, C, Z, Y, X) logits ``z`` against IntervalTargetTransform's bounds and mask."""
        return self.bounds_term(z, bounds, mask) + self.slope_weight * self.slope_term(z)
