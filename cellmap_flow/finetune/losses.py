"""The finetune losses, and the masked means they are all made of.

Every loss here is averaged over the voxels a mask supervises, not over the
patch: a correction labels a few voxels of it, and the rest must not count.
``masked_mean`` is that average, ``balanced_mean`` gives the foreground and
the background half the weight each, and the losses and the distillation
term below are built from the two.

The float operations are kept in the order the trainer always used, so a
run computes the same numbers bit for bit (tests/finetune/
test_training_snapshot.py). lora_trainer re-exports these names.
"""

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "masked_mean",
    "balanced_mean",
    "as_probabilities",
    "soft_target_entropy",
    "distillation_loss",
    "DiceLoss",
    "CombinedLoss",
    "MarginLoss",
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
        """
        Args:
            smooth: Smoothing factor to avoid division by zero (default: 1.0)
        """
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predictions (B, C, Z, Y, X) - raw logits or probabilities
            target: Targets (B, C, Z, Y, X) - binary masks [0, 1]
            mask: Optional mask (B, 1, Z, Y, X) - if provided, only compute loss on masked regions

        Returns:
            Dice loss value (scalar)
        """
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
        """
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
        """
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predictions (B, C, Z, Y, X) - raw logits
            target: Targets (B, C, Z, Y, X) - binary masks [0, 1]
            mask: Optional mask (B, 1, Z, Y, X) - if provided, only compute loss on masked regions

        Returns:
            Combined loss value (scalar)
        """
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
