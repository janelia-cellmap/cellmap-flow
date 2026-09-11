"""SKOOTS-style loss: semantic + skeleton + vector heads, all reduced to Tversky.

Verified against github.com/buswinka/skoots and github.com/buswinka/bism
(not the paper's simplified description).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        # alpha == beta == 0.5 weights FP and FN equally, which is exactly soft Dice;
        # kept as separate knobs since SKOOTS heads may eventually want asymmetric tuning.
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        tp = (pred * target * mask).sum()
        fp = (pred * (1 - target) * mask).sum()
        fn = ((1 - pred) * target * mask).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


class SkootsLoss(nn.Module):
    """Combined loss for a 5-channel SKOOTS head: [semantic, skeleton, vec_z, vec_y, vec_x].

    `target`/`mask` follow the same (B, 5, Z, Y, X) layout produced by
    `SkootsTargetTransform` / `skoots_targets.build_skoots_targets`.
    """

    def __init__(
        self,
        vector_scaling: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        sigma_start: Tuple[float, float, float] = (20.0, 20.0, 20.0),
        sigma_end: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        anneal_steps: int = 10_000,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
        vector_loss_type: str = "gaussian_tversky",
    ) -> None:
        """
        vector_loss_type: "gaussian_tversky" (default) is the SKOOTS-style
            embedding loss above -- a per-voxel Gaussian "do these two points
            agree" probability, annealed via sigma, wrapped in Tversky so its
            scale matches the semantic/skeleton terms. "mse" is a plain
            balanced-fg/bg per-axis MSE on the displacement vector directly
            (same balancing scheme as SkeletonDistanceLoss's distance term),
            with no sigma/annealing at all -- simpler and more directly
            interpretable, at the cost of the embedding-style "agreement"
            property the Gaussian version has (which is what SKOOTS's
            vector-flow clustering during postprocessing was designed
            around). Exists to A/B against the default: see
            lora_trainer.py's "skoots_simple_vector" loss_type.
        """
        super().__init__()
        if vector_loss_type not in ("gaussian_tversky", "mse"):
            raise ValueError(f"Unknown vector_loss_type: {vector_loss_type!r}")
        self.vector_loss_type = vector_loss_type
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.register_buffer(
            "vector_scaling", torch.tensor(vector_scaling).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "sigma_start", torch.tensor(sigma_start).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "sigma_end", torch.tensor(sigma_end).view(1, 3, 1, 1, 1), persistent=False
        )
        self.anneal_steps = anneal_steps
        self.weights = weights or {"semantic": 1.0, "skeleton": 1.0, "vector": 1.0}
        self._step = 0
        self.last_components: Dict[str, float] = {}

    def set_step(self, step: int) -> None:
        # Training loop sets this explicitly instead of the module incrementing a
        # counter on every forward call, which would silently drift under gradient
        # accumulation, validation-mode forwards, or multi-GPU replica copies.
        self._step = step

    def _sigma(self) -> Tensor:
        t = min(self._step / self.anneal_steps, 1.0)
        return self.sigma_start + t * (self.sigma_end - self.sigma_start)

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        semantic_pred = torch.sigmoid(pred[:, 0:1])
        semantic_loss = self.tversky(semantic_pred, target[:, 0:1], mask[:, 0:1])

        skeleton_pred = torch.sigmoid(pred[:, 1:2])
        skeleton_loss = self.tversky(skeleton_pred, target[:, 1:2], mask[:, 1:2])

        # tanh bounds the raw vector head to (-1, 1) per axis before scaling to voxels,
        # so the network can never predict a displacement outside a physically sane range.
        pred_vector = torch.tanh(pred[:, 2:5]) * self.vector_scaling
        target_vector = target[:, 2:5]
        vector_mask = mask[:, 2:3]
        fg_mask = vector_mask * (target[:, 0:1] > 0.5).float()
        bg_mask = vector_mask * (target[:, 0:1] <= 0.5).float()

        if self.vector_loss_type == "mse":
            # Plain balanced-fg/bg MSE on the displacement itself -- same
            # balancing scheme as SkeletonDistanceLoss's distance term, no
            # sigma/annealing/Gaussian-probability machinery. Normalized by
            # vector_scaling first, same reasoning as SkeletonDistanceLoss
            # dividing by distance_scaling: raw voxel^2 error (up to ~20^2
            # per axis) would dwarf the Tversky terms by ~100x in the summed
            # total loss -- confirmed empirically on a real run before this
            # fix (epoch-1 vector=94.2 vs semantic=0.87, skeleton=0.99).
            # `sq_err` sums over the 3 axes (dim=1) so this stays one scalar
            # per voxel, matching the Tversky version's per-voxel shape.
            sq_err = ((pred_vector - target_vector) / self.vector_scaling).pow(2).sum(dim=1, keepdim=True)
            fg_vector_loss = (sq_err * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)
            bg_vector_loss = (sq_err * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)
            vector_loss = (fg_vector_loss + bg_vector_loss) / 2.0
        else:
            # Comparing displacement-to-displacement (instead of reconstructing absolute
            # skeleton coordinates) keeps the target crop-relative and translation-invariant.
            diff = (pred_vector - target_vector) / self._sigma()
            prob = torch.exp(-0.5 * diff.pow(2).sum(dim=1, keepdim=True))

            # Wide sigma early on tolerates coarse vectors while semantic/skeleton heads are
            # still noisy; annealing it down later demands sub-voxel precision once coarse
            # localization has been learned.
            vector_target = torch.ones_like(prob)

            # Balanced fg/bg, same scheme as MarginLoss's `balance_classes`
            # (lora_trainer.py) and SkeletonDistanceLoss's distance term: run the
            # Tversky computation separately per class, then mix 50/50, instead
            # of pooling every voxel into one Tversky call. `bake_vector_targets`
            # zeroes background's vector to (0,0,0) (same convention as
            # `skeleton_distance_target`'s "background stays 0"), so background
            # trivially reaches prob~1 with near-zero learning, and pooling it in
            # with foreground -- via `vector_target=ones` for every voxel, so
            # Tversky's usual fp=0-on-true-negatives immunity doesn't apply here,
            # since there ARE no "negative" voxels in this target -- lets an
            # easy majority background quietly prop up the reported loss while
            # foreground (the only region SkootsPostprocessor's vector-flow
            # clustering actually uses) stays under-trained.
            fg_vector_loss = self.tversky(prob, vector_target, fg_mask)
            bg_vector_loss = self.tversky(prob, vector_target, bg_mask)
            vector_loss = (fg_vector_loss + bg_vector_loss) / 2.0

        total = (
            self.weights["semantic"] * semantic_loss
            + self.weights["skeleton"] * skeleton_loss
            + self.weights["vector"] * vector_loss
        )

        # Stashed rather than returned alongside `total`: the generic training-loop call
        # site (`lora_trainer.py`) expects `criterion(pred, target, mask) -> Tensor`, the
        # same contract every other loss_type follows.
        self.last_components = {
            "semantic": float(semantic_loss.detach()),
            "skeleton": float(skeleton_loss.detach()),
            "vector": float(vector_loss.detach()),
        }
        return total


class SkeletonDistanceLoss(nn.Module):
    """Combined loss for a 3-channel head: [semantic, skeleton, distance-from-skeleton].

    Simpler alternative to `SkootsLoss`'s vector-field head: a scalar "how
    far is this voxel from its own instance's medial axis" regression
    instead of a 3-component displacement vector. This drops the vector
    field's ability to disambiguate *direction* near touching instances --
    the actual mechanism `SkootsPostprocessor`'s embedding trick relies on to
    split them -- so it's only appropriate when instances in the training
    data don't truly touch (established for this dataset in
    corrections/skoots_mito's investigation), where a plain
    distance-transform-style watershed seeded from skeleton-mask components
    is enough to recover per-instance shape without directional guidance.

    `target`/`mask` follow the (B, 3, Z, Y, X) layout produced by
    `SkeletonDistanceTargetTransform` /
    `skoots_targets.build_skeleton_distance_targets`.
    """

    def __init__(
        self,
        distance_scaling: float = 30.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.distance_scaling = distance_scaling
        self.weights = weights or {"semantic": 1.0, "skeleton": 1.0, "distance": 1.0}
        self.last_components: Dict[str, float] = {}

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        semantic_pred = torch.sigmoid(pred[:, 0:1])
        semantic_loss = self.tversky(semantic_pred, target[:, 0:1], mask[:, 0:1])

        skeleton_pred = torch.sigmoid(pred[:, 1:2])
        skeleton_loss = self.tversky(skeleton_pred, target[:, 1:2], mask[:, 1:2])

        # tanh bounds the raw distance head to (-1, 1) before scaling to
        # voxels -- same reasoning as SkootsLoss's vector head: the network
        # can never predict a distance outside a physically sane range, no
        # matter how far training drifts. The loss itself is computed in
        # that same normalized (-1, 1)-ish space (divide both sides by
        # distance_scaling) rather than raw voxel^2 MSE, so this component
        # stays the same order of magnitude as the Tversky losses above
        # instead of dwarfing them by ~100x.
        #
        # Balanced fg/bg, same scheme as MarginLoss's `balance_classes`
        # (lora_trainer.py): average each class's error separately, then mix
        # 50/50, rather than a single average over all voxels. Background is
        # a flat 0 target (see `skeleton_distance_target`'s docstring) that
        # the network can match almost for free -- unlike `semantic_loss`/
        # `skeleton_loss` above, plain MSE has no Tversky-style immunity to
        # class imbalance, so an unweighted average over a mostly-background
        # dense crop is dominated by that trivially-easy majority and stays
        # small even when foreground (the only region
        # SkeletonDistancePostprocessor's watershed actually reads) is barely
        # learned -- measured directly on a real trained checkpoint:
        # whole-patch loss 0.022 vs. foreground-only 0.059 (2.6x), with
        # foreground MAE (~6.6 voxels) nearly as large as the foreground mean
        # target value (~7.6 voxels). Balancing (rather than fully excluding
        # background, as a first pass here did) keeps real gradient pressure
        # on background too, instead of leaving it free to drift once it's
        # no longer supervised at all.
        distance_pred_norm = torch.tanh(pred[:, 2:3])
        distance_target_norm = target[:, 2:3] / self.distance_scaling
        sq_err = (distance_pred_norm - distance_target_norm).pow(2)

        fg_mask = (target[:, 0:1] > 0.5).float() * mask[:, 2:3]
        bg_mask = (target[:, 0:1] <= 0.5).float() * mask[:, 2:3]
        fg_contrib = (sq_err * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)
        bg_contrib = (sq_err * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)
        distance_loss = (fg_contrib + bg_contrib) / 2.0

        total = (
            self.weights["semantic"] * semantic_loss
            + self.weights["skeleton"] * skeleton_loss
            + self.weights["distance"] * distance_loss
        )
        self.last_components = {
            "semantic": float(semantic_loss.detach()),
            "skeleton": float(skeleton_loss.detach()),
            "distance": float(distance_loss.detach()),
        }
        return total


class SkeletonSemanticLoss(nn.Module):
    """Combined loss for a 2-channel head: [semantic, skeleton], no
    instance-splitting channel at all.

    Simplest sibling of `SkootsLoss`/`SkeletonDistanceLoss` -- just the two
    Tversky terms those already share, with the vector/distance term dropped
    entirely. Both remaining channels are Tversky-based, so neither needs the
    fg/bg-balancing fix that vector/distance required (Tversky's fp=0-on-
    true-negatives already makes it immune to class imbalance -- see
    `SkootsLoss`/`SkeletonDistanceLoss` docstrings for why that immunity
    specifically does NOT extend to the other two heads' loss formulas).

    `target`/`mask` follow the (B, 2, Z, Y, X) layout produced by
    `SkeletonSemanticTargetTransform` / `SkootsDataset(target_mode=
    "skeleton_semantic")`.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.weights = weights or {"semantic": 1.0, "skeleton": 1.0}
        self.last_components: Dict[str, float] = {}

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        semantic_pred = torch.sigmoid(pred[:, 0:1])
        semantic_loss = self.tversky(semantic_pred, target[:, 0:1], mask[:, 0:1])

        skeleton_pred = torch.sigmoid(pred[:, 1:2])
        skeleton_loss = self.tversky(skeleton_pred, target[:, 1:2], mask[:, 1:2])

        total = self.weights["semantic"] * semantic_loss + self.weights["skeleton"] * skeleton_loss
        self.last_components = {
            "semantic": float(semantic_loss.detach()),
            "skeleton": float(skeleton_loss.detach()),
        }
        return total
