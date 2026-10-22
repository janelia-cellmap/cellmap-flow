"""The distillation term: its weight, where it applies, and the teacher.

Distillation holds the model to the one it started from where the paint
says nothing: on the unlabeled voxels, on every voxel, or on the good
regions' anchor patches when the session has any (``losses.distillation_loss``).
The teacher is the model as it was before this run changed it: with LoRA
the same module with its adapter switched off, for a full finetune a frozen
copy of the starting weights (``adaptation``'s strategies). The browser's
twin is web/src/finetune/distillation.ts.
"""

import logging

import torch

from cellmap_flow.finetune.losses import distillation_loss

logger = logging.getLogger(__name__)


class Distillation:
    """The distillation term of a run, from its TrainingOptions.

    The weight: ``distillation_lambda``, or when that is None (unset) 1.0 if
    the dataset emits anchors (good regions, or random points of the volume)
    and 0 otherwise. Good regions are useless without a teacher term to apply
    in them: the supervised loss never touches an unannotated voxel, so with
    lambda at 0 every rehearsal patch would contribute exactly nothing and
    the regions the user marked would silently do nothing at all. Marking
    them is an explicit request to be held there. An explicit 0 is honoured:
    it used to be indistinguishable from "unset" and was raised to 1.0 as
    well, so choosing "0 (Disabled)" in the dashboard with good regions
    marked gave 100x the default weight.
    """

    def __init__(self, options, *, anchors_available, model, strategy, device, precision):
        self.all_voxels = options.distillation_scope == "all"
        self.anchors_available = anchors_available
        self.select_channel = options.select_channel
        self.strategy = strategy
        self.precision = precision
        self.weight = options.distillation_lambda
        if self.weight is None:
            self.weight = 1.0 if anchors_available else 0.0
            if anchors_available:
                logger.warning(
                    "The patches include anchors (good regions, or random points "
                    "of the volume) and no distillation weight was given. Using "
                    "lambda=1.0 so the anchors take effect; pass "
                    "an explicit lambda (0 to switch it off) to override."
                )
        elif anchors_available and self.weight <= 0:
            logger.warning(
                "The patches include anchors (good regions, or random points "
                "of the volume) but distillation is switched off (lambda=0), "
                "so they contribute nothing to the loss. Set the rehearsal and "
                "anchor fractions to 0 to skip them, or give distillation a "
                "weight to use them."
            )

        self._teacher = None
        if self.weight > 0:
            # FX-interpreted models (torch.export UnflattenedModule, often
            # wrapped in BatchLoopWrapper) keep every intermediate tensor
            # alive in the FX env, so distillation's two passes can OOM
            # even on H100/A100 80GB. Not auto-disabled here: request a
            # larger node if needed; the OOM handler drops the term as a
            # last resort if memory actually runs out.
            inner = getattr(model, "model", model)
            base = getattr(inner, "model", inner)
            if type(base).__name__ == "UnflattenedModule" or type(inner).__name__ == "BatchLoopWrapper":
                logger.warning(
                    "Distillation enabled with an FX-interpreted base model "
                    "(UnflattenedModule). This requires substantial GPU memory; "
                    "if you OOM, distillation will be disabled automatically as "
                    "a fallback in the OOM handler."
                )
            if anchors_available:
                scope = "anchor patches only: good regions and random points of the volume"
            elif self.all_voxels:
                scope = "all voxels"
            else:
                scope = "unlabeled voxels only"
            logger.info(f"Teacher distillation enabled: lambda={self.weight} ({scope})")
            # Made now, while the weights are the starting weights (see strategy.teacher).
            self._teacher = strategy.teacher(model)
            if strategy.teacher_model is not None:
                strategy.teacher_model.to(device)

    @property
    def on(self) -> bool:
        return self.weight > 0

    @property
    def teacher_model(self):
        """A full finetune's frozen teacher, the copy of its starting weights; else None.

        The CLI hands it to the next iteration's trainer. Setting it to None
        frees it (``drop`` does, with the term).
        """
        return self.strategy.teacher_model

    @teacher_model.setter
    def teacher_model(self, value):
        self.strategy.teacher_model = value
        if value is None:
            self._teacher = None

    def drop(self):
        """Switch the term off for good: the OOM handler's last resort. A full
        finetune's frozen teacher is a whole second copy of the weights, freed
        with the term it served."""
        self.weight = 0
        self.teacher_model = None

    @torch.no_grad()
    def teacher_prediction(self, raw):
        """The starting model's prediction on ``raw``.

        LoRA: the model itself with its adapters switched off, in eval mode.
        Full finetune: the frozen copy taken before training. Only called
        while the term is on, and the teacher is made whenever it is (only
        ``drop`` takes it away). Making one here instead would be wrong for a
        full finetune: it would copy the student as it is now, not as it
        started.
        """
        with self._teacher as model:
            with self.precision.autocast():
                prediction = model(raw)
        if self.select_channel is not None:
            prediction = prediction[:, self.select_channel:self.select_channel + 1]
        prediction = prediction.detach()
        if not torch.isfinite(prediction).all():
            logger.warning(f"NaN/Inf in teacher_pred! range=[{prediction.min():.4f}, {prediction.max():.4f}]")
        return prediction

    def term(self, pred, teacher_pred, mask, anchor):
        """The mean squared distance of ``pred`` from ``teacher_pred`` over the term's scope.

        Good regions decide where the teacher is worth copying. Distilling on
        every unlabeled voxel instead anchors hardest right beside the
        scribbles, which is the one place the teacher is known to be wrong,
        so it partly fights the correction being made; with anchors the term
        is restricted to the regions the user actually vouched for.
        """
        if anchor is not None:
            scope = "anchor"
        elif self.all_voxels or mask is None:
            scope = "all"
        else:
            scope = "unlabeled"
        loss = distillation_loss(
            pred, teacher_pred, scope, anchor_mask=anchor,
            unlabeled_mask=None if mask is None else 1.0 - mask,
        )
        if not torch.isfinite(loss):
            logger.warning(f"NaN/Inf distillation_loss: {loss.item()}")
        return loss
