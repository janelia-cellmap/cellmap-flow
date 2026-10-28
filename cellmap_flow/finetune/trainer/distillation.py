"""The distillation term: its weight, where it applies, and the teacher; and
the hold on the channels a run does not train.

Distillation holds the model to the one it started from where the paint
says nothing: on the unlabeled voxels, on every voxel, or on the good
regions' anchor patches when the session has any (``losses.distillation_loss``).
The teacher is the model as it was before this run changed it: with LoRA
the same module with its adapter switched off, for a full finetune a frozen
copy of the starting weights (``adaptation``'s strategies). The browser's
twin is web/src/finetune/distillation.ts.

A run that trains one channel of a multi-channel model (``select_channel``)
holds every other channel to the teacher's prediction too, on every voxel
(``Distillation.others_term``, at ``OTHER_CHANNELS_WEIGHT``): the channels
share the network, so training one moves the rest, and with the paint about
one of them nothing else would keep them where they were.
"""

import logging

import torch

from cellmap_flow.finetune.losses import distillation_loss

logger = logging.getLogger(__name__)

# The weight of the hold on the channels a run does not train (others_term).
# Fixed, not an option: the hold is what lets one channel train while the
# others stay as they were, not a trade-off to tune. It never reaches the
# trained channel, which the supervised term (and its own distillation term)
# alone decide. The browser's full mode holds them at the same weight.
OTHER_CHANNELS_WEIGHT = 1.0


class Distillation:
    """The distillation term of a run, and the hold on the channels it does not
    train, from its TrainingOptions.

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

    With ``select_channel`` set the term is the selected channel's, and
    ``holds_others`` is on: every other channel is held to the teacher
    (``others_term``), whatever the weight. So the teacher is made whenever
    either is on (``teacher_needed``), with lambda at 0 too.
    """

    def __init__(self, options, *, anchors_available, model, strategy, device, precision):
        self.all_voxels = options.distillation_scope == "all"
        self.anchors_available = anchors_available
        self.select_channel = options.select_channel
        # A model of one channel has no other to hold; the first batch tells
        # (others_term), as nothing before it says how many the model has.
        self.holds_others = options.select_channel is not None
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
        if self.teacher_needed:
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
            if self.on:
                if anchors_available:
                    scope = "anchor patches only: good regions and random points of the volume"
                elif self.all_voxels:
                    scope = "all voxels"
                else:
                    scope = "unlabeled voxels only"
                logger.info(f"Teacher distillation enabled: lambda={self.weight} ({scope})")
            if self.holds_others:
                logger.info(
                    f"Training channel {self.select_channel}: every other channel is held to the "
                    f"starting model's prediction on every voxel (weight {OTHER_CHANNELS_WEIGHT})"
                )
            # Made now, while the weights are the starting weights (see strategy.teacher).
            self._teacher = strategy.teacher(model)
            if strategy.teacher_model is not None:
                strategy.teacher_model.to(device)

    @property
    def on(self) -> bool:
        """Whether the distillation term is in the loss (its weight is above 0)."""
        return self.weight > 0

    @property
    def teacher_needed(self) -> bool:
        """Whether a batch needs the teacher's prediction: for the term, or for the hold."""
        return self.on or self.holds_others

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
        """Switch the term and the hold off for good: the OOM handler's last
        resort. A full finetune's frozen teacher is a whole second copy of the
        weights, freed with the terms it served."""
        self.weight = 0
        self.holds_others = False
        self.teacher_model = None

    @torch.no_grad()
    def teacher_prediction(self, raw):
        """The starting model's prediction on ``raw``, every channel of it.

        LoRA: the model itself with its adapters switched off, in eval mode.
        Full finetune: the frozen copy taken before training. Only called
        while ``teacher_needed``, and the teacher is made whenever it is (only
        ``drop`` takes it away). Making one here instead would be wrong for a
        full finetune: it would copy the student as it is now, not as it
        started. Every channel, as the hold compares the ones the term does
        not (``term`` takes the selected one itself).
        """
        with self._teacher as model:
            with self.precision.autocast():
                prediction = model(raw)
        prediction = prediction.detach()
        if not torch.isfinite(prediction).all():
            logger.warning(f"NaN/Inf in teacher_pred! range=[{prediction.min():.4f}, {prediction.max():.4f}]")
        return prediction

    def term(self, pred, teacher_pred, mask, anchor):
        """The mean squared distance of ``pred`` from ``teacher_pred`` over the term's scope.

        ``pred`` is what the supervised term compares, the selected channel
        with ``select_channel``; ``teacher_pred`` is every channel, and the
        same one is taken from it.

        Good regions decide where the teacher is worth copying. Distilling on
        every unlabeled voxel instead anchors hardest right beside the
        scribbles, which is the one place the teacher is known to be wrong,
        so it partly fights the correction being made; with anchors the term
        is restricted to the regions the user actually vouched for.
        """
        if self.select_channel is not None:
            teacher_pred = teacher_pred[:, self.select_channel:self.select_channel + 1]
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

    def others_term(self, prediction, teacher_pred):
        """The hold: the mean squared distance of every channel but the selected
        one from the teacher's, over every voxel of the patch.

        ``prediction`` and ``teacher_pred`` are every channel, as the model
        emits them (logits, or probabilities for a model that ends in a
        sigmoid). Every voxel, not the term's scope: the paint is about the
        selected channel only, so no voxel of it says another channel should
        change. A model of one channel has no other: the hold goes off at its
        first batch (0 for that one), and the teacher with it unless the
        term needs it.
        """
        others = [c for c in range(prediction.shape[1]) if c != self.select_channel]
        if not others:
            self.holds_others = False
            if not self.on:
                self.teacher_model = None
            return prediction.new_zeros(())
        loss = distillation_loss(prediction[:, others], teacher_pred[:, others], "all")
        if not torch.isfinite(loss):
            logger.warning(f"NaN/Inf other-channels hold: {loss.item()}")
        return loss
