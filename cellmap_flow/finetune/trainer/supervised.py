"""The supervised term of the training loss, and how the model's output is read for it.

``SupervisedLoss`` builds the criterion ``loss_type`` names and compares the
model's output with the target where it is annotated. ``output_class_of``
says what the output is -- a sigmoid baked in, a tanh, or logits -- when
nothing declared it, in the classes the model's server reports
(``serving.probe``) and as the server finds it, off the network's graph
first, so that training, ranking and seeding read one model the same way. The browser's twin is web/src/finetune/full/loss.ts.
"""

import logging

import torch
import torch.nn as nn

from cellmap_flow.finetune.instance_flows import FlowLoss, FlowTargetTransform
from cellmap_flow.finetune.losses import (
    CombinedLoss,
    DiceLoss,
    IntervalLoss,
    MarginLoss,
    as_logits,
    as_probabilities,
    balanced_mean,
    masked_mean,
    soft_target_entropy,
)
from cellmap_flow.finetune.target_transforms import IntervalTargetTransform
from cellmap_flow.serving.probe import UNBOUNDED, classify_output_range

logger = logging.getLogger(__name__)


def output_class_of(model_config, model, example, select_channel=None) -> str:
    """What ``model``'s output is, as its inference server finds it: off
    ``model_config``'s network graph (``ModelConfig.output_graph``,
    ``portable.activation``: the op that ends it), else ``probe_output_class``.
    The graph is the model the job was built from; an adapter or new
    weights change neither the op its network ends in nor the channel read.

    jrc_mus-livers' peroxisome model, a convolution at its end, answered the
    probe with one value inside [-1, 1] and was read as a tanh's output.
    """
    from cellmap_flow.portable.activation import output_activation

    try:
        graph = model_config.output_graph()
        found = output_activation(*graph) if graph is not None else None
    except Exception as e:
        logger.warning(f"Could not read the network's graph ({e}); probing its output instead.")
        found = None
    if found is not None:
        logger.info(f"Model output from its network: {found['output_class']} (it ends in a {found['op']})")
        return found["output_class"]
    return probe_output_class(model, example, select_channel=select_channel)


@torch.no_grad()
def probe_output_class(model, example, select_channel=None) -> str:
    """What ``model``'s output is -- unit, signed_unit or unbounded (serving.probe)
    -- read off its answer to an extreme input.

    ``example`` is an input batch of the shape the model takes; the probe is
    noise of that shape at 100x, so that an output bounded by a sigmoid or a
    tanh shows its bounds and logits show their range. ``select_channel``
    reads the channel training compares. The model is probed in eval mode
    and left as it was: in train mode a full finetune's BatchNorm took the
    noise into its running statistics -- about 7,000x on every channel,
    which takes ~85 batches to decay, longer than most interactive runs --
    and the export and the served model normalize by them. A probe that
    fails reads as logits, with a warning.
    """
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        device = next(model.parameters()).device
        noise = torch.randn(tuple(example[:1].shape), dtype=example.dtype, device=device) * 100
        out = model(noise)
        if select_channel is not None:
            out = out[:, select_channel:select_channel + 1]
        output_class = classify_output_range(float(out.min()), float(out.max()))
        if output_class is None:
            # One value everywhere says nothing of its range (classify_output_range).
            logger.warning("The model answered the probe with one value; reading its output as logits.")
            output_class = UNBOUNDED
        logger.info(f"Model output on the probe: {output_class}")
    except Exception as e:
        logger.warning(f"Output probe failed ({e}); reading the output as logits.")
        output_class = UNBOUNDED
    finally:
        for module, training in modes:
            module.training = training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_class


class SupervisedLoss:
    """The supervised term: the criterion ``options.loss_type`` names, compared with
    the target where it is annotated.

    What the criterion sees depends on what the model's output is
    (``options.output_class``): probabilities when it is bounded -- a sigmoid
    or a tanh baked in, where a second sigmoid would squash [0, 1] into
    [0.5, 0.73] and make a well-fitting prediction look like a constant --
    so BCE-with-logits becomes BCE and Dice and margin skip their sigmoid;
    logits otherwise (losses.as_probabilities). MSE always compares
    probabilities with the 0/1 target, as dice and margin do: on a logit
    model it used to compare the raw logits, training them toward 0 and 1,
    which the served sigmoid turns into 0.5 and 0.73, wrecking any
    threshold. The interval loss bounds the logit, where a distance field is
    a line; the flow loss reads flows and a logit as the model emits them,
    whatever the probe said (its flows can fall within [0, 1]).

    Label smoothing is redundant with the margin loss and means nothing to
    bounds, flows or a soft distance target, so it is 0 for those; class
    balancing does not apply to the flow loss.

    ``targets`` makes a batch's target, mask and bounds from its annotation;
    the call computes the term and keeps the step's metrics (the interval
    terms; on BCE the target's entropy floor and the mean |p - t|), which
    ``tally`` sums over the epoch's supervised batches.
    """

    def __init__(self, options, target_transform=None):
        self.loss_type = options.loss_type
        self.margin = options.margin
        self.balance_classes = options.balance_classes
        self.label_smoothing = options.label_smoothing
        self.mask_unannotated = options.mask_unannotated
        self.target_transform = target_transform
        self.use_bce = self.loss_type == "bce"
        self.use_mse = self.loss_type == "mse"
        self.use_interval = self.loss_type == "interval"
        self.use_flow = self.loss_type == "flow"
        self.output_class = UNBOUNDED if self.use_flow else (options.output_class or UNBOUNDED)
        # Whether the criterion compares probabilities (the model's output is bounded).
        self.takes_probabilities = self.output_class != UNBOUNDED
        self.step_interval_terms = None  # (bounds, slope) terms of the last step
        self.step_bce_metrics = None     # (entropy floor, mean |p - t|) of the last step
        self.reset_epoch()

        if self.loss_type == "dice":
            self.criterion = DiceLoss()
        elif self.loss_type == "bce":
            # reduction='none' so the mask can be applied by hand
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif self.loss_type == "combined":
            self.criterion = CombinedLoss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss(reduction='none')
        elif self.loss_type == "margin":
            self.criterion = MarginLoss(margin=self.margin, balance_classes=self.balance_classes)
        elif self.loss_type == "interval":
            # The bounds are in the transform's geometry, so the loss takes it from there.
            if not isinstance(target_transform, IntervalTargetTransform):
                raise ValueError("loss_type 'interval' needs an IntervalTargetTransform's bounds")
            self.criterion = IntervalLoss(
                target_transform.sigma_nm, target_transform.voxel_size_nm,
                slope_weight=options.slope_weight, balance_classes=self.balance_classes,
            )
        elif self.loss_type == "flow":
            # Flows towards each instance's centre and a foreground logit,
            # with a mask per channel: only FlowTargetTransform makes those.
            if not isinstance(target_transform, FlowTargetTransform):
                raise ValueError("loss_type 'flow' needs a FlowTargetTransform's flow targets")
            self.criterion = FlowLoss()
            if self.balance_classes:
                logger.warning("Class balancing does not apply to the flow loss; ignoring it")
                self.balance_classes = False
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if self.takes_probabilities:
            self._take_probabilities()

        if self.label_smoothing > 0 and (
            self.loss_type in ("margin", "interval", "flow") or options.output_type in ("distance", "flows")
        ):
            logger.warning(
                f"Label smoothing is redundant with {self.loss_type} loss on a {options.output_type} "
                "target, setting to 0"
            )
            self.label_smoothing = 0.0

        if self.balance_classes:
            logger.info("Class balancing enabled: fg and bg scribble voxels weighted equally")
        logger.info(f"Using {self.loss_type} loss")
        if self.label_smoothing > 0:
            logger.info(
                f"Label smoothing: {self.label_smoothing} (targets: {self.label_smoothing / 2:.3f} "
                f"to {1 - self.label_smoothing / 2:.3f})"
            )

    def _take_probabilities(self):
        """Configure the criterion for a model whose output is already a probability."""
        if self.use_bce:
            logger.info("Switching BCEWithLogitsLoss to BCELoss: the model's output is bounded")
            self.criterion = nn.BCELoss(reduction='none')
        if hasattr(self.criterion, 'bce_loss'):
            self.criterion.bce_loss = nn.BCELoss(reduction='none')
        # Tell DiceLoss/MarginLoss to skip their sigmoid
        if hasattr(self.criterion, 'apply_sigmoid'):
            self.criterion.apply_sigmoid = False
        if hasattr(self.criterion, 'dice_loss') and hasattr(self.criterion.dice_loss, 'apply_sigmoid'):
            self.criterion.dice_loss.apply_sigmoid = False

    def targets(self, annotation):
        """A batch's ``(target, mask, bounds, hard_target)`` from its annotation.

        The target transform's target and mask; for the interval loss the
        target is the bounds, and the painted class stands in as the target
        for the class check, the class balance and the TensorBoard image.
        Without a transform, the legacy reading: 0 unannotated, 1 background,
        2 foreground, shifted down by one, and masked when mask_unannotated.
        ``hard_target`` is the target as annotated: class balancing splits
        voxels into fg and bg by it, before smoothing -- split by the smoothed
        target, every bg voxel carried s/2 of fg weight (with 100 fg voxels
        against 100k bg at s = 0.1, 98% of the "fg" half of the loss was bg).
        Smoothing then moves 0 to s/2 and 1 to 1 - s/2.
        """
        target, mask, bounds = annotation, None, None
        if self.target_transform is not None:
            target, mask = self.target_transform(target)
            if self.use_interval:
                bounds = target
                target = (bounds[:, :1] > 0).float()
        elif self.mask_unannotated:
            mask = (target > 0).float()  # (B, C, Z, Y, X)
            target = torch.clamp(target - 1, min=0)
        hard_target = target
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing / 2
        return target, mask, bounds, hard_target

    def _masked_mean(self, per_voxel, hard_target, mask):
        if self.balance_classes:
            # Average fg and bg separately so each contributes equally
            return balanced_mean(per_voxel, hard_target, mask)
        return masked_mean(per_voxel, mask)

    def __call__(self, pred, target, mask, bounds, hard_target):
        """The supervised term of one batch; ``pred`` in fp32 (the loop leaves autocast first)."""
        self.step_interval_terms = self.step_bce_metrics = None
        if self.use_interval:
            # Bounds on the logit, which is where a distance field is a line.
            logits = as_logits(pred, self.output_class)
            bounds_term = self.criterion.bounds_term(logits, bounds, mask)
            slope_term = self.criterion.slope_term(logits)
            self.step_interval_terms = (bounds_term.item(), slope_term.item())
            return bounds_term + self.criterion.slope_weight * slope_term
        if self.use_flow:
            return self.criterion(pred, target, mask)

        probabilities = as_probabilities(pred, self.output_class)
        compared = probabilities if (self.takes_probabilities or self.use_mse) else pred
        if (self.use_bce or self.use_mse) and mask is not None:
            # Per-element losses take the mask by hand
            loss = self._masked_mean(self.criterion(compared, target), hard_target, mask)
            if self.use_bce:
                # The same weighting applied to the target's own entropy gives
                # the floor this batch's BCE cannot go below; mean |p - t| is
                # the loss-independent view of the fit.
                with torch.no_grad():
                    floor = self._masked_mean(soft_target_entropy(target), hard_target, mask)
                    mae = masked_mean((probabilities - target).abs(), mask)
                self.step_bce_metrics = (floor.item(), mae.item())
            return loss
        if 'mask' in self.criterion.forward.__code__.co_varnames:
            # The losses that take a mask themselves (DiceLoss, CombinedLoss, MarginLoss)
            return self.criterion(compared, target, mask)
        loss = self.criterion(compared, target)
        return loss.mean() if (self.use_bce or self.use_mse) else loss

    def reset_epoch(self):
        self._epoch_floor_sum = 0.0
        self._epoch_mae_sum = 0.0
        self._epoch_bce_n = 0

    def tally(self):
        """Add the last step's BCE metrics to the epoch's; for a step that had supervised voxels."""
        if self.step_bce_metrics is not None:
            self._epoch_floor_sum += self.step_bce_metrics[0]
            self._epoch_mae_sum += self.step_bce_metrics[1]
            self._epoch_bce_n += 1

    def epoch_bce_metrics(self):
        """(mean BCE floor, mean |p - t|) over the epoch's supervised steps, or None without BCE."""
        if not self._epoch_bce_n:
            return None
        return self._epoch_floor_sum / self._epoch_bce_n, self._epoch_mae_sum / self._epoch_bce_n
