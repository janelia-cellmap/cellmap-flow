"""What the log says about a run's health, apart from its loss: whether both
classes are annotated at all, what the first batch supervises, and whether
backward reaches the trainable parameters. None of it ever stops training."""

import logging

import torch

logger = logging.getLogger(__name__)


def describe_first_batch(target, mask):
    """Say how many voxels the first batch supervises, and how many are positive.

    One batch is a few patches of the epoch's draw, so it may well hold one
    class only; whether the annotations do is warn_if_single_class's.
    """
    try:
        with torch.no_grad():
            if mask is None:
                supervised = target.numel()
                positive = float((target > 0.5).sum())
            else:
                supervised = float(mask.sum())
                positive = float(((target > 0.5).float() * mask).sum())
            if supervised < 1:
                logger.warning(
                    "No supervised voxels in the first batch: the loss has "
                    "nothing to learn from. Check that the annotations "
                    "overlap the sampled patches."
                )
                return
            logger.info(f"First batch: {supervised:.0f} supervised voxels, {100 * positive / supervised:.1f}% positive")
    except Exception as e:  # never let a diagnostic stop training
        logger.debug(f"First-batch check failed: {e}")


def warn_if_single_class(background, foreground):
    """Say so when every annotated voxel carries the same label.

    ``background`` and ``foreground`` count the voxels annotated as each in
    the whole volume (the dataset's ``class_voxels``). This went by the
    first batch, and warned in every iteration of a session painted with
    both, whose first eight patches happened to hold no background.

    Gradient descent can only do one thing with a target that is all 1:
    raise the prediction everywhere. The model duly does that, globally,
    and the result is worse than the model you started from -- with a
    loss curve that looks unremarkable, because a one-class problem is
    genuinely easy to reduce.

    The usual cause is painting only foreground. An affinity target needs
    both: foreground says "these voxels are one object", background says
    "this is not object, and these two are not joined". Without the
    second, nothing anywhere says 0. (Flow targets are not 0/1 classes: an
    instance patch with no background painted still teaches each object's
    flows, so the flow loss skips this check.)
    """
    if (background > 0) == (foreground > 0):  # both annotated, or nothing is
        return
    only, missing, way = ("foreground", "background", "up") if foreground else ("background", "foreground", "down")
    logger.warning(
        "=" * 70 + "\n"
        f"Nothing is annotated as {missing}: all {background + foreground} annotated "
        f"voxels are {only}.\n"
        "This run cannot teach the model a boundary: gradient descent will "
        f"simply push\nthe prediction {way} everywhere, and the result will be "
        "worse than the model you\nstarted from.\n\n"
        f"Paint some {missing} as well, and put it where it decides something --\n"
        "between two objects that touch, and on the things being confused for one.\n"
        + "=" * 70
    )


class GradientFlow:
    """Whether backward reaches the adapter, over one epoch.

    One LoRA-B parameter is watched across the epoch (its mean |grad| per
    batch, and how far it moved), and at the end of the epoch the trainable
    parameters that received no gradient at all are counted. Together they
    answer: is backward reaching LoRA at all, and if only for some, which?
    """

    def __init__(self, model):
        self.model = model
        self.name, self.param = None, None
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_B" in name:
                self.name, self.param = name, param
                break
        self.initial = self.param.detach().clone() if self.param is not None else None
        self.grad_abs_sum = 0.0
        self.grad_count = 0
        # Which trainable params have seen a nonzero gradient this epoch.
        self.seen_nonzero = {}

    def after_backward(self):
        """Read the gradients, before the optimizer step clears them."""
        if self.param is not None and self.param.grad is not None:
            self.grad_abs_sum += self.param.grad.detach().abs().mean().item()
            self.grad_count += 1
        for name, p in self.model.named_parameters():
            if not p.requires_grad or self.seen_nonzero.get(name, False):
                continue
            if p.grad is not None and p.grad.detach().abs().sum().item() > 0:
                self.seen_nonzero[name] = True

    def report(self, log):
        """The epoch's summary, through ``log``."""
        if self.param is not None:
            mean_grad = self.grad_abs_sum / self.grad_count if self.grad_count else 0.0
            delta = (self.param.detach() - self.initial).abs().mean().item() if self.initial is not None else 0.0
            log(
                f"  [diag] {self.name}: "
                f"mean|grad|={mean_grad:.3e} (over {self.grad_count} batches), "
                f"mean|param_delta|={delta:.3e} this epoch"
            )
        n_trainable = sum(1 for _, p in self.model.named_parameters() if p.requires_grad)
        n_live = sum(1 for v in self.seen_nonzero.values() if v)
        dead = [name for name, p in self.model.named_parameters() if p.requires_grad and not self.seen_nonzero.get(name)]
        log(
            f"  [diag] gradient flow: {n_live}/{n_trainable} trainable "
            f"params got nonzero grad; {n_trainable - n_live} are dead. "
            f"First 5 dead: {dead[:5]}"
        )
