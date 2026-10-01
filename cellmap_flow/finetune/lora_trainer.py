"""
Finetuning trainer for CellMap-Flow models: a LoRA adapter, or every weight.

This module provides a trainer class for finetuning models using user
corrections with mixed-precision training and gradient accumulation.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Dict, Any
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from torch.utils.data import DataLoader

from cellmap_flow.finetune import markers
from cellmap_flow.finetune.adaptation import strategy_for
from cellmap_flow.finetune.losses import (
    CombinedLoss,
    DiceLoss,
    MarginLoss,
    as_probabilities,
    balanced_mean,
    distillation_loss,
    masked_mean,
    soft_target_entropy,
)

logger = logging.getLogger(__name__)


class LoRAFinetuner:
    """
    Trainer for finetuning a model on user corrections: a LoRA adapter, or
    every weight (a full finetune).

    Which of the two the model decides: a PEFT model trains its adapter, a
    plain module trains in full (see adaptation.strategy_for).

    Features:
    - Mixed precision on CUDA: bf16 where the GPU has it, else fp16 with a
      GradScaler; it falls back to fp32 when the model NaNs under either
    - Gradient accumulation to simulate larger batch sizes
    - Partial annotation support (mask unannotated regions)
    - Distillation toward the starting model, on unlabeled voxels, on all
      voxels, or on the good regions' rehearsal patches
    - A best checkpoint, chosen by the supervised loss alone; the export
      loads it first
    - Recovery from an OOM (halve the batch, then drop distillation) and
      from a NaN under mixed precision (restart in fp32)
    - TensorBoard logging, continuing across a job's iterations

    Args:
        model: The model to train: a PEFT model (LoRA) or a plain module (full finetune)
        dataloader: DataLoader for training data
        output_dir: Directory to save checkpoints and logs
        learning_rate: Learning rate (default: 1e-4)
        num_epochs: Number of training epochs (default: 10)
        gradient_accumulation_steps: Steps to accumulate gradients (default: 1)
        use_mixed_precision: Enable mixed precision on CUDA (default: True; off on the CPU)
        loss_type: Loss function: "dice", "bce", "combined" (Dice + BCE), "mse" or "margin"
        device: Training device ("cuda" or "cpu", auto-detected if None)
        select_channel: Optional channel index to select from multi-channel output (default: None)
        mask_unannotated: If True (default), only compute loss on annotated regions (target > 0).
                         Targets are shifted down by 1 (e.g., 1->0, 2->1) after masking.
                         This allows partial annotations where 0=unannotated, 1=background, 2=foreground, etc.
                         Ignored if target_transform is provided.
        label_smoothing: s moves the targets to s/2 and 1 - s/2 (default: 0; forced to 0 for margin)
        distillation_lambda: Weight of the distillation term. None means 1.0 when the
                         dataset has good regions and 0 otherwise; an explicit 0 is honoured.
        distillation_all_voxels: Distil on every voxel, not only the unlabeled ones
                         (no effect when the dataset has good regions: they decide where)
        margin: The margin of the "margin" loss (default: 0.3)
        balance_classes: Weight foreground and background voxels equally
        target_transform: Optional TargetTransform instance that converts raw annotations
                         to (target, mask) pairs. Overrides mask_unannotated when provided.
                         See cellmap_flow.finetune.target_transforms.
        tensorboard: Write TensorBoard logs to output_dir/tensorboard (default: True)
        teacher_model: A full finetune's frozen teacher from a previous iteration, to reuse
        initial_state: A full finetune's starting weights, to reset to; copied from the model if None
        tb_start_step, tb_start_epoch: Where the previous iteration's TensorBoard curves ended

    Examples:
        >>> lora_model = wrap_model_with_lora(model)
        >>> dataloader = create_dataloader("corrections.zarr")
        >>> trainer = LoRAFinetuner(
        ...     lora_model,
        ...     dataloader,
        ...     output_dir="output/fly_organelles_v1.1"
        ... )
        >>> trainer.train()
        >>> trainer.save_adapter()
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        output_dir: str,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        use_mixed_precision: bool = True,
        loss_type: str = "combined",
        device: Optional[str] = None,
        select_channel: Optional[int] = None,
        mask_unannotated: bool = True,
        label_smoothing: float = 0.0,
        distillation_lambda: Optional[float] = None,
        distillation_all_voxels: bool = False,
        margin: float = 0.3,
        balance_classes: bool = False,
        target_transform=None,
        tensorboard: bool = True,
        teacher_model: Optional[nn.Module] = None,
        initial_state: Optional[Dict[str, torch.Tensor]] = None,
        tb_start_step: int = 0,
        tb_start_epoch: int = 0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        # NOTE: self.use_mixed_precision is overridden below if device is CPU,
        # since CUDA AMP utilities can't run on CPU.
        self.select_channel = select_channel
        self.mask_unannotated = mask_unannotated
        self._single_class_checked = False
        self.label_smoothing = label_smoothing
        self.distillation_lambda = distillation_lambda
        self.distillation_all_voxels = distillation_all_voxels
        self.balance_classes = balance_classes
        self.target_transform = target_transform
        # Kept for the TensorBoard config card; the loss objects below do
        # not expose them uniformly.
        self.loss_type = loss_type
        self.margin = margin
        self.learning_rate = learning_rate

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # CUDA AMP utilities (autocast('cuda'), GradScaler('cuda')) error or
        # behave unexpectedly when training on CPU. Force-disable mixed
        # precision unless we're actually on a CUDA device.
        if self.device.type != "cuda" and self.use_mixed_precision:
            logger.warning(
                f"Disabling mixed precision: device={self.device.type} is not CUDA."
            )
            self.use_mixed_precision = False
            use_mixed_precision = False

        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # LoRA or full, and everything that differs between them. The model
        # decides (see adaptation.strategy_for).
        self.strategy = strategy_for(self.model, teacher_model=teacher_model)

        # A full finetune changes the weights themselves, so resetting it --
        # after a NaN, or for a restart -- needs the weights it started from.
        # Kept on the CPU. LoRA resets by re-initialising its adapter instead.
        self.initial_state = (
            initial_state if initial_state is not None else self.strategy.initial_state(self.model)
        )

        # Optimizer, over the parameters that train: the adapter's, or every
        # weight in a full finetune.
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )

        # Loss function
        self._use_bce = False
        self._use_mse = False
        self._model_has_sigmoid = False   # set by _apply_probability_output_mode
        self._step_bce_metrics = None     # (entropy floor, mean |p - t|) of the last step
        self._epoch_bce_floor_sum = 0.0
        self._epoch_mae_sum = 0.0
        self._epoch_bce_n = 0
        if loss_type == "dice":
            self.criterion = DiceLoss()
        elif loss_type == "bce":
            # Use reduction='none' so we can manually apply mask if needed
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
            self._use_bce = True
        elif loss_type == "combined":
            self.criterion = CombinedLoss()
        elif loss_type == "mse":
            self.criterion = nn.MSELoss(reduction='none')
            self._use_mse = True
        elif loss_type == "margin":
            self.criterion = MarginLoss(margin=margin, balance_classes=balance_classes)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Label smoothing is redundant with margin loss
        if loss_type == "margin" and self.label_smoothing > 0:
            logger.warning("Label smoothing is redundant with margin loss, setting to 0")
            self.label_smoothing = 0.0

        if self.balance_classes:
            logger.info("Class balancing enabled: fg and bg scribble voxels weighted equally")

        logger.info(f"Using {loss_type} loss")

        # Good regions are useless without a teacher term to apply in them:
        # the supervised loss never touches an unannotated voxel, so with
        # lambda at 0 every rehearsal patch would contribute exactly nothing
        # and the regions the user marked would silently do nothing at all.
        # Marking them is an explicit request to be held there, so a lambda
        # left unset (None) becomes 1.0 when there are any. An explicit 0 is
        # honoured: it used to be indistinguishable from "unset" and was
        # raised to 1.0 as well, so choosing "0 (Disabled)" in the dashboard
        # with good regions marked gave 100x the default weight.
        self._anchors_available = bool(
            getattr(getattr(self.dataloader, "dataset", None), "emits_anchor", False)
        )
        if self.distillation_lambda is None:
            self.distillation_lambda = 1.0 if self._anchors_available else 0.0
            if self._anchors_available:
                logger.warning(
                    "Good regions are marked and no distillation weight was "
                    "given. Using lambda=1.0 so the anchors take effect; pass "
                    "an explicit lambda (0 to switch it off) to override."
                )
        elif self._anchors_available and self.distillation_lambda <= 0:
            logger.warning(
                "Good regions are marked but distillation is switched off "
                "(lambda=0), so their rehearsal patches contribute nothing to "
                "the loss. Set the rehearsal fraction to 0 to skip them, or "
                "give distillation a weight to use them."
            )

        if self.label_smoothing > 0:
            logger.info(f"Label smoothing: {self.label_smoothing} (targets: {self.label_smoothing/2:.3f} to {1-self.label_smoothing/2:.3f})")
        if self.distillation_lambda > 0:
            # FX-interpreted models (torch.export UnflattenedModule, often
            # wrapped in BatchLoopWrapper) keep every intermediate tensor
            # alive in the FX env, so distillation's two passes can OOM
            # even on H100/A100 80GB. We don't auto-disable here — request
            # a larger node if needed; the OOM handler in train() will
            # disable it as a last resort if memory actually runs out.
            inner = getattr(self.model, "model", self.model)
            base = getattr(inner, "model", inner)
            if type(base).__name__ in ("UnflattenedModule",) or type(inner).__name__ == "BatchLoopWrapper":
                logger.warning(
                    "Distillation enabled with an FX-interpreted base model "
                    "(UnflattenedModule). This requires substantial GPU memory; "
                    "if you OOM, distillation will be disabled automatically as "
                    "a fallback in the OOM handler."
                )
            if self._anchors_available:
                scope_str = "good regions only"
            elif self.distillation_all_voxels:
                scope_str = "all voxels"
            else:
                scope_str = "unlabeled voxels only"
            logger.info(f"Teacher distillation enabled: lambda={self.distillation_lambda} ({scope_str})")

        # The distillation teacher is the model as it was before this run
        # changed it: with LoRA the same module with its adapter switched off,
        # for a full finetune a frozen copy of the starting weights, made now,
        # while they are the starting weights (see strategy.teacher).
        self._teacher = None
        if self.distillation_lambda > 0:
            self._teacher = self.strategy.teacher(self.model)
            if self.teacher_model is not None:
                self.teacher_model.to(self.device)

        # Autocast dtype. This defaulted to fp16 (autocast's CUDA default) and
        # every run on this model NaN'd out on the startup probe and fell back
        # to fp32 -- so "mixed precision" was never once in effect, and the
        # tensor cores sat idle for the whole job.
        #
        # fp16 carries 5 exponent bits, so a UNet this deep overflows in the
        # forward pass. bf16 has fp32's 8, which is exactly the failure mode
        # it exists to fix, and needs no loss scaling. Ampere and newer only
        # (H100/H200 = cc 9.0 yes; the RTX 2080 Ti workstation = cc 7.5 no),
        # so fall back to fp16 where it is unavailable and let the existing
        # probe demote to fp32 if that NaNs too.
        self.amp_dtype = torch.float16
        if self.device.type == "cuda":
            try:
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
            except Exception as e:
                logger.debug(f"bf16 support check failed ({e}); staying on fp16.")

        # GradScaler compensates for fp16's narrow range; bf16 does not need
        # it, and enabling it there costs a little and hides real overflows.
        self.scaler = GradScaler(
            'cuda',
            enabled=use_mixed_precision and self.amp_dtype is torch.float16,
        )

        # Training state
        self.current_epoch = 0
        # Where train() starts counting; load_checkpoint() moves it past the
        # checkpoint's epoch.
        self._start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        # Average supervised loss of the epoch just finished. Checkpoint
        # selection uses this rather than the combined loss -- see the epoch
        # loop for why the combined loss cannot rank epochs.
        self.last_supervised_loss = float('nan')
        self.training_stats = []

        # TensorBoard. File-based, so it works from GPU nodes with no network
        # and needs no service; one `tensorboard --logdir` over the training/
        # tree overlays every run ever made. Silently off if unavailable.
        self.tb = None
        self.tb_dir = self.output_dir / "tensorboard"
        self.tb_image_every = 5          # epochs between patch images
        # Monotonic across restarts: the CLI makes a new trainer for every
        # iteration, writing to the same tensorboard/ directory, and passes
        # the previous one's position on. Starting each at 0 drew every
        # iteration's curves on top of each other.
        self._tb_step = int(tb_start_step)
        self._tb_epoch = int(tb_start_epoch)
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.tb_dir))
            except Exception as e:  # not installed, or logdir not writable
                logger.info(f"TensorBoard logging disabled: {e}")

    @property
    def teacher_model(self) -> Optional[nn.Module]:
        """A full finetune's distillation teacher, the frozen copy of its starting weights; else None.

        The CLI hands it to the next iteration's trainer. Setting it to None
        frees it (the OOM handler does, with distillation).
        """
        return self.strategy.teacher_model

    @teacher_model.setter
    def teacher_model(self, value):
        self.strategy.teacher_model = value
        if value is None:
            self._teacher = None

    def close(self):
        """Close the TensorBoard writer; the CLI calls this when an iteration is done."""
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception as e:
                logger.debug(f"Closing the TensorBoard writer failed: {e}")
            self.tb = None

    def _tb_config_markdown(self) -> str:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        rows = [
            ("epochs", self.num_epochs),
            ("batch size", getattr(self.dataloader, "batch_size", "?")),
            ("gradient accumulation", self.gradient_accumulation_steps),
            ("learning rate", self.learning_rate),
            ("mixed precision", f"{self.use_mixed_precision} ({self.amp_dtype})" if self.use_mixed_precision else "False"),
            ("loss", self.loss_type),
            ("margin", self.margin),
            ("balance classes", self.balance_classes),
            ("label smoothing", self.label_smoothing),
            ("distillation lambda", self.distillation_lambda),
            ("mask unannotated", self.mask_unannotated),
            ("trainable params", f"{trainable:,} of {total:,} ({100 * trainable / max(total, 1):.2f}%)"),
            ("device", str(self.device)),
        ]
        return "| setting | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)

    @torch.no_grad()
    def _tb_log_images(self, raw, target, pred, mask):
        """Mid-Z slice of one sample as one strip: raw (centre-cropped to the output) | target | prediction | mask.

        This is the picture that would have shown augmentation doing nothing
        for five months, and that shows raw and labels moving together once
        it does something. Never lets a display problem stop training.
        """
        try:
            p = as_probabilities(pred[0, 0].detach().float(), self._model_has_sigmoid).cpu()
            t = target[0, 0].detach().float().cpu()
            r = raw[0, 0].detach().float().cpu()
            # Valid-padding models emit a smaller volume than they read.
            c = [(rs - ps) // 2 for rs, ps in zip(r.shape, p.shape)]
            r = r[c[0]:c[0] + p.shape[0], c[1]:c[1] + p.shape[1], c[2]:c[2] + p.shape[2]]
            z = p.shape[0] // 2
            r2 = r[z]
            r2 = (r2 - r2.min()) / (r2.max() - r2.min() + 1e-8)
            m2 = (
                mask[0, 0, z].detach().float().cpu().clamp(0, 1)
                if mask is not None else torch.zeros_like(r2)
            )
            # One strip per epoch, raw | target | prediction | mask, separated
            # by a white line: four tags per epoch was too much to scroll.
            sep = torch.ones(r2.shape[0], 2)
            strip = torch.cat(
                [r2, sep, t[z].clamp(0, 1), sep, p[z].clamp(0, 1), sep, m2], dim=1
            )
            self.tb.add_image("patch/raw|target|prediction|mask", strip[None], self._tb_epoch)
        except Exception as e:
            logger.debug(f"TensorBoard image logging skipped: {e}")

    def _fallback_to_fp32(self):
        """Disable mixed precision training."""
        self.use_mixed_precision = False
        self.scaler = GradScaler('cuda', enabled=False)
        logger.warning(
            f"Mixed precision disabled (was {self.amp_dtype}); training in fp32."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_training_state(self):
        """Reset the weights (in place), optimizer, and training counters for a fresh start."""
        self.model = self.strategy.reset(self.model, self.initial_state)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.defaults['lr'],
        )
        self.current_epoch = 0
        self._start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        # Average supervised loss of the epoch just finished. Checkpoint
        # selection uses this rather than the combined loss -- see the epoch
        # loop for why the combined loss cannot rank epochs.
        self.last_supervised_loss = float('nan')
        self.training_stats = []

    def _halve_batch_size(self):
        """Halve batch size and double gradient accumulation to keep effective batch size.

        Returns True if batch size was reduced, False if already at 1.
        """
        old_bs = self.dataloader.batch_size
        new_bs = max(1, old_bs // 2)
        if new_bs >= old_bs:
            return False
        from cellmap_flow.finetune.data import rebuild_loader

        old_accum = self.gradient_accumulation_steps
        self.gradient_accumulation_steps = old_accum * (old_bs // new_bs)
        # Same workers, persistence and sampling as before, only smaller
        # batches (see rebuild_loader).
        self.dataloader = rebuild_loader(self.dataloader, new_bs)
        getattr(self, "_log_message", logger.info)(
            f"Halved batch size {old_bs} → {new_bs}, "
            f"gradient accumulation {old_accum} → {self.gradient_accumulation_steps} "
            f"(effective batch size unchanged)"
        )
        torch.cuda.empty_cache()
        return True

    def _model_cache_targets(self):
        """Return model objects that may survive LoRA unwrap/rewrap cycles."""
        targets = []
        seen = set()
        stack = [self.model]
        while stack:
            obj = stack.pop(0)
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            targets.append(obj)
            for attr in ("base_model", "model", "module"):
                child = getattr(obj, attr, None)
                if child is not None and id(child) not in seen:
                    stack.append(child)
        return targets

    def _sigmoid_cache_key(self):
        if self.select_channel is not None:
            return ("channel", self.select_channel)
        return ("all", None)

    def _get_cached_model_has_sigmoid(self) -> Optional[bool]:
        key = self._sigmoid_cache_key()
        for target in self._model_cache_targets():
            cache = getattr(target, "_cellmap_flow_model_has_sigmoid_cache", None)
            if isinstance(cache, dict) and key in cache:
                return bool(cache[key])
        return None

    def _cache_model_has_sigmoid(self, value: bool):
        key = self._sigmoid_cache_key()
        for target in self._model_cache_targets():
            try:
                cache = getattr(
                    target, "_cellmap_flow_model_has_sigmoid_cache", None
                )
                if not isinstance(cache, dict):
                    cache = {}
                    setattr(target, "_cellmap_flow_model_has_sigmoid_cache", cache)
                cache[key] = bool(value)
            except Exception:
                pass

    def _apply_probability_output_mode(self, log_message):
        """Configure losses for models that already emit probabilities."""
        self._model_has_sigmoid = True
        if self._use_bce:
            log_message(
                "Switching BCEWithLogitsLoss to BCELoss to avoid double-sigmoid"
            )
            self.criterion = nn.BCELoss(reduction='none')
        if hasattr(self.criterion, 'bce_loss'):
            self.criterion.bce_loss = nn.BCELoss(reduction='none')
        # Tell DiceLoss/MarginLoss to skip their sigmoid
        if hasattr(self.criterion, 'apply_sigmoid'):
            self.criterion.apply_sigmoid = False
        if (
            hasattr(self.criterion, 'dice_loss')
            and hasattr(self.criterion.dice_loss, 'apply_sigmoid')
        ):
            self.criterion.dice_loss.apply_sigmoid = False

    def _warn_if_single_class(self, target, mask):
        """Say so when every supervised voxel carries the same label.

        Gradient descent can only do one thing with a target that is all 1:
        raise the prediction everywhere. The model duly does that, globally,
        and the result is worse than the model you started from -- with a
        loss curve that looks unremarkable, because a one-class problem is
        genuinely easy to reduce.

        The usual cause is painting only foreground. An affinity target needs
        both: foreground says "these voxels are one object", background says
        "this is not object, and these two are not joined". Without the
        second, nothing anywhere says 0.
        """
        if self._single_class_checked:
            return
        self._single_class_checked = True
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
                frac = positive / supervised
                logger.info(
                    f"Supervised target: {supervised:.0f} voxels, "
                    f"{100 * frac:.1f}% positive"
                )
                if frac > 0.999 or frac < 0.001:
                    only = "positive (1)" if frac > 0.5 else "negative (0)"
                    missing = "background" if frac > 0.5 else "foreground"
                    logger.warning(
                        "=" * 70 + "\n"
                        f"EVERY supervised voxel is {only}. This run cannot "
                        "teach the model a boundary:\n"
                        "gradient descent will simply push the prediction that "
                        "way everywhere, and\n"
                        f"the result will be worse than the model you started "
                        f"from.\n\n"
                        f"Paint some {missing} as well, and put it where it "
                        "decides something --\n"
                        "between two objects that touch, and on the things "
                        "being confused for one.\n" + "=" * 70
                    )
        except Exception as e:  # never let a diagnostic stop training
            logger.debug(f"Single-class check failed: {e}")

    def _probe_model(self, log_message):
        """Two forward passes before training, to set up how it runs.

        Whether the model NaNs under mixed precision (then train in fp32),
        and whether it ends in a sigmoid (then the losses take
        probabilities). The second answer is cached on the model, so a
        restart on the same model does not probe again.
        """
        # Probe for FP16 stability: run a single forward pass and check for NaN.
        # Some model+data combinations produce NaN under FP16 autocast.
        if self.use_mixed_precision:
            try:
                probe_raw = next(iter(self.dataloader))[0]
                probe_raw = probe_raw[:1]
                probe_raw = probe_raw.to(self.device)
                with torch.no_grad(), autocast(
                    'cuda', enabled=True, dtype=self.amp_dtype
                ):
                    probe_out = self.model(probe_raw)
                if not torch.isfinite(probe_out).all():
                    log_message("WARNING: Model produces NaN/Inf under FP16 — falling back to FP32.")
                    self._fallback_to_fp32()
                del probe_raw, probe_out
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                log_message("WARNING: FP16 probe OOM — falling back to FP32 with smaller batch.")
                self._fallback_to_fp32()
                self._halve_batch_size()
            except Exception as e:
                log_message(f"WARNING: FP16 probe failed ({e}) — falling back to FP32.")
                self._fallback_to_fp32()

        # Probe for built-in sigmoid: if model outputs are bounded to [0,1]
        # even with extreme inputs, the model has sigmoid baked in.
        # In that case, switch BCEWithLogitsLoss to BCELoss to avoid double-sigmoid,
        # and tell DiceLoss/MarginLoss to skip their sigmoid.
        cached_model_has_sigmoid = self._get_cached_model_has_sigmoid()
        if cached_model_has_sigmoid is not None:
            model_has_sigmoid = cached_model_has_sigmoid
            if model_has_sigmoid:
                log_message("Using cached built-in sigmoid detection")
                self._apply_probability_output_mode(log_message)
        else:
            try:
                probe_raw = next(iter(self.dataloader))[0]
                probe_raw = probe_raw[:1]
                probe_extreme = torch.randn(
                    probe_raw.shape,
                    dtype=probe_raw.dtype,
                    device=self.device,
                ) * 100
                with torch.no_grad(), autocast(
                    'cuda', enabled=self.use_mixed_precision, dtype=self.amp_dtype
                ):
                    probe_out = self.model(probe_extreme)
                    if self.select_channel is not None:
                        probe_out = probe_out[
                            :,
                            self.select_channel : self.select_channel + 1,
                            :,
                            :,
                            :,
                        ]
                model_has_sigmoid = bool(
                    ((probe_out.min() >= 0) & (probe_out.max() <= 1)).item()
                )
                self._cache_model_has_sigmoid(model_has_sigmoid)
                if model_has_sigmoid:
                    log_message("Detected built-in sigmoid in model output")
                    self._apply_probability_output_mode(log_message)
                del probe_extreme, probe_out
                torch.cuda.empty_cache()
            except Exception as e:
                log_message(
                    f"WARNING: Sigmoid probe failed ({e}) — assuming raw logits output."
                )
                torch.cuda.empty_cache()

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Training statistics dictionary with:
            - final_loss: Final epoch loss
            - best_loss: Best loss achieved
            - total_epochs: Number of epochs trained
            - total_steps: Total training steps
        """
        # Create log file
        log_file = self.output_dir / "training_log.txt"

        def log_message(msg):
            """Log to console (tee handles writing to log file).

            Timestamped like the logger's lines so epoch duration can be read
            off the log: the 2026-09-23 A/B runs had none on the per-epoch
            summaries, and per-arm speed had to come from LSF's start/end
            times instead. The progress parsers in finetune/job_manager use
            unanchored re.findall, so the prefix does not affect them.
            """
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp} {msg}" if msg else msg, flush=True)

        log_message("="*60)
        log_message("Starting LoRA Finetuning")
        log_message("="*60)
        log_message(f"Epochs: {self.num_epochs}")
        log_message(f"Batches per epoch: {len(self.dataloader)}")
        log_message(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        log_message(f"Effective batch size: {self.dataloader.batch_size * self.gradient_accumulation_steps}")
        if self.use_mixed_precision:
            log_message(
                f"Mixed precision: {self.use_mixed_precision} "
                f"(dtype={str(self.amp_dtype).replace('torch.', '')}, "
                f"grad_scaler={self.scaler.is_enabled()})"
            )
        else:
            log_message("Mixed precision: False (fp32)")
        log_message(f"Mask unannotated regions: {self.mask_unannotated}")
        log_message(f"Log file: {log_file}")
        if self.tb is not None:
            log_message(f"TensorBoard: tensorboard --logdir {self.output_dir.parent}   (this run: {self.tb_dir})")
            self.tb.add_text("config", self._tb_config_markdown(), self._tb_epoch)
        log_message("")

        start_time = time.time()

        # Store log function for use in _train_epoch and helpers
        self._log_message = log_message

        # The probes only look at what the model computes, so they run in
        # eval mode. In train mode a full finetune's BatchNorm added their
        # batches -- one of them noise at 100x -- to its running statistics:
        # about 7,000x on every channel, which takes ~85 batches to decay,
        # longer than most interactive runs, and the export and the served
        # model normalize by them. LoRA's frozen norms are in eval mode
        # either way.
        self.model.eval()
        try:
            self._probe_model(log_message)
        finally:
            self._set_train_mode()

        stop_signal_path = self.output_dir / "stop_signal.json"
        # Make sure no stale signal from a previous run lingers.
        try:
            if stop_signal_path.exists():
                stop_signal_path.unlink()
        except Exception:
            pass

        # Resumed runs carry on after the checkpoint's epoch. This looped
        # from 0 regardless, so --resume re-ran every epoch it had already
        # done, and the checkpoint's epoch number was only ever logged.
        epoch_loss = None
        for epoch in range(self._start_epoch, self.num_epochs):
            self.current_epoch = epoch
            # User-requested graceful stop: drop out of the training loop so
            # the outer flow (inference server + wait for restart) kicks in.
            if stop_signal_path.exists():
                log_message(
                    f"Stop signal received at epoch {epoch+1}/{self.num_epochs}; "
                    f"exiting training loop early."
                )
                try:
                    stop_signal_path.unlink()
                except Exception:
                    pass
                break
            log_message(f"Starting epoch {epoch+1} of {self.num_epochs}...")
            # Mitigation loop: keep applying mitigations (halve batch, then
            # disable distillation) and retrying until the epoch succeeds or
            # there's nothing left to try. A single try/except wasn't enough —
            # users can restart with a much larger lora_r/batch_size combo and
            # the second attempt also OOMs, so we need to iterate.
            epoch_loss = None
            while True:
                try:
                    epoch_loss = self._train_epoch()
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    mitigated = False
                    if self._halve_batch_size():
                        log_message(
                            f"OOM at epoch {epoch+1} — retrying with smaller batch size "
                            f"(now {self.dataloader.batch_size})."
                        )
                        mitigated = True
                    elif self.distillation_lambda > 0:
                        log_message(
                            f"OOM at epoch {epoch+1} and batch already at 1 — "
                            f"disabling distillation (was lambda={self.distillation_lambda}) and retrying."
                        )
                        self.distillation_lambda = 0
                        # A full finetune's frozen teacher is a whole second
                        # copy of the weights; free it with the term it served.
                        self.teacher_model = None
                        mitigated = True
                    if not mitigated:
                        log_message("ERROR: OOM at batch=1 with no distillation. Cannot continue.")
                        # Every diverged return says so: the job manager
                        # watches for this marker, and without it this path
                        # looked like training that simply went quiet.
                        markers.emit(markers.TRAINING_DIVERGED)
                        return {
                            'final_loss': float('nan'),
                            'best_loss': self.best_loss,
                            'total_epochs': epoch,
                            'total_steps': self.global_step,
                            'training_time': time.time() - start_time,
                            'diverged': True,
                        }
                    # Reset optimizer state (accumulated grads are stale after OOM)
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

            # Handle NaN/Inf loss
            if not math.isfinite(epoch_loss):
                if self.use_mixed_precision:
                    # NaN likely caused by FP16 overflow on specific data —
                    # fall back to FP32 and restart training from scratch
                    log_message(
                        f"WARNING: NaN loss at epoch {epoch+1} under FP16 — "
                        f"falling back to FP32 and restarting training."
                    )
                    self._fallback_to_fp32()
                    self._reset_training_state()
                    return self.train()

                self._log_message(
                    f"ERROR: Loss is {epoch_loss} at epoch {epoch+1}. "
                    f"Stopping training."
                )
                markers.emit(markers.TRAINING_DIVERGED)
                return {
                    'final_loss': epoch_loss,
                    'best_loss': self.best_loss,
                    'total_epochs': epoch + 1,
                    'total_steps': self.global_step,
                    'training_time': time.time() - start_time,
                    'diverged': True,
                }

            # Rank epochs by the supervised term, not the combined loss.
            #
            # loss = supervised + lambda * distillation, and the distillation
            # term is minimized by *not changing the model*: at init LoRA has
            # B=0, so the student is identical to the teacher and distillation
            # is exactly 0. Epoch 1 therefore posts a combined loss no later
            # epoch can beat, and "best" stayed pinned to epoch 1 for the whole
            # run. save_adapter() loads best_checkpoint.pth before exporting,
            # so every finetune shipped a model one optimizer step from its
            # starting point -- measurably so: every LoRA B matrix came out at
            # max|B| = 1e-4, which is Adam's first step at lr=1e-4.
            #
            # Distillation belongs in the objective, where it restrains the
            # update; it cannot also be the yardstick for which epoch is best.
            # With lambda=0 the two terms are equal, so this changes nothing.
            selection_loss = self.last_supervised_loss
            if not math.isfinite(selection_loss):
                selection_loss = epoch_loss

            # Log epoch results. On soft targets the BCE cannot go below the
            # target's entropy, so also say how far above that floor it sits.
            bce_extra = ""
            epoch_floor = epoch_mae = None
            if self._epoch_bce_n:
                epoch_floor = self._epoch_bce_floor_sum / self._epoch_bce_n
                epoch_mae = self._epoch_mae_sum / self._epoch_bce_n
                bce_extra = (
                    f" - Above floor: {selection_loss - epoch_floor:.6f}"
                    f" - MAE: {epoch_mae:.6f}"
                )
            self._log_message(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.6f} - "
                f"Supervised: {selection_loss:.6f} - "
                f"Best supervised: {self.best_loss:.6f}"
                f"{bce_extra}"
            )

            # Save checkpoint if best
            if selection_loss < self.best_loss:
                self.best_loss = selection_loss
                self._log_message("  Saving best checkpoint...")
                self.save_checkpoint(is_best=True)
                self._log_message(f"  → Saved best checkpoint")

            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(is_best=False)

            self.training_stats.append({
                'epoch': epoch + 1,
                'loss': epoch_loss,
                'best_loss': self.best_loss,
            })

            if self.tb is not None:
                self._tb_epoch += 1
                data_wait, compute = getattr(self, "_last_epoch_timing", (0.0, 0.0))
                e = self._tb_epoch
                self.tb.add_scalar("epoch/loss", epoch_loss, e)
                self.tb.add_scalar("epoch/supervised", selection_loss, e)
                self.tb.add_scalar("epoch/best_supervised", self.best_loss, e)
                if epoch_floor is not None:
                    self.tb.add_scalar("epoch/bce_floor", epoch_floor, e)
                    self.tb.add_scalar("epoch/supervised_above_floor", selection_loss - epoch_floor, e)
                    self.tb.add_scalar("epoch/mean_abs_error", epoch_mae, e)
                self.tb.add_scalar("time/epoch_data_wait_s", data_wait, e)
                self.tb.add_scalar("time/epoch_compute_s", compute, e)
                if self.device.type == "cuda":
                    self.tb.add_scalar("memory/peak_gb", torch.cuda.max_memory_allocated() / 1e9, e)
                self.tb.flush()

        # Final checkpoint
        self.save_checkpoint(is_best=False)

        total_time = time.time() - start_time
        self._log_message("")
        self._log_message("="*60)
        self._log_message("Training Complete!")
        self._log_message(f"Total time: {total_time/60:.2f} minutes")
        if self.tb is not None:
            self.tb.flush()
        self._log_message(f"Best loss: {self.best_loss:.6f}")
        # None when no epoch ran: a stop requested before the first one, or a
        # resume from a finished run. Formatting it used to raise TypeError.
        self._log_message(
            f"Final loss: {epoch_loss:.6f}" if epoch_loss is not None
            else "Final loss: n/a (no epoch ran)"
        )
        self._log_message(f"Output directory: {self.output_dir}")
        self._log_message("="*60)

        return {
            'final_loss': epoch_loss,
            'best_loss': self.best_loss,
            'total_epochs': self.num_epochs,
            'total_steps': self.global_step,
            'training_time': total_time,
        }

    def _set_train_mode(self):
        """Train mode, except for the frozen base's norm layers under LoRA.

        A full finetune trains its norm layers, so it keeps them in train
        mode (see strategy.train_mode).
        """
        self.strategy.train_mode(self.model)

    @torch.no_grad()
    def _gradients_finite(self) -> bool:
        """Whether every accumulated gradient is finite.

        With fp16 the GradScaler already skips a step whose gradients
        overflowed, and it must see them scaled, so leave that case to it.
        """
        if self.scaler.is_enabled():
            return True
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return True
        return bool(torch.isfinite(torch.stack([g.float().norm() for g in grads])).all())

    @torch.no_grad()
    def _teacher_forward(self, raw):
        """The starting model's prediction on ``raw``, for the distillation term.

        LoRA: the model itself with its adapters switched off, in eval mode.
        Full finetune: the frozen copy taken before training.

        Only called while distillation is on, and __init__ makes the teacher
        whenever it is (only the OOM handler drops it, along with the term).
        Making one here instead would be wrong for a full finetune: it would
        copy the student as it is now, not as it started.
        """
        with self._teacher as model:
            with autocast('cuda', enabled=self.use_mixed_precision, dtype=self.amp_dtype):
                teacher_pred = model(raw)
        if self.select_channel is not None:
            teacher_pred = teacher_pred[:, self.select_channel:self.select_channel+1, :, :, :]
        return teacher_pred.detach()

    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        epoch_loss = 0.0
        epoch_supervised_loss = 0.0
        # Batches that had any supervised voxel. The supervised mean ranks
        # epochs for the best checkpoint; a batch made only of rehearsal
        # patches has nothing supervised and a supervised loss of exactly 0,
        # so averaging over every batch made "best epoch" partly track how
        # many rehearsal draws an epoch happened to get.
        supervised_batches = 0
        epoch_distill_loss = 0.0
        num_batches = len(self.dataloader)
        self._epoch_bce_floor_sum = 0.0
        self._epoch_mae_sum = 0.0
        self._epoch_bce_n = 0

        # Gradient-flow diagnostic: watch one LoRA-B param across the epoch
        # AND, at end of epoch, count how many trainable params received any
        # gradient at all. Together they answer:
        #   - is backward reaching LoRA at all? (per-param mean|grad|)
        #   - if yes for some, which ones? (zero-grad count + sample names)
        diag_param_name = None
        diag_param = None
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora_B" in name:
                diag_param_name = name
                diag_param = param
                break
        diag_param_initial = (
            diag_param.detach().clone() if diag_param is not None else None
        )
        diag_grad_abs_sum = 0.0
        diag_grad_count = 0

        # Track zero-grad status across all trainable params for the LAST
        # batch of the epoch (cumulative grad before zero_grad fires).
        diag_param_grad_seen_nonzero: dict[str, bool] = {}

        # Fetch explicitly so the wait on the loader is measurable. That is
        # the number that says whether prefetching keeps up, and it was lost
        # when the loss_history.csv instrumentation fell out of the tree.
        epoch_data_wait = 0.0
        epoch_compute = 0.0
        batch_iter = iter(self.dataloader)
        for batch_idx in range(num_batches):
            t_fetch = time.time()
            batch = next(batch_iter)
            t_after_fetch = time.time()
            epoch_data_wait += t_after_fetch - t_fetch
            # The dataset yields a third tensor once the session has good
            # regions: a per-voxel mask marking where the student should be
            # held to the teacher. Older datasets yield the 2-tuple, so both
            # shapes have to work.
            if len(batch) == 3:
                raw, target, anchor = batch
                anchor = anchor.to(self.device, non_blocking=True)
            else:
                raw, target = batch
                anchor = None

            # Move to device
            raw = raw.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Handle partial annotations: create mask and shift labels
            mask = None
            if self.target_transform is not None:
                target, mask = self.target_transform(target)
            elif self.mask_unannotated:
                # Legacy behavior: binary single-channel
                mask = (target > 0).float()  # (B, C, Z, Y, X)
                # Shift labels down by 1 (but keep 0 as 0)
                # e.g., 0->0 (unannotated), 1->0 (background), 2->1 (foreground)
                target = torch.clamp(target - 1, min=0)

            self._warn_if_single_class(target, mask)

            # Class balancing splits voxels into fg and bg by the target as
            # annotated, before smoothing. Split by the smoothed target, every
            # bg voxel carried s/2 of fg weight: with 100 fg voxels against
            # 100k bg at s = 0.1, 98% of the "fg" half of the loss was bg.
            hard_target = target

            # Apply label smoothing: 0 -> s/2, 1 -> 1-s/2
            # This prevents the model from being pushed to extreme 0/1 outputs,
            # preserving gradual distance-like predictions
            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + self.label_smoothing / 2

            # Teacher forward pass for distillation (before student pass): the
            # model as it was before this run (see _teacher_forward).
            teacher_pred = None
            if self.distillation_lambda > 0:
                teacher_pred = self._teacher_forward(raw)
                if not torch.isfinite(teacher_pred).all():
                    logger.warning(f"NaN/Inf in teacher_pred! range=[{teacher_pred.min():.4f}, {teacher_pred.max():.4f}]")

            # Student forward pass with mixed precision
            with autocast(
                'cuda', enabled=self.use_mixed_precision, dtype=self.amp_dtype
            ):
                pred = self.model(raw)

                if not torch.isfinite(pred).all():
                    logger.warning(f"NaN/Inf in student pred! range=[{pred.min():.4f}, {pred.max():.4f}]")

                # Select specific channel if requested (e.g., mito = channel 2 from 8-channel output)
                if self.select_channel is not None:
                    pred = pred[:, self.select_channel:self.select_channel+1, :, :, :]

            # Losses in fp32, outside autocast; only the forward pass runs in
            # reduced precision. Two reasons. Models with a built-in sigmoid
            # (the cellmap *_distance_* UNets) make the trainer swap to
            # BCELoss on probabilities, and autocast refuses to run
            # binary_cross_entropy at all. And a bf16 probability near 0 or 1
            # has too few mantissa bits for log(p) / log(1-p) to mean much.
            with autocast('cuda', enabled=False):
                pred = pred.float()
                if teacher_pred is not None:
                    teacher_pred = teacher_pred.float()

                # Compute supervised loss with optional mask
                # MSE compares probabilities with the 0/1 target, as dice and
                # margin do. On a logit model it used to compare the raw
                # logits, training them toward 0 and 1 -- which the served
                # sigmoid turns into 0.5 and 0.73, wrecking any threshold.
                loss_pred = as_probabilities(pred, self._model_has_sigmoid) if self._use_mse else pred
                if (self._use_bce or self._use_mse) and mask is not None:
                    # For per-element losses (BCE, MSE), manually apply mask
                    per_element_loss = self.criterion(loss_pred, target)

                    def _masked_mean(per_voxel):
                        if self.balance_classes:
                            # Average fg and bg separately so each contributes equally
                            return balanced_mean(per_voxel, hard_target, mask)
                        return masked_mean(per_voxel, mask)

                    supervised_loss = _masked_mean(per_element_loss)
                    if self._use_bce:
                        # Same weighting applied to the target's own entropy
                        # gives the floor this batch's BCE cannot go below;
                        # mean |p - t| is the loss-independent view of the fit.
                        with torch.no_grad():
                            bce_floor = _masked_mean(soft_target_entropy(target))
                            prob = as_probabilities(pred, self._model_has_sigmoid)
                            mae = masked_mean((prob - target).abs(), mask)
                        self._step_bce_metrics = (bce_floor.item(), mae.item())
                elif hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                    # For custom losses that support masking (DiceLoss, CombinedLoss, MarginLoss)
                    supervised_loss = self.criterion(pred, target, mask)
                else:
                    # No masking needed
                    supervised_loss = self.criterion(loss_pred, target)
                    if self._use_bce or self._use_mse:
                        supervised_loss = supervised_loss.mean()

                loss = supervised_loss

                if not torch.isfinite(supervised_loss):
                    logger.warning(f"NaN/Inf supervised_loss: {supervised_loss.item()}")

                # Compute distillation loss
                distill_loss = torch.tensor(0.0, device=self.device)
                if self.distillation_lambda > 0 and teacher_pred is not None:
                    if anchor is not None:
                        # Good regions decide where the teacher is worth
                        # copying. Distilling on every unlabeled voxel
                        # instead -- the scope below -- anchors hardest
                        # right beside the scribbles, which is the one place
                        # the teacher is known to be wrong, so it partly
                        # fights the correction being made. Restrict it to
                        # the regions the user actually vouched for.
                        scope = "anchor"
                    elif self.distillation_all_voxels or mask is None:
                        scope = "all"
                    else:
                        scope = "unlabeled"
                    distill_loss = distillation_loss(
                        pred, teacher_pred, scope, anchor_mask=anchor,
                        unlabeled_mask=None if mask is None else 1.0 - mask,
                    )
                    if not torch.isfinite(distill_loss):
                        logger.warning(f"NaN/Inf distillation_loss: {distill_loss.item()}")
                    loss = loss + self.distillation_lambda * distill_loss

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            if self.tb is not None and batch_idx == 0 and self.current_epoch % self.tb_image_every == 0:
                self._tb_log_images(raw, target, pred, mask)

            # A non-finite loss must never reach the optimizer. This check used
            # to run after scaler.step(), and under bf16 or fp32 -- where the
            # scaler is off and so does not skip inf/NaN steps itself -- AdamW
            # had already written NaN into every trainable weight. LoRA
            # recovered on restart by re-making its adapter; a full finetune
            # kept NaN weights, served them, and trained on from them.
            if not torch.isfinite(loss):
                logger.warning(
                    f"NaN/Inf loss at epoch {self.current_epoch+1}, batch "
                    f"{batch_idx+1}; skipping the update and aborting the epoch."
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.last_supervised_loss = float('nan')
                return float('nan')

            # Backward pass
            self.scaler.scale(loss).backward()

            # Diagnostic: capture grad on the watched LoRA param BEFORE the
            # optimizer step (zero_grad clears it). Also note which trainable
            # params have ever seen a nonzero gradient this epoch so we can
            # report the dead ones at the end.
            if diag_param is not None and diag_param.grad is not None:
                diag_grad_abs_sum += diag_param.grad.detach().abs().mean().item()
                diag_grad_count += 1
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if diag_param_grad_seen_nonzero.get(name, False):
                    continue
                if p.grad is not None and p.grad.detach().abs().sum().item() > 0:
                    diag_param_grad_seen_nonzero[name] = True

            # Update weights after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if not self._gradients_finite():
                    logger.warning(
                        f"NaN/Inf gradient at epoch {self.current_epoch+1}, batch "
                        f"{batch_idx+1}; skipping the update and aborting the epoch."
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.last_supervised_loss = float('nan')
                    return float('nan')
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
                if self.tb is not None:
                    self._tb_step += 1
                    self.tb.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, self._tb_step)
                    self.tb.add_scalar("train/supervised", supervised_loss.item(), self._tb_step)
                    if self._step_bce_metrics is not None:
                        floor, mae = self._step_bce_metrics
                        self.tb.add_scalar("train/bce_floor", floor, self._tb_step)
                        self.tb.add_scalar("train/supervised_above_floor", supervised_loss.item() - floor, self._tb_step)
                        self.tb.add_scalar("train/mean_abs_error", mae, self._tb_step)
                    if self.distillation_lambda > 0:
                        self.tb.add_scalar("train/distillation", distill_loss.item(), self._tb_step)
                    self.tb.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self._tb_step)
                    self.tb.add_scalar("time/step_s", time.time() - t_after_fetch, self._tb_step)
                    self.tb.add_scalar("time/data_wait_s", t_after_fetch - t_fetch, self._tb_step)

            # Accumulate losses (unscaled)
            batch_loss = loss.item() * self.gradient_accumulation_steps
            # .item() above synchronised the device, so this is real compute time.
            epoch_compute += time.time() - t_after_fetch
            self._last_epoch_timing = (epoch_data_wait, epoch_compute)
            if not math.isfinite(batch_loss):
                logger.warning(f"NaN/Inf loss at epoch {self.current_epoch+1}, batch {batch_idx+1}. Aborting epoch.")
                self.last_supervised_loss = float('nan')
                return float('nan')
            epoch_loss += batch_loss
            epoch_distill_loss += distill_loss.item()
            if mask is None or bool(mask.sum() > 0):
                supervised_batches += 1
                epoch_supervised_loss += supervised_loss.item()
                if self._step_bce_metrics is not None:
                    self._epoch_bce_floor_sum += self._step_bce_metrics[0]
                    self._epoch_mae_sum += self._step_bce_metrics[1]
                    self._epoch_bce_n += 1

            # Log progress every batch (since we have few batches)
            avg_loss = epoch_loss / (batch_idx + 1)
            if hasattr(self, '_log_message'):
                if self.distillation_lambda > 0:
                    avg_sup = epoch_supervised_loss / max(supervised_batches, 1)
                    avg_distill = epoch_distill_loss / (batch_idx + 1)
                    self._log_message(
                        f"  Batch {batch_idx+1}/{num_batches} - "
                        f"Loss: {avg_loss:.6f} (sup: {avg_sup:.6f}, distill: {avg_distill:.6f})"
                    )
                else:
                    self._log_message(
                        f"  Batch {batch_idx+1}/{num_batches} - "
                        f"Loss: {avg_loss:.6f}"
                    )
            else:
                # Fallback if _log_message not set
                msg = f"  Batch {batch_idx+1}/{num_batches} - Loss: {avg_loss:.6f}"
                print(msg)
                logger.info(msg)

        # Handle leftover accumulated gradients at end of epoch
        # (in case num_batches is not divisible by gradient_accumulation_steps)
        if num_batches % self.gradient_accumulation_steps != 0:
            if not self._gradients_finite():
                logger.warning("NaN/Inf gradient in the last accumulation step; skipping the update.")
                self.optimizer.zero_grad(set_to_none=True)
                self.last_supervised_loss = float('nan')
                return float('nan')
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.global_step += 1

        # Diagnostic summary: per-watched-param grad/delta + counts of
        # trainable params that received any nonzero gradient this epoch.
        if diag_param is not None and hasattr(self, "_log_message"):
            mean_grad = (
                diag_grad_abs_sum / diag_grad_count if diag_grad_count else 0.0
            )
            param_delta = (
                (diag_param.detach() - diag_param_initial).abs().mean().item()
                if diag_param_initial is not None
                else 0.0
            )
            self._log_message(
                f"  [diag] {diag_param_name}: "
                f"mean|grad|={mean_grad:.3e} (over {diag_grad_count} batches), "
                f"mean|param_delta|={param_delta:.3e} this epoch"
            )

        if hasattr(self, "_log_message"):
            n_trainable = sum(
                1 for _, p in self.model.named_parameters() if p.requires_grad
            )
            n_live = sum(1 for v in diag_param_grad_seen_nonzero.values() if v)
            n_dead = n_trainable - n_live
            dead_names = [
                name for name, p in self.model.named_parameters()
                if p.requires_grad and not diag_param_grad_seen_nonzero.get(name)
            ]
            self._log_message(
                f"  [diag] gradient flow: {n_live}/{n_trainable} trainable "
                f"params got nonzero grad; {n_dead} are dead. "
                f"First 5 dead: {dead_names[:5]}"
            )

        # NaN when nothing in the epoch was supervised: train() then ranks
        # the epoch by its total loss instead.
        self.last_supervised_loss = (
            epoch_supervised_loss / supervised_batches if supervised_batches else float('nan')
        )
        return epoch_loss / num_batches

    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            is_best: If True, saves as "best_checkpoint.pth", which save_adapter
                exports; otherwise as "checkpoint_epoch_<N>.pth"
        """
        checkpoint_name = "best_checkpoint.pth" if is_best else f"checkpoint_epoch_{self.current_epoch+1}.pth"
        checkpoint_path = self.output_dir / checkpoint_name
        # What the strategy keeps: LoRA its adapter and optimizer state every
        # time; a full finetune only its best weights (see strategy.checkpoint).
        state = self.strategy.checkpoint(self.model, self.optimizer, self.scaler, is_best)
        if state is None:
            return
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'training_stats': self.training_stats,
            **state,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def save_adapter(self, adapter_path: Optional[str] = None, export_dir: Optional[str] = None):
        """
        Export the finetune: only the LoRA adapter, or a full finetune's weights.

        Automatically loads the best checkpoint weights before saving
        so the exported adapter reflects the best training epoch.

        Args:
            adapter_path: Path to save adapter. If None, uses output_dir/lora_adapter.
                A full finetune ignores it and writes output_dir/full_finetune.
            export_dir: Instead, the directory to export into: the adapter
                goes to export_dir/lora_adapter, full weights to
                export_dir/full_finetune/model_state_dict.pt. The CLI gives
                every iteration its own.

        Returns:
            The adapter directory, or the full-finetune weights file.
        """
        base = Path(export_dir) if export_dir is not None else self.output_dir
        if export_dir is not None:
            adapter_path = None

        # Load best checkpoint weights before saving
        best_ckpt = self.output_dir / "best_checkpoint.pth"
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            if checkpoint.get('lora_only', False):
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint (epoch {checkpoint['epoch'] + 1}, loss {checkpoint['best_loss']:.6f}) before saving adapter")
        else:
            logger.warning("No best checkpoint found, saving adapter from final epoch weights")

        # The adapter, or for a full finetune the whole state dict, where
        # FinetuneModelConfig(weights_path=...) expects it.
        return str(self.strategy.export(self.model, base, path=adapter_path))

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if checkpoint.get('lora_only', False):
            # Checkpoint contains only trainable (LoRA) params — merge into full state
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self._start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_stats = checkpoint.get('training_stats', [])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Resuming after epoch {self.current_epoch+1}, at epoch {self._start_epoch+1}")
