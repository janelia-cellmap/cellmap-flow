"""
LoRA finetuning trainer for CellMap-Flow models.

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

logger = logging.getLogger(__name__)


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

        # For BCE, manually apply mask if provided
        bce = self.bce_loss(pred, target)
        if mask is not None:
            bce = bce * mask
            bce = bce.sum() / mask.sum().clamp(min=1)  # Average over masked regions
        else:
            bce = bce.mean()

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
            # regardless of how many scribble voxels each has
            fg_mask = target * mask
            bg_mask = (1.0 - target) * mask
            fg_count = fg_mask.sum().clamp(min=1)
            bg_count = bg_mask.sum().clamp(min=1)
            fg_contrib = (fg_loss * fg_mask).sum() / fg_count
            bg_contrib = (bg_loss * bg_mask).sum() / bg_count
            return (fg_contrib + bg_contrib) / 2.0

        # Blend by target: target=1 -> fg_loss, target=0 -> bg_loss
        loss = target * fg_loss + (1.0 - target) * bg_loss

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)
        return loss.mean()


class LoRAFinetuner:
    """
    Trainer for finetuning models with LoRA adapters.

    Features:
    - Mixed precision (FP16) training for memory efficiency
    - Gradient accumulation to simulate larger batch sizes
    - Checkpointing with best model tracking
    - Progress logging
    - Partial annotation support (mask unannotated regions)

    Args:
        model: PEFT model with LoRA adapters
        dataloader: DataLoader for training data
        output_dir: Directory to save checkpoints and logs
        learning_rate: Learning rate (default: 1e-4)
        num_epochs: Number of training epochs (default: 10)
        gradient_accumulation_steps: Steps to accumulate gradients (default: 1)
        use_mixed_precision: Enable FP16 training (default: True)
        loss_type: Loss function ("dice", "bce", or "combined")
        device: Training device ("cuda" or "cpu", auto-detected if None)
        select_channel: Optional channel index to select from multi-channel output (default: None)
        mask_unannotated: If True (default), only compute loss on annotated regions (target > 0).
                         Targets are shifted down by 1 (e.g., 1->0, 2->1) after masking.
                         This allows partial annotations where 0=unannotated, 1=background, 2=foreground, etc.
                         Ignored if target_transform is provided.
        target_transform: Optional TargetTransform instance that converts raw annotations
                         to (target, mask) pairs. Overrides mask_unannotated when provided.
                         See cellmap_flow.finetune.target_transforms.

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
        distillation_lambda: float = 0.0,
        distillation_all_voxels: bool = False,
        margin: float = 0.3,
        balance_classes: bool = False,
        target_transform=None,
        tensorboard: bool = True,
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

        # Optimizer (only LoRA parameters)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )

        # Loss function
        self._use_bce = False
        self._use_mse = False
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
        # Marking them is an explicit request to be held there, so honour it
        # rather than training a no-op and looking like it worked.
        self._anchors_available = bool(
            getattr(getattr(self.dataloader, "dataset", None), "emits_anchor", False)
        )
        if self._anchors_available and self.distillation_lambda <= 0:
            self.distillation_lambda = 1.0
            logger.warning(
                "Good regions are marked but distillation_lambda was 0, which "
                "would make them inert. Setting lambda=1.0 so the anchors take "
                "effect; pass an explicit lambda to override."
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
        self._tb_step = 0                # monotonic: global_step resets on restart
        self._tb_epoch = 0
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.tb_dir))
            except Exception as e:  # not installed, or logdir not writable
                logger.info(f"TensorBoard logging disabled: {e}")

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
        """Mid-Z slice of one sample: raw (centre-cropped to the output), target, prediction, mask.

        This is the picture that would have shown augmentation doing nothing
        for five months, and that shows raw and labels moving together once
        it does something. Never lets a display problem stop training.
        """
        try:
            p = torch.sigmoid(pred[0, 0].detach().float()).cpu()
            t = target[0, 0].detach().float().cpu()
            r = raw[0, 0].detach().float().cpu()
            # Valid-padding models emit a smaller volume than they read.
            c = [(rs - ps) // 2 for rs, ps in zip(r.shape, p.shape)]
            r = r[c[0]:c[0] + p.shape[0], c[1]:c[1] + p.shape[1], c[2]:c[2] + p.shape[2]]
            z = p.shape[0] // 2
            r2 = r[z]
            r2 = (r2 - r2.min()) / (r2.max() - r2.min() + 1e-8)
            step = self._tb_epoch
            self.tb.add_image("patch/raw", r2[None], step)
            self.tb.add_image("patch/target", t[z][None].clamp(0, 1), step)
            self.tb.add_image("patch/prediction", p[z][None], step)
            if mask is not None:
                self.tb.add_image("patch/mask", mask[0, 0, z].detach().float().cpu()[None].clamp(0, 1), step)
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
        """Reset LoRA weights, optimizer, and training counters for a fresh start."""
        from peft import PeftModel
        if isinstance(self.model, PeftModel):
            # Reset LoRA adapter weights to zero (equivalent to base model)
            for name, param in self.model.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    nn.init.zeros_(param) if 'lora_B' in name else nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        else:
            logger.warning(
                "Full finetune: a fresh restart resets the optimizer but NOT the "
                "weights, which continue from where the previous run left them."
            )
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.defaults['lr'],
        )
        self.current_epoch = 0
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
        old_accum = self.gradient_accumulation_steps
        self.gradient_accumulation_steps = old_accum * (old_bs // new_bs)
        self.dataloader = DataLoader(
            self.dataloader.dataset,
            batch_size=new_bs,
            shuffle=True,
            num_workers=self.dataloader.num_workers,
            pin_memory=self.dataloader.pin_memory,
            multiprocessing_context=self.dataloader.multiprocessing_context,
        )
        self._log_message(
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
            times instead. The progress parsers in finetune_job_manager use
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

        self.model.train()
        start_time = time.time()

        # Store log function for use in _train_epoch and helpers
        self._log_message = log_message

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

        stop_signal_path = self.output_dir / "stop_signal.json"
        # Make sure no stale signal from a previous run lingers.
        try:
            if stop_signal_path.exists():
                stop_signal_path.unlink()
        except Exception:
            pass

        for epoch in range(self.num_epochs):
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
                        mitigated = True
                    if not mitigated:
                        log_message("ERROR: OOM at batch=1 with no distillation. Cannot continue.")
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
                print("TRAINING_DIVERGED", flush=True)
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

            # Log epoch results
            self._log_message(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.6f} - "
                f"Supervised: {selection_loss:.6f} - "
                f"Best supervised: {self.best_loss:.6f}"
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
        self._log_message(f"Final loss: {epoch_loss:.6f}")
        self._log_message(f"Output directory: {self.output_dir}")
        self._log_message("="*60)

        return {
            'final_loss': epoch_loss,
            'best_loss': self.best_loss,
            'total_epochs': self.num_epochs,
            'total_steps': self.global_step,
            'training_time': total_time,
        }

    def _train_epoch(self) -> float:
        """Train for one epoch and return average loss."""
        epoch_loss = 0.0
        epoch_supervised_loss = 0.0
        epoch_distill_loss = 0.0
        num_batches = len(self.dataloader)

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

            # Apply label smoothing: 0 -> s/2, 1 -> 1-s/2
            # This prevents the model from being pushed to extreme 0/1 outputs,
            # preserving gradual distance-like predictions
            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + self.label_smoothing / 2

            # Teacher forward pass for distillation (before student pass)
            # Uses the base model without LoRA adapters as the teacher
            teacher_pred = None
            if self.distillation_lambda > 0:
                with torch.no_grad():
                    self.model.disable_adapter_layers()
                    try:
                        with autocast(
                            'cuda', enabled=self.use_mixed_precision,
                            dtype=self.amp_dtype,
                        ):
                            teacher_pred = self.model(raw)
                            if self.select_channel is not None:
                                teacher_pred = teacher_pred[:, self.select_channel:self.select_channel+1, :, :, :]
                        teacher_pred = teacher_pred.detach()
                    finally:
                        self.model.enable_adapter_layers()
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

                # Compute supervised loss with optional mask
                if (self._use_bce or self._use_mse) and mask is not None:
                    # For per-element losses (BCE, MSE), manually apply mask
                    per_element_loss = self.criterion(pred, target)
                    if self.balance_classes:
                        # Average fg and bg separately so each contributes equally
                        fg_mask = target * mask
                        bg_mask = (1.0 - target) * mask
                        fg_count = fg_mask.sum().clamp(min=1)
                        bg_count = bg_mask.sum().clamp(min=1)
                        fg_contrib = (per_element_loss * fg_mask).sum() / fg_count
                        bg_contrib = (per_element_loss * bg_mask).sum() / bg_count
                        supervised_loss = (fg_contrib + bg_contrib) / 2.0
                    else:
                        supervised_loss = (per_element_loss * mask).sum() / mask.sum().clamp(min=1)
                elif hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                    # For custom losses that support masking (DiceLoss, CombinedLoss, MarginLoss)
                    supervised_loss = self.criterion(pred, target, mask)
                else:
                    # No masking needed
                    supervised_loss = self.criterion(pred, target)
                    if self._use_bce or self._use_mse:
                        supervised_loss = supervised_loss.mean()

                loss = supervised_loss

                if not torch.isfinite(supervised_loss):
                    logger.warning(f"NaN/Inf supervised_loss: {supervised_loss.item()}")

                # Compute distillation loss
                distillation_loss = torch.tensor(0.0, device=self.device)
                if self.distillation_lambda > 0 and teacher_pred is not None:
                    distill_loss_map = (pred - teacher_pred) ** 2  # per-element MSE
                    if anchor is not None:
                        # Good regions decide where the teacher is worth
                        # copying. Distilling on every unlabeled voxel
                        # instead -- the branch below -- anchors hardest
                        # right beside the scribbles, which is the one place
                        # the teacher is known to be wrong, so it partly
                        # fights the correction being made. Restrict it to
                        # the regions the user actually vouched for.
                        #
                        # Broadcast over channels: the mask is single-channel
                        # (it is about location) while pred may not be.
                        anchor_mask = anchor.float().expand_as(distill_loss_map)
                        distillation_loss = (
                            distill_loss_map.float() * anchor_mask
                        ).sum() / anchor_mask.sum().clamp(min=1)
                    elif self.distillation_all_voxels or mask is None:
                        # Apply on all voxels
                        distillation_loss = distill_loss_map.mean()
                    else:
                        # Apply only on unlabeled voxels.
                        # Cast to float32 before multiply/sum to avoid FP16 overflow
                        # when summing over many voxels (e.g., 13-channel models).
                        unlabeled_mask = (1.0 - mask).float()
                        distillation_loss = (distill_loss_map.float() * unlabeled_mask).sum() / unlabeled_mask.sum().clamp(min=1)
                    if not torch.isfinite(distillation_loss):
                        logger.warning(f"NaN/Inf distillation_loss: {distillation_loss.item()}")
                    loss = loss + self.distillation_lambda * distillation_loss

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            if self.tb is not None and batch_idx == 0 and self.current_epoch % self.tb_image_every == 0:
                self._tb_log_images(raw, target, pred, mask)

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
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
                if self.tb is not None:
                    self._tb_step += 1
                    self.tb.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, self._tb_step)
                    self.tb.add_scalar("train/supervised", supervised_loss.item(), self._tb_step)
                    if self.distillation_lambda > 0:
                        self.tb.add_scalar("train/distillation", distillation_loss.item(), self._tb_step)
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
            epoch_supervised_loss += supervised_loss.item()
            epoch_distill_loss += distillation_loss.item()

            # Log progress every batch (since we have few batches)
            avg_loss = epoch_loss / (batch_idx + 1)
            if hasattr(self, '_log_message'):
                if self.distillation_lambda > 0:
                    avg_sup = epoch_supervised_loss / (batch_idx + 1)
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

        self.last_supervised_loss = epoch_supervised_loss / num_batches
        return epoch_loss / num_batches

    def _is_peft(self) -> bool:
        try:
            from peft import PeftModel
        except ImportError:
            return False
        return isinstance(self.model, PeftModel)

    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            is_best: If True, saves as "best_model.pth"
        """
        checkpoint_name = "best_checkpoint.pth" if is_best else f"checkpoint_epoch_{self.current_epoch+1}.pth"
        checkpoint_path = self.output_dir / checkpoint_name
        if not self._is_peft():
            # Full finetune: every parameter is trainable, so a LoRA-style
            # checkpoint would be the whole model plus two Adam moments --
            # ~9.5 GB for an 800M-param UNet, twenty times per run. Keep only
            # the best weights, without optimizer state (no resume), which is
            # what save_adapter() exports anyway.
            if not is_best:
                if not getattr(self, "_warned_full_ckpt", False):
                    logger.info("Full finetune: skipping periodic checkpoints; best_checkpoint.pth holds the full weights.")
                    self._warned_full_ckpt = True
                return
            torch.save({
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'training_stats': self.training_stats,
                'lora_only': False,
                'full_model': True,
            }, checkpoint_path)
            logger.debug(f"Full-model checkpoint saved: {checkpoint_path}")
            return

        # Save only trainable (LoRA) parameters to avoid writing the full
        # 800M+ param base model to disk every checkpoint.
        trainable_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
        trainable_state = {k: v for k, v in self.model.state_dict().items() if k in trainable_keys}
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': trainable_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'training_stats': self.training_stats,
            'lora_only': True,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def save_adapter(self, adapter_path: Optional[str] = None):
        """
        Save only the LoRA adapter (not the full model).

        Automatically loads the best checkpoint weights before saving
        so the exported adapter reflects the best training epoch.

        Args:
            adapter_path: Path to save adapter. If None, uses output_dir/lora_adapter
        """
        from cellmap_flow.finetune.lora_wrapper import save_lora_adapter

        if adapter_path is None:
            adapter_path = str(self.output_dir / "lora_adapter")

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

        if not self._is_peft():
            # Full finetune: there is no adapter; export the whole state dict
            # where FinetuneModelConfig(weights_path=...) expects it.
            out = self.output_dir / "full_finetune"
            out.mkdir(parents=True, exist_ok=True)
            weights = out / "model_state_dict.pt"
            torch.save(self.model.state_dict(), weights)
            logger.info(f"Full finetuned weights saved to: {weights}")
            return str(weights)
        save_lora_adapter(self.model, adapter_path)
        logger.info(f"LoRA adapter saved to: {adapter_path}")

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
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_stats = checkpoint.get('training_stats', [])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch+1}")
