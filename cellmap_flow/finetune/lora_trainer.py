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

        # Apply mask if provided
        if mask is not None:
            mask = mask.reshape(mask.size(0), 1, -1)  # (B, 1, N)
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
    ):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision
        self.select_channel = select_channel
        self.mask_unannotated = mask_unannotated
        self.label_smoothing = label_smoothing
        self.distillation_lambda = distillation_lambda
        self.distillation_all_voxels = distillation_all_voxels
        self.balance_classes = balance_classes
        self.target_transform = target_transform

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
        if self.label_smoothing > 0:
            logger.info(f"Label smoothing: {self.label_smoothing} (targets: {self.label_smoothing/2:.3f} to {1-self.label_smoothing/2:.3f})")
        if self.distillation_lambda > 0:
            scope_str = "all voxels" if self.distillation_all_voxels else "unlabeled voxels only"
            logger.info(f"Teacher distillation enabled: lambda={self.distillation_lambda} ({scope_str})")

        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=use_mixed_precision)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_stats = []

    def _fallback_to_fp32(self):
        """Disable mixed precision training."""
        self.use_mixed_precision = False
        self.scaler = GradScaler('cuda', enabled=False)
        torch.cuda.empty_cache()

    def _reset_training_state(self):
        """Reset LoRA weights, optimizer, and training counters for a fresh start."""
        from peft import PeftModel
        if isinstance(self.model, PeftModel):
            # Reset LoRA adapter weights to zero (equivalent to base model)
            for name, param in self.model.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    nn.init.zeros_(param) if 'lora_B' in name else nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.optimizer.defaults['lr'],
        )
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
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
            """Log to console (tee handles writing to log file)."""
            print(msg, flush=True)

        log_message("="*60)
        log_message("Starting LoRA Finetuning")
        log_message("="*60)
        log_message(f"Epochs: {self.num_epochs}")
        log_message(f"Batches per epoch: {len(self.dataloader)}")
        log_message(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        log_message(f"Effective batch size: {self.dataloader.batch_size * self.gradient_accumulation_steps}")
        log_message(f"Mixed precision: {self.use_mixed_precision}")
        log_message(f"Mask unannotated regions: {self.mask_unannotated}")
        log_message(f"Log file: {log_file}")
        log_message("")

        self.model.train()
        start_time = time.time()

        # Store log function for use in _train_epoch and helpers
        self._log_message = log_message

        # Probe for FP16 stability: run a single forward pass and check for NaN.
        # Some model+data combinations produce NaN under FP16 autocast.
        if self.use_mixed_precision:
            try:
                probe_raw, _ = next(iter(self.dataloader))
                probe_raw = probe_raw.to(self.device)
                with torch.no_grad(), autocast('cuda', enabled=True):
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
        try:
            probe_raw, _ = next(iter(self.dataloader))
            probe_extreme = torch.randn_like(probe_raw) * 100
            probe_extreme = probe_extreme.to(self.device)
            with torch.no_grad():
                probe_out = self.model(probe_extreme)
            model_has_sigmoid = probe_out.min() >= 0 and probe_out.max() <= 1
            if model_has_sigmoid:
                log_message("Detected built-in sigmoid in model output")
                if self._use_bce:
                    log_message("Switching BCEWithLogitsLoss to BCELoss to avoid double-sigmoid")
                    self.criterion = nn.BCELoss(reduction='none')
                if hasattr(self.criterion, 'bce_loss'):
                    self.criterion.bce_loss = nn.BCELoss(reduction='none')
                # Tell DiceLoss/MarginLoss to skip their sigmoid
                if hasattr(self.criterion, 'apply_sigmoid'):
                    self.criterion.apply_sigmoid = False
                if hasattr(self.criterion, 'dice_loss') and hasattr(self.criterion.dice_loss, 'apply_sigmoid'):
                    self.criterion.dice_loss.apply_sigmoid = False
            del probe_extreme, probe_out
            torch.cuda.empty_cache()
        except Exception as e:
            log_message(f"WARNING: Sigmoid probe failed ({e}) — assuming raw logits output.")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            try:
                epoch_loss = self._train_epoch()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if not self._halve_batch_size():
                    log_message("ERROR: OOM even at batch_size=1. Cannot continue.")
                    return {
                        'final_loss': float('nan'),
                        'best_loss': self.best_loss,
                        'total_epochs': epoch,
                        'total_steps': self.global_step,
                        'training_time': time.time() - start_time,
                        'diverged': True,
                    }
                log_message(f"OOM at epoch {epoch+1} — retrying with smaller batch size.")
                # Reset optimizer state (accumulated grads are stale after OOM)
                self.optimizer.zero_grad(set_to_none=True)
                epoch_loss = self._train_epoch()

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

            # Log epoch results
            self._log_message(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.6f} - "
                f"Best: {self.best_loss:.6f}"
            )

            # Save checkpoint if best
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
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

        # Final checkpoint
        self.save_checkpoint(is_best=False)

        total_time = time.time() - start_time
        self._log_message("")
        self._log_message("="*60)
        self._log_message("Training Complete!")
        self._log_message(f"Total time: {total_time/60:.2f} minutes")
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

        for batch_idx, (raw, target) in enumerate(self.dataloader):
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
                        with autocast('cuda', enabled=self.use_mixed_precision):
                            teacher_pred = self.model(raw)
                            if self.select_channel is not None:
                                teacher_pred = teacher_pred[:, self.select_channel:self.select_channel+1, :, :, :]
                        teacher_pred = teacher_pred.detach()
                    finally:
                        self.model.enable_adapter_layers()
                if not torch.isfinite(teacher_pred).all():
                    logger.warning(f"NaN/Inf in teacher_pred! range=[{teacher_pred.min():.4f}, {teacher_pred.max():.4f}]")

            # Student forward pass with mixed precision
            with autocast('cuda', enabled=self.use_mixed_precision):
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
                    if self.distillation_all_voxels or mask is None:
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

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses (unscaled)
            batch_loss = loss.item() * self.gradient_accumulation_steps
            if not math.isfinite(batch_loss):
                logger.warning(f"NaN/Inf loss at epoch {self.current_epoch+1}, batch {batch_idx+1}. Aborting epoch.")
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

        return epoch_loss / num_batches

    def save_checkpoint(self, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            is_best: If True, saves as "best_model.pth"
        """
        checkpoint_name = "best_checkpoint.pth" if is_best else f"checkpoint_epoch_{self.current_epoch+1}.pth"
        checkpoint_path = self.output_dir / checkpoint_name

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
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_stats = checkpoint.get('training_stats', [])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch+1}")
