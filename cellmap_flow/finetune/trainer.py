"""
LoRA finetuning trainer for CellMap-Flow models.

This module provides a trainer class for finetuning models using user
corrections with mixed-precision training and gradient accumulation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
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

        # Apply sigmoid if pred is logits (not already in [0, 1])
        if pred.min() < 0 or pred.max() > 1:
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

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        threshold_high = 1.0 - self.margin  # e.g., 0.7
        threshold_low = self.margin          # e.g., 0.3

        # Foreground loss: penalize if pred < threshold_high
        fg_loss = torch.relu(threshold_high - pred) ** 2
        # Background loss: penalize if pred > threshold_low
        bg_loss = torch.relu(pred - threshold_low) ** 2

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
            self.criterion = MarginLoss(margin=margin)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Label smoothing is redundant with margin loss
        if loss_type == "margin" and self.label_smoothing > 0:
            logger.warning("Label smoothing is redundant with margin loss, setting to 0")
            self.label_smoothing = 0.0

        logger.info(f"Using {loss_type} loss")
        if self.label_smoothing > 0:
            logger.info(f"Label smoothing: {self.label_smoothing} (targets: {self.label_smoothing/2:.3f} to {1-self.label_smoothing/2:.3f})")
        if self.distillation_lambda > 0:
            scope_str = "all voxels" if self.distillation_all_voxels else "unlabeled voxels only"
            logger.info(f"Teacher distillation enabled: lambda={self.distillation_lambda} ({scope_str})")

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=use_mixed_precision)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_stats = []

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
            """Log to both console and file."""
            print(msg, flush=True)  # Always print to console with immediate flush
            logger.info(msg)  # Also log normally
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
                f.flush()  # Flush immediately for live streaming

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

        # Store log function for use in _train_epoch
        self._log_message = log_message

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch()

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
            if self.mask_unannotated:
                # Create mask for annotated regions (target > 0)
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
                        with autocast(enabled=self.use_mixed_precision):
                            teacher_pred = self.model(raw)
                            if self.select_channel is not None:
                                teacher_pred = teacher_pred[:, self.select_channel:self.select_channel+1, :, :, :]
                        teacher_pred = teacher_pred.detach()
                    finally:
                        self.model.enable_adapter_layers()

            # Student forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                pred = self.model(raw)

                if batch_idx == 0:
                    print(f"DEBUG trainer: pred.shape after model = {pred.shape}, select_channel = {self.select_channel}")

                # Select specific channel if requested (e.g., mito = channel 2 from 8-channel output)
                if self.select_channel is not None:
                    pred = pred[:, self.select_channel:self.select_channel+1, :, :, :]
                    if batch_idx == 0:
                        print(f"DEBUG trainer: pred.shape after channel selection = {pred.shape}")

                # Compute supervised loss with optional mask
                if (self._use_bce or self._use_mse) and mask is not None:
                    # For per-element losses (BCE, MSE), manually apply mask
                    supervised_loss = self.criterion(pred, target)
                    supervised_loss = supervised_loss * mask
                    supervised_loss = supervised_loss.sum() / mask.sum().clamp(min=1)
                elif hasattr(self.criterion, 'forward') and 'mask' in self.criterion.forward.__code__.co_varnames:
                    # For custom losses that support masking (DiceLoss, CombinedLoss, MarginLoss)
                    supervised_loss = self.criterion(pred, target, mask)
                else:
                    # No masking needed
                    supervised_loss = self.criterion(pred, target)
                    if self._use_bce or self._use_mse:
                        supervised_loss = supervised_loss.mean()

                loss = supervised_loss

                # Compute distillation loss
                distillation_loss = torch.tensor(0.0, device=self.device)
                if self.distillation_lambda > 0 and teacher_pred is not None:
                    distill_loss_map = (pred - teacher_pred) ** 2  # per-element MSE
                    if self.distillation_all_voxels or mask is None:
                        # Apply on all voxels
                        distillation_loss = distill_loss_map.mean()
                    else:
                        # Apply only on unlabeled voxels
                        unlabeled_mask = 1.0 - mask  # 1 where unlabeled
                        distillation_loss = (distill_loss_map * unlabeled_mask).sum() / unlabeled_mask.sum().clamp(min=1)
                    loss = loss + self.distillation_lambda * distillation_loss

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights after accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Debug: Check gradient norms
                if batch_idx == 0:
                    grad_norms = []
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norms.append((name, param.grad.norm().item()))
                    if grad_norms:
                        print(f"DEBUG: First 5 gradient norms:")
                        for name, norm in grad_norms[:5]:
                            print(f"  {name}: {norm:.6f}")
                    else:
                        print("DEBUG: NO GRADIENTS COMPUTED!")

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses (unscaled)
            epoch_loss += loss.item() * self.gradient_accumulation_steps
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

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'training_stats': self.training_stats,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def save_adapter(self, adapter_path: Optional[str] = None):
        """
        Save only the LoRA adapter (not the full model).

        Args:
            adapter_path: Path to save adapter. If None, uses output_dir/lora_adapter
        """
        from cellmap_flow.finetune.lora_wrapper import save_lora_adapter

        if adapter_path is None:
            adapter_path = str(self.output_dir / "lora_adapter")

        save_lora_adapter(self.model, adapter_path)
        logger.info(f"LoRA adapter saved to: {adapter_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.training_stats = checkpoint.get('training_stats', [])

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch+1}")
