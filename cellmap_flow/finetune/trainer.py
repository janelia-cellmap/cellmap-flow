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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predictions (B, C, Z, Y, X) - raw logits or probabilities
            target: Targets (B, C, Z, Y, X) - binary masks [0, 1]

        Returns:
            Dice loss value (scalar)
        """
        # Flatten spatial dimensions
        pred = pred.reshape(pred.size(0), pred.size(1), -1)  # (B, C, N)
        target = target.reshape(target.size(0), target.size(1), -1)  # (B, C, N)

        # Apply sigmoid if pred is logits (not already in [0, 1])
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

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
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predictions (B, C, Z, Y, X) - raw logits
            target: Targets (B, C, Z, Y, X) - binary masks [0, 1]

        Returns:
            Combined loss value (scalar)
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class LoRAFinetuner:
    """
    Trainer for finetuning models with LoRA adapters.

    Features:
    - Mixed precision (FP16) training for memory efficiency
    - Gradient accumulation to simulate larger batch sizes
    - Checkpointing with best model tracking
    - Progress logging

    Args:
        model: PEFT model with LoRA adapters
        dataloader: DataLoader for training data
        output_dir: Directory to save checkpoints and logs
        learning_rate: Learning rate (default: 1e-4)
        num_epochs: Number of training epochs (default: 10)
        gradient_accumulation_steps: Steps to accumulate gradients (default: 4)
        use_mixed_precision: Enable FP16 training (default: True)
        loss_type: Loss function ("dice", "bce", or "combined")
        device: Training device ("cuda" or "cpu", auto-detected if None)

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
        gradient_accumulation_steps: int = 4,
        use_mixed_precision: bool = True,
        loss_type: str = "combined",
        device: Optional[str] = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision

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
        if loss_type == "dice":
            self.criterion = DiceLoss()
        elif loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "combined":
            self.criterion = CombinedLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        logger.info(f"Using {loss_type} loss")

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
        logger.info("="*60)
        logger.info("Starting LoRA Finetuning")
        logger.info("="*60)
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Batches per epoch: {len(self.dataloader)}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.dataloader.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info("")

        self.model.train()
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch()

            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.6f} - "
                f"Best: {self.best_loss:.6f}"
            )

            # Save checkpoint if best
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(is_best=True)
                logger.info(f"  → Saved best checkpoint")

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
        logger.info("")
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Best loss: {self.best_loss:.6f}")
        logger.info(f"Final loss: {epoch_loss:.6f}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*60)

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
        num_batches = len(self.dataloader)

        for batch_idx, (raw, target) in enumerate(self.dataloader):
            # Move to device
            raw = raw.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_mixed_precision):
                pred = self.model(raw)
                loss = self.criterion(pred, target)

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

            # Accumulate loss (unscaled)
            epoch_loss += loss.item() * self.gradient_accumulation_steps

            # Log progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                logger.info(
                    f"  Batch {batch_idx+1}/{num_batches} - "
                    f"Loss: {avg_loss:.6f}"
                )

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
