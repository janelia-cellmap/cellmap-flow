#!/usr/bin/env python
"""
Command-line interface for LoRA finetuning.

Usage:
    python -m cellmap_flow.finetune.cli \
        --model-checkpoint /path/to/checkpoint \
        --corrections corrections.zarr \
        --output-dir output/fly_organelles_v1.1

    # With custom settings
    python -m cellmap_flow.finetune.cli \
        --model-checkpoint /path/to/checkpoint \
        --corrections corrections.zarr \
        --output-dir output/fly_organelles_v1.1 \
        --lora-r 16 \
        --batch-size 4 \
        --num-epochs 20 \
        --learning-rate 2e-4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from cellmap_flow.models.models_config import FlyModelConfig, DaCapoModelConfig
from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora
from cellmap_flow.finetune.dataset import create_dataloader
from cellmap_flow.finetune.trainer import LoRAFinetuner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Finetune CellMap-Flow models with LoRA using user corrections"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="fly",
        choices=["fly", "dacapo"],
        help="Model type (fly or dacapo)"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (for fly models)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (for filtering corrections)"
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["mito"],
        help="Model output channels"
    )
    parser.add_argument(
        "--input-voxel-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Input voxel size (Z Y X)"
    )
    parser.add_argument(
        "--output-voxel-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Output voxel size (Z Y X)"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling (default: 16)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout (default: 0.0)"
    )

    # Data arguments
    parser.add_argument(
        "--corrections",
        type=str,
        required=True,
        help="Path to corrections.zarr directory"
    )
    parser.add_argument(
        "--patch-shape",
        type=int,
        nargs=3,
        default=None,
        help="Patch shape for training (Z Y X). Default: None (use full corrections)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and adapter"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="combined",
        choices=["dice", "bce", "combined"],
        help="Loss function (default: combined)"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision (FP16) training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4)"
    )

    # Resuming
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Print configuration
    logger.info("="*60)
    logger.info("LoRA Finetuning Configuration")
    logger.info("="*60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Corrections: {args.corrections}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    if args.model_type == "fly":
        model_config = FlyModelConfig(
            checkpoint_path=args.model_checkpoint,
            channels=args.channels,
            input_voxel_size=tuple(args.input_voxel_size),
            output_voxel_size=tuple(args.output_voxel_size),
        )
    elif args.model_type == "dacapo":
        # Parse dacapo run name and iteration from checkpoint path
        # Expected format: /path/to/runs/{run_name}/model_checkpoint_{iteration}
        checkpoint_path = Path(args.model_checkpoint)
        iteration = int(checkpoint_path.stem.split('_')[-1])
        run_name = checkpoint_path.parent.name

        model_config = DaCapoModelConfig(
            run_name=run_name,
            iteration=iteration,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    base_model = model_config.config.model
    logger.info(f"✓ Model loaded: {type(base_model).__name__}")

    # Wrap with LoRA
    logger.info(f"Wrapping model with LoRA (r={args.lora_r})...")
    lora_model = wrap_model_with_lora(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Create dataloader
    logger.info(f"Loading corrections from {args.corrections}...")
    dataloader = create_dataloader(
        args.corrections,
        batch_size=args.batch_size,
        patch_shape=tuple(args.patch_shape),
        augment=not args.no_augment,
        num_workers=args.num_workers,
        shuffle=True,
        model_name=args.model_name,
    )
    logger.info(f"✓ DataLoader created: {len(dataloader.dataset)} corrections")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = LoRAFinetuner(
        lora_model,
        dataloader,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=not args.no_mixed_precision,
        loss_type=args.loss_type,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        stats = trainer.train()

        # Save final adapter
        logger.info("\nSaving LoRA adapter...")
        trainer.save_adapter()

        logger.info("\n" + "="*60)
        logger.info("Finetuning Complete!")
        logger.info(f"Best loss: {stats['best_loss']:.6f}")
        logger.info(f"Adapter saved to: {args.output_dir}/lora_adapter")
        logger.info("="*60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving current state...")
        trainer.save_checkpoint(is_best=False)
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
