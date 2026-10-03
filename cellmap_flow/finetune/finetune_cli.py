#!/usr/bin/env python
"""
Command-line interface for LoRA finetuning.

Usage:
    python -m cellmap_flow.finetune.finetune_cli \
        --model-checkpoint /path/to/checkpoint \
        --corrections corrections.zarr \
        --output-dir output/fly_organelles_v1.1

    # With custom settings
    python -m cellmap_flow.finetune.finetune_cli \
        --model-checkpoint /path/to/checkpoint \
        --corrections corrections.zarr \
        --output-dir output/fly_organelles_v1.1 \
        --lora-r 8 \
        --batch-size 8 \
        --num-epochs 20 \
        --learning-rate 2e-4

The job manager launches this module path, and a running job outlives a
dashboard upgrade, so the path, its flags and what it prints only change
compatibly. The flags are finetune.cli's, the job's loop is
finetune.session_loop.TrainingSession, what each iteration writes is
finetune.run_outputs', and its stdout markers are finetune.markers'.
"""

import logging
import sys

from cellmap_flow.finetune.adaptation import FullStrategy, LoraStrategy
from cellmap_flow.finetune.cli import model_config_from_args, parse_args
from cellmap_flow.finetune.model_loading import load_trainable_model
from cellmap_flow.finetune.session_loop import TrainingSession
from cellmap_flow.finetune.trainable import LORA, finetune_modes
from cellmap_flow.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main():
    from cellmap_flow.plugins import load_plugins

    # Here, not at import, so that importing this module leaves the importing
    # process's logging alone. The shared format (logging_setup), the one
    # every cellmap_flow command logs in, so the job's own lines and those of
    # the inference server it starts read the same.
    configure_logging()
    args = parse_args()
    # A plugin's model type, normalizers and postprocessors, which the model
    # entry and the chain it serves with may name.
    load_plugins()

    # Print configuration
    logger.info("=" * 60)
    logger.info("LoRA Finetuning Configuration")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Corrections: {args.corrections}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(
        f"LoRA rank: {args.lora_r} (alpha: {args.lora_alpha}, "
        f"min_channels: {args.lora_min_channels})"
    )
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")

    # === Load model (once) ===
    logger.info("Loading model...")

    model_config = model_config_from_args(args)
    base_model = load_trainable_model(model_config)

    # === Wrap with LoRA (once - same object is reused across restarts) ===
    # The request decides, here only: --lora-r 0 is a full finetune (see
    # FullStrategy). From here on the model does (strategy_for). A finetuned
    # model given as the base still carries its adapter; either way it is
    # folded into the weights first.
    modes = finetune_modes(base_model)
    if not modes:
        raise SystemExit(f"{type(base_model).__name__} has no parameters to train: this model cannot be finetuned")
    if args.lora_r > 0 and LORA not in modes:
        raise SystemExit(
            f"This model ({type(base_model).__name__}) can only be fully finetuned, not with LoRA: its "
            "network is compiled (TorchScript), so adapters cannot be attached to its layers. "
            "Set the LoRA rank to 0 (full finetune)."
        )
    if args.lora_r <= 0:
        strategy = FullStrategy()
    else:
        strategy = LoraStrategy(
            args.lora_r, args.lora_alpha, args.lora_dropout, min_channels=args.lora_min_channels
        )
    model = strategy.prepare(base_model)

    # === Training loop (supports restart via the server or a signal file) ===
    return TrainingSession(args, model_config, model, strategy).run()


if __name__ == "__main__":
    sys.exit(main())
