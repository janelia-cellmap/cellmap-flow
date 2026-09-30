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

logger = logging.getLogger(__name__)


def main():
    # Here, not at import, so that importing this module leaves the importing
    # process's logging alone. cellmap_flow.globals installs the shared log
    # format when it is first imported, replacing whatever was set before,
    # and the job imports it on its first raw read (ImageDataInterface) and
    # when it serves (the inference server). Imported first, so that the
    # format set here holds for the whole run.
    import cellmap_flow.globals  # noqa: F401
    from cellmap_flow.plugins import load_plugins

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
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
