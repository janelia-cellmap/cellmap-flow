#!/usr/bin/env python
"""
Example workflow for training with sparse point annotations.

This demonstrates the complete workflow:
1. Generate sparse point corrections
2. Combine them into a single zarr
3. Train with masked loss (only on annotated regions)

The sparse annotations use:
- Label 0: Unannotated regions (ignored in loss)
- Label 1: Background (included in loss as class 0)
- Label 2: Foreground/mito (included in loss as class 1)

The mask_unannotated=True setting automatically:
- Creates a mask where labels > 0 (annotated regions)
- Shifts labels down by 1 (1→0, 2→1) for loss calculation
"""

import torch
from pathlib import Path

from cellmap_flow.models.models_config import FlyModelConfig
from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora
from cellmap_flow.finetune.dataset import create_dataloader
from cellmap_flow.finetune.trainer import LoRAFinetuner


def main():
    print("=" * 60)
    print("Sparse Annotation Training Workflow")
    print("=" * 60)
    print()

    # Paths
    model_checkpoint = "/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000"

    # Look for the most recent sparse corrections file
    corrections_dir = Path(
        "/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections"
    )
    sparse_files = sorted(corrections_dir.glob("sparse_corrections_*.zarr"))

    if not sparse_files:
        print(f"Error: No sparse corrections found in {corrections_dir}")
        print()
        print("Please run:")
        print("  python scripts/generate_sparse_corrections.py")
        print()
        return

    sparse_corrections = str(sparse_files[-1])  # Use most recent
    output_dir = "output/sparse_annotation_finetuning"

    print(f"Using corrections: {sparse_corrections}")
    print()

    # 1. Load model
    print("1. Loading mito model...")
    model_config = FlyModelConfig(
        checkpoint_path=model_checkpoint,
        channels=["mito"],  # This checkpoint only has 1 channel (mito)
        input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16),
    )
    base_model = model_config.config.model
    print(f"   ✓ Model loaded: {type(base_model).__name__}")

    # 2. Wrap with LoRA
    print("\n2. Wrapping model with LoRA (r=4, alpha=8, dropout=0.1)...")
    lora_model = wrap_model_with_lora(
        base_model,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.1,
    )

    # 3. Create dataloader
    print("\n3. Creating dataloader from sparse corrections...")
    dataloader = create_dataloader(
        sparse_corrections,
        batch_size=5,
        patch_shape=None,  # Use full correction size
        augment=True,
        num_workers=2,
        shuffle=True,
    )
    print(f"   ✓ DataLoader created: {len(dataloader.dataset)} corrections")

    # 4. Create trainer with mask_unannotated=True
    print("\n4. Creating trainer with masked loss...")
    print("   This will:")
    print("   - Only calculate loss on annotated regions (label > 0)")
    print("   - Treat label 1 as background (class 0)")
    print("   - Treat label 2 as foreground (class 1)")
    print()

    trainer = LoRAFinetuner(
        lora_model,
        dataloader,
        output_dir=output_dir,
        learning_rate=1e-4,
        num_epochs=10,
        gradient_accumulation_steps=1,  # batch_size=5 in dataloader, no accumulation
        use_mixed_precision=True,
        loss_type="combined",
        mask_unannotated=True,  # KEY: Only compute loss on annotated regions!
        select_channel=None,  # Model only has 1 channel, no selection needed
    )
    print("   ✓ Trainer created")

    # 5. Train
    print("\n5. Starting training...")
    print("-" * 60)
    stats = trainer.train()
    print("-" * 60)

    # 6. Save adapter
    print("\n6. Saving LoRA adapter...")
    trainer.save_adapter()
    adapter_path = Path(output_dir) / "lora_adapter"
    print(f"   ✓ Adapter saved to: {adapter_path}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)
    print(f"Training stats:")
    print(f"  - Best loss: {stats['best_loss']:.6f}")
    print(f"  - Final loss: {stats['final_loss']:.6f}")
    print(f"  - Training time: {stats['training_time']/60:.2f} minutes")
    print(f"  - Total steps: {stats['total_steps']}")
    print(f"\nAdapter location: {adapter_path}")
    print(f"Checkpoint location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
