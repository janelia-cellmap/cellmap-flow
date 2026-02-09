#!/usr/bin/env python
"""
End-to-end test of LoRA finetuning pipeline.

This script:
1. Loads the fly_organelles model
2. Wraps it with LoRA
3. Creates a dataloader from test corrections
4. Runs finetuning for a few epochs
5. Saves the adapter
6. Tests inference with the finetuned model
"""

import torch
from pathlib import Path

from cellmap_flow.models.models_config import FlyModelConfig
from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora, load_lora_adapter
from cellmap_flow.finetune.dataset import create_dataloader
from cellmap_flow.finetune.trainer import LoRAFinetuner

def main():
    print("="*60)
    print("End-to-End LoRA Finetuning Test")
    print("="*60)

    # Configuration
    model_checkpoint = "/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000"
    corrections_path = "test_corrections.zarr"
    output_dir = "output/test_finetuning"

    # 1. Load model
    print("\n1. Loading fly_organelles model...")
    model_config = FlyModelConfig(
        checkpoint_path=model_checkpoint,
        channels=["mito"],
        input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16),
    )
    base_model = model_config.config.model
    print(f"   ✓ Model loaded: {type(base_model).__name__}")

    # 2. Wrap with LoRA
    print("\n2. Wrapping model with LoRA (r=8)...")
    lora_model = wrap_model_with_lora(
        base_model,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    # 3. Create dataloader
    print("\n3. Creating dataloader...")
    dataloader = create_dataloader(
        corrections_path,
        batch_size=2,
        patch_shape=None,  # Use full correction size (56x56x56)
        augment=True,
        num_workers=2,
        shuffle=True,
    )
    print(f"   ✓ DataLoader created: {len(dataloader.dataset)} corrections")

    # 4. Create trainer
    print("\n4. Creating trainer...")
    trainer = LoRAFinetuner(
        lora_model,
        dataloader,
        output_dir=output_dir,
        learning_rate=1e-4,
        num_epochs=3,  # Just 3 epochs for testing
        gradient_accumulation_steps=2,
        use_mixed_precision=True,
        loss_type="combined",
    )
    print("   ✓ Trainer created")

    # 5. Train
    print("\n5. Starting training (3 epochs)...")
    print("-"*60)
    stats = trainer.train()
    print("-"*60)

    # 6. Save adapter
    print("\n6. Saving LoRA adapter...")
    trainer.save_adapter()
    adapter_path = Path(output_dir) / "lora_adapter"
    print(f"   ✓ Adapter saved to: {adapter_path}")

    # 7. Test loading the adapter
    print("\n7. Testing adapter loading...")
    fresh_model = FlyModelConfig(
        checkpoint_path=model_checkpoint,
        channels=["mito"],
        input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16),
    ).config.model

    loaded_model = load_lora_adapter(
        fresh_model,
        str(adapter_path),
        is_trainable=False,
    )
    print("   ✓ Adapter loaded successfully")

    # 8. Test inference
    print("\n8. Testing inference with finetuned model...")
    loaded_model.eval()
    loaded_model = loaded_model.cuda() if torch.cuda.is_available() else loaded_model

    # Get a sample from dataloader
    for raw_batch, target_batch in dataloader:
        raw_batch = raw_batch.cuda() if torch.cuda.is_available() else raw_batch

        with torch.no_grad():
            pred = loaded_model(raw_batch)

        print(f"   Input shape: {raw_batch.shape}")
        print(f"   Output shape: {pred.shape}")
        print(f"   Output range: [{pred.min():.3f}, {pred.max():.3f}]")
        break

    # 9. Summary
    print("\n" + "="*60)
    print("✓ End-to-End Test Passed!")
    print("="*60)
    print(f"Training stats:")
    print(f"  - Best loss: {stats['best_loss']:.6f}")
    print(f"  - Final loss: {stats['final_loss']:.6f}")
    print(f"  - Training time: {stats['training_time']/60:.2f} minutes")
    print(f"  - Total steps: {stats['total_steps']}")
    print(f"\nAdapter location: {adapter_path}")
    print(f"Checkpoint location: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
