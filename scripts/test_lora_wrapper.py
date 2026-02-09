#!/usr/bin/env python
"""Test LoRA wrapper with fly_organelles model."""

import sys
from cellmap_flow.models.models_config import FlyModelConfig
from cellmap_flow.finetune.lora_wrapper import (
    detect_adaptable_layers,
    wrap_model_with_lora,
    print_lora_parameters,
)

def main():
    print("="*60)
    print("Testing LoRA Wrapper")
    print("="*60)

    # Load fly_organelles model
    print("\n1. Loading fly_organelles model...")
    model_config = FlyModelConfig(
        checkpoint_path="/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000",
        channels=["mito"],
        input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16)
    )

    # Get the model
    model = model_config.config.model
    print(f"✓ Model loaded: {type(model).__name__}")

    # Detect adaptable layers
    print("\n2. Detecting adaptable layers...")
    layers = detect_adaptable_layers(model)
    print(f"✓ Found {len(layers)} adaptable layers")
    print(f"  First 5: {layers[:5]}")
    print(f"  Last 5: {layers[-5:]}")

    # Print original parameters
    print("\n3. Original model parameters:")
    print_lora_parameters(model)

    # Wrap with LoRA
    print("\n4. Wrapping model with LoRA (r=8, alpha=16)...")
    try:
        lora_model = wrap_model_with_lora(
            model,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
        )
        print("✓ LoRA model created successfully")
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("\nTo install PEFT:")
        print("  pip install peft")
        sys.exit(1)

    # Test with different LoRA ranks
    print("\n5. Testing different LoRA ranks...")
    for r in [4, 8, 16]:
        # Load fresh model for each test
        fresh_config = FlyModelConfig(
            checkpoint_path="/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000",
            channels=["mito"],
            input_voxel_size=(16, 16, 16),
            output_voxel_size=(16, 16, 16)
        )
        test_model = wrap_model_with_lora(
            fresh_config.config.model,
            lora_r=r,
            lora_alpha=r*2,
        )
        print(f"  r={r}:")
        print_lora_parameters(test_model)

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    main()
