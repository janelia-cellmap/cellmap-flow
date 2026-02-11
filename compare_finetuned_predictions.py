#!/usr/bin/env python3
"""
Compare predictions before and after LoRA finetuning.

Loads raw data from corrections zarr, runs through finetuned model,
and saves predictions alongside original for comparison.
"""

import argparse
import numpy as np
import zarr
import torch
from pathlib import Path

from cellmap_flow.models.models_config import FlyModelConfig
from cellmap_flow.finetune import load_lora_adapter


def normalize_input(raw_crop):
    """Normalize uint8 [0, 255] to float32 [-1, 1]."""
    return (raw_crop.astype(np.float32) / 127.5) - 1.0


def run_prediction(model, device, raw_crop, select_channel=None):
    """Run model inference on a raw crop.

    Args:
        model: The model (base or finetuned)
        device: torch device
        raw_crop: Raw input (H, W, D) uint8
        select_channel: Optional channel index to select (e.g., 2 for mito)

    Returns:
        Prediction (H, W, D) as float32 [0, 1]
    """
    # Normalize input to [-1, 1]
    input_normalized = normalize_input(raw_crop)

    # Add batch and channel dimensions
    input_tensor = torch.from_numpy(input_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Select channel if specified
    if select_channel is not None:
        output = output[:, select_channel:select_channel+1, :, :, :]

    # Remove batch and channel dimensions and convert to numpy
    prediction = output[0, 0].cpu().numpy().astype(np.float32)

    return prediction


def add_ome_ngff_metadata(group, name, voxel_size, translation_offset=None):
    """Add OME-NGFF v0.4 metadata to a zarr group.

    Args:
        group: Zarr group
        name: Name of the array
        voxel_size: Voxel size in nm [z, y, x]
        translation_offset: Optional translation in VOXELS [z, y, x]
    """
    transforms = []

    # Add scale first
    transforms.append({
        'type': 'scale',
        'scale': voxel_size
    })

    # Then add translation in physical units (nm) if provided
    if translation_offset is not None:
        physical_translation = (np.array(translation_offset) * np.array(voxel_size)).tolist()
        transforms.append({
            'type': 'translation',
            'translation': physical_translation
        })

    group.attrs['multiscales'] = [{
        'version': '0.4',
        'name': name,
        'axes': [
            {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
            {'name': 'x', 'type': 'space', 'unit': 'nanometer'}
        ],
        'datasets': [{
            'path': 's0',
            'coordinateTransformations': transforms
        }]
    }]


def main():
    parser = argparse.ArgumentParser(
        description="Compare predictions before and after LoRA finetuning"
    )
    parser.add_argument(
        "--corrections",
        type=str,
        required=True,
        help="Path to corrections zarr (e.g., corrections/mito_liver.zarr)"
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (e.g., output/fly_organelles_mito_liver/lora_adapter)"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to base model checkpoint"
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

    args = parser.parse_args()

    print("="*60)
    print("Comparing Predictions: Base vs LoRA Finetuned")
    print("="*60)
    print(f"Corrections: {args.corrections}")
    print(f"LoRA adapter: {args.lora_adapter}")
    print(f"Base model: {args.model_checkpoint}")
    print()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load base model
    print("\nLoading base model...")
    model_config = FlyModelConfig(
        checkpoint_path=args.model_checkpoint,
        channels=args.channels,
        input_voxel_size=tuple(args.input_voxel_size),
        output_voxel_size=tuple(args.output_voxel_size),
    )
    base_model = model_config.config.model
    base_model.to(device)
    base_model.eval()
    print(f"✓ Base model loaded")

    # Load LoRA adapter
    print(f"\nLoading LoRA adapter from {args.lora_adapter}...")
    finetuned_model = load_lora_adapter(
        base_model,
        args.lora_adapter,
        is_trainable=False  # For inference
    )
    finetuned_model.to(device)
    finetuned_model.eval()
    print(f"✓ Finetuned model loaded")

    # Determine channel to select (for mito)
    select_channel = None
    if args.channels == ["mito"]:
        select_channel = 2
        print(f"Will select mito channel (index 2) from model output")

    # Open corrections zarr
    print(f"\nLoading corrections from {args.corrections}...")
    corrections_root = zarr.open(args.corrections, mode='a')

    # Get all correction IDs
    correction_ids = [key for key in corrections_root.group_keys()]
    print(f"Found {len(correction_ids)} corrections")

    # Process each correction
    print("\nProcessing corrections...")
    for i, corr_id in enumerate(correction_ids, 1):
        print(f"\n[{i}/{len(correction_ids)}] Processing {corr_id}...")
        corr_group = corrections_root[corr_id]

        # Load raw data
        raw_data = np.array(corr_group['raw/s0'])
        print(f"  Raw shape: {raw_data.shape}, dtype: {raw_data.dtype}")

        # Get voxel size for metadata
        if 'raw' in corr_group and 'multiscales' in corr_group['raw'].attrs:
            multiscales = corr_group['raw'].attrs['multiscales'][0]
            for transform in multiscales['datasets'][0]['coordinateTransformations']:
                if transform['type'] == 'scale':
                    voxel_size = transform['scale']
                    break
        else:
            voxel_size = [16, 16, 16]

        # Calculate translation offset (same as mask)
        raw_shape = np.array(raw_data.shape)
        if 'mask/s0' in corr_group:
            mask_shape = np.array(corr_group['mask/s0'].shape)
            translation_offset = ((raw_shape - mask_shape) // 2).tolist()
        else:
            translation_offset = None

        # Run through finetuned model
        print(f"  Running finetuned model inference...")
        finetuned_pred = run_prediction(
            finetuned_model,
            device,
            raw_data,
            select_channel=select_channel
        )
        print(f"  Finetuned prediction shape: {finetuned_pred.shape}")

        # Save finetuned prediction
        print(f"  Saving finetuned prediction...")
        if 'prediction_finetuned' in corr_group:
            del corr_group['prediction_finetuned']

        pred_finetuned_group = corr_group.create_group('prediction_finetuned')
        pred_finetuned_s0 = pred_finetuned_group.create_dataset(
            's0',
            data=finetuned_pred,
            dtype='float32',
            compression='gzip',
            compression_opts=6,
            chunks=(56, 56, 56)
        )
        add_ome_ngff_metadata(
            pred_finetuned_group,
            'prediction_finetuned',
            voxel_size,
            translation_offset=translation_offset
        )

        print(f"  ✓ Saved finetuned prediction")

        # Print comparison stats
        if 'prediction/s0' in corr_group:
            original_pred = np.array(corr_group['prediction/s0'])
            diff = np.abs(finetuned_pred - original_pred)
            print(f"  Mean absolute difference: {diff.mean():.6f}")
            print(f"  Max absolute difference: {diff.max():.6f}")

    print("\n" + "="*60)
    print(f"✅ Processed {len(correction_ids)} corrections")
    print(f"\nResults saved to: {args.corrections}")
    print("\nComparison structure:")
    print("  - prediction/s0          : Original base model predictions")
    print("  - prediction_finetuned/s0: LoRA finetuned predictions")
    print("  - mask/s0                : Ground truth (eroded) labels")
    print("\nView in Neuroglancer to compare before/after finetuning!")
    print("="*60)


if __name__ == "__main__":
    main()
