#!/usr/bin/env python3
"""
Generate correction zarrs from mito segmentations.
Creates 10 random crops with mito, applies erosion, and saves in corrections format.
Runs fly_organelles_run08_438000 model to generate predictions.
"""

import numpy as np
import zarr
from scipy.ndimage import binary_erosion
import uuid
import torch
from fly_organelles.model import StandardUnet

# Paths
# Using s1 at 16nm resolution
RAW_PATH = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8/s1"
MITO_PATH = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/labels/inference/segmentations/mito/s1"
OUTPUT_PATH = "/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/corrections/mito_liver.zarr"

# Crop sizes (from previous corrections)
RAW_SHAPE = (178, 178, 178)
MASK_SHAPE = (56, 56, 56)
NUM_CROPS = 10
EROSION_ITERATIONS = 5
MIN_MITO_FRACTION = 0.10  # Require at least 10% mito in the center crop

# Model configuration
MODEL_PATH = "/nrs/cellmap/models/saalfeldlab/fly_organelles_run08_438000/model.pt"
MITO_CHANNEL = 2  # Channel index for mito in model output

def load_fly_model(checkpoint_path):
    """Load the fly_organelles model."""
    print(f"Loading fly_organelles model from {checkpoint_path}...")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model directly (model.pt contains the full Sequential model)
    model = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    return model, device


def normalize_input(raw_crop):
    """Normalize uint8 [0, 255] to float32 [-1, 1]."""
    return (raw_crop.astype(np.float32) / 127.5) - 1.0


def run_prediction(model, device, raw_crop, mito_channel=2):
    """Run model inference on a raw crop.

    Args:
        model: The fly_organelles model
        device: torch device
        raw_crop: Raw input (178, 178, 178) uint8
        mito_channel: Channel index for mito output

    Returns:
        Prediction for mito channel (56, 56, 56) as float32 [0, 1]
    """
    # Normalize input to [-1, 1]
    input_normalized = normalize_input(raw_crop)

    # Add batch and channel dimensions: (178, 178, 178) -> (1, 1, 178, 178, 178)
    input_tensor = torch.from_numpy(input_normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)  # (1, 8, 56, 56, 56)

    # Extract mito channel and convert to numpy
    # Keep as float32 [0, 1] for consistency with mask and finetuning
    mito_pred = output[0, mito_channel].cpu().numpy().astype(np.float32)  # (56, 56, 56)

    return mito_pred


def main():
    print("Loading fly_organelles model...")
    model, device = load_fly_model(MODEL_PATH)

    print("\nLoading datasets...")
    # Load using zarr directly
    raw_array = zarr.open(RAW_PATH, 'r')
    mito_array = zarr.open(MITO_PATH, 'r')

    full_shape = np.array(mito_array.shape)

    # Get resolution from parent group multiscales metadata
    # s1 is 16nm resolution
    parent_path = "/nrs/cellmap/data/jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/recon-1/em/fibsem-uint8"
    parent_z = zarr.open(parent_path, 'r')
    if 'multiscales' in parent_z.attrs:
        multiscales = parent_z.attrs['multiscales'][0]
        for dataset in multiscales['datasets']:
            if dataset['path'] == 's1':
                for transform in dataset['coordinateTransformations']:
                    if transform['type'] == 'scale':
                        voxel_size = np.array(transform['scale'])
                        break
    else:
        voxel_size = np.array([16, 16, 16])  # Default for s1

    print(f"Full shape: {full_shape}")
    print(f"Voxel size: {voxel_size}")
    print(f"Raw crop shape: {RAW_SHAPE}")
    print(f"Mask crop shape: {MASK_SHAPE}")

    # Open output zarr
    output_root = zarr.open(OUTPUT_PATH, mode='a')

    crops_created = 0
    attempts = 0
    max_attempts = 1000

    print(f"\nSearching for {NUM_CROPS} crops with mito...")

    # Calculate the offset difference between raw and mask crops
    offset_diff = (np.array(RAW_SHAPE) - np.array(MASK_SHAPE)) // 2

    while crops_created < NUM_CROPS and attempts < max_attempts:
        attempts += 1

        # Sample random position for the MASK (center crop), then calculate raw position around it
        max_mask_offset = full_shape - np.array(MASK_SHAPE)
        if np.any(max_mask_offset < 0):
            print(f"Error: Full shape {full_shape} is smaller than mask shape {MASK_SHAPE}")
            return

        # Sample the mask position first (this is where the prediction will be)
        mask_offset = np.array([
            np.random.randint(0, max_mask_offset[i] + 1) for i in range(3)
        ])

        # Calculate raw position: center the raw crop around the mask
        raw_offset = mask_offset - offset_diff

        # Make sure raw crop is within bounds
        if np.any(raw_offset < 0) or np.any(raw_offset + np.array(RAW_SHAPE) > full_shape):
            continue

        # Calculate slices
        mask_slices = tuple(slice(o, o + s) for o, s in zip(mask_offset, MASK_SHAPE))
        raw_slices = tuple(slice(o, o + s) for o, s in zip(raw_offset, RAW_SHAPE))

        try:
            # Read the FULL mito crop (same size as raw: 178x178x178)
            # This way erosion has full context and no edge artifacts
            mito_full_crop = np.array(mito_array[raw_slices])

            # Check if there's any mito in the CENTER region first
            center_slices_local = tuple(slice(o, o + s) for o, s in zip(offset_diff, MASK_SHAPE))
            mito_center_pre_erosion = mito_full_crop[center_slices_local]

            if not np.any(mito_center_pre_erosion > 0):
                continue

            # Check mito fraction in center BEFORE erosion
            pre_erosion_fraction = np.sum(mito_center_pre_erosion > 0) / mito_center_pre_erosion.size
            if pre_erosion_fraction < MIN_MITO_FRACTION:
                if attempts % 100 == 0:
                    print(f"  Attempt {attempts}: Mito fraction {pre_erosion_fraction:.1%} < {MIN_MITO_FRACTION:.1%}, skipping...")
                continue

            # Apply erosion to the FULL crop (no edge artifacts)
            if EROSION_ITERATIONS > 0:
                mito_binary = mito_full_crop > 0
                eroded_full = binary_erosion(
                    mito_binary,
                    iterations=EROSION_ITERATIONS,
                    structure=np.ones((3, 3, 3))
                )
                mito_full_crop = eroded_full.astype(mito_full_crop.dtype)

            # Now extract the CENTER after erosion
            mask_crop = mito_full_crop[center_slices_local]

            # Check if there's still any mito in center after erosion
            post_erosion_fraction = np.sum(mask_crop > 0) / mask_crop.size
            if post_erosion_fraction == 0:
                if attempts % 100 == 0:
                    print(f"  Attempt {attempts}: Found mito but eroded to nothing, trying again...")
                continue

            # Also check that we still have enough mito after erosion
            if post_erosion_fraction < MIN_MITO_FRACTION * 0.5:  # Allow some reduction from erosion
                if attempts % 100 == 0:
                    print(f"  Attempt {attempts}: After erosion, mito fraction {post_erosion_fraction:.1%} too low, trying again...")
                continue

            # Read the raw crop
            raw_crop = np.array(raw_array[raw_slices])

            # Run prediction through fly model
            print(f"  Running prediction for crop {crops_created + 1}...")
            pred_crop = run_prediction(model, device, raw_crop, mito_channel=MITO_CHANNEL)

            # Create a unique ID for this crop
            crop_id = str(uuid.uuid4())

            # Create group for this crop
            crop_group = output_root.create_group(crop_id, overwrite=False)

            # Helper function to add OME-NGFF metadata
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
                    'scale': voxel_size.tolist()
                })

                # Then add translation in physical units (nm) if provided
                # Translation is applied AFTER scale in the coordinate space
                if translation_offset is not None:
                    # Convert voxel offset to physical coordinates (nm)
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

            # Save raw (with scale structure and OME-NGFF metadata, no translation)
            raw_group = crop_group.create_group('raw')
            raw_s0 = raw_group.create_dataset(
                's0',
                data=raw_crop,
                dtype=raw_crop.dtype,
                compression='gzip',
                compression_opts=6,
                chunks=(64, 64, 64)
            )
            add_ome_ngff_metadata(raw_group, 'raw', voxel_size, translation_offset=None)

            # Save mask (with translation offset to center it in raw)
            # Mask is at offset [61, 61, 61] within the raw crop
            mask_group = crop_group.create_group('mask')
            mask_s0 = mask_group.create_dataset(
                's0',
                data=mask_crop,
                dtype=mask_crop.dtype,
                compression='gzip',
                compression_opts=6,
                chunks=(56, 56, 56)
            )
            add_ome_ngff_metadata(mask_group, 'mask', voxel_size, translation_offset=offset_diff.tolist())

            # Save prediction from model (same size and offset as mask)
            # Keep as float32 [0, 1] for consistency with mask and finetuning
            pred_group = crop_group.create_group('prediction')
            pred_s0 = pred_group.create_dataset(
                's0',
                data=pred_crop,
                dtype='float32',
                compression='gzip',
                compression_opts=6,
                chunks=(56, 56, 56)
            )
            add_ome_ngff_metadata(pred_group, 'prediction', voxel_size, translation_offset=offset_diff.tolist())

            crops_created += 1
            mito_voxels = np.sum(mask_crop > 0)
            mito_fraction = post_erosion_fraction * 100
            print(f"✓ Crop {crops_created}/{NUM_CROPS} created (ID: {crop_id}, mito: {mito_voxels:,} voxels ({mito_fraction:.1f}%), attempts: {attempts})")

        except Exception as e:
            print(f"  Error at attempt {attempts}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if crops_created < NUM_CROPS:
        print(f"\nWarning: Only created {crops_created}/{NUM_CROPS} crops after {max_attempts} attempts")
    else:
        print(f"\n✓ Successfully created all {NUM_CROPS} crops!")

    print(f"\nCorrections saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
