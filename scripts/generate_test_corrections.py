#!/usr/bin/env python
"""
Generate synthetic test corrections for HITL finetuning.

This script:
1. Loads a dataset and model from an existing config
2. Runs inference on random ROIs to get predictions
3. Creates synthetic "corrected" masks by applying transformations
4. Saves corrections in Zarr format: corrections.zarr/<uuid>/{raw, mask, prediction}/s0

Usage:
    python scripts/generate_test_corrections.py \
        --config jrc_mus-salivary-1_mito.yaml \
        --num-corrections 10 \
        --output corrections.zarr
"""

import argparse
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
import zarr
from scipy import ndimage
from funlib.geometry import Coordinate, Roi

# Import cellmap-flow utilities
from cellmap_flow.utils.config_utils import build_models
from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.inferencer import Inferencer


def create_synthetic_correction(
    prediction: np.ndarray,
    correction_type: str = "threshold"
) -> np.ndarray:
    """
    Create a synthetic correction from a prediction.

    Simulates different types of manual corrections:
    - threshold: Apply different threshold
    - erosion: Erode prediction
    - dilation: Dilate prediction
    - fill_holes: Fill small holes
    - remove_small: Remove small objects

    Args:
        prediction: Model prediction (0-255 uint8)
        correction_type: Type of correction to apply

    Returns:
        Corrected mask (0-255 uint8)
    """
    # Convert to binary
    binary_pred = prediction > 127

    if correction_type == "threshold_low":
        # Lower threshold (more permissive)
        corrected = prediction > 80
    elif correction_type == "threshold_high":
        # Higher threshold (more strict)
        corrected = prediction > 180
    elif correction_type == "erosion":
        # Erode to remove small noise
        corrected = ndimage.binary_erosion(binary_pred, iterations=2)
    elif correction_type == "dilation":
        # Dilate to fill gaps
        corrected = ndimage.binary_dilation(binary_pred, iterations=2)
    elif correction_type == "fill_holes":
        # Fill holes in objects
        corrected = ndimage.binary_fill_holes(binary_pred)
    elif correction_type == "remove_small":
        # Remove small objects
        labeled, num_features = ndimage.label(binary_pred)
        sizes = ndimage.sum(binary_pred, labeled, range(num_features + 1))
        mask_size = sizes < 100  # Remove objects smaller than 100 voxels
        remove_pixel = mask_size[labeled]
        corrected = binary_pred.copy()
        corrected[remove_pixel] = 0
    elif correction_type == "open":
        # Morphological opening (erosion then dilation)
        corrected = ndimage.binary_opening(binary_pred, iterations=1)
    elif correction_type == "close":
        # Morphological closing (dilation then erosion)
        corrected = ndimage.binary_closing(binary_pred, iterations=1)
    else:
        # Default: just use prediction as-is
        corrected = binary_pred

    # Convert back to uint8
    return (corrected * 255).astype(np.uint8)


def generate_random_roi(
    data_shape: Coordinate,
    voxel_size: Coordinate,
    roi_shape_voxels: Tuple[int, int, int] = (128, 128, 128),
    prefer_center: bool = True
) -> Roi:
    """
    Generate a random ROI within the dataset bounds.

    Args:
        data_shape: Shape of dataset in voxels
        voxel_size: Voxel size in physical units
        roi_shape_voxels: Desired ROI shape in voxels
        prefer_center: If True, bias towards center of dataset

    Returns:
        Random ROI
    """
    roi_shape = Coordinate(roi_shape_voxels) * voxel_size

    if prefer_center:
        # Generate offset with Gaussian distribution around center
        center = data_shape * voxel_size / 2
        # Standard deviation is 1/4 of dataset size (covers most of dataset)
        std = data_shape * voxel_size / 4

        random_offset = Coordinate(
            max(0, min(
                int(data_shape[i] * voxel_size[i] - roi_shape[i]),
                int(np.random.normal(center[i], std[i]))
            ))
            for i in range(3)
        )
        # Align to voxel grid
        random_offset = Coordinate(
            (random_offset[i] // voxel_size[i]) * voxel_size[i]
            for i in range(3)
        )
    else:
        # Uniform random offset
        max_offset = data_shape * voxel_size - roi_shape
        random_offset = Coordinate(
            np.random.randint(0, int(max_offset[i] / voxel_size[i])) * voxel_size[i]
            for i in range(3)
        )

    return Roi(random_offset, roi_shape)


def save_correction_to_zarr(
    correction_id: str,
    raw_data: np.ndarray,
    prediction: np.ndarray,
    corrected_mask: np.ndarray,
    roi: Roi,
    voxel_size: Coordinate,
    output_path: Path,
    model_name: str,
    dataset_path: str
):
    """
    Save a correction to Zarr format.

    Structure:
        corrections.zarr/
        └── <correction_id>/
            ├── raw/s0/          # Original raw data
            ├── prediction/s0/   # Model prediction
            ├── mask/s0/         # Corrected mask
            └── .zattrs          # Metadata (ROI, model, dataset)
    """
    correction_group = zarr.open_group(str(output_path), mode='a')
    corr_group = correction_group.require_group(correction_id)

    # Save arrays with OME-NGFF compatible structure
    # Structure: raw/s0 (not raw/s0/data)
    corr_group.array('raw/s0', raw_data, chunks=(64, 64, 64), dtype=np.uint8, overwrite=True)
    corr_group.array('prediction/s0', prediction, chunks=(64, 64, 64), dtype=np.uint8, overwrite=True)
    corr_group.array('mask/s0', corrected_mask, chunks=(64, 64, 64), dtype=np.uint8, overwrite=True)

    # Add OME-NGFF metadata for Neuroglancer compatibility
    for name in ['raw', 'prediction', 'mask']:
        group = corr_group.require_group(name)
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
                'coordinateTransformations': [{
                    'type': 'scale',
                    'scale': list(voxel_size)
                }]
            }]
        }]

    # Save metadata
    corr_group.attrs['correction_id'] = correction_id
    corr_group.attrs['model_name'] = model_name
    corr_group.attrs['dataset_path'] = dataset_path
    corr_group.attrs['roi_offset'] = list(roi.offset)
    corr_group.attrs['roi_shape'] = list(roi.shape)
    corr_group.attrs['voxel_size'] = list(voxel_size)

    print(f"✓ Saved correction {correction_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test corrections for HITL finetuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="jrc_mus-salivary-1_mito.yaml",
        help="Path to pipeline config YAML"
    )
    parser.add_argument(
        "--num-corrections",
        type=int,
        default=10,
        help="Number of corrections to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="corrections.zarr",
        help="Output Zarr path for corrections"
    )
    parser.add_argument(
        "--roi-shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="ROI shape in voxels (Z Y X)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s1",
        help="Path to dataset"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000",
        help="Path to model checkpoint"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating {args.num_corrections} test corrections")
    print(f"{'='*60}\n")

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = ImageDataInterface(args.dataset_path)
    voxel_size = dataset.voxel_size
    data_shape = dataset.shape
    print(f"  Shape: {data_shape}")
    print(f"  Voxel size: {voxel_size}")

    # Set up globals for normalization (MUST be done before loading dataset)
    from cellmap_flow.globals import g
    from cellmap_flow.norm.input_normalize import MinMaxNormalizer, LambdaNormalizer

    # Apply same normalization as in the YAML config
    g.input_norms = [
        MinMaxNormalizer(min_value=0, max_value=250, invert=False),
        LambdaNormalizer(expression="x*2-1")
    ]
    g.postprocess = []  # No postprocessing for now
    print(f"  Normalization set up: {len(g.input_norms)} normalizers")

    # Reload dataset to pick up normalization
    dataset = ImageDataInterface(args.dataset_path)

    # Load model using cellmap-flow
    print(f"\nLoading model from: {args.model_checkpoint}")
    from cellmap_flow.models.models_config import FlyModelConfig

    model_config = FlyModelConfig(
        checkpoint_path=args.model_checkpoint,
        channels=["mito"],
        input_voxel_size=(16, 16, 16),
        output_voxel_size=(16, 16, 16)
    )

    # Create inferencer
    inferencer = Inferencer(model_config, use_half_prediction=False)
    print(f"  Model loaded successfully")

    # Correction types to cycle through
    correction_types = [
        "threshold_low",
        "threshold_high",
        "erosion",
        "dilation",
        "fill_holes",
        "remove_small",
        "open",
        "close"
    ]

    # Generate corrections
    print(f"\nGenerating corrections...\n")
    for i in range(args.num_corrections):
        correction_id = str(uuid.uuid4())
        correction_type = correction_types[i % len(correction_types)]

        # Generate random ROI
        roi = generate_random_roi(data_shape, voxel_size, tuple(args.roi_shape))

        print(f"[{i+1}/{args.num_corrections}] Correction {correction_id[:8]}...")
        print(f"  ROI: offset={roi.offset}, shape={roi.shape}")
        print(f"  Type: {correction_type}")

        # Get context from inferencer
        context = inferencer.context

        # Create expanded ROI for reading (includes context)
        read_roi = roi.grow(context, context)

        # Load raw data at FULL INPUT SIZE (for training) WITHOUT normalization
        # This is the data the model needs as input
        original_norms = g.input_norms
        g.input_norms = []
        raw_data_full = dataset.to_ndarray_ts(read_roi)  # Full input size
        raw_data_write = dataset.to_ndarray_ts(roi)       # Output size (for reference)
        g.input_norms = original_norms

        # Ensure uint8
        if raw_data_full.dtype != np.uint8:
            raw_data_full = raw_data_full.astype(np.uint8)

        print(f"  Context: {context}")
        print(f"  Read ROI: {read_roi.get_shape() / dataset.voxel_size}")
        print(f"  Write ROI: {roi.get_shape() / dataset.voxel_size}")

        # Run inference
        # process_chunk handles context internally
        try:
            prediction = inferencer.process_chunk(
                idi=dataset,
                roi=roi
            )
        except Exception as e:
            print(f"  Error during inference: {e}")
            print(f"  Skipping this correction...")
            continue

        print(f"  Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")
        print(f"  Prediction range: [{prediction.min()}, {prediction.max()}]")

        # Convert prediction to uint8 if needed
        if prediction.dtype != np.uint8:
            if prediction.max() <= 1.0:
                prediction = (prediction * 255).astype(np.uint8)
            else:
                prediction = prediction.astype(np.uint8)

        # Handle multi-channel predictions (take first channel if needed)
        if prediction.ndim == 4:
            prediction = prediction[0]

        # Generate synthetic correction
        corrected_mask = create_synthetic_correction(prediction, correction_type)

        # Save to Zarr
        # Note: Save raw at FULL input size, prediction/mask at output size
        save_correction_to_zarr(
            correction_id=correction_id,
            raw_data=raw_data_full,  # Full input size for training
            prediction=prediction,    # Output size
            corrected_mask=corrected_mask,  # Output size
            roi=read_roi,  # Use read_roi for metadata (full size)
            voxel_size=voxel_size,
            output_path=output_path,
            model_name="fly_organelles_mito",
            dataset_path=args.dataset_path
        )

        print()

    print(f"\n{'='*60}")
    print(f"✓ Generated {args.num_corrections} corrections")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
