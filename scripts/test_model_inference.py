#!/usr/bin/env python
"""Quick test to verify model inference works."""

import numpy as np
from funlib.geometry import Roi, Coordinate

# Set up globals for normalization
from cellmap_flow.globals import g
from cellmap_flow.norm.input_normalize import MinMaxNormalizer, LambdaNormalizer

g.input_norms = [
    MinMaxNormalizer(min_value=0, max_value=250, invert=False),
    LambdaNormalizer(expression="x*2-1")
]
g.postprocess = []

# Load dataset
from cellmap_flow.image_data_interface import ImageDataInterface

dataset_path = "/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s1"
dataset = ImageDataInterface(dataset_path)

print(f"Dataset shape: {dataset.shape}")
print(f"Voxel size: {dataset.voxel_size}")

# Load model
from cellmap_flow.models.models_config import FlyModelConfig
from cellmap_flow.inferencer import Inferencer

model_config = FlyModelConfig(
    checkpoint_path="/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000",
    channels=["mito"],
    input_voxel_size=(16, 16, 16),
    output_voxel_size=(16, 16, 16)
)

inferencer = Inferencer(model_config, use_half_prediction=False)

# Test on a specific region (center of dataset)
center = dataset.shape * dataset.voxel_size / 2
roi_shape = Coordinate((56, 56, 56)) * dataset.voxel_size
roi_offset = center - roi_shape / 2
roi = Roi(roi_offset, roi_shape)

print(f"\nTesting ROI: {roi}")

# Load raw data
raw = dataset.to_ndarray_ts(roi)
print(f"Raw data shape: {raw.shape}")
print(f"Raw data range: [{raw.min()}, {raw.max()}]")
print(f"Raw data mean: {raw.mean():.2f}")

# Run inference
pred = inferencer.process_chunk(dataset, roi)
print(f"\nPrediction shape: {pred.shape}")
print(f"Prediction dtype: {pred.dtype}")
print(f"Prediction range: [{pred.min()}, {pred.max()}]")
print(f"Prediction mean: {pred.mean():.6f}")

# Check if prediction is non-zero
if pred.max() > 0:
    print(f"\n✓ Model is working! Found {(pred > 0.5).sum()} positive voxels")
else:
    print(f"\n✗ Model produced all zeros - may need different ROI or settings")
