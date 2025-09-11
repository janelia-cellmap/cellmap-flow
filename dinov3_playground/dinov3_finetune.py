# %%
"""
DINOv3 Fine-tuning Pipeline - Main Execution File

This file contains only function calls and imports.
All function definitions have been extracted to separate modules for better organization.

Modules:
- dinov3_core.py: Core DINOv3 processing functions
- data_processing.py: Data sampling and augmentation functions
- models.py: Neural network model classes
- model_training.py: Training and class balancing functions
- visualization.py: Plotting and visualization functions
- memory_efficient_training.py: Memory-efficient training system

Author: GitHub Copilot
Date: 2025-09-11
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from transformers import AutoImageProcessor, DINOv3ViTModel
import pickle
import os
from datetime import datetime

from funlib.geometry import Coordinate, Roi
from cellmap_flow.image_data_interface import ImageDataInterface

# Import all modularized functions
from memory_efficient_training import (
    MemoryEfficientDataLoader,
    train_classifier_memory_efficient,
    load_checkpoint,
    resume_training_from_checkpoint,
    list_checkpoints,
    train_with_memory_efficient_loader,
    load_model_and_run_inference,
    run_inference_on_new_data
)

from .dinov3_core import (
    enable_amp_inference,
    normalize_features,
    apply_normalization_stats,
    process
)

from .data_processing import (
    sample_training_data,
    apply_intensity_augmentation,
    resize_and_crop_to_target,
    crop_or_pad_to_size,
    create_image_level_split_demo
)

from .models import (
    ImprovedClassifier,
    SimpleClassifier,
    create_model,
    get_model_info,
    print_model_summary
)

from .model_training import (
    balance_classes,
    train_classifier_with_validation,
    evaluate_model
)

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curves,
    visualize_pipeline_results,
    plot_feature_distributions,
    create_comprehensive_report
)

# %%
# Configuration parameters
voxel_size = 128
input_voxel_size = Coordinate((voxel_size, voxel_size, voxel_size))
output_voxel_size = Coordinate((voxel_size, voxel_size, voxel_size))
batch_size = 2
image_size = 224
read_shape = Coordinate((batch_size, image_size, image_size)) * Coordinate(input_voxel_size)
write_shape = Coordinate((batch_size, image_size, image_size)) * Coordinate(output_voxel_size)
output_channels = 380
block_shape = np.array((batch_size, image_size, image_size, output_channels))

# DINOv3 Model Configuration
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
TARGET_IMG_SIZE = 896
UPSAMPLE = True

# Initialize model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = DINOv3ViTModel.from_pretrained(MODEL_ID).to(device)
model = enable_amp_inference(model)
patch_size = getattr(model.config, "patch_size", 16)

print(f"Initialized DINOv3 model on device: {device}")
print(f"Model configuration: {MODEL_ID}")
print(f"Patch size: {patch_size}")

# %%
# Load data
print("Loading datasets...")
raw = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.zarr/recon-1/em/fibsem-uint8/s3").to_ndarray_ts()
gt = ImageDataInterface("/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_nuc/s0").to_ndarray_ts() > 0

print(f"Raw data shape: {raw.shape}")
print(f"Ground truth shape: {gt.shape}")

# %%
# Sample training data using flexible method
print("Sampling training data...")
USE_FLEXIBLE_SAMPLING = True

if USE_FLEXIBLE_SAMPLING:
    print("Using flexible sampling from multiple orientations...")
    raw_select, gt_select = sample_training_data(
        raw, gt, 
        target_size=224, 
        num_samples=10, 
        method="flexible",
        seed=42
    )
else:
    print("Using simple sampling method...")
    raw_select, gt_select = sample_training_data(
        raw, gt,
        target_size=224,
        num_samples=10, 
        method="simple",
        seed=42
    )

print(f"Selected data shapes: raw={raw_select.shape}, gt={gt_select.shape}")

# %%
# Process data through DINOv3
print("Processing data through DINOv3...")
output = process(raw_select)
print(f"DINOv3 output shape: {output.shape}")

# %%
# Create visualization of data
print("Creating data visualization...")
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(raw_select[4,], cmap='gray')
plt.title('Raw Image')
plt.subplot(1, 3, 2)
plt.imshow(gt_select[4])
plt.title('Ground Truth')
plt.subplot(1, 3, 3)
plt.imshow(output[0,4])
plt.title('DINOv3 Features')
plt.show()

# %%
# Example 1: Train classifier with standard pipeline
print("=" * 60)
print("EXAMPLE 1: Standard Training Pipeline")
print("=" * 60)

# Prepare features and labels
output_rearranged = np.moveaxis(output, 0, -1)
features = output_rearranged.reshape(-1, output_channels)
labels = gt_select.copy().reshape(-1)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Normalize features
features_normalized, norm_stats = normalize_features(features, method="standardize")
print(f"Features normalized using method: {norm_stats['method']}")

# Convert to tensors
X = torch.tensor(features_normalized, dtype=torch.float32).to(device)
y = torch.tensor(labels, dtype=torch.long).to(device)

# Create image indices for proper train/val splitting
num_images = raw_select.shape[0]
pixels_per_image = raw_select.shape[1] * raw_select.shape[2]
image_indices = torch.repeat_interleave(torch.arange(num_images), pixels_per_image).to(device)

print(f"Created image indices for {num_images} images")

# Train classifier with all improvements
print("Training classifier with image-level validation and regularization...")
training_results = train_classifier_with_validation(
    X=X,
    y=y,
    num_classes=2,
    device=device,
    validation_split=0.2,
    epochs=100,
    learning_rate=1e-3,
    patience=20,
    balance_classes_method="undersample",
    image_indices=image_indices
)

# Plot training history
plot_training_history(training_results)

print(f"Training completed! Best validation accuracy: {training_results['best_val_acc']:.4f}")

# %%
# Example 2: Memory-efficient training for larger datasets
print("=" * 60)
print("EXAMPLE 2: Memory-Efficient Training")
print("=" * 60)

# Train using memory-efficient data loader
print("Training with memory-efficient data loader...")
memory_results = train_with_memory_efficient_loader(
    raw_data=raw, 
    gt_data=gt,
    train_pool_size=50,    # Sample from 50 possible training images
    val_pool_size=10,      # Use 10 fixed validation images  
    images_per_batch=4,    # 4 images per training batch
    batches_per_epoch=10,  # 10 batches per epoch = 40 images/epoch
    num_classes=2,
    epochs=50,
    balance_method="undersample"
)

print("Memory-efficient training completed!")

# %%
# Example 3: Model inference on new data
print("=" * 60)
print("EXAMPLE 3: Model Inference")
print("=" * 60)

# Load different data for testing
print("Loading test data...")
raw_test = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s2").to_ndarray_ts(
    Roi(np.array((670, 922, 4086))*32, 3*[750*32])
)

print(f"Test data shape: {raw_test.shape}")

# List available checkpoints
print("Available checkpoints:")
checkpoints = list_checkpoints("/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/tmp/results/")
for i, checkpoint in enumerate(checkpoints):
    print(f"  {i+1}. {checkpoint}")

# Run inference with the latest checkpoint (if available)
if checkpoints:
    latest_checkpoint = checkpoints[0]  # Most recent
    print(f"Running inference with checkpoint: {latest_checkpoint}")
    
    inference_results = load_model_and_run_inference(
        latest_checkpoint,
        raw_test, 
        gt_data=None, 
        num_sample_images=5
    )
    
    print(f"Inference completed!")
    if 'accuracy' in inference_results:
        print(f"Test accuracy: {inference_results['accuracy']:.4f}")
else:
    print("No checkpoints found. Train a model first.")

# %%
# Example 4: Demonstrate image-level train/val splitting
print("=" * 60)
print("EXAMPLE 4: Image-Level Train/Val Split Demo")
print("=" * 60)

# Show the conceptual difference between splitting methods
create_image_level_split_demo()

# %%
# Example 5: Model comparison and visualization
print("=" * 60)
print("EXAMPLE 5: Model Architecture Comparison")
print("=" * 60)

# Create and compare different model architectures
print("Creating model architectures...")

# Simple model
simple_model = create_model('simple', input_dim=output_channels, num_classes=2)
print_model_summary(simple_model, input_shape=(1, output_channels))

# Improved model with different configurations
improved_model = create_model(
    'improved', 
    input_dim=output_channels, 
    num_classes=2,
    hidden_dims=[256, 128, 64],
    dropout_rate=0.3,
    use_batch_norm=True
)
print_model_summary(improved_model, input_shape=(1, output_channels))

# %%
# Example 6: Feature analysis and visualization
print("=" * 60)
print("EXAMPLE 6: Feature Analysis")
print("=" * 60)

# Analyze feature distributions
print("Analyzing feature distributions...")
plot_feature_distributions(features, labels)

# %%
# Example 7: Comprehensive training with visualization
print("=" * 60)
print("EXAMPLE 7: Comprehensive Training with Full Visualization")
print("=" * 60)

# Sample fresh data for comprehensive demo
demo_raw, demo_gt = sample_training_data(
    raw, gt,
    target_size=224,
    num_samples=8,
    method="flexible",
    seed=777
)

# Process through DINOv3
demo_features = process(demo_raw)

# Create comprehensive training pipeline
print("Running comprehensive training pipeline...")

# Extract and normalize features
demo_features_flat = np.moveaxis(demo_features, 0, -1).reshape(-1, output_channels)
demo_labels_flat = demo_gt.reshape(-1)

demo_features_norm, demo_norm_stats = normalize_features(demo_features_flat, method="standardize")

# Convert to tensors
demo_X = torch.tensor(demo_features_norm, dtype=torch.float32).to(device)
demo_y = torch.tensor(demo_labels_flat, dtype=torch.long).to(device)

# Create image indices
demo_image_indices = torch.repeat_interleave(
    torch.arange(demo_raw.shape[0]), 
    demo_raw.shape[1] * demo_raw.shape[2]
).to(device)

# Train with comprehensive monitoring
demo_results = train_classifier_with_validation(
    X=demo_X,
    y=demo_y,
    num_classes=2,
    device=device,
    validation_split=0.25,
    epochs=100,
    learning_rate=1e-3,
    patience=25,
    balance_classes_method="hybrid",
    image_indices=demo_image_indices
)

# Create comprehensive visualization report
print("Creating comprehensive visualization report...")
evaluation_results = evaluate_model(
    demo_results['classifier'],
    demo_features_norm,
    demo_labels_flat,
    device
)

create_comprehensive_report(
    demo_results['classifier'],
    demo_results,
    evaluation_results,
    save_dir="/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/tmp/visualization_report/"
)

print("=" * 60)
print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("✅ Modular architecture implemented")
print("✅ All functions extracted to separate modules")
print("✅ Main file contains only imports and function calls")
print("✅ Memory-efficient training system available")
print("✅ Comprehensive visualization and analysis tools")
print("✅ Image-level train/val splitting implemented")
print("✅ Multiple model architectures available")
print("✅ Feature normalization and class balancing")
print("=" * 60)

# %%
