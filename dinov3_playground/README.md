# DINOv3 Playground

This directory contains a modular implementation of DINOv3 fine-tuning for image classification tasks.

## Structure

```
dinov3_playground/
├── __init__.py                 # Package initialization with key imports
├── dinov3_finetune.py         # Main execution file with examples
├── dinov3_core.py             # Core DINOv3 processing functions
├── data_processing.py         # Data sampling and augmentation
├── models.py                  # Neural network model classes
├── model_training.py          # Training and class balancing
├── visualization.py           # Plotting and visualization
└── README.md                  # This file
```

## Key Features

### 🧠 **Core DINOv3 Processing** (`dinov3_core.py`)
- `enable_amp_inference()`: Enable automatic mixed precision for faster inference
- `process()`: Process image batches through DINOv3 model
- `normalize_features()`: Feature normalization with multiple methods
- `apply_normalization_stats()`: Apply pre-computed normalization

### 📊 **Data Processing** (`data_processing.py`)
- `sample_training_data()`: Flexible sampling from 3D volumes (XY, XZ, YZ planes)
- `apply_intensity_augmentation()`: Intensity-based data augmentation
- `resize_and_crop_to_target()`: Smart resizing with augmentation
- `create_image_level_split_demo()`: Demonstrate proper train/val splitting

### 🏗️ **Models** (`models.py`)
- `ImprovedClassifier`: Enhanced neural network with batch norm and dropout
- `SimpleClassifier`: Basic classifier for comparison
- `create_model()`: Factory function for model creation
- `print_model_summary()`: Detailed model architecture analysis

### 🎯 **Training** (`model_training.py`)
- `train_classifier_with_validation()`: Complete training pipeline with early stopping
- `balance_classes()`: Multiple class balancing strategies
- `evaluate_model()`: Comprehensive model evaluation

### 📈 **Visualization** (`visualization.py`)
- `plot_training_history()`: Training/validation curves
- `plot_confusion_matrix()`: Classification performance
- `plot_roc_curves()`: ROC analysis for multi-class
- `visualize_pipeline_results()`: End-to-end pipeline visualization
- `create_comprehensive_report()`: Full analysis report

## Usage

### Basic Usage
```python
from dinov3_playground import (
    process, sample_training_data, 
    ImprovedClassifier, train_classifier_with_validation,
    plot_training_history
)

# Sample data
raw_images, gt_labels = sample_training_data(raw_volume, gt_volume, method="flexible")

# Process through DINOv3
features = process(raw_images)

# Train classifier
results = train_classifier_with_validation(features, gt_labels, ...)

# Visualize results
plot_training_history(results)
```

### Running Examples
```bash
cd dinov3_playground
python dinov3_finetune.py
```

The main file contains 7 comprehensive examples:
1. **Standard Training Pipeline**: Basic classifier training with validation
2. **Memory-Efficient Training**: For large datasets that don't fit in memory
3. **Model Inference**: Loading checkpoints and running inference
4. **Train/Val Split Demo**: Proper image-level splitting explanation
5. **Model Architecture Comparison**: Different model configurations
6. **Feature Analysis**: Understanding feature distributions
7. **Comprehensive Training**: Full pipeline with visualization

## Key Improvements

### ✅ **Modular Architecture**
- Clean separation of concerns
- Easy to test and maintain
- Reusable components

### ✅ **Memory Efficiency**
- On-demand data loading
- Batch processing
- Gradient checkpointing support

### ✅ **Proper Validation**
- Image-level train/val splitting
- Prevents data leakage
- Multiple class balancing methods

### ✅ **Comprehensive Monitoring**
- Training/validation curves
- Early stopping
- Learning rate scheduling
- Gradient clipping

### ✅ **Flexible Data Handling**
- Multi-orientation sampling (XY, XZ, YZ)
- Data augmentation
- Multiple normalization methods

### ✅ **Production Ready**
- Checkpoint saving/loading
- Model serialization
- Inference pipeline
- Error handling

## Dependencies

- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- transformers (for DINOv3)
- PIL/Pillow
- scipy
- seaborn

## Author

Created by GitHub Copilot on 2025-09-11 as part of the cellmap-flow project.
