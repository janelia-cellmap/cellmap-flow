"""
DINOv3 Playground Package

This package contains all the modularized functions for DINOv3 fine-tuning and analysis.

Modules:
- dinov3_core: Core DINOv3 processing functions
- data_processing: Data sampling and augmentation functions  
- models: Neural network model classes
- model_training: Training and class balancing functions
- visualization: Plotting and visualization functions

Author: GitHub Copilot
Date: 2025-09-11
"""

# Import key functions for easy access
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
    create_image_level_split_demo
)

from .models import (
    ImprovedClassifier,
    SimpleClassifier,
    create_model,
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
    create_comprehensive_report
)

__version__ = "1.0.0"
__author__ = "GitHub Copilot"
__description__ = "DINOv3 fine-tuning playground with modular architecture"
