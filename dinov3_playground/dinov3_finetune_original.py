# %%
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

from dinov3_core import (
    enable_amp_inference,
    normalize_features,
    apply_normalization_stats,
    process
)

from data_processing import (
    sample_training_data,
    apply_intensity_augmentation,
    resize_and_crop_to_target,
    crop_or_pad_to_size,
    create_image_level_split_demo
)

from models import (
    ImprovedClassifier,
    SimpleClassifier,
    create_model,
    get_model_info,
    print_model_summary
)

from model_training import (
    balance_classes,
    train_classifier_with_validation,
    evaluate_model
)

from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curves,
    visualize_pipeline_results,
    plot_feature_distributions,
    create_comprehensive_report
)


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


# Load data
raw = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3mb/jrc_22ak351-leaf-3mb.zarr/recon-1/em/fibsem-uint8/s3").to_ndarray_ts()
gt = ImageDataInterface("/groups/cellmap/cellmap/parkg/for Aubrey/3mb_s3.zarr/jrc_22ak351-leaf-3mb_nuc/s0").to_ndarray_ts() > 0


# %%
# Sample training data using flexible method
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
output = process(raw_select)

# %%
# create plot with 3 axes side by side
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(raw_select[4,], cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(gt_select[4])
plt.subplot(1, 3, 3)
plt.imshow(output[0,4])

# %%
# Simple neural network for feature classification
import torch.nn as nn
import torch.optim as optim

# Prepare data for classifier training
num_classes = 3
features = []
labels = []

# Flatten output to (image_size*image_size, output_channels)
output_rearranged = np.moveaxis(output, 0, -1)
features = output_rearranged.reshape(-1, output_channels)

labels = gt_select.copy()
labels = labels.reshape(-1)
    """
    Balance class distribution in the dataset.
    
    Parameters:
    -----------
    X : torch.Tensor
        Feature tensor of shape (n_samples, n_features)
    y : torch.Tensor
        Label tensor of shape (n_samples,)
    method : str, default="undersample"
        Balancing method:
        - "undersample": Reduce majority classes to minority class size
        - "oversample": Increase minority classes to majority class size (with repetition)
        - "weighted": Return class weights for weighted loss (no resampling)
        - "hybrid": Combination of under/oversampling to target size
    max_samples_per_class : int, optional
        Target number of samples per class for hybrid method
        
    Returns:
    --------
    tuple: (balanced_X, balanced_y) or (X, y, class_weights) for weighted method
    """
    
    # Get class distribution
    unique_classes, class_counts = torch.unique(y, return_counts=True)
    print(f"Original class distribution: {dict(zip(unique_classes.cpu().numpy(), class_counts.cpu().numpy()))}")
    
    if method == "weighted":
        # Calculate inverse frequency weights
        total_samples = len(y)
        class_weights = total_samples / (len(unique_classes) * class_counts.float())
        weight_dict = dict(zip(unique_classes.cpu().numpy(), class_weights.cpu().numpy()))
        print(f"Class weights: {weight_dict}")
        return X, y, class_weights
    
    # For resampling methods
    balanced_X_list = []
    balanced_y_list = []
    
    if method == "undersample":
        # Undersample to the size of the minority class
        min_count = torch.min(class_counts).item()
        target_count = min_count
        
    elif method == "oversample":
        # Oversample to the size of the majority class
        max_count = torch.max(class_counts).item()
        target_count = max_count
        
    elif method == "hybrid":
        # Use specified target or median class size
        if max_samples_per_class is None:
            target_count = torch.median(class_counts).item()
        else:
            target_count = max_samples_per_class
    
    print(f"Target samples per class: {target_count}")
    
    for class_id in unique_classes:
        # Get indices for this class
        class_mask = (y == class_id)
        class_indices = torch.where(class_mask)[0]
        class_X = X[class_indices]
        class_y = y[class_indices]
        
        current_count = len(class_indices)
        
        if current_count == target_count:
            # Already balanced
            balanced_X_list.append(class_X)
            balanced_y_list.append(class_y)
            
        elif current_count > target_count:
            # Undersample this class
            selected_indices = torch.randperm(current_count)[:target_count]
            balanced_X_list.append(class_X[selected_indices])
            balanced_y_list.append(class_y[selected_indices])
            
        else:  # current_count < target_count
            # Oversample this class
            repeats_needed = target_count // current_count
            remainder = target_count % current_count
            
            # Repeat the whole class
            repeated_X = class_X.repeat(repeats_needed, 1)
            repeated_y = class_y.repeat(repeats_needed)
            
            # Add random samples for the remainder
            if remainder > 0:
                random_indices = torch.randperm(current_count)[:remainder]
                extra_X = class_X[random_indices]
                extra_y = class_y[random_indices]
                repeated_X = torch.cat([repeated_X, extra_X], dim=0)
                repeated_y = torch.cat([repeated_y, extra_y], dim=0)
            
            balanced_X_list.append(repeated_X)
            balanced_y_list.append(repeated_y)
    
    # Combine all classes
    balanced_X = torch.cat(balanced_X_list, dim=0)
    balanced_y = torch.cat(balanced_y_list, dim=0)
    
    # Shuffle the balanced dataset
    shuffle_indices = torch.randperm(len(balanced_X))
    balanced_X = balanced_X[shuffle_indices]
    balanced_y = balanced_y[shuffle_indices]
    
    # Print new distribution
    new_unique, new_counts = torch.unique(balanced_y, return_counts=True)
    print(f"Balanced class distribution: {dict(zip(new_unique.cpu().numpy(), new_counts.cpu().numpy()))}")
    
    return balanced_X, balanced_y


def create_image_level_split_demo():
    """
    Demonstrate the difference between image-level and pixel-level splitting.
    """
    print("=" * 60)
    print("TRAIN/VALIDATION SPLITTING COMPARISON")
    print("=" * 60)
    
    # Create a simple example with 3 images, 4 pixels each
    n_images = 3
    pixels_per_image = 4
    total_pixels = n_images * pixels_per_image
    
    # Create mock data
    features = torch.randn(total_pixels, 5)  # 5 feature dimensions
    labels = torch.randint(0, 2, (total_pixels,))  # Binary labels
    image_indices = torch.repeat_interleave(torch.arange(n_images), pixels_per_image)
    
    print(f"Example data: {n_images} images, {pixels_per_image} pixels each")
    print(f"Image indices for each pixel: {image_indices.tolist()}")
    print(f"Example labels: {labels.tolist()}")
    
    # Show pixel-level split (old way)
    print(f"\n{'OLD WAY - Pixel-level split:':<40}")
    perm = torch.randperm(total_pixels)
    val_size = int(0.4 * total_pixels)  # 40% validation
    val_pixels_old = perm[:val_size]
    train_pixels_old = perm[val_size:]
    
    print(f"Validation pixels: {val_pixels_old.sort()[0].tolist()}")
    print(f"Training pixels:   {train_pixels_old.sort()[0].tolist()}")
    
    # Show which images these pixels come from
    val_images_old = image_indices[val_pixels_old].unique()
    train_images_old = image_indices[train_pixels_old].unique()
    print(f"Images in validation: {val_images_old.tolist()}")
    print(f"Images in training:   {train_images_old.tolist()}")
    
    # Check for overlap (data leakage)
    overlap = set(val_images_old.tolist()).intersection(set(train_images_old.tolist()))
    if overlap:
        print(f"⚠️  DATA LEAKAGE: Images {list(overlap)} appear in BOTH train and val!")
    
    # Show image-level split (new way)
    print(f"\n{'NEW WAY - Image-level split:':<40}")
    unique_images = torch.unique(image_indices)
    shuffled_images = unique_images[torch.randperm(len(unique_images))]
    num_val_images = max(1, int(len(unique_images) * 0.4))
    
    val_images_new = shuffled_images[:num_val_images]
    train_images_new = shuffled_images[num_val_images:]
    
    print(f"Validation images: {val_images_new.tolist()}")
    print(f"Training images:   {train_images_new.tolist()}")
    
    # Get pixels for each split
    val_mask = torch.isin(image_indices, val_images_new)
    train_mask = torch.isin(image_indices, train_images_new)
    
    val_pixels_new = torch.where(val_mask)[0]
    train_pixels_new = torch.where(train_mask)[0]
    
    print(f"Validation pixels: {val_pixels_new.tolist()}")
    print(f"Training pixels:   {train_pixels_new.tolist()}")
    
    # Check for overlap
    overlap_new = set(val_images_new.tolist()).intersection(set(train_images_new.tolist()))
    if overlap_new:
        print(f"⚠️  DATA LEAKAGE: Images {list(overlap_new)} appear in BOTH train and val!")
    else:
        print(f"✅ NO DATA LEAKAGE: Complete separation of images")
    
    print(f"\nKey Benefits of Image-Level Split:")
    print(f"• Prevents data leakage between train/val sets")
    print(f"• More realistic validation performance")
    print(f"• Better generalization assessment")
    print(f"• Proper evaluation of model's ability to handle new images")


def train_classifier_with_validation(X, y, num_classes, device, 
                                   validation_split=0.2, epochs=500, 
                                   learning_rate=1e-3, weight_decay=1e-4,
                                   patience=50, min_delta=0.001,
                                   use_improved_classifier=True,
                                   batch_size=1024,
                                   balance_classes_method="undersample",
                                   image_indices=None):
    """
    Train classifier with proper train/validation split and early stopping.
    
    Key Feature: Image-level train/validation splitting to prevent data leakage.
    The train/val split is done at the IMAGE level first, then class balancing 
    is applied only to the training set. This ensures no pixels from the same 
    image appear in both training and validation sets.
    
    Parameters:
    -----------
    X : torch.Tensor
        Feature tensor of shape (n_samples, n_features)
    y : torch.Tensor
        Label tensor of shape (n_samples,)
    num_classes : int
        Number of classes
    device : torch.device
        Device to train on
    validation_split : float, default=0.2
        Fraction of data to use for validation
    epochs : int, default=500
        Maximum number of epochs
    learning_rate : float, default=1e-3
        Learning rate
    weight_decay : float, default=1e-4
        L2 regularization strength
    patience : int, default=50
        Early stopping patience (epochs without improvement)
    min_delta : float, default=0.001
        Minimum improvement to reset patience
    use_improved_classifier : bool, default=True
        Whether to use ImprovedClassifier or SimpleClassifier
    batch_size : int, default=1024
        Batch size for training (helps with large datasets)
    balance_classes_method : str, default="undersample"
        Method to balance classes: "undersample", "oversample", "weighted", "hybrid", or "none"
    image_indices : torch.Tensor, optional
        Tensor indicating which image each pixel belongs to (for proper train/val splitting).
        If provided, train/val split will be done at image level instead of pixel level.
        Shape: (n_samples,) with values from 0 to num_images-1
        
    Returns:
    --------
    dict containing:
        - 'classifier': trained model
        - 'train_losses': list of training losses
        - 'val_losses': list of validation losses
        - 'train_accs': list of training accuracies
        - 'val_accs': list of validation accuracies
        - 'best_val_acc': best validation accuracy achieved
        - 'epochs_trained': number of epochs actually trained
    """
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Create train/validation split FIRST (before class balancing)
    if image_indices is not None:
        # Image-level splitting: split based on images, not individual pixels
        unique_images = torch.unique(image_indices)
        num_images = len(unique_images)
        
        # Shuffle image indices for random split
        shuffled_image_indices = unique_images[torch.randperm(num_images)]
        
        # Split images into train/val
        num_val_images = max(1, int(num_images * validation_split))
        val_image_indices = shuffled_image_indices[:num_val_images]
        train_image_indices = shuffled_image_indices[num_val_images:]
        
        # Create pixel-level masks based on image assignment
        val_mask = torch.isin(image_indices, val_image_indices)
        train_mask = torch.isin(image_indices, train_image_indices)
        
        # Split data
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        
        print(f"Image-level split: {len(train_image_indices)} train images, {len(val_image_indices)} val images")
        print(f"Train images: {train_image_indices.tolist()}")
        print(f"Val images: {val_image_indices.tolist()}")
        
    else:
        # Original pixel-level splitting (fallback for backward compatibility)
        print("Warning: Using pixel-level split. For proper validation, provide image_indices.")
        indices = torch.randperm(n_samples)
        val_size = int(n_samples * validation_split)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_train, X_val = X[train_indices], X[val_indices] 
        y_train, y_val = y[train_indices], y[val_indices]
    
    # Balance classes if requested (applied to training set only)
    class_weights = None
    if balance_classes_method != "none":
        print(f"Applying class balancing method '{balance_classes_method}' to training set only")
        
        if balance_classes_method == "weighted":
            # For weighted method, we compute weights from full training set but don't resample
            _, _, class_weights = balance_classes(X_train, y_train, method="weighted")
        else:
            # For resampling methods, we resample the training set
            X_train, y_train = balance_classes(X_train, y_train, method=balance_classes_method)
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Training set class distribution: {torch.bincount(y_train)}")
    print(f"Validation set class distribution: {torch.bincount(y_val)}")
    
    # Create model
    if use_improved_classifier:
        classifier = ImprovedClassifier(n_features, num_classes).to(device)
        print("Using ImprovedClassifier with regularization")
    else:
        classifier = SimpleClassifier(n_features, num_classes).to(device)
        print("Using SimpleClassifier")
    
    # Loss and optimizer with weight decay (L2 regularization)
    if class_weights is not None:
        # Use weighted loss for class imbalance
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using weighted CrossEntropyLoss for class balancing")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20
    )
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    print(f"Starting training for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Batch processing for memory efficiency
        n_train_batches = (len(X_train) + batch_size - 1) // batch_size
        
        for i in range(n_train_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            logits = classifier(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * len(X_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total += len(X_batch)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            n_val_batches = (len(X_val) + batch_size - 1) // batch_size
            
            for i in range(n_val_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_val))
                
                X_batch = X_val[start_idx:end_idx]
                y_batch = y_val[start_idx:end_idx]
                
                logits = classifier(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * len(X_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                val_total += len(X_batch)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Check if learning rate was reduced
        if new_lr < old_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Early stopping check
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"  Best Val Acc: {best_val_acc:.4f}, LR: {current_lr:.6f}")
            print(f"  Epochs without improvement: {epochs_without_improvement}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    return {
        'classifier': classifier,
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(train_losses)
    }


def plot_training_history(training_results, figsize=(15, 5)):
    """
    Plot training history to visualize overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(training_results['train_losses']) + 1)
    
    # Plot losses
    axes[0].plot(epochs, training_results['train_losses'], label='Training Loss', color='blue')
    axes[0].plot(epochs, training_results['val_losses'], label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axes[1].plot(epochs, training_results['train_accs'], label='Training Accuracy', color='blue')
    axes[1].plot(epochs, training_results['val_accs'], label='Validation Accuracy', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    print(f"Final Training Accuracy: {training_results['train_accs'][-1]:.4f}")
    print(f"Final Validation Accuracy: {training_results['val_accs'][-1]:.4f}")
    print(f"Best Validation Accuracy: {training_results['best_val_acc']:.4f}")
    print(f"Total epochs trained: {training_results['epochs_trained']}")
    
    # Check for overfitting
    final_gap = training_results['train_accs'][-1] - training_results['val_accs'][-1]
    if final_gap > 0.1:
        print(f"⚠️  Potential overfitting detected! Training-Validation accuracy gap: {final_gap:.4f}")
    else:
        print(f"✅ Good generalization! Training-Validation accuracy gap: {final_gap:.4f}")


# net = SimpleClassifier(output_channels, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=1e-3)

# # Prepare data for training
# X = torch.tensor(features, dtype=torch.float32).to(device)
# y = torch.tensor(labels, dtype=torch.long).to(device)

# # Training loop (very simple, for demonstration)
# epochs = 1500
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     logits = net(X)
#     loss = criterion(logits, y)
#     loss.backward()
#     optimizer.step()
#     if (epoch+1) % 2 == 0:
#         pred = logits.argmax(dim=1)
#         acc = (pred == y).float().mean().item()
#         print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Acc: {acc:.4f}")

# # run the network
# with torch.no_grad():
#     logits = net(X)
#     pred = logits.argmax(dim=1).to("cpu").numpy()

# pred = pred.reshape((-1,224,224))


# # %%
# # create plot with 3 axes side by side
# image_number = 0
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(raw_select[image_number], cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow(gt_select[image_number])
# plt.subplot(1, 3, 3)
# plt.imshow(pred[image_number])

# %%
# single_output = process(raw[-20:-19,-224:,-224:])
# output_rearranged = np.moveaxis(output, 0, -1)

# features = output_rearranged.reshape(-1, output_channels)
# plt.imshow(single_output[0,0])
# %%

def process_image_pipeline(image_data, ground_truth=None, 
                          trained_classifier=None, num_classes=3,
                          train_classifier=True, epochs=500, 
                          learning_rate=1e-3, device=None, 
                          normalize_features_method="standardize",
                          feature_norm_stats=None,
                          use_data_augmentation=True,
                          balance_classes_method="undersample"):
    """
    Process a set of images through the complete pipeline:
    1. DINOv3 feature extraction with proper reshaping
    2. Feature normalization for better classifier training
    3. Simple neural network training/inference
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Input image data of shape (batch_size, height, width) or (height, width) for single image
    ground_truth : numpy.ndarray, optional
        Ground truth labels for training the classifier. Same spatial dimensions as image_data.
        If provided and train_classifier=True, will train a new classifier.
    trained_classifier : torch.nn.Module, optional
        Pre-trained classifier to use for inference. If None and train_classifier=False, 
        will return only DINOv3 features.
    num_classes : int, default=3
        Number of classes for classification
    train_classifier : bool, default=True
        Whether to train a new classifier or use existing one
    epochs : int, default=1000
        Number of training epochs if training classifier
    learning_rate : float, default=1e-3
        Learning rate for classifier training
    device : torch.device, optional
        Device to use for computation. If None, will auto-detect.
    normalize_features_method : str, default="standardize"
        Method for normalizing DINOv3 features: "standardize", "minmax", "robust", or "none"
    feature_norm_stats : dict, optional
        Pre-computed normalization statistics for inference. If None, will compute from data.
    use_data_augmentation : bool, default=True
        Whether to apply data augmentation during training (intensity shifts, etc.)
    balance_classes_method : str, default="undersample"
        Method to balance classes: "undersample", "oversample", "weighted", "hybrid", or "none"
        
    Returns:
    --------
    dict containing:
        - 'features': Raw DINOv3 features of shape (output_channels, batch_size, H, W)
        - 'features_normalized': Normalized features for classifier input
        - 'predictions': Classification predictions of shape (batch_size, H, W) if classifier used
        - 'classifier': Trained classifier model if train_classifier=True
        - 'training_history': Training metrics (losses, accuracies) if train_classifier=True
        - 'normalization_stats': Feature normalization statistics
        - 'processed_images': Preprocessed images that went into DINOv3
    """
    
    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure image_data is 3D (add batch dimension if needed)
    if image_data.ndim == 2:
        image_data = image_data[np.newaxis, ...]
    elif image_data.ndim != 3:
        raise ValueError(f"image_data must be 2D or 3D, got shape {image_data.shape}")
    
    # Ensure ground_truth has same spatial dimensions if provided
    if ground_truth is not None:
        if ground_truth.ndim == 2:
            ground_truth = ground_truth[np.newaxis, ...]
        elif ground_truth.ndim != 3:
            raise ValueError(f"ground_truth must be 2D or 3D, got shape {ground_truth.shape}")
        
        if ground_truth.shape != image_data.shape:
            raise ValueError(f"ground_truth shape {ground_truth.shape} doesn't match image_data shape {image_data.shape}")
    
    print(f"Processing {image_data.shape[0]} images of size {image_data.shape[1]}x{image_data.shape[2]}")
    print(f"Using device: {device}")
    
    # Step 1: Extract DINOv3 features
    print("Extracting DINOv3 features...")
    dinov3_features = process(image_data)  # Shape: (output_channels, batch_size, H, W)
    print(f"DINOv3 features shape: {dinov3_features.shape}")
    
    results = {
        'features': dinov3_features,
        'features_normalized': None,
        'processed_images': None,  # process() function doesn't return processed images in current version
        'predictions': None,
        'classifier': None,
        'training_history': None,
        'normalization_stats': None
    }
    
    # Step 2: Feature normalization and classifier training/inference
    if train_classifier or trained_classifier is not None:
        print("Preparing classifier data...")
        
        # Rearrange features for classifier: (output_channels, batch_size, H, W) -> (batch_size*H*W, output_channels)
        features_rearranged = np.moveaxis(dinov3_features, 0, -1)  # (batch_size, H, W, output_channels)
        features_flat = features_rearranged.reshape(-1, output_channels)
        
        # Normalize features for better classifier training
        if feature_norm_stats is None:
            # Compute normalization statistics from current data
            print(f"Normalizing features using method: {normalize_features_method}")
            features_normalized, norm_stats = normalize_features(
                features_flat, method=normalize_features_method
            )
            results['normalization_stats'] = norm_stats
        else:
            # Use pre-computed statistics (for inference)
            print("Applying pre-computed normalization statistics")
            features_normalized = apply_normalization_stats(features_flat, feature_norm_stats)
            results['normalization_stats'] = feature_norm_stats
        
        results['features_normalized'] = features_normalized
        X = torch.tensor(features_normalized, dtype=torch.float32).to(device)
        
        if train_classifier:
            if ground_truth is None:
                raise ValueError("ground_truth is required for training classifier")
            
            print(f"Training classifier on {features_flat.shape[0]} pixels...")
            
            # Prepare labels
            labels_flat = ground_truth.reshape(-1)
            y = torch.tensor(labels_flat, dtype=torch.long).to(device)
            
            # Create image indices for proper train/val splitting
            # Each pixel gets an index indicating which image it came from
            batch_size, height, width = image_data.shape
            image_indices = []
            for img_idx in range(batch_size):
                image_indices.extend([img_idx] * (height * width))
            image_indices = torch.tensor(image_indices, dtype=torch.long).to(device)
            
            print(f"Created image indices: {batch_size} images, {len(image_indices)} total pixels")
            print(f"Image index distribution: {torch.bincount(image_indices)}")
            
            # Train classifier with improved methods
            print("Training classifier with image-level validation split and regularization...")
            training_results = train_classifier_with_validation(
                X=X,
                y=y,
                num_classes=num_classes,
                device=device,
                validation_split=0.2,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=1e-4,  # L2 regularization
                patience=50,        # Early stopping
                use_improved_classifier=True,  # Use improved architecture
                batch_size=min(1024, len(X) // 4),  # Adaptive batch size
                balance_classes_method=balance_classes_method,  # Class balancing
                image_indices=image_indices  # Enable image-level splitting
            )
            
            classifier = training_results['classifier']
            results['classifier'] = classifier
            results['training_history'] = training_results
            
        else:
            classifier = trained_classifier
            
        # Step 3: Generate predictions
        if classifier is not None:
            print("Generating predictions...")
            classifier.eval()
            with torch.no_grad():
                logits = classifier(X)
                pred_flat = logits.argmax(dim=1).cpu().numpy()
            
            # Reshape predictions back to original spatial dimensions
            batch_size, height, width = image_data.shape
            predictions = pred_flat.reshape(batch_size, height, width)
            results['predictions'] = predictions
            print(f"Predictions shape: {predictions.shape}")
    
    return results


def visualize_pipeline_results(image_data, ground_truth, results, image_idx=0, figsize=(15, 5)):
    """
    Visualize the results of the image processing pipeline.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Original input images
    ground_truth : numpy.ndarray
        Ground truth labels
    results : dict
        Results dictionary from process_image_pipeline()
    image_idx : int, default=0
        Index of the image to visualize (for batch processing)
    figsize : tuple, default=(15, 5)
        Figure size for the plot
    """
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    axes[0].imshow(image_data[image_idx], cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    if ground_truth is not None:
        axes[1].imshow(ground_truth[image_idx])
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')
    
    # DINOv3 features (show first channel)
    if 'features' in results:
        axes[2].imshow(results['features'][0, image_idx])  # First feature channel
        axes[2].set_title('DINOv3 Features (Ch 0)')
        axes[2].axis('off')
    
    # Predictions
    if 'predictions' in results and results['predictions'] is not None:
        axes[3].imshow(results['predictions'][image_idx])
        axes[3].set_title('Predictions')
        axes[3].axis('off')
    else:
        axes[3].text(0.5, 0.5, 'No Predictions', ha='center', va='center', transform=axes[3].transAxes)
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_single_image_inference(image_data, trained_classifier, 
                                  normalization_stats=None, device=None):
    """
    Process a single image or batch of images through the pipeline for inference only.
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Input image data
    trained_classifier : torch.nn.Module
        Pre-trained classifier
    normalization_stats : dict, optional
        Pre-computed normalization statistics from training
    device : torch.device, optional
        Device to use for computation
        
    Returns:
    --------
    dict containing predictions and features
    """
    return process_image_pipeline(
        image_data=image_data,
        ground_truth=None,
        trained_classifier=trained_classifier,
        train_classifier=False,
        feature_norm_stats=normalization_stats,
        device=device
    )

# %%
# Example usage of the complete pipeline with flexible sampling
print("=" * 50)
print("EXAMPLE: Complete Pipeline Usage with Flexible Sampling")
print("=" * 50)

# Sample training data using flexible method (different orientations)
example_images, example_gt = sample_training_data(
    raw, gt,
    target_size=224,
    num_samples=10,
    method="flexible",  # Try different orientations
    seed=123  # Different seed for variety
)

print(f"Sampled training data shapes: raw={example_images.shape}, gt={example_gt.shape}")

# Process through complete pipeline with training
pipeline_results = process_image_pipeline(
    image_data=example_images,
    ground_truth=example_gt,
    num_classes=2,  # Binary classification (background vs nucleus)
    train_classifier=True,
    epochs=500,
    learning_rate=1e-3,
    normalize_features_method="standardize"  # Use z-score normalization
)

# Visualize results for multiple images to see variety
for idx in range(10):  # Show different samples
    print(f"\nVisualization for sample {idx}:")
    visualize_pipeline_results(
        image_data=example_images,
        ground_truth=example_gt,
        results=pipeline_results,
        image_idx=idx,
        figsize=(20, 5)
    )

# %%
# Example: Using trained classifier for inference on new data
print("=" * 50)
print("EXAMPLE: Inference with Trained Classifier")
print("=" * 50)

# Sample new data for inference using flexible method
new_image_data, new_gt_data = sample_training_data(
    raw, gt,
    target_size=224,
    num_samples=5,  # Fewer samples for inference
    method="flexible",
    seed=999  # Different seed for new samples
)

print(f"New inference data shape: {new_image_data.shape}")

# Use the trained classifier for inference
inference_results = process_single_image_inference(
    image_data=new_image_data,
    trained_classifier=pipeline_results['classifier'],
    normalization_stats=pipeline_results['normalization_stats']  # Use same normalization as training
)

# Visualize inference results for all samples
for idx in range(new_image_data.shape[0]):
    print(f"\nInference result for sample {idx}:")
    visualize_pipeline_results(
        image_data=new_image_data,
        ground_truth=new_gt_data,  # No ground truth for inference
        results=inference_results,
        image_idx=idx,
        figsize=(15, 4)
    )

# %%
# Comparison: Simple vs Flexible sampling
# print("=" * 60)
# print("COMPARISON: Simple vs Flexible Sampling Methods")
# print("=" * 60)

# # Sample using both methods for comparison
# simple_raw, simple_gt = sample_training_data(
#     raw, gt, target_size=224, num_samples=5, method="simple", seed=42
# )

# flexible_raw, flexible_gt = sample_training_data(
#     raw, gt, target_size=224, num_samples=5, method="flexible", seed=42
# )

# print(f"Simple method shapes: {simple_raw.shape}")
# print(f"Flexible method shapes: {flexible_raw.shape}")

# # Visualize comparison
# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# for i in range(5):
#     # Simple method
#     axes[0, i].imshow(simple_raw[i], cmap='gray')
#     axes[0, i].set_title(f'Simple #{i}')
#     axes[0, i].axis('off')
    
#     # Flexible method  
#     axes[1, i].imshow(flexible_raw[i], cmap='gray')
#     axes[1, i].set_title(f'Flexible #{i}')
#     axes[1, i].axis('off')

# plt.suptitle('Comparison: Simple vs Flexible Sampling Methods', fontsize=16)
# plt.tight_layout()
# plt.show()

# # %%
# # Feature Normalization Comparison
# print("=" * 60)
# print("FEATURE NORMALIZATION COMPARISON")
# print("=" * 60)

# # Test different normalization methods
# test_images, test_gt = sample_training_data(
#     raw, gt, target_size=224, num_samples=3, method="flexible", seed=555
# )

# normalization_methods = ["none", "standardize", "minmax", "robust"]
# results_comparison = {}

# for method in normalization_methods:
#     print(f"\nTesting normalization method: {method}")
    
#     result = process_image_pipeline(
#         image_data=test_images,
#         ground_truth=test_gt,
#         num_classes=2,
#         train_classifier=True,
#         epochs=100,  # Fewer epochs for comparison
#         learning_rate=1e-3,
#         normalize_features_method=method
#     )
    
#     results_comparison[method] = result
    
#     # Print some statistics about the features
#     if method != "none":
#         features_norm = result['features_normalized']
#         print(f"  Normalized features - Mean: {np.mean(features_norm):.4f}, Std: {np.std(features_norm):.4f}")
#         print(f"  Min: {np.min(features_norm):.4f}, Max: {np.max(features_norm):.4f}")
#     else:
#         features_flat = np.moveaxis(result['features'], 0, -1).reshape(-1, output_channels)
#         print(f"  Raw features - Mean: {np.mean(features_flat):.4f}, Std: {np.std(features_flat):.4f}")
#         print(f"  Min: {np.min(features_flat):.4f}, Max: {np.max(features_flat):.4f}")

# print(f"\nRecommendation: 'standardize' is typically best for neural network training")
# print(f"It ensures zero mean and unit variance, which helps with gradient flow.")

# # %%
# # Advanced Training Example with Overfitting Prevention
# print("=" * 70)
# print("ADVANCED TRAINING: Overfitting Prevention & Validation Monitoring")
# print("=" * 70)

# # Sample more diverse training data with augmentation
# advanced_images, advanced_gt = sample_training_data(
#     raw, gt,
#     target_size=224,
#     num_samples=10,  # More samples for better training
#     method="flexible",
#     seed=777,
#     use_augmentation=True  # Enable data augmentation
# )

# print(f"Training with {advanced_images.shape[0]} augmented samples...")
# print(f"Total pixels for training: {advanced_images.shape[0] * 224 * 224:,}")
# print(f"Feature dimension per pixel: {output_channels}")

# # Train with improved regularization
# advanced_results = process_image_pipeline(
#     image_data=advanced_images,
#     ground_truth=advanced_gt,
#     num_classes=2,
#     train_classifier=True,
#     epochs=300,  # Reduced epochs due to early stopping
#     learning_rate=1e-3,
#     normalize_features_method="standardize",
#     use_data_augmentation=True
# )

# # Plot training history to check for overfitting
# print("\nTraining History:")
# plot_training_history(advanced_results['training_history'], figsize=(15, 6))

# # %%
# # Compare Simple vs Advanced Training
# print("=" * 60)
# print("COMPARISON: Simple vs Advanced Training")
# print("=" * 60)

# # Simple training (old method, prone to overfitting)
# simple_images, simple_gt = sample_training_data(
#     raw, gt, target_size=224, num_samples=5, method="simple", seed=42, use_augmentation=False
# )

# # Use old-style training for comparison
# simple_X = torch.tensor(
#     np.moveaxis(process(simple_images), 0, -1).reshape(-1, output_channels), 
#     dtype=torch.float32
# ).to(device)
# simple_y = torch.tensor(simple_gt.reshape(-1), dtype=torch.long).to(device)

# # Quick simple training (without validation split)
# simple_net = SimpleClassifier(output_channels, 2).to(device)
# simple_optimizer = optim.Adam(simple_net.parameters(), lr=1e-3)
# simple_criterion = nn.CrossEntropyLoss()

# print("Training simple classifier (prone to overfitting)...")
# simple_net.train()
# for epoch in range(200):
#     simple_optimizer.zero_grad()
#     logits = simple_net(simple_X)
#     loss = simple_criterion(logits, simple_y)
#     loss.backward()
#     simple_optimizer.step()
    
#     if (epoch + 1) % 50 == 0:
#         acc = (logits.argmax(dim=1) == simple_y).float().mean().item()
#         print(f"Simple training - Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")

# # Test both on new data
# test_images, test_gt = sample_training_data(
#     raw, gt, target_size=224, num_samples=3, method="flexible", seed=999, use_augmentation=False
# )

# print(f"\nTesting both models on {test_images.shape[0]} new samples...")

# # Test advanced model
# advanced_test_results = process_single_image_inference(
#     image_data=test_images,
#     trained_classifier=advanced_results['classifier'],
#     normalization_stats=advanced_results['normalization_stats']
# )

# # Test simple model
# test_features = np.moveaxis(process(test_images), 0, -1).reshape(-1, output_channels)
# test_X = torch.tensor(test_features, dtype=torch.float32).to(device)

# simple_net.eval()
# with torch.no_grad():
#     simple_pred = simple_net(test_X).argmax(dim=1).cpu().numpy().reshape(test_images.shape)

# # Compare results
# fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# for i in range(3):
#     # Original image
#     axes[i, 0].imshow(test_images[i], cmap='gray')
#     axes[i, 0].set_title(f'Test Image {i+1}')
#     axes[i, 0].axis('off')
    
#     # Ground truth
#     axes[i, 1].imshow(test_gt[i])
#     axes[i, 1].set_title('Ground Truth')
#     axes[i, 1].axis('off')
    
#     # Simple model prediction
#     axes[i, 2].imshow(simple_pred[i])
#     axes[i, 2].set_title('Simple Model\n(Overfitting)')
#     axes[i, 2].axis('off')
    
#     # Advanced model prediction
#     axes[i, 3].imshow(advanced_test_results['predictions'][i])
#     axes[i, 3].set_title('Advanced Model\n(Regularized)')
#     axes[i, 3].axis('off')

# plt.suptitle('Model Comparison: Simple vs Advanced Training', fontsize=16)
# plt.tight_layout()
# plt.show()

# print("Key improvements in advanced training:")
# print("✅ Data augmentation (intensity shifts, rotations, flips)")
# print("✅ Train/validation split for monitoring")
# print("✅ Early stopping to prevent overfitting") 
# print("✅ L2 regularization (weight decay)")
# print("✅ Dropout and batch normalization")
# print("✅ Learning rate scheduling")
# print("✅ Gradient clipping")
# print("✅ Feature normalization")

# # %%
# # Class Balancing Comparison
# print("=" * 70)
# print("CLASS BALANCING METHODS COMPARISON")
# print("=" * 70)

# # Create a deliberately imbalanced dataset for demonstration
# imbalanced_images, imbalanced_gt = sample_training_data(
#     raw, gt,
#     target_size=224,
#     num_samples=8,
#     method="flexible",
#     seed=444,
#     use_augmentation=False  # No augmentation for cleaner comparison
# )

# # Check class distribution in the original data
# unique_labels, label_counts = np.unique(imbalanced_gt, return_counts=True)
# print(f"Original class distribution: {dict(zip(unique_labels, label_counts))}")
# total_pixels = imbalanced_gt.size
# print(f"Total pixels: {total_pixels:,}")
# for label, count in zip(unique_labels, label_counts):
#     percentage = (count / total_pixels) * 100
#     print(f"  Class {label}: {count:,} pixels ({percentage:.1f}%)")

# print("\nTesting different balancing methods...")

# # Test different balancing methods
# balancing_methods = [
#     ("none", "No Balancing"),
#     ("weighted", "Weighted Loss"), 
#     ("undersample", "Undersampling"),
#     ("oversample", "Oversampling"),
#     ("hybrid", "Hybrid Sampling")
# ]

# balancing_results = {}

# for method, description in balancing_methods:
#     print(f"\n{'-' * 50}")
#     print(f"Testing: {description} (method='{method}')")
#     print(f"{'-' * 50}")
    
#     try:
#         result = process_image_pipeline(
#             image_data=imbalanced_images,
#             ground_truth=imbalanced_gt,
#             num_classes=2,
#             train_classifier=True,
#             epochs=100,  # Fewer epochs for comparison
#             learning_rate=1e-3,
#             normalize_features_method="standardize",
#             balance_classes_method=method
#         )
        
#         balancing_results[method] = result
        
#         # Print final accuracy
#         final_val_acc = result['training_history']['best_val_acc']
#         print(f"Best validation accuracy: {final_val_acc:.4f}")
        
#     except Exception as e:
#         print(f"Error with method {method}: {e}")
#         balancing_results[method] = None

# # Compare results
# print(f"\n{'=' * 70}")
# print("BALANCING METHODS COMPARISON SUMMARY")
# print(f"{'=' * 70}")

# comparison_data = []
# for method, description in balancing_methods:
#     if balancing_results[method] is not None:
#         best_acc = balancing_results[method]['training_history']['best_val_acc']
#         epochs_trained = balancing_results[method]['training_history']['epochs_trained']
#         comparison_data.append((method, description, best_acc, epochs_trained))

# # Sort by validation accuracy
# comparison_data.sort(key=lambda x: x[2], reverse=True)

# print(f"{'Method':<12} {'Description':<20} {'Best Val Acc':<12} {'Epochs':<8}")
# print(f"{'-' * 60}")
# for method, desc, acc, epochs in comparison_data:
#     print(f"{method:<12} {desc:<20} {acc:.4f}      {epochs:<8}")

# # Plot comparison of training histories
# if len(comparison_data) > 1:
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
#     colors = ['blue', 'red', 'green', 'orange', 'purple']
    
#     for i, (method, desc, _, _) in enumerate(comparison_data):
#         if balancing_results[method] is not None:
#             history = balancing_results[method]['training_history']
#             epochs = range(1, len(history['train_losses']) + 1)
#             color = colors[i % len(colors)]
            
#             # Plot losses
#             axes[0].plot(epochs, history['val_losses'], 
#                         label=f'{desc}', color=color, linewidth=2)
            
#             # Plot accuracies  
#             axes[1].plot(epochs, history['val_accs'],
#                         label=f'{desc}', color=color, linewidth=2)
    
#     axes[0].set_xlabel('Epoch')
#     axes[0].set_ylabel('Validation Loss')
#     axes[0].set_title('Validation Loss Comparison')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
    
#     axes[1].set_xlabel('Epoch')
#     axes[1].set_ylabel('Validation Accuracy') 
#     axes[1].set_title('Validation Accuracy Comparison')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# print(f"\nRecommendations:")
# print(f"• 'undersample': Best for balanced training, prevents overfitting")
# print(f"• 'weighted': Good when you can't lose any data")
# print(f"• 'oversample': Use when minority class is very small")
# print(f"• 'hybrid': Balanced approach, good middle ground")
# print(f"• 'none': Only if classes are naturally balanced")

# # %%
# # Image-Level Train/Validation Split Demonstration
# print("=" * 70)
# print("DEMONSTRATION: Image-Level vs Pixel-Level Train/Val Split")
# print("=" * 70)

# # First, show a conceptual demo of the splitting logic
# create_image_level_split_demo()

# # %%
# # Now demonstrate with real data
# print("\n" + "=" * 70)
# print("REAL DATA DEMONSTRATION - FIXED VERSION")
# print("=" * 70)

# print("🔧 IMPORTANT FIX: Train/val split now happens BEFORE class balancing!")
# print("This prevents the IndexError and ensures proper image-level separation.")

# # Sample a few images for clear demonstration
# demo_images, demo_gt = sample_training_data(
#     raw, gt,
#     target_size=224,
#     num_samples=4,  # Use 4 images to clearly show the split
#     method="flexible",
#     seed=123,
#     use_augmentation=False  # No augmentation for cleaner comparison
# )

# print(f"Demo dataset: {demo_images.shape[0]} images of size {demo_images.shape[1]}x{demo_images.shape[2]}")
# print(f"Total pixels: {demo_images.size:,}")

# # The improved pipeline now automatically does image-level splitting
# print(f"\nRunning pipeline with FIXED IMAGE-LEVEL train/val split...")
# print(f"Expected: 80% of images (3-4 images) for training, 20% (1 image) for validation")
# print(f"Class balancing will be applied to training set only (after split)")

# # Run the pipeline - it will automatically use image-level splitting
# demo_results = process_image_pipeline(
#     image_data=demo_images,
#     ground_truth=demo_gt,
#     num_classes=2,
#     train_classifier=True,
#     epochs=50,  # Fewer epochs for demo
#     learning_rate=1e-3,
#     normalize_features_method="standardize",
#     balance_classes_method="undersample"  # This will work now!
# )

# print(f"\nKey Improvements in the Fix:")
# print(f"✅ Train/val split happens BEFORE class balancing")
# print(f"✅ Class balancing is applied only to training set")
# print(f"✅ No shape mismatch between image_indices and features")
# print(f"✅ Validation set remains untouched by class balancing")
# print(f"✅ Complete separation of images between train and val")

# print(f"\nTraining completed successfully with the fixed approach!")
# if 'training_history' in demo_results and demo_results['training_history']:
#     final_val_acc = demo_results['training_history']['val_accs'][-1]
#     print(f"Final validation accuracy: {final_val_acc:.4f}")
#     print(f"This validation accuracy is trustworthy because:")
#     print(f"  • Validation images are completely separate from training")
#     print(f"  • No data leakage between train and validation sets")
#     print(f"  • Class balancing doesn't affect validation distribution")

# %%
# Example usage - train on large dataset without memory issues
results = train_with_memory_efficient_loader(
    raw_data=raw, 
    gt_data=gt,
    train_pool_size=50,    # Sample from 50 possible training images
    val_pool_size=10,      # Use 10 fixed validation images  
    images_per_batch=4,    # 4 images per training batch
    batches_per_epoch=10,  # 10 batches per epoch = 40 images/epoch
    num_classes=3,
    epochs=200,
    balance_method="undersample"
)
# %%
# Test your latest model on 3 sample images
from funlib.geometry import Roi
raw = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s2").to_ndarray_ts(Roi(np.array((670, 922, 4086))*32,3*[750*32]))

results = load_model_and_run_inference(
    "/groups/cellmap/cellmap/ackermand/Programming/cellmap-flow/tmp/results/memory_efficient_training_20250911_143601_epoch_0100.pkl",
    raw, gt_data=None, num_sample_images=10
)
print(f"Test accuracy: {results['accuracy']:.4f}")
# %%
