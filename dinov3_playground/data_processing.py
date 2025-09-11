"""
Data Processing Module for DINOv3 Training

This module contains functions for:
- Data sampling and augmentation
- Image preprocessing and resizing
- Data splitting and preparation

Author: GitHub Copilot
Date: 2025-09-11
"""

import numpy as np
from scipy import ndimage
from skimage import transform, exposure
import random


def sample_training_data(raw_data, gt_data, target_size=224, num_samples=10, 
                        method="flexible", seed=None, use_augmentation=True):
    """
    Sample training images from 3D volumes with flexible plane selection.
    
    Parameters:
    -----------
    raw_data : numpy.ndarray
        3D raw data (z, y, x)
    gt_data : numpy.ndarray  
        3D ground truth data (z, y, x)
    target_size : int, default=224
        Target size for output images
    num_samples : int, default=10
        Number of samples to generate
    method : str, default="flexible"
        Sampling method: "xy", "xz", "yz", or "flexible"
    seed : int, optional
        Random seed for reproducibility
    use_augmentation : bool, default=True
        Whether to apply data augmentation
        
    Returns:
    --------
    tuple: (sampled_raw, sampled_gt) arrays of shape (num_samples, target_size, target_size)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    z, y, x = raw_data.shape
    sampled_raw = []
    sampled_gt = []
    
    for _ in range(num_samples):
        if method == "flexible":
            # Randomly choose a plane orientation
            plane = np.random.choice(['xy', 'xz', 'yz'])
        else:
            plane = method
        
        # Sample based on chosen plane
        if plane == 'xy':
            # Sample along Z axis
            z_idx = np.random.randint(0, z)
            raw_slice = raw_data[z_idx, :, :]
            gt_slice = gt_data[z_idx, :, :]
            
        elif plane == 'xz':
            # Sample along Y axis
            y_idx = np.random.randint(0, y)
            raw_slice = raw_data[:, y_idx, :]
            gt_slice = gt_data[:, y_idx, :]
            
        elif plane == 'yz':
            # Sample along X axis
            x_idx = np.random.randint(0, x)
            raw_slice = raw_data[:, :, x_idx]
            gt_slice = gt_data[:, :, x_idx]
        
        # Resize and crop to target size
        raw_processed, gt_processed = resize_and_crop_to_target(
            raw_slice, gt_slice, target_size, 
            resize_method='crop_or_pad', use_augmentation=use_augmentation
        )
        
        sampled_raw.append(raw_processed)
        sampled_gt.append(gt_processed)
    
    return np.array(sampled_raw), np.array(sampled_gt)


def apply_intensity_augmentation(raw_slice, augment_prob=0.7):
    """
    Apply intensity-based data augmentation to raw image slice.
    
    Parameters:
    -----------
    raw_slice : numpy.ndarray
        Input image slice
    augment_prob : float, default=0.7
        Probability of applying each augmentation
        
    Returns:
    --------
    numpy.ndarray: Augmented image slice
    """
    augmented = raw_slice.copy()
    
    if np.random.random() < augment_prob:
        # Brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = augmented * brightness_factor
    
    if np.random.random() < augment_prob:
        # Contrast adjustment using histogram stretching
        p2, p98 = np.percentile(augmented, (2, 98))
        augmented = exposure.rescale_intensity(augmented, in_range=(p2, p98))
    
    if np.random.random() < augment_prob:
        # Gamma correction
        gamma = np.random.uniform(0.8, 1.2)
        augmented = exposure.adjust_gamma(augmented, gamma)
    
    if np.random.random() < augment_prob:
        # Add slight Gaussian noise
        noise_std = np.random.uniform(0.01, 0.05) * np.std(augmented)
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented = augmented + noise
    
    return augmented


def resize_and_crop_to_target(raw_slice, gt_slice, target_size, 
                             resize_method='crop_or_pad', use_augmentation=True):
    """
    Resize and crop image slices to target size with optional augmentation.
    
    Parameters:
    -----------
    raw_slice : numpy.ndarray
        Raw image slice
    gt_slice : numpy.ndarray
        Ground truth slice
    target_size : int
        Target output size
    resize_method : str, default='crop_or_pad'
        Method: 'resize', 'crop_or_pad', or 'random_crop'
    use_augmentation : bool, default=True
        Whether to apply augmentation
        
    Returns:
    --------
    tuple: (processed_raw, processed_gt)
    """
    h, w = raw_slice.shape
    
    if resize_method == 'resize':
        # Simple resize to target size
        raw_resized = transform.resize(raw_slice, (target_size, target_size), preserve_range=True)
        gt_resized = transform.resize(gt_slice, (target_size, target_size), 
                                    preserve_range=True, order=0)  # Nearest neighbor for labels
        
    elif resize_method == 'crop_or_pad':
        # Crop or pad to target size
        raw_resized = crop_or_pad_to_size(raw_slice, target_size)
        gt_resized = crop_or_pad_to_size(gt_slice, target_size)
        
    elif resize_method == 'random_crop' and min(h, w) >= target_size:
        # Random crop if image is large enough
        top = np.random.randint(0, h - target_size + 1)
        left = np.random.randint(0, w - target_size + 1)
        raw_resized = raw_slice[top:top+target_size, left:left+target_size]
        gt_resized = gt_slice[top:top+target_size, left:left+target_size]
        
    else:
        # Fall back to crop_or_pad if random_crop can't be applied
        raw_resized = crop_or_pad_to_size(raw_slice, target_size)
        gt_resized = crop_or_pad_to_size(gt_slice, target_size)
    
    # Apply augmentation if requested
    if use_augmentation:
        # Intensity augmentation on raw data
        raw_resized = apply_intensity_augmentation(raw_resized)
        
        # Geometric augmentation on both raw and GT
        if np.random.random() < 0.5:
            # Random rotation
            angle = np.random.uniform(-15, 15)
            raw_resized = ndimage.rotate(raw_resized, angle, reshape=False, order=1)
            gt_resized = ndimage.rotate(gt_resized, angle, reshape=False, order=0)
        
        if np.random.random() < 0.5:
            # Random horizontal flip
            raw_resized = np.fliplr(raw_resized)
            gt_resized = np.fliplr(gt_resized)
        
        if np.random.random() < 0.5:
            # Random vertical flip
            raw_resized = np.flipud(raw_resized)
            gt_resized = np.flipud(gt_resized)
    
    return raw_resized.astype(np.float32), gt_resized.astype(np.int64)


def crop_or_pad_to_size(image, target_size):
    """
    Crop or pad image to target size.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    target_size : int
        Target size
        
    Returns:
    --------
    numpy.ndarray: Processed image of size (target_size, target_size)
    """
    h, w = image.shape
    
    # Calculate padding or cropping needed
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    crop_h = max(0, h - target_size)
    crop_w = max(0, w - target_size)
    
    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                      mode='reflect')
    
    # Apply cropping if needed
    if crop_h > 0 or crop_w > 0:
        crop_top = crop_h // 2
        crop_left = crop_w // 2
        image = image[crop_top:crop_top+target_size, crop_left:crop_left+target_size]
    
    return image


def create_image_level_split_demo():
    """
    Create a demonstration of proper image-level train/validation splitting.
    
    This function shows how to split data at the image level to prevent data leakage.
    """
    print("=" * 60)
    print("IMAGE-LEVEL TRAIN/VALIDATION SPLITTING DEMONSTRATION")
    print("=" * 60)
    
    # Simulate image sampling with image indices
    n_images = 20
    pixels_per_image = 1000
    
    # Create mock image indices for all pixels
    image_indices = []
    for img_idx in range(n_images):
        image_indices.extend([img_idx] * pixels_per_image)
    image_indices = np.array(image_indices)
    
    print(f"Total images: {n_images}")
    print(f"Pixels per image: {pixels_per_image}")
    print(f"Total pixels: {len(image_indices)}")
    
    # Get unique images for splitting
    unique_images = np.unique(image_indices)
    np.random.shuffle(unique_images)
    
    # Split images 80/20
    split_idx = int(0.8 * len(unique_images))
    train_images = unique_images[:split_idx]
    val_images = unique_images[split_idx:]
    
    print(f"\nImage-level split:")
    print(f"Training images: {len(train_images)} ({train_images})")
    print(f"Validation images: {len(val_images)} ({val_images})")
    
    # Create pixel-level masks
    train_mask = np.isin(image_indices, train_images)
    val_mask = np.isin(image_indices, val_images)
    
    print(f"\nPixel-level results:")
    print(f"Training pixels: {train_mask.sum()}")
    print(f"Validation pixels: {val_mask.sum()}")
    print(f"Total: {train_mask.sum() + val_mask.sum()}")
    
    # Verify no overlap
    overlap = np.intersect1d(train_images, val_images)
    print(f"Image overlap: {len(overlap)} (should be 0)")
    
    return {
        'train_images': train_images,
        'val_images': val_images,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'image_indices': image_indices
    }
