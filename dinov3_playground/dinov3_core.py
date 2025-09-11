"""
DINOv3 Core Processing Module

This module contains the core DINOv3 feature extraction functionality including:
- Model initialization and configuration
- Feature processing and normalization
- Utility functions for tensor operations

Author: GitHub Copilot
Date: 2025-09-11
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DINOv3ViTModel
import functools


def enable_amp_inference(model, amp_dtype=torch.bfloat16):
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def amp_forward(*args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            return orig_forward(*args, **kwargs)

    model.forward = amp_forward
    return model


# Initialize DINOv3 model and processor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_id)
model = DINOv3ViTModel.from_pretrained(model_id)
model = enable_amp_inference(model, torch.bfloat16)
model.to(DEVICE)
model.eval()

# Get output dimensions from model configuration
output_channels = model.config.hidden_size  # 384 for ViT-S/16


def get_img():
    """Load and return a sample image for testing."""
    import requests
    from io import BytesIO
    
    url = 'https://github.com/facebookresearch/dinov2/blob/main/assets/dinov2_vitb14_pretrain.jpg?raw=true'
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def round_to_multiple(x: int, k: int) -> int:
    """Round x to the nearest multiple of k."""
    return round(x / k) * k


def ensure_multiple(size: int, patch: int) -> int:
    """Ensure size is a multiple of patch size."""
    return round_to_multiple(size, patch)


def normalize_01(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def normalize_features(features, method="standardize", eps=1e-6):
    """
    Normalize feature vectors using various methods.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature array of shape (n_samples, n_features)
    method : str, default="standardize"
        Normalization method: "standardize", "minmax", "unit", or "robust"
    eps : float, default=1e-6
        Small epsilon for numerical stability
        
    Returns:
    --------
    tuple: (normalized_features, normalization_stats)
    """
    if method == "standardize":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std < eps, eps, std)  # Avoid division by zero
        normalized = (features - mean) / std
        stats = {"method": "standardize", "mean": mean, "std": std}
        
    elif method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val < eps, eps, range_val)  # Avoid division by zero
        normalized = (features - min_val) / range_val
        stats = {"method": "minmax", "min": min_val, "max": max_val, "range": range_val}
        
    elif method == "unit":
        # Unit vector normalization: x / ||x||
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < eps, eps, norms)  # Avoid division by zero
        normalized = features / norms
        stats = {"method": "unit", "eps": eps}
        
    elif method == "robust":
        # Robust normalization using median and IQR
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        iqr = np.where(iqr < eps, eps, iqr)  # Avoid division by zero
        normalized = (features - median) / iqr
        stats = {"method": "robust", "median": median, "q25": q25, "q75": q75, "iqr": iqr}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def apply_normalization_stats(features, stats):
    """
    Apply previously computed normalization statistics to new features.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature array to normalize
    stats : dict
        Normalization statistics from normalize_features()
        
    Returns:
    --------
    numpy.ndarray: Normalized features
    """
    method = stats["method"]
    eps = 1e-6
    
    if method == "standardize":
        normalized = (features - stats["mean"]) / stats["std"]
        
    elif method == "minmax":
        normalized = (features - stats["min"]) / stats["range"]
        
    elif method == "unit":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < eps, eps, norms)
        normalized = features / norms
        
    elif method == "robust":
        normalized = (features - stats["median"]) / stats["iqr"]
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def process(data):    
    """
    Process raw image data through DINOv3 to extract features.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input images of shape (batch_size, height, width) or (height, width)
        
    Returns:
    --------
    numpy.ndarray: Features of shape (output_channels, batch_size, patch_h, patch_w)
    """
    global model, processor, DEVICE
    
    # Handle single image
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add batch dimension
    
    batch_size, height, width = data.shape
    
    # Convert to PIL Images for the processor
    pil_images = []
    for i in range(batch_size):
        # Normalize to 0-255 range for PIL
        img_normalized = ((data[i] - data[i].min()) / (data[i].max() - data[i].min() + 1e-8) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_normalized).convert('RGB')
        pil_images.append(pil_image)
    
    # Process through DINOv3 processor
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        # Use last_hidden_state which has shape (batch_size, num_patches + 1, hidden_size)
        features = outputs.last_hidden_state
        
        # Remove CLS token (first token) to get only patch features
        patch_features = features[:, 1:, :]  # Shape: (batch_size, num_patches, hidden_size)
        
        # Reshape to spatial format
        # DINOv3 ViT-S/16 has patch size 16, so for 224x224 input we get 14x14 patches
        patch_size = 16
        patch_h = patch_w = height // patch_size  # Assuming square patches
        
        # Reshape: (batch_size, num_patches, hidden_size) -> (batch_size, patch_h, patch_w, hidden_size)
        spatial_features = patch_features.reshape(batch_size, patch_h, patch_w, -1)
        
        # Rearrange to (hidden_size, batch_size, patch_h, patch_w) to match expected output format
        output = spatial_features.permute(3, 0, 1, 2).cpu().numpy()
    
    return output
