import random
# %%
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from transformers import AutoImageProcessor, DINOv3ViTModel
from funlib.geometry import Coordinate
from cellmap_flow.image_data_interface import ImageDataInterface
import functools
def enable_amp_inference(model, amp_dtype=torch.bfloat16):
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    @torch.inference_mode()
    def wrapped_forward(*args, **kwargs):
        device = next(model.parameters()).device
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                return orig_forward(*args, **kwargs)
        else:
            return orig_forward(*args, **kwargs)

    model.forward = wrapped_forward
    model.eval()  # turn off dropout/batchnorm training behavior
    return model


voxel_size = 128
input_voxel_size = Coordinate((voxel_size, voxel_size, voxel_size))
output_voxel_size = Coordinate((voxel_size, voxel_size, voxel_size))
batch_size = 2
image_size = 224
read_shape = Coordinate((batch_size, image_size, image_size)) * Coordinate(input_voxel_size)
write_shape = Coordinate((batch_size, image_size, image_size)) * Coordinate(output_voxel_size)
output_channels = 380
block_shape = np.array((batch_size, image_size, image_size, output_channels))

# -------------------------
# Config
# -------------------------
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"#dinov3-vitl16-pretrain-sat493m"
TARGET_IMG_SIZE = 896
UPSAMPLE = True   # <--- toggle upsampling here

# -------------------------
# Helpers
# -------------------------
def get_img():
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")

def round_to_multiple(x: int, k: int) -> int:
    return int(k * round(float(x) / k))

def ensure_multiple(size: int, patch: int) -> int:
    s = round_to_multiple(size, patch)
    return max(patch, s)

def normalize_01(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + 1e-6)

# -------------------------
# Load model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = DINOv3ViTModel.from_pretrained(MODEL_ID).to(device)
# after loading:
model = enable_amp_inference(model)     # do this once

#model.eval()

patch_size = getattr(model.config, "patch_size", 16)

# -------------------------
# Prepare input
# -------------------------
img = get_img()
size_rounded = ensure_multiple(TARGET_IMG_SIZE, patch_size)

inputs = processor(
    images=img,
    size={"height": size_rounded, "width": size_rounded},
    return_tensors="pt"
)
pixel_values = inputs["pixel_values"].to(device)


def process_chunk(idi: ImageDataInterface, input_roi):
    data = idi.to_ndarray_ts(input_roi)  # Shape: (batch_size, Y, X)
    
    # Convert each slice to RGB format for the model
    batch_images = []
    for slice_2d in data:
        # Normalize the slice to 0-255 range and convert to RGB
        slice_normalized = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(np.uint8)
        # Convert to 3-channel RGB by repeating the grayscale values
        slice_rgb = np.stack([slice_normalized] * 3, axis=-1)
        batch_images.append(Image.fromarray(slice_rgb))
    
    # Process the batch through the model
    inputs = processor(
        images=batch_images,
        size={"height": size_rounded, "width": size_rounded},
        return_tensors="pt"
    )
    pixel_values_batch = inputs["pixel_values"].to(device)
    
    # -------------------------
    # Forward pass
    # -------------------------
    outputs = model(pixel_values=pixel_values_batch)

    last_hidden = outputs.last_hidden_state        # [B, 1 + N + R, C]
    cls_token = last_hidden[:, 0]                  # [B, C]
    tokens_wo_cls = last_hidden[:, 1:]             # [B, N + R, C]

    B, NR, C = tokens_wo_cls.shape
    H = W = size_rounded // patch_size
    N_expected = H * W
    
    if NR > N_expected:
        patch_tokens = tokens_wo_cls[:, (NR-N_expected):, :]  # drop register tokens
    else:
        patch_tokens = tokens_wo_cls
    
    feat_map = patch_tokens.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

    # Select only the desired number of output channels to reduce memory usage
    feat_map = feat_map[:, :output_channels, :, :]  # [B, output_channels, H, W]

    # -------------------------
    # Optionally upsample
    # -------------------------
    if UPSAMPLE:
        feat_map = F.interpolate(
            feat_map, size=(image_size, image_size),
            mode="bilinear", align_corners=False
        )
    
    # Convert to numpy and ensure C-contiguous layout
    output = feat_map.float().cpu().numpy()  # Shape: (batch_size, output_channels, H, W)
    
    # Transpose to desired shape: (batch_size, output_channels, H, W) -> (output_channels, batch_size, H, W)
    output = output.transpose(1, 0, 2, 3)  # Now shape: (output_channels, batch_size, H, W)
    
    # Also return the processed images that went into the model for comparison
    processed_images = []
    for img_pil in batch_images:
        processed_img = np.array(img_pil.resize((size_rounded, size_rounded)))
        processed_images.append(processed_img[:, :, 0])  # Just take one channel since all 3 are the same
    processed_images = np.array(processed_images)
    print(f"output_shape: {output.shape}")
    return np.ascontiguousarray(output)
    # return data, np.ascontiguousarray(output), processed_images, pixel_values_batch.to("cpu").numpy()

# import cellmap_flow.image_data_interface
# import cellmap_flow.utils.ds
# from importlib import reload

# reload(cellmap_flow.utils.ds)
# reload(cellmap_flow.image_data_interface)
# from cellmap_flow.image_data_interface import ImageDataInterface
# from funlib.geometry import Roi

# idi = ImageDataInterface("/nrs/cellmap/data/jrc_22ak351-leaf-3m/jrc_22ak351-leaf-3m.zarr/recon-1/em/fibsem-uint8/s4")
# data, output, processed_images,pixel_values_batch = process_chunk(idi, Roi((2560, 2560, 2560), read_shape))

# # First, visualize the original input data
# print(f"Original data shape: {data.shape}")  # Should be (batch_size, H, W)
# print(f"Output shape: {output.shape}")  # Should be (batch_size, output_channels, H, W)
# print(f"Target image size (for model): {TARGET_IMG_SIZE}")
# print(f"Size rounded (actual model input): {size_rounded}")
# print(f"Patch size: {patch_size}")
# print(f"Feature map resolution: {size_rounded // patch_size}")
# print(f"UPSAMPLE enabled: {UPSAMPLE}")

# # Plot original images and processed images side by side
# fig_orig, axes_orig = plt.subplots(batch_size, batch_size, figsize=(10, 8))

# for batch_idx in range(batch_size):
#     # Original data
#     ax_orig = axes_orig[batch_idx, 0]
#     orig_data = data[batch_idx]
#     orig_norm = (orig_data - orig_data.min()) / (orig_data.max() - orig_data.min() + 1e-8)
#     im1 = ax_orig.imshow(orig_norm, cmap='gray')
#     ax_orig.set_title(f'Original Image {batch_idx} ({orig_data.shape})')
#     ax_orig.axis('off')
#     plt.colorbar(im1, ax=ax_orig, shrink=0.8)
    
#     # Processed data (what actually goes into DINOv3)
#     ax_proc = axes_orig[batch_idx, 1]
#     proc_data = processed_images[batch_idx]
#     proc_norm = (proc_data - proc_data.min()) / (proc_data.max() - proc_data.min() + 1e-8)
#     im2 = ax_proc.imshow(proc_norm, cmap='gray')
#     ax_proc.set_title(f'Processed Image {batch_idx} ({proc_data.shape})')
#     ax_proc.axis('off')
#     plt.colorbar(im2, ax=ax_proc, shrink=0.8)

# plt.tight_layout()
# plt.show()

# # Visualize output channels for both images in the batch

# # Create visualization for both images in the batch
# # Note: output shape is now (channels, batches, H, W)
# fig, axes = plt.subplots(batch_size, output_channels, figsize=(output_channels * 2, 4))
# if output_channels == 1:
#     axes = axes.reshape(2, 1)

# # for batch_idx in range(batch_size):  # For both images in the batch
# #     for channel_idx in range(output_channels):
# #         ax = axes[batch_idx, channel_idx]
        
# #         # Get the channel data using new dimension order: (channel_idx, batch_idx, H, W)
# #         channel_data = output[channel_idx, batch_idx, :, :]
        
# #         # Normalize to 0-1 for better visualization
# #         channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
        
# #         # Display the channel
# #         im = ax.imshow(channel_norm, cmap='viridis')
# #         ax.set_title(f'Batch {batch_idx}, Ch {channel_idx}')
# #         ax.axis('off')
        
# #         # Add colorbar for the first row
# #         if batch_idx == 0:
# #             plt.colorbar(im, ax=ax, shrink=0.8)

# plt.tight_layout()
# plt.show()


# %%
# # Simple neural network for feature classification
# import torch.nn as nn
# import torch.optim as optim
# import random

# # Example: Let's use all feature vectors from the first image in the batch as training data
# # and assign random labels (for demonstration; replace with your real labels)

# num_classes = 3  # Set the number of categories you want to classify

# # For each pixel, the feature vector is all channel values at that pixel (for the first image in the batch)
# features = []
# labels = []
# for x in range(image_size):
#     for y in range(image_size):
#         vec = output[:, 0, x, y]  # shape: (output_channels,)
#         features.append(vec)
#         # Assign category based on pixel location
#         if (175 <= x < 200 and 110 <= y < 125):
#             labels.append(0)  # Category 1
#         elif (150 <= x < 190 and 150 <= y < 215):
#             labels.append(1)  # Category 2
#         else:
#             labels.append(-1)  # Ignore/other
# features = np.stack(features)  # shape: (image_size*image_size, output_channels)
# labels = np.array(labels)

# # Filter out ignored pixels (label == -1)
# mask = labels != -1
# features = features[mask]
# labels = labels[mask]

# # Define a simple neural network
# class SimpleClassifier(nn.Module):
#     def __init__(self, in_dim, num_classes):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_classes)
#         )
#     def forward(self, x):
#         return self.net(x)

# net = SimpleClassifier(output_channels, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=1e-3)

# # Prepare data for training
# X = torch.tensor(features, dtype=torch.float32).to(device)
# y = torch.tensor(labels, dtype=torch.long).to(device)

# # Training loop (very simple, for demonstration)
# epochs = 1000
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


# # After training, run the classifier on all pixels in the first image and visualize the results
# all_features = []
# for x in range(image_size):
#     for y in range(image_size):
#         vec = output[:, 0, x, y]
#         all_features.append(vec)
# all_features = np.stack(all_features)
# all_X = torch.tensor(all_features, dtype=torch.float32).to(device)
# with torch.no_grad():
#     all_logits = net(all_X)
#     all_preds = all_logits.argmax(dim=1).cpu().numpy()

# # Reshape predictions to image
# pred_img = np.full((image_size, image_size), -1, dtype=int)
# idx = 0
# for x in range(image_size):
#     for y in range(image_size):
#         pred_img[x, y] = all_preds[idx]
#         idx += 1



# # Overlay the initial labeling and predicted label map with some opacity on both images in the batch
# for batch_idx in range(batch_size):
#     # Build initial label mask for this image (same logic as training, but for all pixels)
#     initial_label_mask = np.full((image_size, image_size), -1, dtype=int)
#     for x in range(image_size):
#         for y in range(image_size):
#             if (175 <= x < 200 and 110 <= y < 125):
#                 initial_label_mask[x, y] = 0
#             elif (150 <= x < 190 and 150 <= y < 215):
#                 initial_label_mask[x, y] = 1
#     # Overlay initial labeling
#     orig_data = data[batch_idx]
#     orig_norm = (orig_data - orig_data.min()) / (orig_data.max() - orig_data.min() + 1e-8)
#     plt.figure(figsize=(8, 8))
#     plt.imshow(orig_norm, cmap='gray')
#     plt.imshow(initial_label_mask, cmap='tab10', vmin=-1, vmax=1, alpha=0.4)
#     plt.title(f'Initial Labeling Overlay (image {batch_idx})')
#     plt.colorbar(ticks=[0, 1], label='Category')
#     plt.axis('off')
#     plt.show()

#     # Run classifier on all pixels in this image
#     all_features = []
#     for x in range(image_size):
#         for y in range(image_size):
#             vec = output[:, batch_idx, x, y]
#             all_features.append(vec)
#     all_features = np.stack(all_features)
#     all_X = torch.tensor(all_features, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         all_logits = net(all_X)
#         all_preds = all_logits.argmax(dim=1).cpu().numpy()
#     pred_img = np.full((image_size, image_size), -1, dtype=int)
#     idx = 0
#     for x in range(image_size):
#         for y in range(image_size):
#             pred_img[x, y] = all_preds[idx]
#             idx += 1
#     # Overlay prediction
#     plt.figure(figsize=(8, 8))
#     plt.imshow(orig_norm, cmap='gray')
#     plt.imshow(pred_img, cmap='tab10', vmin=-1, vmax=1, alpha=0.4)
#     plt.title(f'Predicted Categories Overlay (image {batch_idx})')
#     plt.colorbar(ticks=[-1, 1], label='Category')
#     plt.axis('off')
#     plt.show()

# %%
