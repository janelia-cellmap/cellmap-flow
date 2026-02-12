# Annotation Workflows for LoRA Finetuning

This document describes the available annotation workflows for LoRA model finetuning in CellMapFlow.

## Overview

CellMapFlow provides two complementary annotation workflows:

1. **Dashboard-Based Interactive Annotation** (Recommended for dense corrections)
   - Create annotation crops directly from the Neuroglancer viewer
   - Edit annotations interactively in the browser
   - Automatic bidirectional syncing between viewer and local disk
   - Ideal for correcting specific model errors with dense labels

2. **Sparse Point Annotation System** (For partial volume labeling)
   - Label only a subset of your volume by placing annotations at specific points
   - Avoid ambiguous unlabeled regions using a 3-level labeling scheme
   - Train efficiently by only computing loss on annotated regions
   - Ideal for large-scale annotation with focused corrections

---

# Workflow 1: Dashboard-Based Interactive Annotation

## Overview

The dashboard provides an integrated workflow for creating and editing annotation crops directly from the Neuroglancer viewer. Annotations are automatically synced between the browser (via MinIO) and your local filesystem, enabling a seamless annotation-to-training pipeline.

## Features

- **One-Click Crop Creation:** Create annotation crops at your current view position
- **Interactive Editing:** Edit annotations directly in Neuroglancer browser
- **Automatic Syncing:** Background sync keeps local disk updated with browser edits
- **Manual Save:** Force-sync all annotations to disk before training
- **Model-Aware Sizing:** Crops are automatically sized to match your model's output shape

## Step-by-Step Guide

### Step 1: Start the Dashboard

Launch the dashboard with your dataset and models:

```bash
cellmap_flow_app
```

Navigate to the Finetune tab in the web interface.

### Step 2: Select Your Model

1. Choose the model you want to finetune from the dropdown
   - The crop will be automatically sized to the model's output inference shape
   - Models must be configured with `write_shape`, `output_voxel_size`, and `output_channels`
2. If you don't see your model, click the refresh button (↻)

**Note:** Models submitted from the GUI after app startup may need a restart with proper YAML configuration.

### Step 3: Configure Output Path

Specify where to save annotation crops:

```
/path/to/output/corrections
```

This path:
- Will store all crop zarr files (e.g., `5d291ea8-20260212-132326.zarr`)
- Must be accessible for training later
- Will have a `.minio` subdirectory for MinIO storage
- Is saved to localStorage for convenience

### Step 4: Navigate in Neuroglancer

Position your view at the location where you want to create an annotation crop. The crop will be created at the **current view center position** automatically.

### Step 5: Create Annotation Crop

Click **"Create Annotation Crop"**

The system will:
1. Auto-detect your current view center and coordinate scales
2. Extract raw data at the model's input size
3. Run model inference to generate a prediction
4. Create a zarr file with:
   - `raw/s0`: Input image data
   - `annotation/s0`: Empty annotation array (for you to edit)
   - `prediction/s0`: Model's current prediction
   - Metadata with crop ID, timestamps, coordinates
5. Upload to MinIO server for browser access
6. Add an editable segmentation layer to Neuroglancer

**Output Example:**
```
✓ Created crop: 5d291ea8-20260212-132326
  Center (nm): [125430.0, 89234.5, 102938.0]
  Zarr path: /path/to/output/corrections/5d291ea8-20260212-132326.zarr
  MinIO URL: http://192.168.1.100:9000/annotations/5d291ea8-20260212-132326.zarr
```

### Step 6: Edit Annotations in Neuroglancer

The new annotation layer (`annotation_<crop_id>`) is now available in your viewer:

1. Select the annotation layer
2. Use Neuroglancer's segmentation tools to paint corrections:
   - **Paint:** Add foreground annotations (label 1)
   - **Erase:** Mark as background (label 0)
   - **Fill:** Fill regions
3. Edits are automatically saved to MinIO

**Annotation Guidelines:**
- Mark model **false positives** with background (0)
- Mark model **false negatives** with foreground (1)
- Focus on boundary corrections and clear errors

### Step 7: Sync Annotations to Disk

Before training, ensure all browser edits are saved locally:

**Option A - Automatic Background Sync:**
- Annotations auto-sync every 30 seconds
- Only modified crops are synced
- Runs in background thread

**Option B - Manual Force Sync:**
1. Click **"💾 Save Annotations to Disk"**
2. All crops are synced immediately
3. Check the log for sync confirmation:
   ```
   ✓ Synced 3 annotations
     Synced 3 / 5 crops
   ```

### Step 8: Train with Annotation Crops

Once annotations are synced, use them for finetuning:

```bash
cellmap_flow_finetune \
  --model-name fly_organelles_mito \
  --corrections /path/to/output/corrections \
  --output-dir output/finetuned_model \
  --batch-size 1 \
  --num-epochs 10 \
  --learning-rate 1e-4
```

**Key Training Parameters:**
- `--corrections`: Path to the directory containing your crop zarr files
- `mask_unannotated=False`: Dashboard annotations are dense (fully labeled)
- `normalize=False`: Dashboard corrections are already normalized

The trainer will automatically:
- Discover all crops in the corrections directory
- Load `raw` and `annotation` arrays from each crop
- Train only on your corrections using LoRA

## Architecture Details

### Data Flow

```
Neuroglancer Browser
       ↕ (writes via S3 protocol)
    MinIO Server
       ↕ (background sync every 30s)
  Local Filesystem
       ↕ (training reads)
   LoRA Finetuner
```

### Zarr Structure

Each crop creates a zarr file with this structure:

```
5d291ea8-20260212-132326.zarr/
├── raw/
│   └── s0/              # Input EM data (model input size)
├── annotation/
│   └── s0/              # Your corrections (model output size)
├── prediction/
│   └── s0/              # Model's original prediction
└── .zattrs              # Metadata (crop_id, timestamp, coordinates)
```

### MinIO Integration

- **Storage Location:** `<output_path>/.minio/annotations/`
- **Access:** Read/write via S3 protocol at `http://<ip>:9000`
- **Credentials:** Default `minio/minio123` (local only)
- **Bucket:** `annotations` (auto-created, public read/write)

### Sync Behavior

**Background Sync:**
- Checks modification timestamps
- Only syncs changed annotations
- Tracks last sync time per crop
- Runs continuously in daemon thread

**Manual Sync:**
- Forces sync of all crops
- Ignores modification timestamps
- Useful before starting training

## Troubleshooting

### Problem: "No models available for finetuning"

**Cause:** Models need full configuration metadata (write_shape, output_voxel_size, output_channels)

**Solution:**
1. Ensure models are loaded from proper YAML configuration
2. Click the refresh button (↻) after submitting models
3. Restart the app with models configured in YAML if needed

---

### Problem: MinIO not accessible

**Cause:** Firewall or network configuration

**Solution:**
1. Check MinIO is running: Look for MinIO status in the log
2. Verify port is open (default: 9000)
3. Check IP address is reachable from browser

---

### Problem: Annotations not syncing to disk

**Cause:** Background sync may not have run yet

**Solution:**
1. Wait 30 seconds for automatic sync
2. Or click "💾 Save Annotations to Disk" to force sync
3. Check the log for sync messages

---

### Problem: Crop created at wrong location

**Cause:** View center was not at intended location

**Solution:**
- The crop is created at the **current view center** (where your cursor is)
- Navigate to the exact location before clicking "Create Annotation Crop"
- Check the logged center position in nanometers

---

# Workflow 2: Sparse Point Annotation System

## Overview

The sparse annotation system allows you to:
- **Label only a subset of your volume** by placing annotations at specific points
- **Avoid ambiguous unlabeled regions** by using a 3-level labeling scheme
- **Train efficiently** by only computing loss on annotated regions

## Label Scheme

The system uses a 3-level label scheme:

| Label | Meaning | Training Behavior |
|-------|---------|-------------------|
| **0** | Unannotated | **Ignored** in loss calculation |
| **1** | Background | **Included** as class 0 (background) |
| **2** | Foreground (e.g., mito) | **Included** as class 1 (foreground) |

During training with `mask_unannotated=True`:
1. A mask is created where `label > 0` (annotated regions)
2. Labels are shifted down by 1: `1→0`, `2→1`
3. Loss is only computed on the masked (annotated) regions

## Workflow

### Step 1: Generate Sparse Point Corrections

This script samples random points from eroded mito regions (foreground) and surrounding background space, creating sparse spherical annotations around each point. All corrections are written to a single timestamped zarr file.

```bash
python scripts/generate_sparse_corrections.py
```

**Configuration (edit in script):**
```python
NUM_FOREGROUND_POINTS = 1000  # Points to sample from mito
NUM_BACKGROUND_POINTS = 1000  # Points to sample from background
ANNOTATION_RADIUS = 3         # Radius of annotation sphere (voxels)
MIN_POINT_DISTANCE = 5        # Minimum spacing between points
```

**Output:**
- Creates a single zarr with timestamp: `sparse_corrections_YYYYMMDD_HHMMSS.zarr`
- Contains all corrections as separate groups
- Each correction has:
  - `raw/s0`: Input image (178×178×178)
  - `mask/s0`: Sparse annotation mask (56×56×56) with labels 0/1/2
  - `prediction/s0`: Model prediction (56×56×56)
  - Metadata: timestamp, point counts, annotation fraction

**Example output:**
```
Processing correction abc123...
  Sampled 856 foreground points
  Sampled 943 background points
  Annotated voxels: 45,328 (2.58%)
    - Foreground (2): 23,156
    - Background (1): 22,172
  ✓ Added as correction 1

✓ Complete! Created 10 sparse corrections
Output: corrections/sparse_corrections_20260211_143022.zarr
```

### Step 2: Train with Masked Loss

Train the model using only the annotated regions:

```bash
python scripts/example_sparse_annotation_workflow.py
```

Or use in your own code:

```python
from cellmap_flow.finetune.trainer import LoRAFinetuner

trainer = LoRAFinetuner(
    lora_model,
    dataloader,
    output_dir="output/sparse_finetuning",
    learning_rate=1e-4,
    num_epochs=10,
    loss_type="combined",
    mask_unannotated=True,  # Enable masked loss
)

trainer.train()
trainer.save_adapter()
```

## Key Parameters

### Sparse Point Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_FOREGROUND_POINTS` | 1000 | Points sampled from eroded mito regions |
| `NUM_BACKGROUND_POINTS` | 1000 | Points sampled from background shell |
| `ANNOTATION_RADIUS` | 3 | Radius of annotation sphere (voxels) |
| `MIN_POINT_DISTANCE` | 5 | Minimum spacing between points (voxels) |
| `BACKGROUND_MIN_DIST` | 2 | Min distance from mito for background |
| `BACKGROUND_MAX_DIST` | 10 | Max distance from mito for background |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask_unannotated` | `True` | Only compute loss on labeled regions |
| `loss_type` | `"combined"` | Loss function (dice/bce/combined) |
| `learning_rate` | `1e-4` | Learning rate |
| `num_epochs` | `10` | Training epochs |

## How the Masking Works

When `mask_unannotated=True` in the trainer:

1. **Mask Creation:**
   ```python
   mask = (target > 0).float()  # Binary mask of annotated regions
   ```

2. **Label Shifting:**
   ```python
   target = torch.clamp(target - 1, min=0)  # 0→0, 1→0, 2→1
   ```

3. **Loss Computation:**
   - Dice Loss: Only computes intersection/union over masked regions
   - BCE Loss: Multiplies loss by mask, averages over masked voxels
   - Combined: Applies masking to both components

## Annotation Statistics

Typical annotation coverage with default parameters:
- **~2-5% of volume annotated** (very sparse!)
- **~1,000-2,000 annotated spheres** per correction
- **Training is fast** due to focused loss computation

## Benefits

1. **Partial Labeling:** You don't need to label the entire volume
2. **Unambiguous:** Background is explicitly labeled (not just "not foreground")
3. **Efficient:** Loss computation focuses on annotated regions
4. **Scalable:** Can annotate at specific points of interest
5. **Trackable:** Datetime stamps allow version control of annotations

## Example Use Cases

1. **Correcting Model Errors:**
   - Find regions where model fails
   - Add background annotations where false positives occur
   - Add foreground annotations where false negatives occur

2. **Refining Boundaries:**
   - Add annotations at unclear boundaries
   - Label edge cases the model struggles with

3. **Class Imbalance:**
   - Oversample rare structures by adding more foreground points
   - Balance training by adjusting point ratios

## Files

- `scripts/generate_sparse_corrections.py`: Generate sparse annotations (single zarr output)
- `scripts/example_sparse_annotation_workflow.py`: Complete training example
- `cellmap_flow/finetune/trainer.py`: Trainer with masked loss support

## Sparse Workflow Troubleshooting

**Problem:** Not enough points sampled

**Solution:** Check that MIN_POINT_DISTANCE isn't too large, or increase the sampling region

---

**Problem:** Annotations too sparse/dense

**Solution:** Adjust NUM_FOREGROUND/BACKGROUND_POINTS and ANNOTATION_RADIUS

---

**Problem:** Background points not sampled

**Solution:** Increase BACKGROUND_MAX_DIST or check that mito regions aren't too isolated

---

# Choosing the Right Workflow

## Use Dashboard Workflow When:

✅ You want to **correct specific model errors** visually
✅ You need **dense, high-quality annotations** for small regions
✅ You prefer an **interactive, visual editing** experience
✅ You're working with **<10 correction crops**
✅ You want **fast iteration** between annotation and training

**Example Use Cases:**
- Fixing false positives/negatives in a specific region
- Refining boundary predictions
- Creating gold-standard training examples
- Quick prototyping of model corrections

## Use Sparse Point Workflow When:

✅ You need to annotate **large volumes** efficiently
✅ You want **programmatic control** over annotation placement
✅ You're labeling **thousands of points** across the dataset
✅ You can accept **partial annotations** (not every voxel labeled)
✅ You want to **balance foreground/background** systematically

**Example Use Cases:**
- Labeling at scale across entire datasets
- Systematic sampling of structures
- Class balancing for rare organelles
- Batch correction generation

## Combining Both Workflows

You can use both workflows together:

1. **Generate sparse annotations** programmatically for broad coverage
2. **Create dashboard crops** for specific problem areas
3. **Combine all corrections** into a single training directory
4. **Train once** on the merged dataset

```bash
# Merge corrections from both workflows
mkdir all_corrections/
cp -r sparse_corrections_20260212/*.zarr all_corrections/
cp -r dashboard_corrections/*.zarr all_corrections/

# Train on combined dataset
cellmap_flow_finetune \
  --corrections all_corrections/ \
  --model-name my_model \
  --output-dir output/combined_finetune
```

---

# Related Files

## Dashboard Workflow
- `cellmap_flow/dashboard/app.py`: Dashboard server with MinIO integration
- `cellmap_flow/dashboard/templates/_finetune_tab.html`: Finetune tab UI
- `sync_annotations.py`: Standalone annotation syncing utility
- `sync_all_annotations.sh`: Batch sync script

## Sparse Workflow
- `scripts/generate_sparse_corrections.py`: Sparse annotation generator
- `scripts/example_sparse_annotation_workflow.py`: Training example

## Shared Components
- `cellmap_flow/finetune/cli.py`: Finetuning CLI
- `cellmap_flow/finetune/trainer.py`: LoRA trainer
- `cellmap_flow/finetune/dataset.py`: Correction dataset loader
