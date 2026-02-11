# Sparse Annotation Workflow for LoRA Finetuning

This document describes how to use the sparse point annotation system for partial volume labeling with LoRA finetuning.

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

## Troubleshooting

**Problem:** Not enough points sampled

**Solution:** Check that MIN_POINT_DISTANCE isn't too large, or increase the sampling region

---

**Problem:** Annotations too sparse/dense

**Solution:** Adjust NUM_FOREGROUND/BACKGROUND_POINTS and ANNOTATION_RADIUS

---

**Problem:** Background points not sampled

**Solution:** Increase BACKGROUND_MAX_DIST or check that mito regions aren't too isolated
