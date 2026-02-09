# Human-in-the-Loop Finetuning - Test Data

## Overview

We've created synthetic test corrections to develop and test the finetuning pipeline without needing browser-based correction capture first.

## Generated Files

### Scripts

1. **`scripts/generate_test_corrections.py`** - Generates synthetic corrections
   - Runs inference on random ROIs
   - Applies synthetic corrections (erosion, dilation, thresholding, etc.)
   - Saves in Zarr format

2. **`scripts/inspect_corrections.py`** - Inspects and validates corrections
   - Shows statistics for each correction
   - Can save PNG slices for visualization
   - Validates data quality

3. **`scripts/test_model_inference.py`** - Simple model inference test
   - Verifies model works correctly
   - Useful for debugging

### Data

**`test_corrections.zarr/`** - 20 test corrections in standardized format:
```
test_corrections.zarr/
└── <correction_uuid>/
    ├── raw/s0/data          # Original EM data (uint8, unnormalized)
    ├── prediction/s0/data    # Model prediction (uint8, 0-255)
    ├── mask/s0/data         # Corrected mask (uint8, 0-255)
    └── .zattrs              # Metadata (ROI, model, dataset, voxel_size)
```

## Data Quality

Inspecting the 20 corrections:

- **Raw data**: Proper EM intensities (e.g., [75, 186], [1, 108])
- **Predictions**: Range from all zeros (no mito) to 240/255 (strong mito signal)
- **Corrections**: Synthetic edits using morphological operations
  - `threshold_low`: More permissive threshold (>80)
  - `threshold_high`: Stricter threshold (>180)
  - `erosion`: Remove noise
  - `dilation`: Fill gaps
  - `fill_holes`: Fill internal holes
  - `remove_small`: Remove small objects
  - `open`: Erosion + dilation
  - `close`: Dilation + erosion

Example correction with good data:
```
Correction: be6b9d4a...
  Raw data range: [0, 255]
  Prediction range: [0, 240]
  Corrected mask coverage: 2.20%
  Changed pixels: 18.02%
```

## Usage

### Generate More Corrections

```bash
python scripts/generate_test_corrections.py \
    --num-corrections 50 \
    --roi-shape 56 56 56 \
    --output test_corrections.zarr
```

### Inspect Corrections

```bash
# View statistics
python scripts/inspect_corrections.py \
    --corrections test_corrections.zarr \
    --limit 10

# Save PNG slices
python scripts/inspect_corrections.py \
    --corrections test_corrections.zarr \
    --save-slices \
    --output-dir correction_slices
```

### Test Model Inference

```bash
python scripts/test_model_inference.py
```

## Dataset & Model Info

- **Dataset**: `/nrs/cellmap/data/jrc_mus-salivary-1/jrc_mus-salivary-1.zarr/recon-1/em/fibsem-uint8/s1`
  - Shape: (7443, 6933, 7696) voxels
  - Voxel size: (16, 16, 16) nm
  - Total size: ~350 GB

- **Model**: fly_organelles mitochondria model
  - Checkpoint: `/groups/cellmap/cellmap/zouinkhim/exp_c-elegen/v3/train/runs/20250806_mito_mouse_distance_16nm/model_checkpoint_362000`
  - Architecture: StandardUnet (3D UNet)
  - Input size: (178, 178, 178) voxels
  - Output size: (56, 56, 56) voxels
  - Output: Single channel (mito probability, 0-1)

- **Normalization**:
  - MinMaxNormalizer: [0, 250] → [0, 1]
  - LambdaNormalizer: x*2-1 (maps to [-1, 1])

## Next Steps

Now that we have test correction data, we can build the finetuning pipeline:

1. **Phase 2: LoRA Integration**
   - Create `cellmap_flow/finetune/lora_wrapper.py`
   - Implement generic layer detection
   - Wrap model with LoRA adapters

2. **Phase 3: Training Data Pipeline**
   - Create `cellmap_flow/finetune/dataset.py`
   - Implement PyTorch Dataset that loads from `test_corrections.zarr`
   - Add 3D augmentation

3. **Phase 4: Finetuning Loop**
   - Create `cellmap_flow/finetune/trainer.py`
   - Implement training loop with FP16
   - Create CLI to trigger finetuning

4. **Phase 5: Test End-to-End**
   - Train LoRA adapter on test corrections
   - Verify improved predictions on corrected regions
   - Save and deploy adapter

## Storage Format Rationale

### Why Zarr + UUID?

- **Zarr**: Efficient for 3D volumes, supports compression, OME-NGFF compatible
- **UUID**: Unique IDs prevent collisions, enables distributed correction collection
- **Flat structure**: Easy to iterate, scales to 100K+ corrections

### Why Save Raw + Prediction + Mask?

- **Raw**: Input for training (X)
- **Mask**: Target for training (Y)
- **Prediction**: For analysis, debugging, and active learning (future)

### Metadata in .zattrs

Stores essential info for:
- Filtering corrections by model
- Querying by dataset/ROI
- Tracking voxel sizes for proper data loading
- Future: user ID, timestamp, correction type
