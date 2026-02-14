# LoRA Finetuning Implementation - Changes and Fixes

This document tracks all changes made to implement and fix LoRA finetuning for the fly_organelles model with mito channel selection.

## Problem Summary

The fly_organelles model outputs 8 channels (all_mem, organelle, mito, er, nucleus, pm, vesicle, ld), but we only want to finetune on the mito channel (index 2). Initial attempts to wrap the model with a `ChannelSelector` caused PEFT compatibility issues.

## Solution

Instead of wrapping the model before LoRA, we select the channel **after** the forward pass but **before** computing the loss in the trainer. This avoids PEFT introspection issues.

---

## Files Modified

### 1. `cellmap_flow/finetune/trainer.py`

**Changes:**
- Added `select_channel` parameter to `LoRAFinetuner.__init__`
- Modified `_train_epoch()` to select channel from predictions before loss computation

**Code added:**
```python
# In __init__:
select_channel: Optional[int] = None,

# In _train_epoch:
# Select specific channel if requested (e.g., mito = channel 2 from 8-channel output)
if self.select_channel is not None:
    pred = pred[:, self.select_channel:self.select_channel+1, :, :, :]
```

**Why:**
- Allows trainer to handle multi-channel models
- Channel selection happens after forward pass, avoiding PEFT compatibility issues
- Clean separation of concerns: model outputs all channels, trainer selects what's needed

---

### 2. `cellmap_flow/finetune/cli.py`

**Changes:**
- Removed `ChannelSelector` wrapper class (PEFT incompatible)
- Added logic to set `select_channel=2` when `channels=["mito"]`
- Pass `select_channel` to trainer initialization

**Code removed:**
```python
# OLD: ChannelSelector wrapper (caused PEFT errors)
class ChannelSelector(nn.Module):
    def __init__(self, model, channel_idx):
        super().__init__()
        self.model = model
        self.channel_idx = channel_idx

    def forward(self, x):
        output = self.model(x)
        return output[:, self.channel_idx:self.channel_idx+1, :, :, :]

base_model = ChannelSelector(base_model, channel_idx=2)
```

**Code added:**
```python
# NEW: Simple channel selection in trainer
select_channel = None
if args.channels == ["mito"]:
    select_channel = 2
    logger.info("Will select mito channel (index 2) from model output during training")

# Pass to trainer
trainer = LoRAFinetuner(
    ...
    select_channel=select_channel,
)
```

**Why:**
- PEFT library had issues with custom wrapper modules
- Errors included:
  - `TypeError: forward() got an unexpected keyword argument 'input_ids'`
  - `TypeError: forward() missing 1 required positional argument: 'x'`
- PEFT is designed for transformers and passes transformer-specific arguments
- Wrapping before LoRA caused introspection and calling convention issues

---

### 3. `cellmap_flow/models/models_config.py`

**Changes:**
- Modified `load_eval_model()` to handle `model.pt` files (full Sequential models)
- Added special case for `.pt` files to load with `weights_only=False`

**Code added:**
```python
elif checkpoint_path.endswith("model.pt"):
    # Load full model directly (for trusted fly_organelles models)
    model = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.to(device)
    model.eval()
    return model
```

**Why:**
- fly_organelles `model.pt` files contain full `Sequential(UNet, Sigmoid)` models
- Not state dicts like typical checkpoints
- Need `weights_only=False` to unpickle the full model structure

---

### 4. `.gitignore`

**Changes:**
- Added project-specific directories to ignore

**Lines added:**
```
corrections/
output/
```

**Why:**
- Corrections zarr files are large and dataset-specific
- Output directories contain training checkpoints and adapters
- These shouldn't be committed to version control

---

## Files Created

### 1. `generate_mito_corrections.py`

**Purpose:**
Generate correction zarrs from mito segmentations for training data.

**Key features:**
- Loads raw EM from `jrc_mus-liver-zon-1` at s1 (16nm resolution)
- Loads mito segmentations from same dataset
- Creates 10 random crops with:
  - Raw: 178³ voxels
  - Mask: 56³ voxels (center crop)
- Applies 5 iterations of binary erosion to mito masks
- Runs fly_organelles_run08_438000 model to generate predictions
- Saves in OME-NGFF v0.4 format with proper metadata

**Erosion strategy:**
- Crops full 178³ region from segmentation
- Applies erosion to full crop (no edge artifacts)
- Extracts center 56³ after erosion
- Ensures mito fraction > 10% pre-erosion, > 5% post-erosion

**Output format:**
```
corrections/mito_liver.zarr/
└── <uuid>/
    ├── raw/s0              # 178³ uint8, no translation
    ├── mask/s0             # 56³ uint8, translation [976, 976, 976] nm
    └── prediction/s0       # 56³ float32, translation [976, 976, 976] nm
```

**Usage:**
```bash
python generate_mito_corrections.py
```

---

### 2. `compare_finetuned_predictions.py`

**Purpose:**
Compare predictions before and after LoRA finetuning.

**Key features:**
- Loads base model and LoRA adapter
- For each correction:
  - Loads raw data
  - Runs through finetuned model
  - Saves as `prediction_finetuned/s0`
  - Prints comparison stats (mean/max difference)

**Output:**
Adds `prediction_finetuned/s0` to each correction group for side-by-side comparison in Neuroglancer.

**Usage:**
```bash
python compare_finetuned_predictions.py \
    --corrections corrections/mito_liver.zarr \
    --lora-adapter output/fly_organelles_mito_liver/lora_adapter \
    --model-checkpoint /nrs/cellmap/models/saalfeldlab/fly_organelles_run08_438000/model.pt \
    --channels mito \
    --input-voxel-size 16 16 16 \
    --output-voxel-size 16 16 16
```

---

## Training Configuration

### Default Settings

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| Mixed precision | `True` | FP16 enabled by default |
| LoRA rank | 8 | Can adjust with `--lora-r` |
| LoRA alpha | 16 | Can adjust with `--lora-alpha` |
| Batch size | 2 | Can adjust with `--batch-size` |
| Learning rate | 1e-4 | Can adjust with `--learning-rate` |
| Gradient accumulation | 4 | Can adjust with `--gradient-accumulation-steps` |
| Loss type | `combined` | Dice + BCE, can use `--loss-type` |

### Memory Usage

With FP16 enabled:
- Batch size 2: ~10-12 GB GPU memory
- Batch size 1: ~6-8 GB GPU memory

Disable FP16 if needed:
```bash
--no-mixed-precision
```

---

## Complete Training Workflow

### 1. Generate Corrections

```bash
python generate_mito_corrections.py
```

Creates `corrections/mito_liver.zarr` with 10 corrections.

### 2. Run Finetuning

```bash
python -m cellmap_flow.finetune.cli \
    --model-checkpoint /nrs/cellmap/models/saalfeldlab/fly_organelles_run08_438000/model.pt \
    --corrections corrections/mito_liver.zarr \
    --output-dir output/fly_organelles_mito_liver \
    --channels mito \
    --input-voxel-size 16 16 16 \
    --output-voxel-size 16 16 16 \
    --lora-r 4 \
    --lora-alpha 8 \
    --num-epochs 15 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --loss-type combined \
    --lora-dropout 0.1
```

Creates:
- `output/fly_organelles_mito_liver/lora_adapter/` - LoRA weights
- `output/fly_organelles_mito_liver/checkpoint_epoch_*.pth` - Checkpoints
- `output/fly_organelles_mito_liver/best_checkpoint.pth` - Best model

### 3. Compare Predictions

```bash
python compare_finetuned_predictions.py \
    --corrections corrections/mito_liver.zarr \
    --lora-adapter output/fly_organelles_mito_liver/lora_adapter \
    --model-checkpoint /nrs/cellmap/models/saalfeldlab/fly_organelles_run08_438000/model.pt \
    --channels mito \
    --input-voxel-size 16 16 16 \
    --output-voxel-size 16 16 16
```

Adds `prediction_finetuned/s0` to corrections for comparison.

### 4. Visualize in Neuroglancer

Open `corrections/mito_liver.zarr` and compare:
- `raw/s0` - Original EM data
- `prediction/s0` - Base model predictions
- `prediction_finetuned/s0` - Finetuned predictions
- `mask/s0` - Ground truth (eroded) labels

---

## Technical Details

### Channel Selection Logic

**Why channel 2?**

From fly_organelles model metadata:
```python
classes = ["all_mem", "organelle", "mito", "er", "nuc", "pm", "vesicle", "ld"]
# Index:     0          1           2      3     4      5     6         7
```

Mito is at index 2.

**How it works:**

1. Model outputs shape: `(B, 8, Z, Y, X)`
2. After forward pass in trainer:
   ```python
   if self.select_channel is not None:
       pred = pred[:, 2:3, :, :, :]  # (B, 1, Z, Y, X)
   ```
3. Loss computed on single-channel prediction vs single-channel target

### Input/Output Normalization

**Input (raw EM):**
- Storage: uint8 [0, 255]
- Model input: float32 [-1, 1]
- Normalization: `(x / 127.5) - 1.0`

**Output (predictions/masks):**
- Storage: float32 [0, 1]
- Model output: float32 [0, 1] (after Sigmoid)
- No additional normalization needed

### OME-NGFF Metadata

All arrays include proper OME-NGFF v0.4 metadata:

```python
{
  'multiscales': [{
    'version': '0.4',
    'name': 'raw',
    'axes': [
      {'name': 'z', 'type': 'space', 'unit': 'nanometer'},
      {'name': 'y', 'type': 'space', 'unit': 'nanometer'},
      {'name': 'x', 'type': 'space', 'unit': 'nanometer'}
    ],
    'datasets': [{
      'path': 's0',
      'coordinateTransformations': [
        {'type': 'scale', 'scale': [16, 16, 16]},
        {'type': 'translation', 'translation': [976, 976, 976]}  # For mask/prediction
      ]
    }]
  }]
}
```

Translation offset: `[61, 61, 61] voxels × 16 nm/voxel = [976, 976, 976] nm`

---

## Troubleshooting

### Training Issues

**Error: Out of memory**
```bash
--batch-size 1 \
--gradient-accumulation-steps 8
```

**Error: NaN loss**
```bash
--learning-rate 5e-5 \
--no-mixed-precision
```

**Error: Model loading failed**
- Ensure using `model.pt` not state dict checkpoint
- Check `weights_only=False` for model.pt files

### PEFT Compatibility

**Previous errors (now fixed):**
- ❌ `forward() got an unexpected keyword argument 'input_ids'`
- ❌ `forward() missing 1 required positional argument: 'x'`

**Solution:**
- Don't wrap model with custom modules before LoRA
- Use `select_channel` parameter in trainer instead

---

## Recent Changes

### Auto-Serve Finetuned Model After Training

After training completes, the CLI automatically starts an inference server on the same GPU and prints a `CELLMAP_FLOW_SERVER_IP` marker. The dashboard's job monitor detects this marker and adds the finetuned model as a Neuroglancer layer.

**Files changed:**
- `cellmap_flow/finetune/cli.py`: Added `--auto-serve` and `--serve-data-path` flags; starts `CellMapFlowServer` in a daemon thread after training
- `cellmap_flow/finetune/job_manager.py`: Added `_parse_inference_server_ready()` and `_add_finetuned_neuroglancer_layer()` to detect server startup and add layers
- `cellmap_flow/dashboard/app.py`: Added `/api/finetune/job/<id>/inference-server` status endpoint
- `cellmap_flow/dashboard/templates/_finetune_tab.html`: Added "Auto-load model after training" checkbox and inference server status display

### Iterative Training (Restart on Same GPU)

Users can restart training after completion without needing a new GPU allocation. The dashboard writes a `restart_signal.json` file, which the CLI detects in a polling loop.

**Files changed:**
- `cellmap_flow/finetune/cli.py`: Added restart signal polling loop with `_wait_for_restart_signal()`; retrains with updated parameters
- `cellmap_flow/finetune/job_manager.py`: Added `restart_finetuning_job()`, `_archive_job_logs()`, and `_parse_training_restart()` for restart orchestration
- `cellmap_flow/dashboard/app.py`: Added `/api/finetune/job/<id>/restart` endpoint
- `cellmap_flow/dashboard/templates/_finetune_tab.html`: Added "Restart Training" button and modal with parameter override UI

### Log Stream Filtering

Noisy debug and werkzeug lines are filtered from the training log stream displayed in the dashboard.

**Files changed:**
- `cellmap_flow/dashboard/app.py`: Added regex-based line filtering in `stream_job_logs()` SSE endpoint

### Model File Generation

After training, model script (`.py`) and config (`.yaml`) files are automatically generated so the finetuned model can be loaded independently.

**Files created:**
- `cellmap_flow/finetune/model_templates.py`: Templates for generating model scripts and YAML configs

## Future Improvements

1. **Active Learning:**
   - Model suggests uncertain regions
   - User prioritizes corrections on hard cases

2. **Validation Set:**
   - Split corrections into train/val
   - Track validation metrics during training

3. **Multi-channel Finetuning:**
   - Extend to finetune multiple channels simultaneously
   - Joint optimization across organelles

---

## References

- Main README: [HITL_FINETUNING_README.md](HITL_FINETUNING_README.md)
- LoRA Paper: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- PEFT Library: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- OME-NGFF Spec: [https://ngff.openmicroscopy.org/latest/](https://ngff.openmicroscopy.org/latest/)
