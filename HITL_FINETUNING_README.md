# Human-in-the-Loop Finetuning for CellMap-Flow

## Overview

This implements a complete LoRA-based finetuning pipeline for CellMap-Flow models using user corrections as training data.

## Features

- ✅ **Lightweight**: Only 0.4-0.8% of parameters trained with LoRA
- ✅ **Fast**: 2-4 hours to finetune vs days for full retraining
- ✅ **Memory Efficient**: FP16 mixed precision, gradient accumulation, patch-based training
- ✅ **Generic**: Works with any PyTorch model (UNets, CNNs, etc.)
- ✅ **Production Ready**: Checkpointing, resume, error handling, logging

## Quick Start

### 1. Install Dependencies

```bash
pip install 'peft>=0.7.0' 'transformers>=4.35.0' 'accelerate>=0.20.0'
```

Or install the finetune extras:
```bash
pip install -e ".[finetune]"
```

### 2. Generate Test Corrections

```bash
python scripts/generate_test_corrections.py \
    --num-corrections 50 \
    --roi-shape 64 64 64 \
    --output test_corrections.zarr
```

### 3. Run Finetuning

```bash
python -m cellmap_flow.finetune.cli \
    --model-checkpoint /path/to/checkpoint \
    --corrections test_corrections.zarr \
    --output-dir output/fly_organelles_v1.1 \
    --lora-r 8 \
    --num-epochs 10 \
    --batch-size 2
```

### 4. Use Finetuned Model

```python
from cellmap_flow.finetune import load_lora_adapter
from cellmap_flow.models.models_config import FlyModelConfig

# Load base model
model_config = FlyModelConfig(
    checkpoint_path="/path/to/base_checkpoint",
    channels=["mito"],
    input_voxel_size=(16, 16, 16),
    output_voxel_size=(16, 16, 16),
)
base_model = model_config.config.model

# Load LoRA adapter
finetuned_model = load_lora_adapter(
    base_model,
    "output/fly_organelles_v1.1/lora_adapter",
    is_trainable=False  # For inference
)

# Use for inference
finetuned_model.eval()
pred = finetuned_model(raw_input)
```

## Architecture

### Components

1. **Test Data Generation** (`scripts/generate_test_corrections.py`)
   - Runs inference on random ROIs
   - Creates synthetic corrections (erosion, dilation, thresholding, etc.)
   - Stores in Zarr format: `corrections.zarr/<uuid>/{raw, mask, prediction}/`

2. **LoRA Wrapper** (`cellmap_flow/finetune/lora_wrapper.py`)
   - Auto-detects adaptable layers (Conv/Linear)
   - Wraps models with HuggingFace PEFT LoRA adapters
   - Saves/loads adapters separately from base model

3. **Dataset** (`cellmap_flow/finetune/dataset.py`)
   - Loads corrections from Zarr
   - 3D augmentation (flips, rotations, intensity, noise)
   - Efficient DataLoader with persistent workers

4. **Trainer** (`cellmap_flow/finetune/trainer.py`)
   - FP16 mixed precision training
   - Gradient accumulation (simulate larger batches)
   - DiceLoss / BCE / Combined loss
   - Automatic checkpointing

5. **CLI** (`cellmap_flow/finetune/cli.py`)
   - Command-line interface for training
   - Supports fly_organelles and DaCaPo models
   - Configurable hyperparameters

## Data Format

### Corrections Storage

```
corrections.zarr/
└── <correction_uuid>/
    ├── raw/s0/data          # Original EM data (uint8)
    ├── prediction/s0/data    # Model prediction (uint8)
    ├── mask/s0/data         # Corrected mask (uint8)
    └── .zattrs              # Metadata
        ├── correction_id
        ├── model_name
        ├── dataset_path
        ├── roi_offset       # [z, y, x]
        ├── roi_shape        # [dz, dy, dx]
        └── voxel_size       # [16, 16, 16]
```

### LoRA Adapter Output

```
output/fly_organelles_v1.1/
├── lora_adapter/
│   ├── adapter_config.json      # LoRA configuration
│   └── adapter_model.bin        # Adapter weights (~10 MB)
├── best_checkpoint.pth          # Best model checkpoint
├── checkpoint_epoch_5.pth       # Periodic checkpoints
└── checkpoint_epoch_10.pth
```

## Training Configuration

### Memory Requirements

| Patch Size | Batch Size | GPU Memory | Training Time (10 epochs) |
|------------|------------|------------|---------------------------|
| 64³        | 2          | ~10 GB     | ~1-2 hours                |
| 96³        | 2          | ~16 GB     | ~2-3 hours                |
| 128³       | 1          | ~20 GB     | ~3-4 hours                |

### Recommended Settings

**For quick iteration (testing)**:
```bash
--lora-r 4 \
--num-epochs 5 \
--batch-size 4 \
--patch-shape 48 48 48
```

**For production (best results)**:
```bash
--lora-r 8 \
--num-epochs 20 \
--batch-size 2 \
--patch-shape 64 64 64 \
--gradient-accumulation-steps 4
```

**For large models (memory constrained)**:
```bash
--lora-r 8 \
--num-epochs 10 \
--batch-size 1 \
--patch-shape 64 64 64 \
--gradient-accumulation-steps 8 \
--no-mixed-precision  # Disable FP16 if causing issues
```

## LoRA Parameters

### Rank (r)

Controls adapter capacity:
- **r=4**: Minimal params (1.6M), fast, may underfit
- **r=8**: Balanced (3.2M), recommended default
- **r=16**: High capacity (6.5M), slower, may overfit on small datasets

### Alpha

Controls scaling of LoRA updates:
- Typically set to `2*r` (e.g., alpha=16 for r=8)
- Higher alpha = stronger LoRA influence
- Lower alpha = more conservative updates

### Dropout

Regularization for LoRA layers:
- **0.0**: No dropout (default, good for small datasets)
- **0.1-0.2**: Light regularization
- **0.3-0.5**: Heavy regularization (for large datasets)

## Loss Functions

### Dice Loss
- Best for **sparse targets** (e.g., mitochondria, small organelles)
- Optimizes overlap between prediction and ground truth
- Less sensitive to class imbalance

### BCE Loss
- Good for **dense targets** or balanced datasets
- Pixel-wise binary cross-entropy
- Faster convergence in some cases

### Combined Loss (Recommended)
- Uses both Dice and BCE (50/50 weight by default)
- Best of both worlds: good overlap + pixel accuracy
- More stable training

## Advanced Usage

### Resume Training

```bash
python -m cellmap_flow.finetune.cli \
    --model-checkpoint /path/to/checkpoint \
    --corrections corrections.zarr \
    --output-dir output/model_v1.1 \
    --resume output/model_v1.1/checkpoint_epoch_5.pth
```

### Custom Loss Weights

```python
from cellmap_flow.finetune import LoRAFinetuner, CombinedLoss

# Create custom loss with different weights
criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)

trainer = LoRAFinetuner(
    model, dataloader, output_dir,
    loss_type="combined"  # Will use default weights
)
# Or replace with custom:
trainer.criterion = criterion
```

### Filter Corrections by Model

```python
from cellmap_flow.finetune import create_dataloader

# Only load corrections for specific model
dataloader = create_dataloader(
    "corrections.zarr",
    model_name="fly_organelles_mito",  # Filter by model name
    batch_size=2
)
```

## Validation Scripts

### Test LoRA Wrapper
```bash
python scripts/test_lora_wrapper.py
```
Expected output:
- Detects 18 layers in fly_organelles UNet
- Shows trainable params: 3.2M (0.41%) for r=8

### Test Dataset
```bash
python scripts/test_dataset.py
```
Expected output:
- Loads corrections from Zarr
- Shows augmentation working (samples differ)
- Creates batches: [2, 1, 64, 64, 64]

### Test End-to-End
```bash
python scripts/test_end_to_end_finetuning.py
```
Expected output:
- Trains for 3 epochs
- Saves LoRA adapter
- Loads adapter and tests inference

## Performance Tips

1. **Use FP16**: Halves memory usage, ~30% faster
2. **Gradient Accumulation**: Simulate larger batches without more memory
3. **Persistent Workers**: `num_workers > 0` with `persistent_workers=True`
4. **Pin Memory**: Faster GPU transfers
5. **Patch-based**: Use smaller patches (64³) for memory efficiency

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (try 1)
- Reduce `--patch-shape` (try 48 48 48)
- Increase `--gradient-accumulation-steps` (try 8)
- Disable FP16: `--no-mixed-precision`

### Training Unstable

- Lower `--learning-rate` (try 5e-5)
- Use `--loss-type combined`
- Increase `--lora-dropout` (try 0.1)

### Poor Results

- Increase `--lora-r` (try 16)
- Increase `--num-epochs` (try 20)
- Check correction quality: `python scripts/inspect_corrections.py`
- Ensure sufficient corrections (50+ recommended)

## Next Steps

### Browser Integration (Future Work)

1. **Correction Capture**:
   - Add annotation layer to Neuroglancer viewer
   - Implement `/api/corrections/submit` endpoint
   - Store corrections to Zarr automatically

2. **Auto-trigger**:
   - Background daemon monitors correction count
   - Auto-submits LSF finetuning job when threshold met
   - Notifies user when finetuned model ready

3. **A/B Testing**:
   - Load base + finetuned models side-by-side in Neuroglancer
   - User compares and votes
   - System tracks which model performs better

4. **Active Learning**:
   - Model suggests regions where it's uncertain
   - User prioritizes corrections on hard cases
   - Improves efficiency of human corrections

## References

- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **PEFT Library**: [HuggingFace PEFT](https://github.com/huggingface/peft)
- **Dice Loss**: [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
