"""
Human-in-the-loop finetuning for CellMap-Flow models.

This package provides lightweight LoRA-based finetuning for pre-trained models
using user corrections as training data.
"""

from cellmap_flow.finetune.lora_wrapper import (
    detect_adaptable_layers,
    wrap_model_with_lora,
    print_lora_parameters,
    load_lora_adapter,
    save_lora_adapter,
)

from cellmap_flow.finetune.dataset import (
    CorrectionDataset,
    create_dataloader,
)

__all__ = [
    "detect_adaptable_layers",
    "wrap_model_with_lora",
    "print_lora_parameters",
    "load_lora_adapter",
    "save_lora_adapter",
    "CorrectionDataset",
    "create_dataloader",
]
