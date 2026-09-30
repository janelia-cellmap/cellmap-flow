"""
Human-in-the-loop finetuning for CellMap-Flow models.

This package provides lightweight LoRA-based finetuning for pre-trained models
using user corrections as training data.

The names below are imported when first used (PEP 562): importing a light
submodule -- ``markers`` or ``job_log``, which the dashboard reads a
training job's log with -- used to import torch, the trainer and the
dataset code first.
"""

import importlib

# name -> the submodule it lives in
_EXPORTS = {
    "detect_adaptable_layers": "lora_wrapper",
    "wrap_model_with_lora": "lora_wrapper",
    "print_lora_parameters": "lora_wrapper",
    "load_lora_adapter": "lora_wrapper",
    "save_lora_adapter": "lora_wrapper",
    "VirtualPatchDataset": "virtual_dataset",
    "create_dataloader": "virtual_dataset",
    "LoRAFinetuner": "lora_trainer",
    "DiceLoss": "losses",
    "CombinedLoss": "losses",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
