"""
Generic LoRA wrapper for PyTorch models.

This module provides automatic detection of adaptable layers and wraps
PyTorch models with LoRA (Low-Rank Adaptation) adapters using the
HuggingFace PEFT library.

LoRA enables efficient finetuning by training only a small number of
additional parameters (typically 1-2% of the original model) while
keeping the base model frozen.
"""

import logging
from typing import List, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def detect_adaptable_layers(
    model: nn.Module,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Automatically detect layers suitable for LoRA adaptation.

    Searches for Conv2d, Conv3d, and Linear layers, filtering by name patterns.
    By default, excludes batch norm, layer norm, and final output layers.

    Args:
        model: PyTorch model to inspect
        include_patterns: List of regex patterns for layer names to include
                         If None, includes all Conv/Linear layers
        exclude_patterns: List of substrings for layer names to exclude
                         Default: ['bn', 'norm', 'final', 'head']

    Returns:
        List of layer names suitable for LoRA adaptation

    Examples:
        >>> model = my_unet_model()
        >>> layers = detect_adaptable_layers(model)
        >>> print(f"Found {len(layers)} adaptable layers")
        Found 24 adaptable layers

        >>> # Only adapt encoder layers
        >>> layers = detect_adaptable_layers(
        ...     model,
        ...     include_patterns=[r".*encoder.*"]
        ... )
    """
    import re

    if exclude_patterns is None:
        exclude_patterns = ['bn', 'norm', 'final', 'head', 'output']

    adaptable = []

    for name, module in model.named_modules():
        # Check if it's a convolutional or linear layer
        if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            continue

        # Apply include patterns if specified
        if include_patterns is not None:
            if not any(re.match(pattern, name) for pattern in include_patterns):
                continue

        # Apply exclude patterns
        if any(exclude in name.lower() for exclude in exclude_patterns):
            logger.debug(f"Excluding layer: {name} (matched exclude pattern)")
            continue

        adaptable.append(name)

    logger.info(f"Detected {len(adaptable)} adaptable layers")
    if len(adaptable) > 0:
        logger.debug(f"Adaptable layers: {adaptable[:5]}..." if len(adaptable) > 5 else f"Adaptable layers: {adaptable}")

    return adaptable


class SequentialWrapper(nn.Module):
    """
    Wrapper for Sequential models to make them compatible with PEFT.

    PEFT expects models to accept **kwargs, but Sequential only accepts
    positional args. This wrapper provides that interface.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x=None, input_ids=None, **kwargs):
        # PEFT may pass input as 'input_ids' kwarg for transformers
        # For vision models, we expect 'x' as positional or kwarg
        if x is None and input_ids is not None:
            x = input_ids
        if x is None:
            raise ValueError("Input tensor not provided")
        # Ignore other kwargs and just pass x
        return self.model(x)


def wrap_model_with_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    modules_to_save: Optional[List[str]] = None,
    task_type: str = "FEATURE_EXTRACTION",
) -> nn.Module:
    """
    Wrap a PyTorch model with LoRA adapters using HuggingFace PEFT.

    This creates a PEFT model with LoRA adapters on specified layers.
    The base model is frozen, and only LoRA parameters are trainable.

    Args:
        model: PyTorch model to wrap (e.g., UNet, CNN)
        target_modules: List of layer names to adapt. If None, auto-detects.
        lora_r: LoRA rank (number of low-rank dimensions)
                Higher = more capacity, more parameters
                Typical values: 4-32, default 8
        lora_alpha: LoRA alpha (scaling factor)
                    Controls strength of LoRA updates
                    Typical: 2*r, default 16
        lora_dropout: Dropout probability for LoRA layers (0.0-0.5)
        modules_to_save: Additional modules to make trainable (e.g., final layer)
        task_type: PEFT task type. Options:
                   - "FEATURE_EXTRACTION" (default, for general models)
                   - "SEQ_CLS" (sequence classification)
                   - "TOKEN_CLS" (token classification)
                   - "CAUSAL_LM" (causal language modeling)

    Returns:
        PEFT model with LoRA adapters

    Raises:
        ImportError: If peft library is not installed
        ValueError: If no adaptable layers found

    Examples:
        >>> # Auto-detect and wrap all Conv/Linear layers
        >>> lora_model = wrap_model_with_lora(model, lora_r=8)

        >>> # Wrap specific layers with custom config
        >>> lora_model = wrap_model_with_lora(
        ...     model,
        ...     target_modules=["encoder.conv1", "encoder.conv2"],
        ...     lora_r=16,
        ...     lora_alpha=32,
        ...     modules_to_save=["final_conv"]
        ... )

        >>> # Check trainable parameters
        >>> print_lora_parameters(lora_model)
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "peft library is required for LoRA finetuning. "
            "Install with: pip install peft"
        )

    # Wrap Sequential models to make them compatible with PEFT
    if isinstance(model, nn.Sequential):
        logger.info("Wrapping Sequential model for PEFT compatibility")
        model = SequentialWrapper(model)

    # Auto-detect target modules if not specified
    if target_modules is None:
        target_modules = detect_adaptable_layers(model)
        if len(target_modules) == 0:
            raise ValueError(
                "No adaptable layers found in model. "
                "Specify target_modules manually or check model architecture."
            )
        logger.info(f"Auto-detected {len(target_modules)} target modules for LoRA")

    # Map task type string to PEFT TaskType enum
    task_type_map = {
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
    }

    if task_type not in task_type_map:
        logger.warning(
            f"Unknown task_type '{task_type}', using FEATURE_EXTRACTION. "
            f"Valid options: {list(task_type_map.keys())}"
        )
        task_type = "FEATURE_EXTRACTION"

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=task_type_map[task_type],
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",  # Don't adapt bias terms
    )

    logger.info(
        f"Creating LoRA model with r={lora_r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}"
    )

    # Wrap model with PEFT
    peft_model = get_peft_model(model, lora_config)

    logger.info("LoRA model created successfully")
    print_lora_parameters(peft_model)

    return peft_model


def print_lora_parameters(model: nn.Module):
    """
    Print statistics about trainable and total parameters in a LoRA model.

    Args:
        model: PEFT model with LoRA adapters

    Examples:
        >>> lora_model = wrap_model_with_lora(model)
        >>> print_lora_parameters(lora_model)
        Trainable params: 294,912 (1.2% of total)
        Total params: 24,567,890
    """
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model.print_trainable_parameters()
            return
    except ImportError:
        pass

    # Fallback if not a PEFT model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if total_params > 0:
        percentage = 100 * trainable_params / total_params
        logger.info(
            f"Trainable params: {trainable_params:,} ({percentage:.2f}% of total)"
        )
        logger.info(f"Total params: {total_params:,}")
    else:
        logger.warning("Model has no parameters")


def load_lora_adapter(
    model: nn.Module,
    adapter_path: str,
    is_trainable: bool = False,
) -> nn.Module:
    """
    Load a pretrained LoRA adapter into a base model.

    Args:
        model: Base PyTorch model (without LoRA)
        adapter_path: Path to saved LoRA adapter directory
        is_trainable: If True, adapter parameters are trainable (for continued training)
                     If False, adapter parameters are frozen (for inference)

    Returns:
        PEFT model with loaded adapter

    Examples:
        >>> # Load adapter for inference
        >>> model = load_lora_adapter(
        ...     base_model,
        ...     "models/fly_organelles/v1.1.0/lora_adapter"
        ... )

        >>> # Load adapter for continued training
        >>> model = load_lora_adapter(
        ...     base_model,
        ...     "models/fly_organelles/v1.1.0/lora_adapter",
        ...     is_trainable=True
        ... )
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft library is required. Install with: pip install peft"
        )

    logger.info(f"Loading LoRA adapter from: {adapter_path}")

    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=is_trainable,
    )

    if is_trainable:
        logger.info("Adapter loaded in trainable mode")
    else:
        logger.info("Adapter loaded in inference mode (frozen)")

    print_lora_parameters(peft_model)

    return peft_model


def save_lora_adapter(
    model: nn.Module,
    output_path: str,
):
    """
    Save only the LoRA adapter parameters (not the full model).

    This saves only the trained LoRA weights (~5-20 MB) rather than
    the entire model (~200-500 MB).

    Args:
        model: PEFT model with LoRA adapters
        output_path: Directory to save adapter

    Examples:
        >>> save_lora_adapter(
        ...     lora_model,
        ...     "models/fly_organelles/v1.1.0/lora_adapter"
        ... )
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft library is required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        raise ValueError(
            "Model must be a PeftModel. Use wrap_model_with_lora() first."
        )

    logger.info(f"Saving LoRA adapter to: {output_path}")
    model.save_pretrained(output_path)
    logger.info("Adapter saved successfully")


def merge_lora_into_base(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights back into the base model.

    This creates a standalone model with LoRA weights merged in,
    removing the need for PEFT at inference time.

    Warning: This increases model size back to the full model size.
    Only use if you need a standalone model without PEFT dependency.

    Args:
        model: PEFT model with LoRA adapters

    Returns:
        Base model with merged weights

    Examples:
        >>> merged_model = merge_lora_into_base(lora_model)
        >>> torch.save(merged_model.state_dict(), "merged_model.pt")
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft library is required. Install with: pip install peft"
        )

    if not isinstance(model, PeftModel):
        raise ValueError(
            "Model must be a PeftModel to merge adapters"
        )

    logger.info("Merging LoRA adapters into base model")
    merged_model = model.merge_and_unload()
    logger.info("Adapters merged successfully")

    return merged_model
