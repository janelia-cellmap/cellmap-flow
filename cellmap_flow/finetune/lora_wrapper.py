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


def _narrowest_dim(module: nn.Module) -> Optional[int]:
    """min(in, out) width of a conv/linear layer, or None if it has neither."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return min(module.in_channels, module.out_channels)
    if isinstance(module, nn.Linear):
        return min(module.in_features, module.out_features)
    w = getattr(module, "weight", None)
    if isinstance(w, torch.Tensor) and w.ndim >= 2:
        return min(w.shape[0], w.shape[1])
    return None


def detect_adaptable_layers(
    model: nn.Module,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    min_channels: int = 0,
) -> List[str]:
    """
    Automatically detect layers suitable for LoRA adaptation.

    Searches for Conv2d, Conv3d, and Linear layers, filtering by name patterns.
    By default, only excludes batch/layer-norm style modules. Output/head
    layers are deliberately INCLUDED so the model can fully adapt its
    feature→output mapping for cross-domain finetuning. (Previously
    'final', 'head', 'output' were excluded; that left the output projection
    frozen, which prevented learning when the base model's predictions on
    the target dataset were poor.)

    Args:
        model: PyTorch model to inspect
        include_patterns: List of regex patterns for layer names to include
                         If None, includes all Conv/Linear layers
        exclude_patterns: List of substrings for layer names to exclude
                         Default: ['bn', 'norm']
        min_channels: Skip layers whose narrower side (in or out) is below
                     this. 0 adapts everything. Narrow layers are where LoRA
                     is expensive for nothing: PEFT builds lora_A as a
                     full-kernel conv Cin -> r, so on a 16-channel layer at
                     full resolution the adapter is 4x the FLOPs of the layer
                     it adapts and runs bandwidth-bound. On
                     mito-aff-unet-setup-16 the seven layers under 96
                     channels hold ~1% of the adapter's parameters and cost
                     41% of every training step (0.095 -> 0.056 s at 178^3).

    Returns:
        List of layer names suitable for LoRA adaptation
    """
    import re

    if exclude_patterns is None:
        exclude_patterns = ['bn', 'norm']

    adaptable = []
    skipped_narrow = []

    for name, module in model.named_modules():
        is_adaptable = isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))

        # An unflattened (torch.export) conv/linear, which wrap_model_with_lora
        # turns into a real one before PEFT sees it. Not "anything with a 2D+
        # weight": a transposed conv has one too, and PEFT cannot adapt it.
        if not is_adaptable:
            is_adaptable = _interpreter_layer_spec(module) is not None

        if not is_adaptable:
            continue

        # PEFT builds lora_A with the base conv's kernel, stride and padding
        # but not its dilation, so on a dilated conv the adapter's output is a
        # different size from the layer's and the first forward pass fails.
        dilation = getattr(module, "dilation", None)
        if dilation is None:
            spec = _interpreter_layer_spec(module)
            dilation = spec[1].get("dilation") if spec else None
        if dilation is not None and any(d != 1 for d in _as_tuple(dilation)):
            logger.info(f"Not adapting {name}: PEFT's LoRA does not support dilated convolutions")
            continue

        # Apply include patterns if specified
        if include_patterns is not None:
            if not any(re.match(pattern, name) for pattern in include_patterns):
                continue

        # Apply exclude patterns
        if any(exclude in name.lower() for exclude in exclude_patterns):
            logger.debug(f"Excluding layer: {name} (matched exclude pattern)")
            continue

        if min_channels > 0:
            width = _narrowest_dim(module)
            if width is not None and width < min_channels:
                skipped_narrow.append((name, width))
                continue

        adaptable.append(name)

    logger.info(f"Detected {len(adaptable)} adaptable layers")
    if skipped_narrow:
        # Loud on purpose: a run that meant to skip these and did not is
        # indistinguishable from the log otherwise, and costs 1.7x.
        logger.info(
            f"LoRA min_channels={min_channels}: skipping {len(skipped_narrow)} "
            f"narrow layer(s), adapting {len(adaptable)}. Skipped: "
            + ", ".join(f"{n} ({w} ch)" for n, w in skipped_narrow)
        )
    if len(adaptable) > 0:
        logger.debug(f"Adaptable layers: {adaptable[:5]}..." if len(adaptable) > 5 else f"Adaptable layers: {adaptable}")

    return adaptable


_CONV_CLASSES = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}


def _tupled(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _as_tuple(value):
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _interpreter_layer_spec(module: nn.Module):
    """How to rebuild an unflattened leaf as a real layer, or None.

    torch.export's unflatten turns every nn.Conv3d/nn.Linear into an
    InterpreterModule that holds the weight and a one-op FX graph, e.g.
    ``aten.conv3d.default(x, weight, bias, stride, padding, dilation,
    groups)``. The stride, padding, dilation and groups live only in that
    call, so they are read from it. Anything else -- a transposed conv, which
    PEFT cannot adapt, or a module whose graph does more than one op -- gives
    None and is left as it is.

    Returns ``(layer_class, kwargs)``.
    """
    import re

    if type(module).__name__ != "InterpreterModule":
        return None
    weight = getattr(module, "weight", None)
    graph = getattr(module, "graph", None)
    if not isinstance(weight, torch.Tensor) or graph is None:
        return None
    calls = [node for node in graph.nodes if node.op == "call_function"]
    if len(calls) != 1:
        return None
    call = calls[0]
    args = list(call.args)
    target = str(call.target)
    has_bias = isinstance(getattr(module, "bias", None), torch.Tensor)

    if re.match(r"aten\.linear\.", target) and weight.ndim == 2:
        return nn.Linear, dict(in_features=weight.shape[1], out_features=weight.shape[0], bias=has_bias)

    if re.match(r"aten\.convolution\.", target):
        # convolution(input, weight, bias, stride, padding, dilation,
        #             transposed, output_padding, groups)
        if len(args) < 9 or args[6]:
            return None
        stride, padding, dilation, groups = args[3], args[4], args[5], args[8]
    else:
        # conv{N}d(input, weight, bias=None, stride=1, padding=0, dilation=1,
        # groups=1). conv_transpose{N}d does not match, and is left alone.
        match = re.match(r"aten\.conv([123])d\.", target)
        if match is None or weight.ndim != int(match.group(1)) + 2:
            return None
        values = args + [None, None, None, 1, 0, 1, 1][len(args):]
        for key, index in (("stride", 3), ("padding", 4), ("dilation", 5), ("groups", 6)):
            if key in call.kwargs:
                values[index] = call.kwargs[key]
        stride, padding, dilation, groups = values[3:7]
    dims = weight.ndim - 2
    if dims not in _CONV_CLASSES:
        return None
    groups = int(groups)
    return _CONV_CLASSES[dims], dict(
        in_channels=weight.shape[1] * groups,
        out_channels=weight.shape[0],
        kernel_size=tuple(weight.shape[2:]),
        stride=_tupled(stride),
        padding=_tupled(padding),
        dilation=_tupled(dilation),
        groups=groups,
        bias=has_bias,
    )


def _replace_interpreter_modules(model: nn.Module) -> int:
    """Replace unflattened conv/linear leaves (InterpreterModule from torch.export
    unflatten) with real nn.Conv*/nn.Linear that share the same weight/bias tensors.

    PEFT's dispatch only accepts nn.Conv1d/2d/3d, nn.Linear, etc., so unflattened
    modules need to be swapped before LoRA wrapping. The FX graph's call_module
    will invoke whatever module is registered under the name, so the swap doesn't
    break the forward pass.

    Only InterpreterModule leaves whose graph is a single convolution or linear
    call are replaced, with that call's stride, padding, dilation and groups.
    This used to replace any module with a weight by a stride-1, unpadded
    conv built from the weight's shape alone: a real nn.ConvTranspose3d became
    a Conv3d with its channels swapped, and an unflattened strided or padded
    conv lost its stride and padding, so LoRA on a UNet with transposed-conv
    upsampling died at the first forward pass.

    Returns the number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        spec = _interpreter_layer_spec(module)
        if spec is None:
            continue
        layer_class, kwargs = spec
        w = module.weight
        b = getattr(module, 'bias', None)
        new_mod = layer_class(**kwargs)
        new_mod.weight = nn.Parameter(w)
        if kwargs["bias"]:
            new_mod.bias = nn.Parameter(b)

        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)
        count += 1

    if count > 0:
        logger.info(f"Replaced {count} non-standard modules with nn.Conv/Linear for PEFT compatibility")
    return count


class BatchLoopWrapper(nn.Module):
    """Wraps a model with fixed batch_size=1 (e.g. UnflattenedModule from
    torch.export without dynamic shapes) so it accepts arbitrary batch sizes
    by looping over the batch dim.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, *args, **kwargs):
        if x.shape[0] == 1:
            return self.model(x, *args, **kwargs)
        outs = [self.model(x[i:i + 1], *args, **kwargs) for i in range(x.shape[0])]
        return torch.cat(outs, dim=0)


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
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.1,
    modules_to_save: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    lora_min_channels: int = 0,
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
        lora_dropout: Dropout probability for LoRA layers (0.0-0.5, default 0.1)
        lora_min_channels: When auto-detecting, skip layers narrower than this
                on either side; see detect_adaptable_layers. Ignored when
                target_modules is given explicitly. Default 0 (adapt all).
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

    # Bake in any adapter the model already carries, before adding ours.
    #
    # Calling get_peft_model() on something that is already a PeftModel does
    # not stack: both adapters are named "default", so the second injection
    # replaces the first and its weights are dropped on the floor. PEFT says
    # as much ("modify a model with PEFT for a second time... call .unload()
    # before"), but only as a warning, so finetuning a model that already had
    # an adapter -- which is every "continue from my last run" -- silently
    # started from the bare base instead. The only visible trace was the total
    # parameter count going *down* after wrapping.
    #
    # Merging makes the existing adapter part of the frozen base weights, so
    # the new LoRA starts from the model you were actually looking at, and the
    # distillation teacher (adapters disabled) is that same model rather than
    # the untuned original.
    model = _merge_existing_adapters(model)

    # Wrap Sequential models to make them compatible with PEFT
    if isinstance(model, nn.Sequential):
        logger.info("Wrapping Sequential model for PEFT compatibility")
        model = SequentialWrapper(model)

    # Replace any non-standard leaf modules (e.g. InterpreterModule) with
    # real nn.Conv*/Linear so PEFT's dispatch can wrap them.
    _replace_interpreter_modules(model)

    # Auto-detect target modules if not specified
    if target_modules is None:
        target_modules = detect_adaptable_layers(model, min_channels=lora_min_channels)
        if len(target_modules) == 0:
            raise ValueError(
                "No adaptable layers found in model. "
                "Specify target_modules manually or check model architecture."
            )
        logger.info(f"Auto-detected {len(target_modules)} target modules for LoRA")

    # Map task type string to PEFT TaskType enum
    # None means PEFT uses the base PeftModel with a clean forward() passthrough,
    # which is correct for custom nn.Module models (not HuggingFace transformers).
    task_type_map = {
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
    }

    peft_task_type = task_type_map.get(task_type) if task_type else None

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=peft_task_type,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",  # Don't adapt bias terms
    )

    logger.info(
        f"Creating LoRA model with r={lora_r}, alpha={lora_alpha}, "
        f"dropout={lora_dropout}, min_channels={lora_min_channels}"
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


def _lora_conv_delta_weight(layer, adapter):
    """The LoRA delta for a conv layer: a contraction over the rank axis.

    peft computes this itself, but takes a conv2d shortcut whenever
    `weight.size()[2:4] == (1, 1)` -- which is also true of a *3D* conv with a
    1x1x1 kernel, like an affinity head. The squeeze then leaves a trailing
    spatial axis, the matmul becomes a batched one over the channel counts,
    and it fails with a shape mismatch instead of merging. lora_B is always
    pointwise, so the delta is just lora_B summed against lora_A over rank.
    """
    weight_A = layer.lora_A[adapter].weight
    weight_B = layer.lora_B[adapter].weight
    delta = torch.einsum(
        "or,ri...->oi...", weight_B.flatten(1).float(), weight_A.float()
    )
    return (delta * layer.scaling[adapter]).to(weight_A.dtype)


def _fix_conv_delta_weights(model: nn.Module) -> int:
    """Give the conv layers peft would mis-merge a delta it can merge.

    Returns how many were patched. Only layers that would hit the broken
    branch are touched; every other layer keeps peft's own implementation.
    """
    import types

    patched = 0
    for module in model.modules():
        if not hasattr(module, "get_base_layer") or not hasattr(module, "lora_A"):
            continue
        base = module.get_base_layer()
        weight = getattr(base, "weight", None)
        if weight is None or weight.dim() != 5 or tuple(weight.shape[2:4]) != (1, 1):
            continue
        if getattr(base, "groups", 1) != 1:
            continue
        if any(getattr(module, "use_dora", {}).values()):
            continue
        module.get_delta_weight = types.MethodType(_lora_conv_delta_weight, module)
        patched += 1
    return patched


def _merge_existing_adapters(model: nn.Module) -> nn.Module:
    """Fold any already-attached LoRA adapter into the base weights.

    Returns the plain module to wrap. A model with no adapter passes straight
    through. See the note in create_lora_model() for why stacking is not an
    option.
    """
    try:
        from peft import PeftModel
    except ImportError:
        return model

    if not isinstance(model, PeftModel):
        return model

    patched = _fix_conv_delta_weights(model)
    if patched:
        logger.info(
            f"Computing the LoRA delta for {patched} 1x1x1 3D conv layer(s) "
            f"here rather than in peft, whose shortcut for them is conv2d-only."
        )

    before = sum(p.numel() for p in model.parameters())
    try:
        merged = model.merge_and_unload()
    except Exception as e:
        # Losing the adapter silently is what caused the original bug, so
        # refuse rather than carry on and train from the wrong starting point.
        raise RuntimeError(
            "This model already has a LoRA adapter, and it could not be "
            f"merged into the base weights ({e}). Training on top of it would "
            "silently discard that adapter and start from the untuned base "
            "model instead."
        ) from e

    after = sum(p.numel() for p in merged.parameters())
    logger.info(
        f"Merged the model's existing LoRA adapter into its base weights "
        f"({before:,} -> {after:,} params); the new adapter will train on top "
        f"of it."
    )
    return merged


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

    # Fold in any adapter the model already carries, for the same reason
    # create_lora_model() does -- and additionally because the adapter being
    # loaded here was saved against the *merged* module tree. Calling
    # from_pretrained() on a PeftModel wraps it a second time, which both
    # drops the existing adapter and double-nests every module name
    # ("base_model.model.base_model.model...."), so none of the saved keys
    # match. PEFT reports that as a warning about missing adapter keys and
    # then returns a model with no adapter loaded at all: the served
    # "finetuned" model was the untouched base.
    model = _merge_existing_adapters(model)

    # Wrap Sequential models to make them compatible with PEFT
    if isinstance(model, nn.Sequential):
        logger.info("Wrapping Sequential model for PEFT compatibility")
        model = SequentialWrapper(model)

    # Replace any non-standard leaf modules (e.g. InterpreterModule from
    # torch.export unflatten) with real nn.Conv*/Linear so PEFT's dispatch
    # can find the target modules named in the saved adapter config. Must
    # mirror create_lora_model()'s call to this before training, since the
    # adapter's target_modules names were recorded against the post-replacement
    # module tree.
    _replace_interpreter_modules(model)

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
