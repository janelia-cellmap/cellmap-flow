"""How a finetune adapts the model: a LoRA adapter, or every weight.

Everything that differs between the two lives here: how the model is made
trainable, what stays frozen, the distillation teacher, the in-place reset
(the trainer's retry after a NaN) and the restart between a job's
iterations, what a checkpoint keeps, the export and the merge into plain
weights. The request decides once, when the model is prepared (``--lora-r``,
0 for full); after that the model does (``strategy_for``), since a restart
can change ``args.lora_r`` but not the model.

The adapter's names depend on the module tree it was trained on
(``BatchLoopWrapper``'s ``model.`` prefix included), so the wrappers stay in
lora_wrapper and this module only calls them. peft is imported lazily.
"""

import copy
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Dict, Literal, Optional, Protocol

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptationStrategy",
    "LoraStrategy",
    "FullStrategy",
    "strategy_for",
    "is_peft_model",
    "cpu_state_copy",
    "frozen_teacher_copy",
]


def is_peft_model(model) -> bool:
    """Whether ``model`` carries a LoRA adapter (is a PeftModel); False without peft."""
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


def cpu_state_copy(model: nn.Module) -> Dict[str, torch.Tensor]:
    """A CPU copy of ``model``'s state dict, to reset a full finetune to."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def frozen_teacher_copy(model: nn.Module) -> nn.Module:
    """A frozen, eval-mode copy of ``model``: the distillation teacher of a full finetune."""
    try:
        teacher = copy.deepcopy(model)
    except Exception as e:
        raise ValueError(
            "Distillation on a full finetune (--lora-r 0) needs a frozen copy "
            f"of the model as its teacher, and this model could not be copied "
            f"({e}). Set the distillation weight to 0, or train a LoRA adapter "
            "(rank > 0), whose teacher is the base model itself."
        ) from e
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.eval()


class AdaptationStrategy(Protocol):
    kind: Literal["lora", "full"]
    export_name: str        # "lora_adapter" | "full_finetune": the export's directory
    r: int                  # the LoRA rank; 0 for a full finetune, as --lora-r has it
    teacher_model: Optional[nn.Module]  # a full finetune's frozen teacher, once made

    def prepare(self, model: nn.Module) -> nn.Module: ...
    def train_mode(self, model: nn.Module) -> None: ...
    def teacher(self, model: nn.Module) -> ContextManager[nn.Module]: ...
    def initial_state(self, model: nn.Module) -> Optional[dict]: ...
    def reset(self, model: nn.Module, initial_state) -> nn.Module: ...
    def restart(self, model: nn.Module, initial_state) -> nn.Module: ...
    def trainable_state(self, model: nn.Module) -> dict: ...
    def checkpoint(self, model: nn.Module, optimizer, scaler, is_best: bool) -> Optional[dict]: ...
    def export(self, model: nn.Module, export_dir: Path, path=None) -> Path: ...
    def merge(self, model: nn.Module) -> nn.Module: ...


class _AdapterOff:
    """LoRA's teacher: the model with its adapter off, in eval mode (no dropout
    noise, no batch statistics). On exit the adapter is back on and every
    module has its mode back. Reusable: the trainer enters it every batch.
    """

    def __init__(self, model):
        self.model = model
        self._modes = None

    def __enter__(self):
        self._modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        self.model.disable_adapter_layers()
        return self.model

    def __exit__(self, *exc):
        self.model.enable_adapter_layers()
        for module, training in self._modes:
            module.training = training
        return False


class LoraStrategy:
    """Train a LoRA adapter on the frozen model; export only the adapter."""

    kind = "lora"
    export_name = "lora_adapter"
    # The teacher is the model itself with its adapter off: nothing to keep.
    teacher_model = None

    def __init__(self, r: int, alpha: float, dropout: float, min_channels=None, target_modules=None):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.min_channels = min_channels
        self.target_modules = target_modules

    def prepare(self, model):
        """``model`` in a fresh adapter; one it already carries is folded into its weights first."""
        from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

        logger.info(f"Wrapping model with LoRA (r={self.r})...")
        return wrap_model_with_lora(
            model,
            target_modules=self.target_modules,
            lora_r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            lora_min_channels=self.min_channels or 0,
        )

    def train_mode(self, model):
        """Train mode, except for the frozen base's norm layers: they stay in
        eval mode, as served, instead of updating running statistics that the
        adapter does not save.
        """
        model.train()
        norm_types = (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm)
        for module in model.modules():
            if isinstance(module, norm_types) and not any(
                p.requires_grad for p in module.parameters(recurse=False)
            ):
                module.eval()

    def teacher(self, model):
        """The model as it was before this run: itself, adapter off. Costs nothing."""
        return _AdapterOff(model)

    def initial_state(self, model):
        """None: LoRA resets by re-initialising its adapter, not by reloading weights."""
        return None

    def reset(self, model, initial_state=None):
        """Re-initialise the adapter in place: B to zero, so it adds nothing, and A as peft does."""
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                nn.init.zeros_(param) if 'lora_B' in name else nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        return model

    def restart(self, model, initial_state=None):
        """A fresh adapter with this strategy's rank; the old one is unloaded, not merged."""
        logger.info("Resetting LoRA adapter weights for fresh restart...")
        return self.prepare(model.unload())

    def trainable_state(self, model):
        """The adapter's weights: the parameters that train."""
        trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        return {k: v for k, v in model.state_dict().items() if k in trainable_keys}

    def checkpoint(self, model, optimizer, scaler, is_best):
        """The adapter's weights (not the 800M-param base), optimizer and scaler: resumable."""
        return {
            'model_state_dict': self.trainable_state(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'lora_only': True,
        }

    def export(self, model, export_dir, path=None):
        """``save_pretrained`` into ``export_dir``/lora_adapter (or ``path``, the adapter's own directory)."""
        from cellmap_flow.finetune.lora_wrapper import save_lora_adapter

        path = Path(path) if path is not None else Path(export_dir) / self.export_name
        save_lora_adapter(model, str(path))
        logger.info(f"LoRA adapter saved to: {path}")
        return path

    @staticmethod
    def merge(model):
        """The adapter folded into the base weights: a plain module, served without peft.

        A 1x1x1 Conv3d's delta is computed by lora_wrapper: peft's 1x1
        shortcut is conv2d-only and fails on it.
        """
        from cellmap_flow.finetune.lora_wrapper import _merge_existing_adapters

        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft library is required. Install with: pip install peft")
        if not isinstance(model, PeftModel):
            raise ValueError("Model must be a PeftModel to merge adapters")
        return _merge_existing_adapters(model)


class FullStrategy:
    """Train every weight; export the whole state dict (FinetuneModelConfig(weights_path=...)).

    Against LoRA r=64 on mito-aff-unet-setup-16 (2026-09-23): 0.50 vs 0.90 s
    a step, 31 vs 50 GB at batch 8, and a lower loss at every checkpoint.
    """

    kind = "full"
    export_name = "full_finetune"
    r = 0

    def __init__(self, teacher_model: Optional[nn.Module] = None):
        self.teacher_model = teacher_model  # the frozen teacher: made once, or a previous one's
        self._warned_periodic = False

    def prepare(self, model):
        """Every weight trainable; an adapter the model carries (a finetuned base) is folded into them first."""
        from cellmap_flow.finetune.lora_wrapper import _merge_existing_adapters

        logger.info("lora_r=0: full finetuning -- every parameter trainable, no adapter. "
                    "Restarts start again from the starting weights.")
        model = _merge_existing_adapters(model)
        for p in model.parameters():
            p.requires_grad_(True)
        n_train = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable params: {n_train:,} || all params: {n_train:,} || trainable%: 100.0000")
        return model

    def train_mode(self, model):
        """Train mode throughout: a full finetune trains its norm layers too."""
        model.train()

    def teacher(self, model):
        """A frozen copy of the weights training starts from, made at the first call
        (the trainer's, before any step) and kept: the CLI hands it to each later
        iteration's trainer, so the teacher is the job's starting weights, copied once.
        """
        if self.teacher_model is None:
            self.teacher_model = frozen_teacher_copy(model)
        logger.info(
            "Full finetune with distillation: the teacher is a frozen copy "
            "of the starting weights."
        )
        return nullcontext(self.teacher_model)

    def initial_state(self, model):
        """The weights training starts from, on the CPU: what resetting it means."""
        return cpu_state_copy(model)

    def reset(self, model, initial_state):
        """Load the starting weights back, in place."""
        if initial_state is not None:
            model.load_state_dict(initial_state)
            logger.info("Full finetune: weights reset to the ones training started from.")
        return model

    def restart(self, model, initial_state):
        """The same model, back at its starting weights."""
        if initial_state is not None:
            logger.info("Resetting the full finetune to its starting weights for a fresh restart...")
            model.load_state_dict(initial_state)
        return model

    def trainable_state(self, model):
        """Every weight."""
        return model.state_dict()

    def checkpoint(self, model, optimizer, scaler, is_best):
        """Only the best epoch's weights, no optimizer state; None otherwise.

        With every parameter trainable, a resumable checkpoint would be the
        model plus two Adam moments: ~9.5 GB for an 800M-param UNet, each time.
        """
        if not is_best:
            if not self._warned_periodic:
                logger.info("Full finetune: skipping periodic checkpoints; best_checkpoint.pth holds the full weights.")
                self._warned_periodic = True
            return None
        return {
            'model_state_dict': self.trainable_state(model),
            'lora_only': False,
            'full_model': True,
        }

    def export(self, model, export_dir, path=None):
        """The state dict, to ``export_dir``/full_finetune/model_state_dict.pt (``path`` is LoRA's)."""
        out = Path(export_dir) / self.export_name
        out.mkdir(parents=True, exist_ok=True)
        weights = out / "model_state_dict.pt"
        torch.save(model.state_dict(), weights)
        logger.info(f"Full finetuned weights saved to: {weights}")
        return weights

    @staticmethod
    def merge(model):
        """``model`` itself: its weights are the finetune."""
        return model


def strategy_for(
    model: nn.Module,
    lora_r: Optional[int] = None,
    *,
    alpha=None,
    dropout=None,
    min_channels=None,
    target_modules=None,
    teacher_model: Optional[nn.Module] = None,
) -> AdaptationStrategy:
    """The strategy ``model`` is trained with: LoRA if it carries an adapter, else full.

    The model decides, not the request: it is built once and shared with the
    inference server, so a job never switches kinds. For a LoRA model,
    ``lora_r`` (when positive) and the keywords set the adapter ``restart``
    makes; the adapter's own rank, alpha and dropout fill in the rest.
    ``teacher_model`` is a full finetune's existing teacher, to reuse.
    """
    if not is_peft_model(model):
        return FullStrategy(teacher_model=teacher_model)
    active = model.active_adapter
    config = model.peft_config[active[0] if isinstance(active, (list, tuple)) else active]
    return LoraStrategy(
        r=lora_r if lora_r is not None and lora_r > 0 else config.r,
        alpha=alpha if alpha is not None else config.lora_alpha,
        dropout=dropout if dropout is not None else config.lora_dropout,
        min_channels=min_channels,
        target_modules=target_modules,
    )
