"""The trainable nn.Module behind a model config.

A served model is often not trainable as it stands: the cellmap and
Hugging Face models are TorchScript, which LoRA cannot wrap and autograd
cannot train. ``cellmap_model.train()`` rebuilds them as an unflattened
torch.export module, which is what the trainer needs. The adapter and
full-finetune exports are keyed by the names in that exact module tree
(``BatchLoopWrapper``'s ``model.`` prefix included), so training and
serving must build it the same way: both call ``load_trainable_model``
(serving through ``FinetuneModelConfig``).

The finetune CLI used to do this itself for fly/dacapo/huggingface/script
models only, so a ``type: cellmap`` model -- which is exactly what
export_merged produces -- or a finetuned model registered back into the
dashboard could not be finetuned at all.
"""

import inspect
import json
import logging

import torch

logger = logging.getLogger(__name__)


def decode_model_entry(text: str) -> dict:
    """A model entry given on the command line: JSON, or encode_to_str() of it."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    from cellmap_flow.utils.web_utils import decode_to_json

    return decode_to_json(text)


def model_config_from_entry(entry: dict, name=None):
    """Build a ModelConfig from a model entry, as ``ModelConfig.to_dict()`` writes it.

    Keys the class does not take are dropped first: ``to_dict()`` surfaces
    some for display (FinetuneModelConfig copies its base model's channels
    and voxel sizes up, say), and the constructor would reject them.
    """
    from cellmap_flow.models import registry
    from cellmap_flow.config.yaml import ConfigError

    entry = dict(entry)
    try:
        config_class = registry.model_type(str(entry.get("type", "")))
    except ConfigError:
        # No such type, or none given: build_model reports it, naming the model.
        config_class = None
    if config_class is not None:
        accepted = set(inspect.signature(config_class.__init__).parameters) - {"self"}
        dropped = sorted(k for k in entry if k != "type" and k not in accepted)
        if dropped:
            logger.debug(f"Model entry keys not taken by {config_class.__name__}: {dropped}")
        entry = {k: v for k, v in entry.items() if k == "type" or k in accepted}
    if name and not entry.get("name"):
        entry["name"] = name
    return registry.build_model(entry, entry.get("name") or name or "model")


def _cellmap_model_for(model_config):
    """The CellmapModel behind a TorchScript model config, if there is one."""
    repo = getattr(model_config, "repo", None)
    if getattr(type(model_config), "cli_name", None) == "huggingface" and repo:
        from cellmap_models.model_export.cellmap_model import get_huggingface_model

        return get_huggingface_model(repo, getattr(model_config, "revision", None))
    return getattr(model_config, "cellmap_model", None)


def load_trainable_model(model_config) -> torch.nn.Module:
    """The module to train for ``model_config``.

    - TorchScript (cellmap, Hugging Face): the unflattened module from
      ``cellmap_model.train()``, wrapped in BatchLoopWrapper when it is fixed
      at batch 1.
    - A finetuned model (``type: finetune``): the model it serves -- the base
      tree with its LoRA adapter attached (a PeftModel), or with its full
      finetuned weights loaded. Wrapping it in a new adapter folds the old
      one in first (lora_wrapper._merge_existing_adapters), so training
      continues from the finetuned model rather than from its base.
    - Anything else: ``model_config.config.model`` as it is.
    """
    base_model = model_config.config.model
    logger.info(f"Model loaded: {type(base_model).__name__}")

    if not isinstance(base_model, torch.jit.ScriptModule):
        return base_model

    logger.info("TorchScript model detected — loading trainable model via cellmap_model.train()...")
    cellmap_model = _cellmap_model_for(model_config)
    if cellmap_model is None:
        logger.warning("No CellmapModel available — LoRA may fail on TorchScript model")
        return base_model

    trainable = cellmap_model.train()
    if trainable is None:
        logger.warning("cellmap_model.train() returned None — LoRA may fail")
        return base_model
    # UnflattenedModule (from torch.export) often has fixed batch=1.
    # Wrap it so the trainer can use any batch size.
    if type(trainable).__name__ == "UnflattenedModule":
        from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper

        trainable = BatchLoopWrapper(trainable)
        logger.info("Wrapped UnflattenedModule with BatchLoopWrapper for variable batch sizes")
    logger.info(f"Trainable model loaded: {type(trainable).__name__}")
    return trainable


def root_base_model_dict(model_config) -> dict:
    """The innermost non-finetune model entry under ``model_config``.

    A full finetune's weights replace every parameter, so what it needs from
    its base is only the module tree -- and a finetune-of-a-finetune base
    would hand the weights a PeftModel, whose names no longer match.
    """
    entry = model_config.to_dict()
    while entry.get("type") == "finetune" and isinstance(entry.get("base_model"), dict):
        entry = entry["base_model"]
    return entry
