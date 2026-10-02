"""The model config classes: ``ModelConfig`` and the built-in model types.

This is their public import path. The docs import them from here, and
plugins subclass ``ModelConfig``, ``ScriptModelConfig`` and the rest by
it. The classes live in ``cellmap_flow.models.configs``, one module per
type. ``Config``, what a type's ``_get_config`` returns, is here too, for
plugins that define a type.
"""

from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.models.configs.script import ScriptModelConfig
from cellmap_flow.models.configs.dacapo import DaCapoModelConfig
from cellmap_flow.models.configs.fly import FlyModelConfig
from cellmap_flow.models.configs.bio import BioModelConfig
from cellmap_flow.models.configs.cellmap import CellMapModelConfig
from cellmap_flow.models.configs.finetune import FinetuneModelConfig
from cellmap_flow.models.configs.huggingface import HuggingFaceModelConfig
from cellmap_flow.models.configs.cellpose import CellposeModelConfig

__all__ = [
    "Config",
    "ModelConfig",
    "ScriptModelConfig",
    "DaCapoModelConfig",
    "FlyModelConfig",
    "BioModelConfig",
    "CellMapModelConfig",
    "FinetuneModelConfig",
    "HuggingFaceModelConfig",
    "CellposeModelConfig",
]
