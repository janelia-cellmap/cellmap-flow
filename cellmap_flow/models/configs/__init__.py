"""The built-in model types, one module per type.

- ``base``: ``ModelConfig``, the base of every type (a plugin's too), and
  the helpers the types share.
- ``script``: ``ScriptModelConfig``, a model that a Python script defines.
- ``dacapo``: ``DaCapoModelConfig``, a DaCapo run at one iteration.
- ``fly``: ``FlyModelConfig``, a fly_organelles checkpoint.
- ``bio``: ``BioModelConfig``, a bioimage.io model.
- ``cellmap``: ``CellMapModelConfig``, a cellmap_models export folder.
- ``finetune``: ``FinetuneModelConfig``, a base model with a finetune's
  LoRA adapter or full weights on top.
- ``huggingface``: ``HuggingFaceModelConfig``, a cellmap_models export on
  the Hugging Face Hub.

Import the classes from ``cellmap_flow.models.models_config``, their public
path: the docs name it, and plugins subclass the classes by it.

Each type imports its framework (torch, dacapo, bioimageio,
cellmap_models, ...) inside the methods that build the model, because the
CLIs import every type to build their commands.

Every type module is imported here, in this order, however the package is
first reached. The registry lists the types in the order ModelConfig's
subclasses were defined, and when two classes claim one name the first
keeps it; so the built-in types are always defined first, in this order.
"""

from cellmap_flow.models.configs import (  # noqa: F401
    base,
    script,
    dacapo,
    fly,
    bio,
    cellmap,
    finetune,
    huggingface,
)
