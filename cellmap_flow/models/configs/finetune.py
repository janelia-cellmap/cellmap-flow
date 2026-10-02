"""``FinetuneModelConfig``: a base model with a finetune's weights on top.

The weights are a LoRA adapter or, from a full finetune, a whole state
dict. Either is loaded onto the module the trainer trained
(``finetune.model_loading.load_trainable_model``), and the served model
keeps the base model's geometry, at the voxel sizes the finetune was
trained at when those differ (``trained_at_geometry``).
"""

import numpy as np

from cellmap_flow.models.configs.base import Config, ModelConfig
from cellmap_flow.models.geometry import _numbers, _voxels


def _voxel_sizes(value):
    """A voxel size given as a sequence or as the CLI's "10.48,8,8"; None stays None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split(",")
    return _numbers(float(v) for v in value)


def trained_at_geometry(base_cfg, input_voxel_size=None, output_voxel_size=None) -> dict:
    """The base model's geometry at the voxel sizes a finetune was trained at.

    A volume made without resampling lies on the raw level nearest the
    model's voxel size, and the trainer read that level as it is: the model
    learnt that scale, so the finetuned model really is a model at the
    level's voxel size (16 nm data read as 8 nm by an 8 nm model gives a 16
    nm model). Declaring it, the served model reads that level exactly,
    instead of the nearest one relabelled with a warning. The shapes in nm
    scale with the voxel size; the voxel counts are the base's.

    Returns ``input_voxel_size``, ``output_voxel_size``, ``read_shape`` and
    ``write_shape``; each size None keeps the base's.
    """
    geometry = {}
    for which, trained in (("input", input_voxel_size), ("output", output_voxel_size)):
        shape_key = "read_shape" if which == "input" else "write_shape"
        base_size = getattr(base_cfg, f"{which}_voxel_size")
        base_shape = getattr(base_cfg, shape_key)
        if trained is None:
            geometry[f"{which}_voxel_size"], geometry[shape_key] = base_size, base_shape
            continue
        voxels = np.asarray(_voxels(base_shape, base_size), dtype=float)
        geometry[f"{which}_voxel_size"] = _numbers(trained)
        geometry[shape_key] = _numbers(voxels * np.asarray(trained, dtype=float))
    return geometry


class FinetuneModelConfig(ModelConfig):
    """Configuration class for a finetuned model.

    Wraps any base ModelConfig with the finetune's weights, which are
    exactly one of: a LoRA adapter (``lora_adapter_path``), applied with
    PEFT, or a full finetune's state dict (``weights_path``, from a run
    with ``--lora-r 0``), loaded strictly. The base model is built by its
    own ModelConfig, as the trainable module the trainer trained, and the
    weights go on top of it.
    """

    cli_name = "finetune"

    def __init__(
        self,
        lora_adapter_path: str = None,
        base_model: dict = None,
        name: str = None,
        scale=None,
        weights_path: str = None,
        input_voxel_size: tuple = None,
        output_voxel_size: tuple = None,
    ):
        """
        Args:
            lora_adapter_path: Path to the saved LoRA adapter directory.
            weights_path: Alternative to lora_adapter_path: a full state dict
                (torch.save of model.state_dict()) from a full finetune, i.e.
                a run with --lora-r 0. Loaded strictly onto the same trainable
                module the base model config produces. Exactly one of
                lora_adapter_path / weights_path must be given.
            base_model: Dict describing the base model (same format as a YAML
                model entry, e.g. {"type": "fly", "checkpoint_path": "...", ...}).
                May also be passed as a string produced by
                ``cellmap_flow.serving.protocol.encode_to_str`` -- the
                dynamic server CLI (see ``command`` below) can only pass
                plain strings, so ``command`` encodes the dict and this
                constructor decodes it back on the receiving end.
            name: Display name for this model.
            scale: Optional scale override.
            input_voxel_size, output_voxel_size: the voxel sizes the finetune
                was trained at, when not the base model's (see
                ``trained_at_geometry``); the trainer writes them into the
                serving YAML. A sequence, or "10.48,8,8" from the CLI.
        """
        super().__init__()
        self.lora_adapter_path = lora_adapter_path
        self.weights_path = weights_path
        if bool(lora_adapter_path) == bool(weights_path):
            raise ValueError(
                "FinetuneModelConfig needs exactly one of lora_adapter_path "
                "(a LoRA adapter directory) or weights_path (a full state dict "
                f"from a --lora-r 0 run); got lora_adapter_path={lora_adapter_path!r}, "
                f"weights_path={weights_path!r}"
            )
        if base_model is None:
            raise ValueError("FinetuneModelConfig requires base_model (a model entry dict)")
        if isinstance(base_model, str):
            from cellmap_flow.serving.protocol import decode_to_json

            base_model = decode_to_json(base_model)
        self.base_model_dict = base_model
        self.name = name
        self.scale = scale
        self.input_voxel_size = _voxel_sizes(input_voxel_size)
        self.output_voxel_size = _voxel_sizes(output_voxel_size)
        self._base_model_config = None

    @property
    def env(self):
        """Its own ``env`` if it was given one, else its base model's: the
        weights go on the base model, so they need the base's packages."""
        own = self.__dict__.get("_env")
        if own:
            return own
        base = self.base_model_dict
        return base.get("env") if isinstance(base, dict) else None

    @env.setter
    def env(self, value):
        self._env = value

    @property
    def base_model_config(self):
        """Lazily build the base ModelConfig from the stored dict."""
        if self._base_model_config is None:
            from cellmap_flow.models.registry import build_model

            base_name = self.base_model_dict.get("name", "base_model")
            self._base_model_config = build_model(self.base_model_dict, base_name)
        return self._base_model_config

    def _get_config(self):
        # Imported here rather than at module scope: importing torch costs
        # ~7s, and the CLI builds its command list from this module, so
        # `cellmap_flow --help` paid that before printing anything.
        import torch

        from cellmap_flow.finetune.lora_wrapper import load_lora_adapter
        from cellmap_flow.finetune.model_loading import load_trainable_model

        # Get the fully-populated config from the base model. The served model
        # has the base's geometry, so the base is checked the way this config
        # is (on the warmup forward, or not at all) rather than by a forward of
        # its own.
        self.base_model_config.validate_model_shapes = self.validate_model_shapes
        self.base_model_config.check_shapes_on_warmup = self.check_shapes_on_warmup
        base_cfg = self.base_model_config.config

        # The module the finetune was trained on, built the way the trainer
        # builds it: a TorchScript base (cellmap, Hugging Face) becomes
        # cellmap_model.train()'s unflattened module, in BatchLoopWrapper.
        # The adapter and full-finetune weights are keyed by that exact tree.
        base_model = load_trainable_model(self.base_model_config)

        device = next(base_model.parameters()).device
        if self.weights_path:
            # Full finetune: the trained weights are the whole module, saved
            # from the same trainable tree built above, so strict is right --
            # a key mismatch here means the served model is not the trained one.
            state = torch.load(self.weights_path, map_location=device, weights_only=True)
            missing, unexpected = base_model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"weights_path {self.weights_path} does not match the base model: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected keys "
                    f"(e.g. {(missing or unexpected)[:3]})"
                )
            model = base_model
        else:
            model = load_lora_adapter(base_model, self.lora_adapter_path, is_trainable=False)
        model.to(device)
        model.eval()

        # Replace the model in the config, keep everything else: the base's
        # geometry, at the voxel sizes the finetune was trained at.
        config = Config()
        config.model = model
        for key, value in trained_at_geometry(base_cfg, self.input_voxel_size, self.output_voxel_size).items():
            setattr(config, key, value)
        config.output_channels = base_cfg.output_channels
        config.block_shape = base_cfg.block_shape

        # Copy optional attributes from base config
        for attr in ("channels", "axes_names", "chunk_output_axes", "output_dtype"):
            if hasattr(base_cfg, attr):
                setattr(config, attr, getattr(base_cfg, attr))

        return config

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds.

        Surfaces key base model fields at the top level so the pipeline
        builder UI can display them alongside the finetune-specific fields.
        """
        result = self._with_name_scale({
            "type": "finetune",
            "lora_adapter_path": self.lora_adapter_path,
            "weights_path": self.weights_path,
            "base_model": self.base_model_dict,
        })
        # Its own voxel sizes, before the base's are surfaced below.
        for key in ("input_voxel_size", "output_voxel_size"):
            if getattr(self, key) is not None:
                result[key] = list(getattr(self, key))

        # Surface base model fields for UI display. Not its voxel sizes: those
        # keys are this config's own, the sizes the finetune was trained at,
        # and the base's would be read back as them.
        base = self.base_model_dict
        for key in (
            "channels",
            "checkpoint_path",
            "input_size",
            "output_size",
        ):
            if key in base and key not in result:
                result[key] = base[key]
        if "type" in base:
            result["base_type"] = base["type"]

        return result
