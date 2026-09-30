"""``ModelConfig``, the base of every model type, and the helpers the types share.

A model type is a ModelConfig subclass. The CLIs, YAML ``type:`` entries
and the dashboard's model form are built from its constructor's signature,
and its ``_get_config`` builds the model and returns a ``Config`` that
describes it. What the rest of cellmap-flow reads:

- ``config``: that Config, built on first use and checked: the required
  attributes, then the declared shapes against a forward pass, unless
  ``validate_model_shapes`` is off or ``check_shapes_on_warmup`` leaves it
  to a ModelRunner's warmup forward, which calls ``check_output_shape``.
- ``geometry``: the Config's shapes and voxel sizes, as a ``ModelGeometry``.
- ``chunk_output_axes``, ``output_dtype``: what a processed chunk is.
- ``to_dict()``: the model entry ``registry.build_model`` rebuilds it from.
- ``command``: the ``cellmap_flow_server`` arguments that rebuild it.
"""

import inspect
import logging
import shlex

import numpy as np

from cellmap_flow.models.geometry import ModelGeometry

logger = logging.getLogger(__name__)


def _get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    # Imported here rather than at module scope: importing torch costs
    # ~7s, and the CLI builds its command list from this module, so
    # `cellmap_flow --help` paid that before printing anything.
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def _as_int_tuple(value):
    """(178, 178, 178) from 178, "178,178,178", [178, 178, 178] or a Coordinate.

    The server CLI hands unannotated constructor arguments over as strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace("(", "").replace(")", "").split(",") if p.strip()]
        value = [float(p) for p in parts]
        if len(value) == 1:
            value = value[0]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (int(value),) * 3
    return tuple(int(v) for v in value)


def _plain(value):
    """``value`` with tuples, arrays and numpy numbers as lists and Python numbers.

    A tuple in a model's to_dict() would reach the exported YAML as a
    ``!!python/tuple`` tag, which ``yaml.safe_load`` refuses.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _given_init_params(cls, args, kwargs):
    """The arguments ``cls(*args, **kwargs)`` passes, by parameter name, or None.

    None when they do not fit the signature; the constructor then raises
    its own error.
    """
    try:
        sig = inspect.signature(cls.__init__)
        bound = sig.bind(None, *args, **kwargs)  # None stands in for self
    except (TypeError, ValueError):
        return None
    params = {}
    for pname, value in list(bound.arguments.items())[1:]:
        kind = sig.parameters[pname].kind
        if kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if kind is inspect.Parameter.VAR_KEYWORD:
            params.update(value)
        else:
            params[pname] = value
    return params


def _cli_value(value):
    """One constructor argument as the server CLI parses it back."""
    if isinstance(value, dict):
        # FinetuneModelConfig decodes this back into the dict.
        from cellmap_flow.utils.web_utils import encode_to_str

        return encode_to_str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def command_argv(cls, params: dict) -> list:
    """``cellmap_flow_server`` arguments that build ``cls(**params)``.

    The name the server CLI registers ``cls`` under, then ``--arg value``
    for each constructor argument in signature order, leaving out None.
    Each value is written the way the server CLI parses it back.
    """
    from cellmap_flow.models.registry import cli_name_of

    argv = [str(cli_name_of(cls))]
    for name in list(inspect.signature(cls.__init__).parameters)[1:]:
        value = params.get(name)
        if value is None:
            continue
        argv += [f"--{name.replace('_', '-')}", _cli_value(value)]
    return argv


class ModelConfig:
    def __new__(cls, *args, **kwargs):
        # Remember the constructor arguments, for the default to_dict().
        # Recorded here rather than by wrapping each subclass's __init__,
        # whose signature the CLIs and the model form are built from.
        # Unpickling and copying call __new__ with no arguments, record
        # None, and then restore the original's.
        self = super().__new__(cls)
        self._init_params = _given_init_params(cls, args, kwargs)
        return self

    def __init__(self):
        self._config = None
        self.validate_model_shapes = True
        # Set by an Inferencer before it builds the config: the declared shapes
        # are then checked on its warmup forward, on the device that serves,
        # instead of by a separate forward here, on whatever device the loader
        # left the model on. For a script model that is the CPU, where the
        # extra forward took 7-13 s of the server's start.
        self.check_shapes_on_warmup = False

    def __str__(self) -> str:
        elms = []
        for k, v in vars(self).items():
            if k in ("_init_params", "_geometry"):
                continue
            if isinstance(v, np.ndarray):
                elms.append(f"{k}: type={type(v)} shape={v.shape}\n")
            else:
                elms.append(f"{k}: {v}\n")
        return f"{type(self).__name__}({', '.join(elms)})"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            self._config = self._get_config()
            self._validate_config()
        return self._config

    def _validate_config(self):
        """Ensure config has required attributes and shapes are consistent."""
        required = [
            "read_shape",
            "write_shape",
            "input_voxel_size",
            "output_voxel_size",
            "output_channels",
            "block_shape",
        ]
        if not (hasattr(self._config, "model") or hasattr(self._config, "predict")):
            required.insert(0, "model or predict")

        for attr in required:
            if not hasattr(self._config, attr):
                raise AttributeError(f"{attr} not found in config")

        if not self.validate_model_shapes:
            logger.info("Skipping model shape validation for %s", type(self).__name__)
        elif getattr(self, "check_shapes_on_warmup", False):
            logger.info("Model shapes will be checked on the warmup forward")
        else:
            self._validate_model_shapes()
        logger.warning(f"Model config validated: {self.__str__()}")

    def _validate_model_shapes(self):
        """Run a dummy forward pass to verify declared shapes match actual model output.

        For a config built without an Inferencer to serve it (the dashboard
        resolving geometry, a script); an Inferencer checks its warmup
        forward's output instead, with check_output_shape.
        """
        # Imported here rather than at module scope: importing torch costs
        # ~7s, and the CLI builds its command list from this module, so
        # `cellmap_flow --help` paid that before printing anything.
        import torch

        config = self._config
        if not hasattr(config, "model"):
            return

        model = config.model
        if not isinstance(model, torch.nn.Module):
            # TensorFlow/ONNX/cellpose scripts set model to None or a non-torch
            # object and run through process_chunk; there is nothing to forward.
            return
        input_size = np.array(config.read_shape) // np.array(config.input_voxel_size)

        try:
            first_param = next(model.parameters(), None)
            device = first_param.device if first_param is not None else "cpu"
            dummy = torch.zeros(
                (1, 1, *[int(s) for s in input_size]), device=device
            )
            was_training = model.training
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            if was_training:
                model.train()
        except (RuntimeError, TypeError) as e:
            logger.warning(f"Could not validate model shapes (forward pass failed): {e}")
            return
        self.check_output_shape(out.shape)

    def check_output_shape(self, output_shape):
        """Raise ValueError if a forward's ``output_shape`` (batch first)
        contradicts the declared write_shape, block_shape or output_channels.
        """
        config = self._config
        input_size = np.array(config.read_shape) // np.array(config.input_voxel_size)
        declared_output_size = np.array(config.write_shape) // np.array(
            config.output_voxel_size
        )
        declared_block_spatial = np.array(config.block_shape)[:3]
        actual_output = np.array(tuple(output_shape)[1:])  # drop batch dim
        # Determine actual spatial shape (skip channel dim if present)
        if len(actual_output) == 4:
            actual_channels = actual_output[0]
            actual_spatial = actual_output[1:]
        elif len(actual_output) == 3:
            actual_channels = 1
            actual_spatial = actual_output
        else:
            logger.warning(
                f"Unexpected model output ndim={len(actual_output)}, "
                "skipping shape validation"
            )
            return

        errors = []
        if not np.array_equal(actual_spatial, declared_output_size):
            errors.append(
                f"write_shape mismatch: declared write_shape / output_voxel_size = "
                f"{declared_output_size.tolist()} but model actually outputs "
                f"spatial shape {actual_spatial.tolist()}. "
                f"Expected write_shape = "
                f"{(actual_spatial * np.array(config.output_voxel_size)).tolist()}"
            )
        if not np.array_equal(actual_spatial, declared_block_spatial):
            errors.append(
                f"block_shape mismatch: declared block_shape spatial dims = "
                f"{declared_block_spatial.tolist()} but model actually outputs "
                f"spatial shape {actual_spatial.tolist()}. "
                f"Expected block_shape = "
                f"{[*actual_spatial.tolist(), int(actual_channels)]}"
            )
        if int(actual_channels) != int(config.output_channels):
            errors.append(
                f"output_channels mismatch: declared {config.output_channels} "
                f"but model actually outputs {int(actual_channels)} channels"
            )
        if errors:
            msg = (
                f"Script config shape validation failed for "
                f"{getattr(self, 'script_path', 'unknown')}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
            raise ValueError(msg)

        logger.info(
            f"Shape validation passed: input {input_size.tolist()} -> "
            f"output spatial {actual_spatial.tolist()}, "
            f"channels {int(actual_channels)}"
        )

    @property
    def geometry(self) -> ModelGeometry:
        """The config's geometry as one ModelGeometry, read once per built config."""
        config = self.config
        cached = self.__dict__.get("_geometry")
        if cached is None or cached[0] is not config:
            cached = (
                config,
                ModelGeometry.from_config(
                    config,
                    chunk_output_axes=self.chunk_output_axes,
                    output_dtype=self.output_dtype,
                ),
            )
            self._geometry = cached
        return cached[1]

    @property
    def chunk_output_axes(self) -> tuple[str, ...]:
        """Returns the axes order of processed chunk output. Defaults to ('c', 'z', 'y', 'x').

        Models can override by setting config.chunk_output_axes.
        Note: this is distinct from config.output_axes used by BioModelConfig
        for raw bioimageio model axes.
        """
        if hasattr(self.config, "chunk_output_axes"):
            return tuple(self.config.chunk_output_axes)
        return ("c", "z", "y", "x")

    @property
    def output_dtype(self):
        """Returns the output dtype of the model. Defaults to np.float32."""
        if hasattr(self.config, "output_dtype"):
            return self.config.output_dtype
        # ModelConfig does not set name; a plugin type need not either.
        label = getattr(self, "name", None) or type(self).__name__
        logger.warning(
            f"Model {label} does not define output_dtype, defaulting to np.float32"
        )
        return np.float32

    def to_dict(self):
        """This config as a model entry, which ``registry.build_model`` rebuilds.

        ``{"type": <registered name>, **the constructor arguments given}``,
        leaving out None as the built-in types leave out an unset name or
        scale. The built-in types write their own; this one serves a plugin
        type that does not, so exporting its config and launching its server
        work without one.
        """
        params = self.__dict__.get("_init_params")
        if params is None:
            raise NotImplementedError(
                f"{type(self).__name__} was not built through its constructor, "
                "so it needs a to_dict() of its own"
            )
        from cellmap_flow.models.registry import cli_name_of

        result = {"type": cli_name_of(type(self))}
        result.update({k: _plain(v) for k, v in params.items() if v is not None})
        return result

    def _with_name_scale(self, result: dict) -> dict:
        """``result`` with name and scale added after its keys, when they are set."""
        if getattr(self, "name", None) is not None:
            result["name"] = self.name
        if getattr(self, "scale", None) is not None:
            result["scale"] = self.scale
        return result

    def _launch_params(self) -> dict:
        """The constructor arguments a launched server needs; to_dict() by default."""
        return self.to_dict()

    @property
    def command(self) -> str:
        """``cellmap_flow_server`` arguments that rebuild this exact config.

        Generated from to_dict() so no constructor argument is left behind
        (hand-written commands dropped Fly's input/output size, which then
        silently fell back to 178/56, and Bio's required voxel size), with
        every token shell-quoted: bsub runs it through ``bash -c`` and local
        launches through ``shlex.split``. The class is named as the server
        CLI registers it (``registry.cli_name_of``), so it rebuilds this
        class and not a parent it inherited cli_name from.
        """
        return shlex.join(command_argv(type(self), self._launch_params()))
