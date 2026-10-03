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
- ``launch_entry``: the entry a launched server rebuilds it from
  (``cellmap_flow serve --model <it, as JSON>``).
- ``command``: the ``cellmap_flow_server <type>`` arguments that rebuild
  it, which launchers passed before 0.3.0; that form of the server goes in
  the release after it.
- ``env``: the environment its entry names for its server, when not this
  one (``models.envs``). Set after construction, never a constructor
  argument; to_dict() and launch_entry carry it.
- ``default_env``: the environment a type's models run in when their
  entry names none; ``effective_env``: the one a model runs in, of the two.
"""

import functools
import inspect
import logging
import shlex
from types import ModuleType
from typing import Any

import numpy as np

from cellmap_flow.models.geometry import DEFAULT_OUTPUT_AXES, ModelGeometry, _numbers, _voxels

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


def _voxel_size(value):
    """A voxel size given as one number, "5.24,4,4" or one per axis: three numbers, ints kept ints.

    Unlike _as_int_tuple, which truncates a 5.24 nm voxel to 5.
    """
    if isinstance(value, str):
        value = [float(v) for v in value.replace("(", "").replace(")", "").split(",") if v.strip()]
    if np.ndim(value) == 0:
        value = [value] * 3
    elif len(value) == 1:
        value = list(value) * 3
    return _numbers(float(v) for v in value)


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
        from cellmap_flow.serving.protocol import encode_to_str

        return encode_to_str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def model_entry(cls, params: dict) -> dict:
    """The model entry ``cellmap_flow serve --model`` builds ``cls(**params)`` from.

    ``{"type": <the name the registry has cls under>, **params}``, leaving
    out None. The server rebuilds it with ``registry.build_model``, as a
    YAML's entry is rebuilt, so ``params`` may be to_dict()'s, extra keys
    and all.
    """
    from cellmap_flow.models.registry import cli_name_of

    rest = {k: v for k, v in params.items() if k != "type" and v is not None}
    return {"type": cli_name_of(cls), **rest}


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


class ModelEnvError(RuntimeError):
    """A model that runs in its own environment was built in another, which
    lacks its packages (``ModelConfig.config``)."""


def _missing_module(error: BaseException):
    """The ImportError behind ``error``, as text, or None.

    A script model's error is load_safe_config's RuntimeError, raised from
    the script's own, so the causes are followed.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, ImportError):
            return str(error)
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return None


DEFAULT_AXES_NAMES = ["x", "y", "z"]


class Config:
    """What a model type's ``_get_config`` returns: the model, its geometry
    and anything else its script or checkpoint defines, as attributes.

    Built from keyword arguments (a script's globals, for a script model),
    which ``to_dict()`` and ``get()`` read back. ``axes_names`` defaults to
    x, y, z.
    """

    def __init__(self, **kwargs):
        self.axes_names = kwargs.get("axes_names", DEFAULT_AXES_NAMES)
        self.__dict__.update(kwargs)
        self.kwargs = kwargs

    def __str__(self) -> str:
        elms = []
        for k, v in vars(self).items():
            if any(x in k for x in ["kwargs", "__"]) or isinstance(v, ModuleType):
                continue
            if ["model","checkpoint"].__contains__(k):
                elms.append(f"{k}")
                continue
            if inspect.ismethod(v) or inspect.isfunction(v):
                # A method bound to this Config (BioModelConfig's
                # process_chunk) reprs as "<bound method ... of Config(...)>",
                # which recursed until RecursionError.
                elms.append(f"{k}: <function {getattr(v, '__name__', '?')}>")
                continue
            elms.append(f"{k}: {v}")
        newline = '\n'
        return f"{type(self).__name__}({newline.join(elms)})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self):
        """
        Returns the configuration as a dictionary.
        """
        return self.kwargs

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets the value of a configuration key.
        """
        return self.kwargs.get(key, default)


def _with_env(to_dict):
    """``to_dict`` with the config's ``env`` added to the entry it returns."""

    @functools.wraps(to_dict)
    def to_dict_with_env(self):
        result = to_dict(self)
        if getattr(self, "env", None) and isinstance(result, dict):
            result["env"] = self.env
        return result

    to_dict_with_env._adds_env = True
    return to_dict_with_env


class ModelConfig:
    # The environment this model's server and finetuning job run in, when it
    # is not cellmap-flow's own: a pixi environment's name or a directory
    # (models.envs). registry.build_model sets it from a model entry's
    # `env`; it is not a constructor argument, because the server rebuilds
    # the model from the same entry and must not get it back.
    env = None
    # The environment this type's models run in when their entry gives no
    # `env` (`env: current` opts out). A subclass sets a name, or makes it a
    # property that decides per model. Never written by to_dict(), which
    # says only what the entry said; read through effective_env.
    default_env = None
    # Whether the finetune trainer can train this type's models: it trains
    # the module ``trainable_model()`` gives, and serves the result through
    # ``serve_trained``. A type sets it once those two do the right thing for
    # it; the trainer refuses the others before submitting a job.
    finetunable = False

    def finetune_modes(self):
        """What this model can be finetuned with: ("lora", "full"), ("full",)
        or (). For the dashboard, so it must not build the network: a type
        whose networks differ (a compiled one takes no LoRA adapter) reads
        it from what describes the model. The trainer checks the module it
        gets the same way (``finetune.trainable.finetune_modes``)."""
        return ("lora", "full") if type(self).finetunable else ()

    def trainable_model(self):
        """The torch module the finetune trainer trains, or None for
        ``config.model`` as it is (``finetune.model_loading.load_trainable_model``).

        The trainer feeds it a float tensor (B, 1, Z, Y, X), normalized as
        the dashboard's input normalization does, at the input voxel size and
        read shape, and wants (B, C, Z', Y', X') at the write shape back. A
        type whose ``config.model`` is not such a module (a bioimage.io
        prediction pipeline, a Cellpose model object) returns one that is,
        sharing its parameters, so that what is trained is what is served.
        """
        return None

    def serve_trained(self, config, module):
        """Make ``config`` serve ``module``, the trained ``trainable_model()``.

        ``config`` is this model's own Config (the trainer's live server) or
        a new one with its geometry (a finetuned model's,
        ``FinetuneModelConfig``). By default the module is the model, run by
        the inferencer's own forward; a type that predicts through its own
        ``process_chunk`` sets that up here instead.
        """
        config.model = module

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # No type's to_dict() knows env, which is not among the arguments it
        # writes, so add it to each type's own, plugins' included: an
        # exported YAML, a finetuned model's base_model and the trainer's
        # model entry then keep the environment.
        own = cls.__dict__.get("to_dict")
        if own is not None and not getattr(own, "_adds_env", False):
            cls.to_dict = _with_env(own)

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

    @property
    def effective_env(self):
        """The environment this model's server and finetuning job run in, or
        None for this one: ``env``, else ``default_env`` (``envs.effective``)."""
        from cellmap_flow.models import envs

        return envs.model_env(self)

    def _get_config(self):
        raise NotImplementedError()

    @property
    def config(self):
        if self._config is None:
            try:
                self._config = self._get_config()
            except Exception as e:
                from cellmap_flow.models import envs

                missing = _missing_module(e)
                env = self.effective_env if missing else None
                if env and not envs.is_running_in(env):
                    # The model's packages are in its own environment, not in
                    # this process's; say so rather than show a bare
                    # ImportError from deep inside its script.
                    label = getattr(self, "name", None) or type(self).__name__
                    raise ModelEnvError(
                        f"Model {label} runs in its own environment ({env}), and this "
                        f"process cannot build it: {missing}. Read what you need from its "
                        "running server instead."
                    ) from e
                raise
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
        # Not //: 1865.44 / 10.48 is a hair under 178 in floats.
        input_size = _voxels(config.read_shape, config.input_voxel_size)

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
        input_size = np.array(_voxels(config.read_shape, config.input_voxel_size))
        declared_output_size = np.array(_voxels(config.write_shape, config.output_voxel_size))
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
            # A script is named by its file, the one to fix; the others by name.
            which = getattr(self, "script_path", None) or getattr(self, "name", None)
            msg = (
                f"{type(self).__name__} shape validation failed"
                + (f" for {which}" if which else "")
                + ":\n"
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
        """The axes of a processed chunk; by default ``geometry.DEFAULT_OUTPUT_AXES``,
        ("c", "z", "y", "x").

        Models can override by setting config.chunk_output_axes. The server
        and blockwise read it only for whether a chunk has a channel axis:
        one that has must have it first, and its spatial axes are the raw
        data's, in the raw data's order.
        """
        if hasattr(self.config, "chunk_output_axes"):
            return tuple(self.config.chunk_output_axes)
        return DEFAULT_OUTPUT_AXES

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
        if self.env:
            result["env"] = self.env
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
    def launch_entry(self) -> dict:
        """The model entry a launched server rebuilds this exact config from.

        to_dict(), less what only the pipeline builder shows (a Hugging Face
        repo's downloaded metadata), under the name the registry has this
        class under (``registry.cli_name_of``), so that the server rebuilds
        this class and not a parent it inherited cli_name from. With the
        entry's own ``env``, if it has one, which ``serving.launch`` takes off
        again; the server's environment is ``effective_env``, which it reads
        from the config.
        """
        entry = model_entry(type(self), self._launch_params())
        if self.env:
            entry["env"] = self.env
        return entry

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
