"""The model types cellmap-flow knows, and how a model config is built from an entry.

Every ModelConfig subclass is a model type: the built-in ones in
``models/configs`` (imported from models_config) and any that a plugin
defines. The model CLIs
(``cellmap_flow infer``, ``cellmap_flow serve``, ``cellmap_flow yaml``), YAML
configs, the dashboard's model form and the finetune CLI's model entries
all look types up here, so a plugin type is a type everywhere.

Nothing is cached. Plugins are loaded when ``cellmap_flow`` is imported, and
a plugin or a test can define a type at any time, so each call walks the
subclasses as they are at that moment.

A model entry (``models.entry``) becomes a config in one place,
``build_model``: the entry's keys are matched to the constructor's arguments
(``YAML_ALIASES`` first), its values coerced by the constructor's
annotations (``entry.coerce``), its ``env`` checked, and the required
arguments checked before the constructor, which may load files, runs. Every
launcher and the server build models through it, so the same entry gives
the same config from a YAML, ``cellmap_flow infer``, the dashboard's form,
its pickers and ``serve``; ``build_models`` builds a YAML's whole ``models``.

A bad YAML entry or type name is a ``config.yaml.ConfigError``.

A model entry's ``env`` (the environment its server runs in, see
``models.envs``) is not a constructor argument: ``build_model`` checks it
and sets it on the built config, and the dashboard's form offers it beside
each type's arguments. Left out, the type's ``default_env`` applies
(``ModelConfig.effective_env``).

Importing this module imports neither torch, flask, huggingface_hub nor
``cellmap_flow.globals``. The model config classes, which bring numpy and
funlib (most of a second), are imported only when a function needs them.
``describe_types`` reads constructor signatures only, so it never imports
bioimageio, dacapo or cellmap_models either.
"""

import inspect
import logging
import os
from typing import Any, Dict, List, Tuple

from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.models.entry import coerce

logger = logging.getLogger(__name__)

# YAML keys accepted in place of a constructor argument's name.
YAML_ALIASES = {
    "checkpoint": "checkpoint_path",
    "classes": "channels",
    "resolution": "input_voxel_size",
    "output_resolution": "output_voxel_size",
    "config_folder": "folder_path",
    # The bioimage type's model was model_name until 0.3.0, and the docs
    # offered model_path for it.
    "model_name": "model",
    "model_path": "model",
}

# Constructor arguments that are never required from a user, even without a
# default: a YAML entry's key supplies the name, and scale is optional.
_NEVER_REQUIRED = ("name", "scale")

# The model-entry key that is not a constructor argument (models.envs).
ENV_KEY = "env"


def _checked_env(env, model_name: str):
    """A model entry's ``env``, checked; None when it has none (or a blank one).

    Raises:
        ConfigError: ``env`` names no environment that can be used.
    """
    if env is None or env == "":
        return None
    from cellmap_flow.models import envs

    return envs.validate(env, model_name)


def _model_config_base():
    from cellmap_flow.models.models_config import ModelConfig

    return ModelConfig


def _subclasses(base) -> List[type]:
    """Every subclass of ``base``, each once, breadth first.

    All direct subclasses come first, in definition order, then theirs. So
    when two classes claim one name, a built-in type (a direct subclass)
    keeps it against a plugin's subclass of another type.
    """
    found, seen = [], set()
    queue = list(base.__subclasses__())
    while queue:
        cls = queue.pop(0)
        if cls in seen:
            continue
        seen.add(cls)
        found.append(cls)
        queue.extend(cls.__subclasses__())
    return found


def _name_from_class(cls, base_name: str = "ModelConfig") -> str:
    # The class name minus the base's, lower-cased: DaCapoModelConfig is
    # "dacapo", HuggingFaceModelConfig "huggingface". (The CLI's old copy of
    # this also ran a camelCase-to-kebab substitution, but after
    # lower-casing, so it never changed anything.)
    return cls.__name__.replace(base_name, "").lower()


def cli_name_of(cls: type) -> str:
    """The name ``cls`` is registered under: its own ``cli_name``, else one from its class name.

    Only a ``cli_name`` the class sets itself counts. A plugin's
    ``class MyScript(ScriptModelConfig)`` inherits ``cli_name = "script"``;
    taking that as its name would either replace the script type or lose
    the name clash and leave MyScript without a type, and its ``command``
    would ask the server for a plain ScriptModelConfig. It is "myscript".
    """
    name = cls.__dict__.get("cli_name")
    return name if name else _name_from_class(cls)


def model_types() -> Dict[str, type]:
    """Every model type, by the name the CLIs and YAML ``type:`` use.

    Plugins included, subclasses of subclasses too. When two classes claim a
    name the first one found keeps it, and the other is left out with a
    warning.
    """
    types: Dict[str, type] = {}
    for cls in _subclasses(_model_config_base()):
        name = cli_name_of(cls)
        taken = types.get(name)
        if taken is None:
            types[name] = cls
        elif taken is not cls:
            logger.warning(
                f"Model type {name!r} is already {taken.__module__}.{taken.__qualname__}; "
                f"{cls.__module__}.{cls.__qualname__} is not registered under it "
                "(give it a cli_name of its own)"
            )
    return types


def _match_type(mtype: str, types: Dict[str, type]):
    """The class for a ``type``: its registered name in any case, with ``_``
    or ``-`` or neither; or its class name ("ScriptModelConfig"), which the
    dashboard's form posted before 0.3.0."""
    normalized = mtype.lower().replace("_", "-")
    for type_name, cls in types.items():
        if type_name == normalized or mtype.lower() == type_name.replace("-", ""):
            return cls
    return next((cls for cls in types.values() if cls.__name__ == mtype), None)


def model_type(name: str) -> type:
    """The class registered as ``name``: a type name as a YAML ``type:``
    gives it, or a class name.

    Raises:
        ConfigError: no type has that name; the message lists the valid ones.
    """
    types = model_types()
    cls = _match_type(str(name), types)
    if cls is None:
        raise ConfigError(
            f"Unrecognized model type {name!r}. Valid types are: {', '.join(sorted(types))}"
        )
    return cls


def required_params(cls) -> List[str]:
    """The constructor arguments a user must give: no default, and not name or scale."""
    return [
        p
        for p, info in inspect.signature(cls.__init__).parameters.items()
        if p != "self"
        and info.default is inspect.Parameter.empty
        and p not in _NEVER_REQUIRED
    ]


def build_model(entry: Dict[str, Any], name: str = None, base_dir: str = None):
    """The model config a model entry describes (``models.entry``).

    ``name`` is the model's name when the entry gives none: its key under a
    YAML's ``models``, or what a launcher calls it. ``base_dir`` is the YAML
    file's folder, when there is one: a relative path in one of the type's
    ``config_relative_paths`` is taken from there rather than from the
    working directory.

    Raises:
        ConfigError: the entry does not describe a model that can be built.
    """
    if not isinstance(entry, dict):
        raise ConfigError(f"Model '{name or '?'}' must be a mapping, got {entry!r}")
    mtype = entry.get("type")
    model_name = entry.get("name") or name or mtype
    if not mtype:
        raise ConfigError(f"Model '{model_name}' missing 'type' field")

    types = model_types()
    config_class = _match_type(str(mtype), types)
    if config_class is None:
        raise ConfigError(
            f"Model '{model_name}' has unrecognized type '{mtype}'. "
            f"Valid types are: {', '.join(sorted(types))}"
        )

    kwargs = {YAML_ALIASES.get(key, key): value for key, value in entry.items() if key not in ("type", ENV_KEY)}
    if kwargs.get("name") is None and name is not None:
        kwargs["name"] = name
    for param_name in getattr(config_class, "config_relative_paths", ()):
        value = kwargs.get(param_name)
        relative = isinstance(value, str) and value and "://" not in value and not os.path.isabs(
            os.path.expanduser(value))
        if base_dir and relative:
            kwargs[param_name] = os.path.join(base_dir, value)
    # Checked before the constructor, which may load files, runs.
    env = _checked_env(entry.get(ENV_KEY), model_name)

    try:
        arguments = coerce(config_class, kwargs)
    except ValueError as e:
        raise ConfigError(f"Model '{model_name}' ({mtype}): {e}") from e

    required = required_params(config_class)
    for param_name in required:
        if param_name in arguments:
            continue
        if param_name == "output_voxel_size" and "input_voxel_size" in arguments:
            arguments["output_voxel_size"] = arguments["input_voxel_size"]
            logger.warning(
                f"Model '{model_name}' ({mtype}): 'output_voxel_size' not specified, "
                f"using 'input_voxel_size' ({arguments['input_voxel_size']}) as default"
            )
            continue
        # Given under an alias, but as null: left for the constructor to reject.
        if not any(
            mapped == param_name and yaml_key in entry
            for yaml_key, mapped in YAML_ALIASES.items()
        ):
            raise ConfigError(
                f"Model '{model_name}' ({mtype}) missing required parameter '{param_name}'"
            )

    try:
        model = config_class(**arguments)
        logger.debug(f"Created model '{model_name}': {model}")
    except TypeError as e:
        raise ConfigError(
            f"Error creating model '{model_name}' ({mtype}): {e}. "
            f"Provided parameters: {arguments}. "
            f"Required parameters: {required}"
        ) from e
    except (ValueError, OSError) as e:
        # Some constructors read files straight away (a cellmap model's
        # metadata.json), so a wrong path shows up here.
        raise ConfigError(f"Error creating model '{model_name}' ({mtype}): {e}") from e
    if env:
        model.env = env
    return model


def build_models(model_entries, base_dir: str = None) -> list:
    """The model configs for a YAML's ``models``: ``build_model`` for each entry.

    ``base_dir``, the YAML file's folder, is where a relative path of a
    type's ``config_relative_paths`` is taken from.

    ``model_entries`` maps each model's name to its entry, or is a list of
    entries that each have a ``name``::

        models:
          my_model_1:
            type: cellmap
            checkpoint_path: /path/to/checkpoint
          my_model_2:
            type: dacapo
            run_name: my_run
            iteration: 50000

    Raises:
        ConfigError: an entry does not describe a model that can be built.
    """
    if isinstance(model_entries, list):
        entries = {}
        for entry in model_entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ConfigError("Each model entry in the list must have a 'name' field.")
            entries[entry["name"]] = entry
        model_entries = entries

    return [build_model(entry, model_name, base_dir) for model_name, entry in model_entries.items()]


# --- click's strings -----------------------------------------------------------

def _parse_type_annotation(annotation) -> Tuple[type, bool]:
    """(base type, optional) for an annotation, as the CLI options use it.

    The option types of every model command come from this, so it keeps
    its quirks: the Union check compares against ``type`` and so never
    matches a Union, which makes ``Optional[int]`` a str option.
    """
    if isinstance(annotation, str):
        if annotation == "str":
            return str, False
        elif annotation == "int":
            return int, False
        elif annotation == "float":
            return float, False
        elif annotation == "bool":
            return bool, False
        return str, False

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is type(None) or (
        hasattr(annotation, "__name__") and annotation.__name__ == "NoneType"
    ):
        return str, True

    if origin is type(None.__class__.__bases__[0]):
        if type(None) in args:
            non_none_types = [t for t in args if t is not type(None)]
            if non_none_types:
                base_type, _ = _parse_type_annotation(non_none_types[0])
                return base_type, True
        return str, False

    # A list option is one comma-separated string of its element type.
    if origin is list:
        if args:
            base_type, _ = _parse_type_annotation(args[0])
            return base_type, False
        return str, False

    if origin is tuple:
        return str, False

    if annotation in (str, int, float, bool):
        return annotation, False

    return str, False


def _click_option(param_name: str, param_info: inspect.Parameter, used_short_names=None):
    """The ``click.option`` arguments for one constructor argument, or None for self.

    The short flag is the argument's first letter when no earlier option has
    taken it; ``used_short_names`` records the ones taken.
    """
    annotation = param_info.annotation
    default = param_info.default

    if param_name == "self":
        return None

    if used_short_names is None:
        used_short_names = set()

    base_type, is_optional = _parse_type_annotation(annotation)

    is_required = default is inspect.Parameter.empty and param_name not in _NEVER_REQUIRED

    option_name = param_name.replace("_", "-")

    short_name = None
    candidate_short = "-" + param_name[0] if len(param_name) > 0 else None
    if candidate_short and candidate_short not in used_short_names:
        short_name = candidate_short
        used_short_names.add(short_name)

    long_name = "--" + option_name

    option_config = {
        "param_decls": [short_name, long_name] if short_name else [long_name],
        "required": is_required,
        "type": base_type if base_type in (str, int, float, bool) else str,
        "help": f"Parameter: {param_name}",
    }

    if default is not inspect.Parameter.empty and default is not None:
        option_config["default"] = default
        option_config["required"] = False
        option_config["help"] += f" (default: {default})"
    elif not is_required:
        option_config["default"] = None
        option_config["help"] += " (optional)"

    if annotation != inspect.Parameter.empty:
        origin = getattr(annotation, "__origin__", None)
        if origin in (list, tuple):
            option_config["help"] += " [comma-separated values]"

    return option_config


def click_options(cls, reserved_short) -> List[dict]:
    """The ``click.option`` arguments for each of ``cls``'s constructor arguments.

    In signature order, which is also the order short flags are handed out
    in: an argument gets its first letter unless an earlier argument has it
    already (``script -s`` is ``--script-path``, and ``--scale`` has no short
    flag). ``reserved_short`` holds the command's own short flags (``-d`` and
    so on), which no argument gets.
    """
    used = set(reserved_short)
    options = []
    for param_name, param in inspect.signature(cls.__init__).parameters.items():
        option = _click_option(param_name, param, used)
        if option:
            options.append(option)
    return options


def print_available_models(cli_command_name: str = "cellmap_flow"):
    """Print every model type and its constructor arguments, for ``list-models``.

    ``cli_command_name`` is the command the closing hint names.
    """
    import click

    click.echo("Available model configurations:\n")
    for cli_name, config_class in sorted(model_types().items()):
        click.echo(f"  {cli_name:20s} - {config_class.__name__}")

        params = [p for p in inspect.signature(config_class.__init__).parameters if p != "self"]
        if params:
            click.echo(f"                       Parameters: {', '.join(params)}")

    click.echo(f"\nUse '{cli_command_name} <model-name> --help' for detailed parameter information.")


# --- the dashboard's model form ------------------------------------------------

def parameter_info(cls) -> Dict[str, Any]:
    """Each constructor argument of ``cls``: name, required, description, type and default."""
    sig = inspect.signature(cls.__init__)
    params = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_info = {
            "name": param_name,
            "required": param.default == inspect.Parameter.empty,
            "description": param_name.replace("_", " ").title(),
        }

        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
            # list[str] is "list", tuple is "tuple".
            if hasattr(annotation, "__origin__"):
                param_info["type"] = str(annotation.__origin__.__name__)
            else:
                param_info["type"] = (
                    annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
                )
        else:
            param_info["type"] = "string"

        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default

        params[param_name] = param_info

    return params


def _input_type(param_name: str, param_info: Dict[str, Any]) -> str:
    """Which form control the dashboard draws for an argument."""
    lowered = param_name.lower()
    if "path" in lowered or "checkpoint" in lowered:
        return "file"
    if "channels" in lowered or "voxel_size" in lowered:
        return "textarea"  # for multi-line JSON
    if param_info.get("type") in ("dict",):
        return "textarea"  # for JSON dicts
    if param_name in ("input_size", "output_size", "iteration"):
        return "number"
    return "text"


def describe_types() -> Dict[str, Dict[str, Any]]:
    """What the dashboard's model form offers, by type name (``model_types``).

    Each type's ``display_name``, ``description``, ``parameters``
    (``parameter_info`` with the form control for each), its ``type``, the
    name the form posts, and its ``class_name``.
    """
    registry = {}

    for type_name, cls in model_types().items():
        class_name = cls.__name__
        # ScriptModelConfig -> "Script Model", CellMapModelConfig -> "Cell Map Model".
        display_name = class_name.replace("Config", "").replace("ModelConfig", "")
        display_name = "".join(
            [f" {c}" if c.isupper() and i > 0 else c for i, c in enumerate(display_name)]
        )

        params = parameter_info(cls)
        for param_name, param_info in params.items():
            param_info["input_type"] = _input_type(param_name, param_info)
        # Every type's, though no constructor takes it; build_model takes it
        # off before building.
        params.setdefault(ENV_KEY, {
            "name": ENV_KEY,
            "required": False,
            "description": "Environment (pixi env, alias or path; blank: the type's default)",
            "type": "str",
            "input_type": "text",
        })

        registry[type_name] = {
            "display_name": display_name,
            "description": f"Create a {display_name} model configuration",
            "parameters": params,
            "type": type_name,
            "class_name": class_name,
        }

    return registry
