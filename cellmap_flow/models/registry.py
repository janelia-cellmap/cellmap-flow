"""The model types cellmap-flow knows, and how a model config is built from strings.

Every ModelConfig subclass is a model type: the built-in ones in
``models/configs`` (imported from models_config) and any that a plugin
defines. The model CLIs
(``cellmap_flow``, ``cellmap_flow_server``, ``cellmap_flow_yaml``), YAML
configs, the dashboard's model form and the finetune CLI's model entries
all look types up here, so a plugin type is a type everywhere.

Nothing is cached. Plugins are loaded when ``cellmap_flow`` is imported, and
a plugin or a test can define a type at any time, so each call walks the
subclasses as they are at that moment.

Strings become constructor arguments in three ways, kept apart on purpose
because each reproduces what its caller has always done:

- ``coerce_cli_args``: click's strings. ``"8,8,8"`` for a tuple becomes
  ``(8, 8, 8)``, ints.
- ``coerce_form_params``: the dashboard form's strings. JSON, or
  ``"8,8,8"`` becomes ``(8.0, 8.0, 8.0)``, floats.
- ``build_model``: YAML values, after ``YAML_ALIASES`` and the voxel-size
  shorthands, then ``coerce_cli_args``.

Importing this module imports neither torch, flask, huggingface_hub nor
``cellmap_flow.globals``. The model config classes, which bring numpy and
funlib (most of a second), are imported only when a function needs them.
``describe_types`` reads constructor signatures only, so it never imports
bioimageio, dacapo or cellmap_models either.
"""

import inspect
import json
import logging
from typing import Any, Dict, List, Tuple, get_type_hints

logger = logging.getLogger(__name__)

# YAML keys accepted in place of a constructor argument's name.
YAML_ALIASES = {
    "checkpoint": "checkpoint_path",
    "classes": "channels",
    "resolution": "input_voxel_size",
    "output_resolution": "output_voxel_size",
    "config_folder": "folder_path",
    "model_path": "model_name",
}

# Constructor arguments that are never required from a user, even without a
# default: a YAML entry's key supplies the name, and scale is optional.
_NEVER_REQUIRED = ("name", "scale")


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
    # "dacapo", HuggingFaceModelConfig "huggingface". (cli_utils also ran a
    # camelCase-to-kebab substitution, but after lower-casing, so it never
    # changed anything.)
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
    """The class for a YAML ``type:``: any case, with ``_`` or ``-`` or neither."""
    normalized = mtype.lower().replace("_", "-")
    for type_name, cls in types.items():
        if type_name == normalized or mtype.lower() == type_name.replace("-", ""):
            return cls
    return None


def model_type(name: str) -> type:
    """The class registered as ``name``, matched as a YAML ``type:`` is.

    Raises:
        ConfigError: no type has that name; the message lists the valid ones.
    """
    from cellmap_flow.utils.config_utils import ConfigError

    types = model_types()
    cls = _match_type(str(name), types)
    if cls is None:
        raise ConfigError(
            f"Unrecognized model type {name!r}. Valid types are: {', '.join(sorted(types))}"
        )
    return cls


def model_classes() -> Dict[str, type]:
    """Every model type by class name (``"ScriptModelConfig"``), plugins included."""
    classes: Dict[str, type] = {}
    for cls in _subclasses(_model_config_base()):
        taken = classes.get(cls.__name__)
        if taken is None:
            classes[cls.__name__] = cls
        elif taken is not cls:
            logger.warning(
                f"Two model config classes are called {cls.__name__}; "
                f"{cls.__module__}.{cls.__qualname__} is left out"
            )
    return classes


def required_params(cls) -> List[str]:
    """The constructor arguments a user must give: no default, and not name or scale."""
    return [
        p
        for p, info in inspect.signature(cls.__init__).parameters.items()
        if p != "self"
        and info.default is inspect.Parameter.empty
        and p not in _NEVER_REQUIRED
    ]


def build_model(entry: Dict[str, Any], name: str):
    """A model config from a YAML model entry; ``name`` is its key in the YAML.

    Raises:
        ConfigError: the entry does not describe a model that can be built.
    """
    from cellmap_flow.utils.config_utils import ConfigError

    model_name = name
    if not isinstance(entry, dict):
        raise ConfigError(f"Model '{model_name}' must be a mapping, got {entry!r}")
    mtype = entry.get("type")
    if not mtype:
        raise ConfigError(f"Model '{model_name}' missing 'type' field")

    types = model_types()
    config_class = _match_type(mtype, types)
    if config_class is None:
        raise ConfigError(
            f"Model '{model_name}' has unrecognized type '{mtype}'. "
            f"Valid types are: {', '.join(sorted(types))}"
        )

    kwargs = {}
    for yaml_key, yaml_value in entry.items():
        if yaml_key == "type":
            continue
        param_name = YAML_ALIASES.get(yaml_key, yaml_key)
        # A voxel size may be written as one number or as a list.
        if param_name in ("input_voxel_size", "output_voxel_size"):
            if isinstance(yaml_value, int):
                yaml_value = (yaml_value, yaml_value, yaml_value)
            elif isinstance(yaml_value, list):
                yaml_value = tuple(yaml_value)
        kwargs[param_name] = yaml_value

    if "name" not in kwargs:
        kwargs["name"] = model_name

    processed_kwargs = coerce_cli_args(config_class, kwargs)

    required = required_params(config_class)
    for param_name in required:
        if param_name in processed_kwargs:
            continue
        if param_name == "output_voxel_size" and "input_voxel_size" in processed_kwargs:
            processed_kwargs["output_voxel_size"] = processed_kwargs["input_voxel_size"]
            logger.warning(
                f"Model '{model_name}' ({mtype}): 'output_voxel_size' not specified, "
                f"using 'input_voxel_size' ({processed_kwargs['input_voxel_size']}) as default"
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
        model = config_class(**processed_kwargs)
        logger.debug(f"Created model '{model_name}': {model}")
        return model
    except TypeError as e:
        raise ConfigError(
            f"Error creating model '{model_name}' ({mtype}): {e}. "
            f"Provided parameters: {processed_kwargs}. "
            f"Required parameters: {required}"
        ) from e
    except (ValueError, OSError) as e:
        # Some constructors read files straight away (a cellmap model's
        # metadata.json), so a wrong path shows up here.
        raise ConfigError(f"Error creating model '{model_name}' ({mtype}): {e}") from e


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


def _parse_comma_separated_values(value: str, target_type: type) -> Any:
    """``"1,2,3"`` as ``[1, 2, 3]`` for int; a single value is a one-item list."""
    if value is None:
        return None
    if "," in value:
        values = [v.strip() for v in value.split(",")]
        if target_type == int:
            return [int(v) for v in values]
        elif target_type == float:
            return [float(v) for v in values]
        return values
    if target_type == int:
        return [int(value)]
    elif target_type == float:
        return [float(value)]
    return [value]


def coerce_cli_args(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """``kwargs`` from the command line, as constructor arguments for ``cls``.

    Only list and tuple arguments given as strings are converted (from
    comma-separated values); None values and names ``cls`` does not take
    are dropped, and everything else is passed through as it is.
    """
    sig = inspect.signature(cls.__init__)
    processed_kwargs = {}

    try:
        type_hints = get_type_hints(cls.__init__)
    except Exception:
        type_hints = {}

    for param_name, param_info in sig.parameters.items():
        if param_name == "self":
            continue

        value = kwargs.get(param_name)
        if value is None:
            continue

        annotation = type_hints.get(param_name, param_info.annotation)
        origin = getattr(annotation, "__origin__", None)

        if origin in (list, tuple) and isinstance(value, str):
            args = getattr(annotation, "__args__", ())
            element_type = args[0] if args else str
            base_element_type, _ = _parse_type_annotation(element_type)
            parsed_values = _parse_comma_separated_values(value, base_element_type)
            processed_kwargs[param_name] = (
                tuple(parsed_values) if origin == tuple else parsed_values
            )
        elif annotation == tuple and isinstance(value, str):
            # A bare `tuple` annotation: ints if they parse, else strings.
            try:
                parsed_values = _parse_comma_separated_values(value, int)
            except (ValueError, TypeError):
                parsed_values = _parse_comma_separated_values(value, str)
            processed_kwargs[param_name] = tuple(parsed_values)
        else:
            processed_kwargs[param_name] = value

    return processed_kwargs


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

    In reverse signature order, the order the options are applied in, and
    so the order short flags are handed out: the last argument gets its
    first letter before an earlier one with the same letter. Every existing
    command's short flags depend on that order. ``reserved_short`` holds the
    command's own short flags (``-d`` and so on), which no argument gets.
    """
    used = set(reserved_short)
    options = []
    for param_name, param in reversed(list(inspect.signature(cls.__init__).parameters.items())):
        option = _click_option(param_name, param, used)
        if option:
            options.append(option)
    return options


# --- the dashboard's model form ------------------------------------------------

def coerce_form_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """The model form's values, as constructor arguments for ``cls``.

    Raises:
        ValueError: a required argument is missing or empty.
    """
    parsed_params = {}
    sig = inspect.signature(cls.__init__)

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls") or param_name not in params:
            continue

        value = params[param_name]
        if value is None or value == "":
            # An empty field leaves an optional argument at its default.
            if param.default != inspect.Parameter.empty:
                continue
            else:
                raise ValueError(f"Required parameter '{param_name}' is missing")

        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation

            if hasattr(annotation, "__origin__") and annotation.__origin__ in (list, tuple):
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = [v.strip() for v in value.split(",")]
                if annotation.__origin__ == tuple:
                    value = tuple(value)

            # A dict (a finetuned model's base_model) comes as JSON.
            elif annotation == dict:
                if isinstance(value, str):
                    value = json.loads(value)

            elif annotation in (int, float):
                value = annotation(value)

            # A bare tuple: JSON, or "16, 16, 16" as floats.
            elif "tuple" in str(annotation).lower():
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except Exception:
                        value = tuple(float(v.strip()) for v in value.split(","))
                if not isinstance(value, tuple):
                    value = tuple(value)

        parsed_params[param_name] = value

    return parsed_params


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
    if param_name in ("input_size", "output_size", "edge_length_to_process", "iteration"):
        return "number"
    return "text"


def describe_types(classes=None) -> Dict[str, Dict[str, Any]]:
    """What the dashboard's model form offers, by class name.

    ``classes`` (class name -> class) defaults to ``model_classes()``.
    """
    if classes is None:
        classes = model_classes()
    registry = {}

    for class_name, cls in classes.items():
        # ScriptModelConfig -> "Script Model", CellMapModelConfig -> "Cell Map Model".
        display_name = class_name.replace("Config", "").replace("ModelConfig", "")
        display_name = "".join(
            [f" {c}" if c.isupper() and i > 0 else c for i, c in enumerate(display_name)]
        )

        params = parameter_info(cls)
        for param_name, param_info in params.items():
            param_info["input_type"] = _input_type(param_name, param_info)

        registry[class_name] = {
            "display_name": display_name,
            "description": f"Create a {display_name} model configuration",
            "parameters": params,
            "class_name": class_name,
        }

    return registry
