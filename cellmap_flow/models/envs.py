"""Running a model in an environment of its own.

Some models need packages that cannot share cellmap-flow's environment:
Cellpose 4 (Cellpose-SAM) wants a newer torch and numpy than the default
environment's cellpose 3 allows, and transformer models pin their own
stacks. A model entry's ``env`` says where such a model's inference server,
and its finetuning job, run instead::

    models:
      cpsam:
        type: script
        script_path: example/cellpose_sam_model.py
        env: cellpose4

``env`` is one of:

- the name of an environment in cellmap-flow's ``pixi.toml``. The program
  runs as ``pixi run --frozen --manifest-path <pixi.toml> -e <name> ...``,
  which installs the environment from the lockfile on first use. The
  manifest is the one in the checkout cellmap-flow is installed from, or
  ``CELLMAP_FLOW_PIXI_MANIFEST``; pixi is ``PIXI_EXE`` (which ``pixi run``
  sets), else the ``pixi`` on PATH.
- an absolute path to a conda environment or virtualenv that has
  cellmap-flow installed. The program runs as ``<path>/bin/python -P -m
  <module>``.

A value with a ``/`` in it, or starting with ``~``, is a path; anything
else is a pixi environment's name.

A name that ``~/.cellmap_flow/envs.yaml`` maps to a path is that path: an
alias, which wins over a pixi environment of the same name. It is how a
machine without pixi (a conda-only cluster account) supplies the
environments the model types default to::

    cellpose4: /groups/lab/home/me/miniconda3/envs/cellpose4

The file is read each time it is needed, and ``CELLMAP_FLOW_ENVS_FILE``
names another.

A model type can name the environment its models run in when the entry
gives none, as ``ModelConfig.default_env``: a class attribute, or a
property that decides per model. The environment a model runs in
(``model_env``) is its entry's ``env``, else its type's ``default_env``,
else this one; ``env: current`` runs it in this one whatever its type's
default. A default that cannot be used here falls back to this
environment with a warning (``_usable_default`` says when and why), where
an explicit ``env`` that cannot be used is an error.

``env`` is not a constructor argument of any model type, so it is not on
the model's own form or command line: ``registry.build_model`` takes it
off the entry and sets ``ModelConfig.env``, ``to_dict()`` and
``launch_entry`` write it back (never the type's default, so an exported
YAML says only what its author chose), and ``serving.launch`` takes it off
again to put the server in the environment, so the server never sees it.

Only the server and the trainer move. Everything else (the dashboard,
blockwise, ``infer --server-check``) runs the model, when it runs it at
all, in the process's own environment.
"""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from cellmap_flow.config.yaml import ConfigError

logger = logging.getLogger(__name__)

PIXI_MANIFEST_ENV = "CELLMAP_FLOW_PIXI_MANIFEST"
ALIASES_FILE_ENV = "CELLMAP_FLOW_ENVS_FILE"

# `env: current` runs a model in this process's environment, whatever its
# type's default. Not "default": pixi always has an environment of that
# name (the one Fileglancer deploys), and `env: default` already means it.
CURRENT = "current"

# The defaults already warned about. The dashboard reads a model's
# environment on many requests, and once per process is enough.
_warned = set()


def is_path(env: str) -> bool:
    """Whether ``env`` names a directory rather than a pixi environment."""
    return "/" in env or env.startswith("~")


def aliases_file() -> Path:
    """``CELLMAP_FLOW_ENVS_FILE``, else ``~/.cellmap_flow/envs.yaml``."""
    override = os.environ.get(ALIASES_FILE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cellmap_flow" / "envs.yaml"


def aliases() -> Dict[str, str]:
    """The aliases file's names and their paths, ``~`` expanded; {} without one.

    Read on each call, so an alias added while the dashboard runs is used
    by the next launch.

    Raises:
        ConfigError: the file is not a mapping of names to paths.
    """
    path = aliases_file()
    if not path.is_file():
        return {}
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ConfigError(
            f"{path} must map environment names to paths, e.g. "
            "cellpose4: /groups/lab/home/me/miniconda3/envs/cellpose4"
        )
    return {name: os.path.expanduser(value) for name, value in data.items()}


def pixi_manifest() -> Path:
    """The pixi.toml whose environments ``env`` names.

    ``CELLMAP_FLOW_PIXI_MANIFEST``, else the one at the root of the checkout
    this package is installed from (pixi installs it editable from there).
    """
    override = os.environ.get(PIXI_MANIFEST_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "pixi.toml"


def _read_manifest(manifest: Path) -> dict:
    import tomllib

    with open(manifest, "rb") as f:
        return tomllib.load(f)


def pixi_environments(manifest: Path) -> Dict[str, dict]:
    """Each environment ``manifest`` declares, as a table with ``features``.

    ``default`` is always there, as pixi has it even when it is not
    declared. The list form (``name = ["feature", ...]``) becomes
    ``{"features": [...]}``.
    """
    declared = _read_manifest(manifest).get("environments", {})
    environments = {"default": {"features": []}}
    for name, value in declared.items():
        environments[name] = {"features": value} if isinstance(value, list) else dict(value)
    return environments


def pixi_program() -> str:
    """The pixi to run: ``PIXI_EXE``, else the one on PATH, else ``"pixi"``."""
    # An absolute path where one is known: the LSF job's login shell need
    # not have pixi on its PATH.
    return os.environ.get("PIXI_EXE") or shutil.which("pixi") or "pixi"


def has_pixi() -> bool:
    """Whether this machine has a pixi to run (``PIXI_EXE`` or one on PATH)."""
    return bool(os.environ.get("PIXI_EXE") or shutil.which("pixi"))


def _check_directory(what: str, path: str, model_name: str) -> None:
    if not os.path.isabs(path):
        raise ConfigError(f"Model '{model_name}': {what} must be an absolute path")
    if not os.path.exists(os.path.join(path, "bin", "python")):
        raise ConfigError(
            f"Model '{model_name}': {what} has no bin/python; it must be a "
            "conda environment or virtualenv with cellmap-flow installed"
        )


def validate(env, model_name: str = "model") -> str:
    """``env`` checked, with a path's ``~`` expanded; what the model keeps.

    An alias stays a name, so an exported YAML names the environment the
    same way on every machine, and each machine's aliases file says where
    it is. ``current`` is kept as it is.

    Raises:
        ConfigError: ``env`` is not a string, a path (an alias's included)
            is relative or has no ``bin/python``, or no pixi.toml declares
            an environment by that name.
    """
    if not isinstance(env, str) or not env.strip():
        raise ConfigError(f"Model '{model_name}': env must be a pixi environment's name or a path, got {env!r}")
    env = env.strip()
    if env == CURRENT:
        return env
    if is_path(env):
        path = os.path.expanduser(env)
        _check_directory(f"env {env!r}", path, model_name)
        return path
    alias = aliases().get(env)
    if alias is not None:
        _check_directory(f"env {env!r} ({alias}, in {aliases_file()})", alias, model_name)
        return env

    manifest = pixi_manifest()
    if not manifest.is_file():
        raise ConfigError(
            f"Model '{model_name}': env {env!r} names a pixi environment, but there is no "
            f"pixi.toml at {manifest}. Set {PIXI_MANIFEST_ENV} to cellmap-flow's pixi.toml, "
            f"give env as the path of an environment, or map {env} to one in {aliases_file()}"
        )
    environments = pixi_environments(manifest)
    if env not in environments:
        raise ConfigError(
            f"Model '{model_name}': {manifest} has no environment {env!r}; "
            f"it has {', '.join(sorted(environments))}"
        )
    return env


def _directory(env: str) -> Optional[str]:
    """The directory of a path or an alias; None for a pixi environment."""
    if is_path(env):
        return os.path.expanduser(env)
    return aliases().get(env)


def prefix(env: str) -> str:
    """The directory ``env`` is installed in.

    For a pixi environment that is where pixi installs it by default,
    ``.pixi/envs/<name>`` beside the manifest; a pixi configured with
    detached environments puts it elsewhere.
    """
    directory = _directory(env)
    if directory is not None:
        return directory
    return str(pixi_manifest().parent / ".pixi" / "envs" / env)


def is_installed(env: str) -> bool:
    """Whether ``env`` has a python yet (a pixi one is installed on first use)."""
    return os.path.exists(os.path.join(prefix(env), "bin", "python"))


def is_running_in(env: str) -> bool:
    """Whether this process already runs in ``env``: the server or trainer a
    model was moved into, where building the model is what it is there for."""
    try:
        return Path(sys.prefix).resolve() == Path(prefix(env)).resolve()
    except (OSError, ConfigError):
        return False


def _warn_once(env: str, message: str) -> None:
    if env not in _warned:
        _warned.add(env)
        logger.warning(message)


def _usable_default(env: str, model_name: str) -> Optional[str]:
    """``env``, a model type's default environment, or None to run the model
    in this one, with a warning saying why.

    The type chose it, not the user, so a machine that cannot provide it
    is no reason to refuse the model: without a pixi.toml declaring it, or
    without pixi (a conda-only cluster account), the model runs here, as
    it did before its type had a default, and the warning says how to
    provide one (an alias). A pixi environment that is declared but not
    installed is used anyway, since ``pixi run --frozen`` installs it: the
    warning is that the first job then takes minutes, and how to install
    it first. A path or an alias is the user's own, so one that cannot be
    used is an error, as an explicit ``env`` is.
    """
    if is_path(env) or env in aliases():
        return validate(env, model_name)
    manifest = pixi_manifest()
    if not manifest.is_file():
        reason = f"there is no pixi.toml at {manifest}"
    elif env not in pixi_environments(manifest):
        reason = f"{manifest} has no environment {env!r}"
    elif not has_pixi():
        reason = "pixi is not installed"
    else:
        if not is_installed(env):
            _warn_once(env, (
                f"Model '{model_name}' runs in env {env!r}, which is not installed: its first job "
                f"installs it (several minutes). To install it now: cellmap_flow envs install {env}"
            ))
        return env
    _warn_once(env, (
        f"Model '{model_name}' runs in env {env!r} by default, but {reason}, so it runs in this "
        f"environment. To give it one, map {env} to a conda env in {aliases_file()}; "
        f"env: {CURRENT} silences this."
    ))
    return None


def effective(env, default_env, model_name: str = "model") -> Optional[str]:
    """The environment a model runs in, or None for this one.

    ``env`` is the entry's own (validated when the entry was read):
    itself, unless it is ``current``. Without one, ``default_env``, its
    type's, when it can be used here (``_usable_default``).
    """
    if env:
        return None if env == CURRENT else env
    if not default_env or default_env == CURRENT:
        return None
    return _usable_default(default_env, model_name)


def model_env(model_config) -> Optional[str]:
    """``effective`` of a model config's ``env`` and ``default_env``."""
    label = getattr(model_config, "name", None) or type(model_config).__name__
    return effective(getattr(model_config, "env", None), getattr(model_config, "default_env", None), label)


def declared_default(cls):
    """``cls.default_env`` as the class declares it: a name, None, or the
    property (or other descriptor) that decides it per model."""
    import inspect

    return inspect.getattr_static(cls, "default_env", None)


def decides_per_model(declared) -> bool:
    """Whether a ``declared_default`` is decided per model, not one name."""
    return declared is not None and not isinstance(declared, str)


def type_default(cls) -> Optional[str]:
    """The default environment of ``cls``'s models when it is one for all of
    them, without building one; None when it has none or decides per model."""
    declared = declared_default(cls)
    return None if decides_per_model(declared) else declared


def _pixi_run(env: str) -> List[str]:
    return [pixi_program(), "run", "--frozen", "--manifest-path", str(pixi_manifest()), "-e", env]


def server_argv(env: str) -> List[str]:
    """The program that serves a model in ``env``, before its ``--model`` and ``-d``.

    In a pixi environment that is the environment's ``cellmap_flow serve``,
    as the deployment's own CELLMAP_FLOW_SERVER_COMMAND is. A directory need
    not be on PATH, so there it is its python running the CLI; -P keeps the
    job's working directory off sys.path, so a job started inside another
    checkout does not run that checkout's cellmap_flow.
    """
    if _directory(env) is not None:
        return python_argv(env, "cellmap_flow.cli.main") + ["serve"]
    return _pixi_run(env) + ["cellmap_flow", "serve"]


def python(env: str) -> List[str]:
    """The python of ``env``: a directory's ``bin/python``, or ``pixi run ... python``."""
    directory = _directory(env)
    if directory is not None:
        return [os.path.join(directory, "bin", "python")]
    return _pixi_run(env) + ["python"]


def python_argv(env: str, module: str) -> List[str]:
    """``python -P -m <module>`` in ``env``."""
    return python(env) + ["-P", "-m", module]


def lib_dir(env: str) -> str:
    """The lib directory of ``env``, for LD_LIBRARY_PATH.

    A pixi environment's is in its default place (``prefix``); with
    detached environments it is elsewhere, and the job then loads
    libraries as ``pixi run`` alone would.
    """
    return os.path.join(prefix(env), "lib")


def _normalized(package: str) -> str:
    return package.lower().replace("_", "-")


def _brings_peft(pypi_dependencies: dict) -> bool:
    """Whether a feature's pypi-dependencies install peft, the trainer's one
    import a model's own environment is likely to lack: directly, or as
    cellmap-flow's ``finetune`` extra."""
    for package, spec in pypi_dependencies.items():
        if _normalized(package) == "peft":
            return True
        if _normalized(package) == "cellmap-flow" and isinstance(spec, dict):
            if "finetune" in spec.get("extras", []):
                return True
    return False


def finetune_problem(env: str) -> Optional[str]:
    """Why the trainer cannot run in ``env``, or None when it can (as far as
    can be told cheaply).

    A pixi environment is checked against the manifest: one of its features,
    the default feature included unless it sets ``no-default-feature``, must
    install peft. A directory (a path or an alias) is not checked; a trainer
    that cannot import there fails in its job's log.
    """
    if _directory(env) is not None:
        return None
    manifest_path = pixi_manifest()
    manifest = _read_manifest(manifest_path)
    environment = pixi_environments(manifest_path).get(env)
    if environment is None:
        return f"{manifest_path} has no environment {env!r}"
    tables = [manifest.get("feature", {}).get(f, {}) for f in environment.get("features", [])]
    if not environment.get("no-default-feature", False):
        tables.append(manifest)
    if any(_brings_peft(table.get("pypi-dependencies", {})) for table in tables):
        return None
    return (
        f"the pixi environment {env!r} does not install peft, which the trainer needs; "
        f"add the finetune extra of cellmap-flow to it in {manifest_path}"
    )
