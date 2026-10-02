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

``env`` is not a constructor argument of any model type, so it is not on
the model's own form or command line: ``registry.build_model`` takes it
off the entry and sets ``ModelConfig.env``, ``to_dict()`` and
``launch_entry`` write it back, and ``serving.launch`` takes it off again
to put the server in the environment, so the server never sees it.

Only the server and the trainer move. Everything else (the dashboard,
blockwise, ``infer --server-check``) runs the model, when it runs it at
all, in the process's own environment.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from cellmap_flow.config.yaml import ConfigError

PIXI_MANIFEST_ENV = "CELLMAP_FLOW_PIXI_MANIFEST"


def is_path(env: str) -> bool:
    """Whether ``env`` names a directory rather than a pixi environment."""
    return "/" in env or env.startswith("~")


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


def _pixi_program() -> str:
    # An absolute path where one is known: the LSF job's login shell need
    # not have pixi on its PATH.
    return os.environ.get("PIXI_EXE") or shutil.which("pixi") or "pixi"


def validate(env, model_name: str = "model") -> str:
    """``env`` checked, with a path's ``~`` expanded; what the model keeps.

    Raises:
        ConfigError: ``env`` is not a string, a path is relative or has no
            ``bin/python``, or no pixi.toml declares an environment by that
            name.
    """
    if not isinstance(env, str) or not env.strip():
        raise ConfigError(f"Model '{model_name}': env must be a pixi environment's name or a path, got {env!r}")
    env = env.strip()
    if is_path(env):
        path = os.path.expanduser(env)
        if not os.path.isabs(path):
            raise ConfigError(f"Model '{model_name}': env {env!r} must be an absolute path")
        if not os.path.exists(os.path.join(path, "bin", "python")):
            raise ConfigError(
                f"Model '{model_name}': env {env!r} has no bin/python; it must be a "
                "conda environment or virtualenv with cellmap-flow installed"
            )
        return path

    manifest = pixi_manifest()
    if not manifest.is_file():
        raise ConfigError(
            f"Model '{model_name}': env {env!r} names a pixi environment, but there is no "
            f"pixi.toml at {manifest}. Set {PIXI_MANIFEST_ENV} to cellmap-flow's pixi.toml, "
            "or give env as the path of an environment"
        )
    environments = pixi_environments(manifest)
    if env not in environments:
        raise ConfigError(
            f"Model '{model_name}': {manifest} has no environment {env!r}; "
            f"it has {', '.join(sorted(environments))}"
        )
    return env


def _pixi_run(env: str) -> List[str]:
    return [_pixi_program(), "run", "--frozen", "--manifest-path", str(pixi_manifest()), "-e", env]


def server_argv(env: str) -> List[str]:
    """The program that serves a model in ``env``, before its ``--model`` and ``-d``.

    In a pixi environment that is the environment's ``cellmap_flow serve``,
    as the deployment's own CELLMAP_FLOW_SERVER_COMMAND is. A directory need
    not be on PATH, so there it is its python running the CLI; -P keeps the
    job's working directory off sys.path, so a job started inside another
    checkout does not run that checkout's cellmap_flow.
    """
    if is_path(env):
        return python_argv(env, "cellmap_flow.cli.main") + ["serve"]
    return _pixi_run(env) + ["cellmap_flow", "serve"]


def python_argv(env: str, module: str) -> List[str]:
    """``python -P -m <module>`` in ``env``."""
    if is_path(env):
        return [os.path.join(os.path.expanduser(env), "bin", "python"), "-P", "-m", module]
    return _pixi_run(env) + ["python", "-P", "-m", module]


def lib_dir(env: str) -> str:
    """The lib directory of ``env``, for LD_LIBRARY_PATH.

    For a pixi environment that is where pixi installs it by default,
    ``.pixi/envs/<name>`` beside the manifest; a pixi configured with
    detached environments puts it elsewhere, and the job then loads
    libraries as ``pixi run`` alone would.
    """
    if is_path(env):
        return os.path.join(os.path.expanduser(env), "lib")
    return str(pixi_manifest().parent / ".pixi" / "envs" / env / "lib")


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
    install peft. A directory is not checked; a trainer that cannot import
    there fails in its job's log.
    """
    if is_path(env):
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
