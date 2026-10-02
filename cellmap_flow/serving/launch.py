"""The command line that starts an inference server for a model.

Every launcher builds it here: ``cellmap_flow infer <type>``,
``cellmap_flow yaml`` and the dashboard's catalog and Hugging Face models.
It is ``<SERVER_COMMAND> --model <entry> -d <data path>``, where the entry
is ``ModelConfig.launch_entry`` as JSON, which the server rebuilds as it
rebuilds a YAML's model entry, then ``--resample`` or ``--no-resample``:
always one of them, so a server reads the data as its launcher was told to
(a YAML's ``resample``, ``infer --no-resample``, the dashboard's checkbox)
whatever ``serve``'s own default is.

The serve program is ``jobs.launch.SERVER_COMMAND``, and deployments change
it: the fileglancer deploy sets ``"pixi run cellmap_flow serve"``. So it is
read each time a command is built, never copied at import (a copy misses
the override), and split into words (quoted as one token, the shell looks
for a program called "pixi run cellmap_flow serve"). ``cellmap_flow_server``,
the program before 0.3.0, takes ``--model`` too, so a deployment that
still sets that works.

A model with an environment of its own (``models.envs``: its entry's
``env``, else its type's ``default_env``) is served from it instead: the
entry's ``env`` is taken off, and the program is that environment's (``pixi run --frozen --manifest-path <pixi.toml> -e <env>
cellmap_flow serve``, or ``<env>/bin/python -P -m cellmap_flow.cli.main
serve``) in place of SERVER_COMMAND, which names this deployment's own.

The launchers submit it as a shell line, which ``jobs.spec.shell_join``
quotes: the entry's JSON is full of double quotes and braces, and
``shlex.join`` would single-quote it, which LSF's own quoting splits apart
(``jobs.spec.shell_quote`` says how).
"""

import json
import shlex

from cellmap_flow.jobs import launch as jobs_launch
from cellmap_flow.jobs.spec import shell_join


def _program(env) -> list:
    """The serve program: ``env``'s, or this deployment's SERVER_COMMAND."""
    if not env:
        return shlex.split(jobs_launch.SERVER_COMMAND)
    from cellmap_flow.models import envs

    # Checked again: server_argv_for's params never went through build_model.
    return envs.server_argv(envs.validate(env))


def _serve_argv(entry: dict, env, data_path, resample=True) -> list:
    """The server's argv for ``entry``, served from ``env`` (None: this one's SERVER_COMMAND)."""
    # The server must not get env back: in its environment it is at home.
    entry = dict(entry)
    entry.pop("env", None)
    # Compact, and a value JSON has no type for (a plugin's Path) as its str.
    entry_json = json.dumps(entry, separators=(",", ":"), default=str)
    argv = [*_program(env), "--model", entry_json, "-d", str(data_path)]
    return argv + ["--resample" if resample else "--no-resample"]


def server_argv(model_config, data_path: str, resample: bool = True) -> list:
    """The server's argv for ``model_config`` reading ``data_path``, resampling
    it to the model's input voxel size when ``resample``."""
    from cellmap_flow.models import envs

    return _serve_argv(model_config.launch_entry, envs.model_env(model_config), data_path, resample)


def server_argv_for(model_type: str, params: dict, data_path: str, resample: bool = True) -> list:
    """The server's argv for a model given as its type and constructor arguments.

    For launchers that have no model config to hand and should not build
    one (a Hugging Face config fetches its repo's metadata); None arguments
    are left out, as in ``ModelConfig.launch_entry``. An ``env`` among
    ``params`` serves the model from that environment, else the type's
    ``default_env`` does. Only a default the class itself sets: one decided
    per model by a property needs the model, and ``server_argv``.
    """
    from cellmap_flow.models import envs
    from cellmap_flow.models.configs.base import model_entry
    from cellmap_flow.models.registry import model_type as lookup

    cls = lookup(model_type)
    env = envs.effective(params.get("env"), envs.type_default(cls), params.get("name") or model_type)
    return _serve_argv(model_entry(cls, params), env, data_path, resample)


def server_command(model_config, data_path: str, resample: bool = True) -> str:
    """``server_argv`` as the shell line start_hosts takes."""
    return shell_join(server_argv(model_config, data_path, resample))


def server_command_for(model_type: str, params: dict, data_path: str, resample: bool = True) -> str:
    """``server_argv_for`` as the shell line start_hosts takes."""
    return shell_join(server_argv_for(model_type, params, data_path, resample))
