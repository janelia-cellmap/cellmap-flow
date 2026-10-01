"""The command line that starts an inference server for a model.

Every launcher builds it here: ``cellmap_flow infer <type>``,
``cellmap_flow yaml`` and the dashboard's catalog and Hugging Face models.
It is ``<SERVER_COMMAND> --model <entry> -d <data path>``, where the entry
is ``ModelConfig.launch_entry`` as JSON, which the server rebuilds as it
rebuilds a YAML's model entry, then ``--resample`` when the launcher was
asked to resample (a YAML's ``resample: true``, ``infer --resample``). The
flag is left out otherwise, so a command without it is what it always was.

The serve program is ``jobs.launch.SERVER_COMMAND``, and deployments change
it: the fileglancer deploy sets ``"pixi run cellmap_flow serve"``. So it is
read each time a command is built, never copied at import (a copy misses
the override), and split into words (quoted as one token, the shell looks
for a program called "pixi run cellmap_flow serve"). ``cellmap_flow_server``,
the program before 0.3.0, takes ``--model`` too, so a deployment that
still sets that works.
"""

import json
import shlex

from cellmap_flow.jobs import launch as jobs_launch


def _serve_argv(entry: dict, data_path, resample=False) -> list:
    # Compact, and a value JSON has no type for (a plugin's Path) as its str.
    entry_json = json.dumps(entry, separators=(",", ":"), default=str)
    argv = [*shlex.split(jobs_launch.SERVER_COMMAND), "--model", entry_json, "-d", str(data_path)]
    return argv + ["--resample"] if resample else argv


def server_argv(model_config, data_path: str, resample: bool = False) -> list:
    """The server's argv for ``model_config`` reading ``data_path``, resampling
    it to the model's input voxel size when ``resample``."""
    return _serve_argv(model_config.launch_entry, data_path, resample)


def server_argv_for(model_type: str, params: dict, data_path: str) -> list:
    """The server's argv for a model given as its type and constructor arguments.

    For launchers that have no model config to hand and should not build
    one (a Hugging Face config fetches its repo's metadata); None arguments
    are left out, as in ``ModelConfig.launch_entry``.
    """
    from cellmap_flow.models.configs.base import model_entry
    from cellmap_flow.models.registry import model_type as lookup

    return _serve_argv(model_entry(lookup(model_type), params), data_path)


def server_command(model_config, data_path: str, resample: bool = False) -> str:
    """``server_argv`` as one shell-quoted string, which is what start_hosts takes."""
    return shlex.join(server_argv(model_config, data_path, resample))
