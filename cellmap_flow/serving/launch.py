"""The command line that starts an inference server for a model.

Every launcher builds it here: ``cellmap_flow <type>``, ``cellmap_flow run``,
``cellmap_flow_yaml`` and the dashboard's catalog and Hugging Face models.

The serve program is ``bsub_utils.SERVER_COMMAND``, and deployments change
it: the fileglancer deploy sets ``"pixi run cellmap_flow_server"``. So it is
read each time a command is built, never copied at import (a copy misses
the override), and split into words (quoted as one token, the shell looks
for a program called "pixi run cellmap_flow_server").

bsub_utils imports ``cellmap_flow.globals``, so it is imported only when a
command is built; importing this module stays cheap.
"""

import shlex


def _serve_words():
    from cellmap_flow.utils import bsub_utils

    return shlex.split(bsub_utils.SERVER_COMMAND)


def server_argv(model_config, data_path: str) -> list:
    """The server's argv for ``model_config`` reading ``data_path``."""
    return [*_serve_words(), *shlex.split(model_config.command), "-d", str(data_path)]


def server_argv_for(model_type: str, params: dict, data_path: str) -> list:
    """The server's argv for a model given as its type and constructor arguments.

    For launchers that have no model config to hand and should not build
    one (a Hugging Face config fetches its repo's metadata): the arguments
    go in signature order, and None ones are left out, as in
    ``ModelConfig.command``.
    """
    from cellmap_flow.models.models_config import command_argv
    from cellmap_flow.models.registry import model_type as lookup

    return [*_serve_words(), *command_argv(lookup(model_type), params), "-d", str(data_path)]


def server_command(model_config, data_path: str) -> str:
    """``server_argv`` as one shell-quoted string, which is what start_hosts takes."""
    return shlex.join(server_argv(model_config, data_path))
