"""Starting an inference server for a model: the one launcher, and the
command line it runs.

``launch(model, data_path, ...)`` is what every launcher calls:
``cellmap_flow infer <type>``, ``cellmap_flow yaml`` and ``add --run``, the
dashboard's Models tab (its catalog, Hugging Face, zoo and Cellpose picks,
Add a model, and a model ticked again) and the Training tab's ranking. It
applies the one data path and scale rule (``config.yaml.resolve_data_path``),
falls back to the saved launcher settings for the queue and charge group,
builds the command and starts the job (``jobs.launch.start_hosts``), which
it returns.

The command is ``<SERVER_COMMAND> --model <entry> -d <data path>``, where the entry
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
import logging
import shlex

from cellmap_flow.jobs import launch as jobs_launch
from cellmap_flow.jobs.spec import shell_join

logger = logging.getLogger(__name__)


def _program(env) -> list:
    """The serve program: ``env``'s, or this deployment's SERVER_COMMAND."""
    if not env:
        return shlex.split(jobs_launch.SERVER_COMMAND)
    from cellmap_flow.models import envs

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


def server_command(model_config, data_path: str, resample: bool = True) -> str:
    """``server_argv`` as the shell line start_hosts takes."""
    return shell_join(server_argv(model_config, data_path, resample))


def remembered(queue, charge_group):
    """``(queue, charge_group)`` for a command-line launch: the ones given,
    else the saved launcher settings'; the ones given are saved for the next
    run and the dashboard."""
    from cellmap_flow.jobs.settings import launcher_settings

    settings = launcher_settings()
    queue = queue or settings.queue
    charge_group = charge_group or settings.charge_group
    settings.queue = queue
    if charge_group:
        settings.charge_group = charge_group
    settings.save()
    return queue, charge_group


def launch(model, data_path: str, *, queue=None, charge_group=None, resample: bool = True, wait_for_host: bool = True):
    """Start an inference server for ``model`` on ``data_path``, and return its job.

    ``model`` is a model config, or a model entry (``models.entry``), which
    is built first (``registry.build_model``). The server reads
    ``data_path`` with the model's ``scale`` applied
    (``config.yaml.resolve_data_path``), resampling it to the model's voxel
    size when ``resample``. ``queue`` and ``charge_group`` default to the
    saved launcher settings (``jobs.settings``). The job is named after the
    model, and is returned once its server has announced its address, or at
    once when ``wait_for_host`` is off.

    Raises:
        ConfigError: ``model`` is an entry that does not describe a model.
        JobStartError: no queue accepted the job, or its server never came up.
    """
    from cellmap_flow.config.yaml import resolve_data_path
    from cellmap_flow.jobs.settings import launcher_settings
    from cellmap_flow.models.registry import build_model, cli_name_of

    if isinstance(model, dict):
        model = build_model(model)
    name = getattr(model, "name", None) or cli_name_of(type(model))
    path = resolve_data_path(data_path, getattr(model, "scale", None))
    if path != data_path:
        logger.info(f"Model {name} specifies scale {model.scale}; reading {path}")
    if queue is None or charge_group is None:
        settings = launcher_settings()
        queue = queue if queue is not None else settings.queue
        charge_group = charge_group if charge_group is not None else settings.charge_group
    command = server_command(model, path, resample)
    logger.info(f"Starting a server for {name}: {command}")
    # Through the module, so that a test's stand-in for start_hosts stands in here too.
    return jobs_launch.start_hosts(command, job_name=name, queue=queue, charge_group=charge_group,
                                   wait_for_host=wait_for_host)
