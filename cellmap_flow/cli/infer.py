"""``cellmap_flow infer <type>``: start a model's inference server, then open
the viewer on its predictions and serve the dashboard.

There is a subcommand for each model type (``models.registry``), built when
click asks for it. It takes the type's constructor arguments as options
(``registry.click_options``), and the dataset (``-d``), queue (``-q``),
billing project (``-P``), ``--resample``, ``--server-check`` and ``--env``
(the environment the server runs in, ``models.envs``) as its own.

``run``, the generic form before 0.3.0 (``cellmap_flow run -m TYPE -c
key=value``), is a deprecated alias: it says which ``infer`` command it
stands for, and runs it.
"""

import click
import logging
import shlex
from typing import Type
from cellmap_flow.jobs.launch import install_cleanup_handlers
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.models import registry
from cellmap_flow.serving.launch import launch, remembered
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.config.yaml import ConfigError, resolve_data_path
from cellmap_flow.cli.common import ModelTypeGroup, deprecation_notice, resample_option

logger = logging.getLogger(__name__)


@click.command(name="run", hidden=True, help="Deprecated: use `cellmap_flow infer <type>`.")
@click.option(
    "-m",
    "--model-type",
    required=True,
    help="Model type (e.g., dacapo, script, cellmap)",
)
@click.option("-d", "--data-path", required=True, help="Path to the dataset")
@click.option(
    "-q",
    "--queue",
    default=None,
    help="GPU queue for the job, or several, comma-separated, to go to the freest "
        "(default: the saved queues)",
)
@click.option(
    "-P", "--project", default=None, help="Project/chargeback group for billing"
)
@click.option(
    "-c", "--config", multiple=True, help="Model configuration as key=value pairs"
)
@click.option(
    "--server-check", is_flag=True, help="Run server check instead of full inference"
)
@click.pass_context
def run_generic(ctx, model_type, data_path, queue, project, config, server_check):
    """``cellmap_flow run -m TYPE -c key=value``, the form before 0.3.0:
    ``cellmap_flow infer TYPE --key value``, which it prints and runs.
    """
    command = infer.get_command(ctx, model_type)
    if command is None:
        raise click.BadParameter(
            f"unknown model type {model_type!r}; the types are "
            f"{', '.join(sorted(registry.model_types()))}",
            param_hint="'-m'",
        )
    argv = []
    for item in config:
        key, sep, value = item.partition("=")
        if not sep:
            raise click.BadParameter(f"{item!r} is not key=value", param_hint="'-c'")
        argv += [f"--{key.replace('_', '-')}", value]
    argv += ["-d", data_path]
    argv += ["-q", queue] if queue else []
    argv += ["-P", project] if project else []
    argv += ["--server-check"] if server_check else []

    deprecation_notice("cellmap_flow run", shlex.join(["cellmap_flow", "infer", model_type, *argv]))
    # No parent, so that a usage error names `cellmap_flow infer TYPE`.
    with command.make_context(f"cellmap_flow infer {model_type}", argv) as sub_ctx:
        return command.invoke(sub_ctx)


def create_dynamic_command(cli_name: str, config_class: Type[ModelConfig]):
    """A click command for one model type: its constructor arguments as
    options, built into a model entry and served through ``launch``."""
    def command_func(**kwargs):
        data_path = kwargs.pop("data_path")
        queue = kwargs.pop("queue", None)
        project = kwargs.pop("project", None)
        server_check = kwargs.pop("server_check", False)
        resample = kwargs.pop("resample", True)
        env = kwargs.pop("env", None)

        # The options as a model entry, built as a YAML's or the form's is.
        entry = {"type": cli_name, **{key: value for key, value in kwargs.items() if value is not None}}
        if env:
            entry["env"] = env
        try:
            model_config = registry.build_model(entry)
        except ConfigError as e:
            raise click.ClickException(str(e))
        queue, project = remembered(queue, project)

        if server_check:
            from cellmap_flow.serving.engine import check_server, make_server

            data_path = resolve_data_path(data_path, getattr(model_config, "scale", None))
            check_server(make_server(data_path, model_config, resample=resample))
            click.echo("Server check passed")
            return
        # Ctrl+C or SIGTERM from here on kills the job this starts.
        install_cleanup_handlers()
        try:
            launch(model_config, data_path, queue=queue, charge_group=project, resample=resample)
        except JobStartError as e:
            raise click.ClickException(str(e))
        from cellmap_flow.dashboard.services.startup import generate_neuroglancer_url

        # Serves the dashboard, on the level the model reads; does not return.
        generate_neuroglancer_url(resolve_data_path(data_path, getattr(model_config, "scale", None)))

    # Add docstring
    command_func.__doc__ = f"""
    Run inference using {config_class.__name__}.

    Model parameters are auto-generated from the class constructor.
    """

    # Add common options
    command_func = click.option(
        "-d", "--data-path", required=True, type=str, help="Path to the dataset"
    )(command_func)

    command_func = click.option(
        "-q",
        "--queue",
        default=None,
        type=str,
        help="GPU queue for the job, or several, comma-separated, to go to the freest "
        "(default: the saved queues)",
    )(command_func)

    command_func = click.option(
        "-P",
        "--project",
        default=None,
        type=str,
        help="Project/chargeback group for billing",
    )(command_func)

    command_func = click.option(
        "--server-check",
        is_flag=True,
        help="Run server check instead of full inference",
    )(command_func)

    command_func = resample_option()(command_func)

    command_func = click.option(
        "--env",
        default=None,
        type=str,
        help="Run the server in this environment: a pixi environment of "
        "cellmap-flow's pixi.toml, an alias (cellmap_flow envs), or the absolute "
        "path of one with cellmap-flow installed; current: this one "
        "(default: the model type's, else this one)",
    )(command_func)

    # Add model-specific options based on constructor parameters; -d, -q
    # and -P are the command's own.
    # Applied last to first, because each decorator puts its option before
    # the ones already applied.
    for option_config in reversed(registry.click_options(config_class, {"-d", "-q", "-P"})):
        command_func = click.option(
            *option_config.pop("param_decls"), **option_config
        )(command_func)

    return click.command(name=cli_name)(command_func)


infer = ModelTypeGroup(
    "infer",
    make_command=create_dynamic_command,
    help="""Start a model's inference server, then open the viewer on its
    predictions and serve the dashboard.

    There is a subcommand for each model type; `cellmap_flow models` lists
    them, and `cellmap_flow infer <type> --help` shows a type's options.

    \b
      cellmap_flow infer dacapo -r my_run -i 100 -d /path/to/data
      cellmap_flow infer script -s /path/to/script.py -d /path/to/data
      cellmap_flow infer cellmap -f /path/to/model -n mymodel -d /path/to/data
    """,
)
