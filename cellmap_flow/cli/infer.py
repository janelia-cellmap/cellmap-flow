"""``cellmap_flow infer <type>``: start a model's inference server, then open
the viewer on its predictions and serve the dashboard.

There is a subcommand for each model type (``models.registry``), built when
click asks for it. It takes the type's constructor arguments as options
(``registry.click_options``), and the dataset (``-d``), queue (``-q``),
billing project (``-P``) and ``--server-check`` as its own.

``run`` is the older generic form, ``cellmap_flow run -m TYPE -c key=value``.
"""

import click
import logging
import inspect
import sys
from typing import Type
from cellmap_flow.jobs.launch import install_cleanup_handlers, start_hosts
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.models import registry
from cellmap_flow.serving.launch import server_command
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.globals import g
from cellmap_flow.config.yaml import resolve_data_path
from cellmap_flow.cli.common import ModelTypeGroup

logger = logging.getLogger(__name__)


@click.command(name="run")
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
    help="Queue for job submission (default: the saved queue)",
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
def run_generic(model_type, data_path, queue, project, config, server_check):
    """
    Generic run command that accepts any model type with dynamic configuration.

    Example:
        cellmap_flow run -m dacapo -d /data/path -c run_name=myrun -c iteration=100
    """
    # Fall back to cached values if not provided
    if project is None:
        project = g.charge_group
    if queue is None:
        queue = g.queue

    model_configs = registry.model_types()

    if model_type not in model_configs:
        click.echo(f"Error: Unknown model type '{model_type}'", err=True)
        click.echo(
            f"Available types: {', '.join(sorted(model_configs.keys()))}", err=True
        )
        sys.exit(1)

    config_class = model_configs[model_type]

    # Parse config key=value pairs
    kwargs = {}
    for item in config:
        if "=" not in item:
            click.echo(
                f"Error: Invalid config format '{item}'. Use key=value", err=True
            )
            sys.exit(1)
        key, value = item.split("=", 1)
        kwargs[key] = value

    # Process the kwargs
    processed_kwargs = registry.coerce_cli_args(config_class, kwargs)

    # Create model config
    try:
        model_config = config_class(**processed_kwargs)
    except TypeError as e:
        click.echo(f"Error creating model config: {e}", err=True)
        click.echo(f"Required parameters for {model_type}: ", err=True)
        sig = inspect.signature(config_class.__init__)
        for param_name, param_info in sig.parameters.items():
            if param_name != "self" and param_info.default is inspect.Parameter.empty:
                click.echo(f"  - {param_name}", err=True)
        sys.exit(1)

    # The scale selects a level of a multiscale data_path; see resolve_data_path.
    final_data_path = resolve_data_path(data_path, getattr(model_config, "scale", None))

    # Save server config to cache
    g.queue = queue
    if project:
        g.charge_group = project
    g.save_server_config()

    # Run the server check or full inference
    if server_check:
        from cellmap_flow.server import CellMapFlowServer

        server = CellMapFlowServer(final_data_path, model_config)
        server._chunk_impl(None, None, 2, 2, 2)
        click.echo("Server check passed")
    else:
        command = server_command(model_config, final_data_path)
        logger.info(f"Executing command: {command}")
        # Ctrl+C or SIGTERM from here on kills the job this starts.
        install_cleanup_handlers()
        try:
            start_hosts(command, queue, project, model_config.name or model_type)
        except JobStartError as e:
            raise click.ClickException(str(e))
        from cellmap_flow.dashboard.services.startup import generate_neuroglancer_url

        # Serves the dashboard; does not return.
        generate_neuroglancer_url(final_data_path)


def create_dynamic_command(cli_name: str, config_class: Type[ModelConfig]):
    """
    Dynamically create a Click command for a ModelConfig subclass.
    """
    # Create the command function
    def command_func(**kwargs):
        # Separate model config kwargs from CLI kwargs
        model_kwargs = {}
        data_path = kwargs.pop("data_path")
        queue = kwargs.pop("queue", None)
        project = kwargs.pop("project", None)
        server_check = kwargs.pop("server_check", False)

        # Fall back to cached values if not provided
        if project is None:
            project = g.charge_group
        if queue is None:
            queue = g.queue

        # Process kwargs for the model config
        for key, value in kwargs.items():
            if value is not None:
                model_kwargs[key] = value

        # Process constructor args (handle list/tuple conversions)
        processed_kwargs = registry.coerce_cli_args(config_class, model_kwargs)

        # Create model config instance
        try:
            model_config = config_class(**processed_kwargs)
        except TypeError as e:
            logger.error(f"Error creating {config_class.__name__}: {e}")
            logger.error(f"Provided arguments: {processed_kwargs}")
            sys.exit(1)

        # The scale selects a level of a multiscale data_path; see resolve_data_path.
        final_data_path = resolve_data_path(
            data_path, getattr(model_config, "scale", None)
        )

        # Save server config to cache
        g.queue = queue
        if project:
            g.charge_group = project
        g.save_server_config()

        # Run server check or full inference
        if server_check:
            from cellmap_flow.server import CellMapFlowServer

            server = CellMapFlowServer(final_data_path, model_config)
            server._chunk_impl(None, None, 2, 2, 2)
            click.echo("Server check passed")
        else:
            command = server_command(model_config, final_data_path)
            logger.info(f"Executing command: {command}")
            base_name = getattr(model_config, "name", None) or cli_name
            # Ctrl+C or SIGTERM from here on kills the job this starts.
            install_cleanup_handlers()
            try:
                start_hosts(command, queue, project, base_name)
            except JobStartError as e:
                raise click.ClickException(str(e))
            from cellmap_flow.dashboard.services.startup import generate_neuroglancer_url

            # Serves the dashboard; does not return.
            generate_neuroglancer_url(final_data_path)

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
        help="Queue for job submission (default: the saved queue)",
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
