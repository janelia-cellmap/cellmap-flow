"""
Dynamic CLI generator that automatically detects ModelConfig subclasses
and creates CLI commands based on their __init__ parameters.
"""

import os
import click
import logging
import inspect
import sys
from typing import Type, Dict
from typing import Type, get_type_hints
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.utils.cli_utils import (
    get_all_subclasses,
    create_click_option_from_param,
    process_constructor_args,
    get_all_model_configs,
    print_available_models,
)
from cellmap_flow.utils.plugin_manager import (
    register_plugin,
    unregister_plugin,
    list_plugins,
    load_plugins,
)

logging.basicConfig()
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level",
)
def cli(log_level):
    """
    CellMap Flow - Dynamic CLI for model inference.

    Automatically generates commands for all available ModelConfig subclasses.

    Examples:
        cellmap_flow_v2 dacapo -r my_run -i 100 -d /path/to/data
        cellmap_flow_v2 script -s /path/to/script.py -d /path/to/data
        cellmap_flow_v2 cellmap -f /path/to/model -n mymodel -d /path/to/data
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command(name="list-models")
def list_models():
    """List all available model configurations."""
    print_available_models("cellmap_flow")


@cli.command(name="register")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite existing plugin with the same name.")
def register_cmd(filepath, force):
    """Register a custom plugin (normalizer, postprocessor, or model config).

    FILEPATH is the path to a .py file defining subclasses of
    InputNormalizer, PostProcessor, or ModelConfig.

    Example:
        cellmap_flow register my_normalizer.py
    """
    try:
        dest = register_plugin(filepath, force=force)
        click.echo(f"Registered plugin: {dest.name}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command(name="unregister")
@click.argument("name")
def unregister_cmd(name):
    """Unregister a previously registered plugin by name.

    NAME is the plugin filename (with or without .py extension).

    Example:
        cellmap_flow unregister my_normalizer
    """
    try:
        unregister_plugin(name)
        click.echo(f"Unregistered plugin: {name}")
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command(name="list-plugins")
def list_plugins_cmd():
    """List all registered plugins."""
    plugins = list_plugins()
    if not plugins:
        click.echo("No plugins registered.")
        return
    click.echo("Registered plugins:\n")
    for plugin_path in plugins:
        click.echo(f"  {plugin_path.name}  ({plugin_path})")



@cli.command(name="run")
@click.option(
    "-m",
    "--model-type",
    required=True,
    help="Model type (e.g., dacapo, script, cellmap)",
)
@click.option("-d", "--data-path", required=True, help="Path to the dataset")
@click.option("-q", "--queue", default="gpu_h100", help="Queue for job submission")
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
        cellmap_flow_v2 run -m dacapo -d /data/path -c run_name=myrun -c iteration=100
    """
    model_configs = get_all_model_configs()

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
    processed_kwargs = process_constructor_args(config_class, kwargs)

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

    # Append scale to data_path if present in model config
    final_data_path = data_path
    if hasattr(model_config, 'scale') and model_config.scale:
        final_data_path = os.path.join(data_path, model_config.scale)

    # Run the server check or full inference
    if server_check:
        server = CellMapFlowServer(final_data_path, model_config)
        server._chunk_impl(None, None, 2, 2, 2, None)
        click.echo("Server check passed")
    else:
        command = f"{SERVER_COMMAND} {model_config.command} -d {final_data_path}"
        logger.info(f"Executing command: {command}")
        start_hosts(command, queue, project, model_config.name or model_type)
        neuroglancer_url = generate_neuroglancer_url(final_data_path)
        click.echo(f"Neuroglancer URL: {neuroglancer_url}")


def create_dynamic_command(cli_name: str, config_class: Type[ModelConfig]):
    """
    Dynamically create a Click command for a ModelConfig subclass.
    """
    # Get constructor signature
    sig = inspect.signature(config_class.__init__)

    # Get type hints if available
    try:
        type_hints = get_type_hints(config_class.__init__)
    except:
        type_hints = {}

    # Track used short names to avoid duplicates
    used_short_names = set(["-d", "-q", "-P"])  # Reserved for common options

    # Create the command function
    def command_func(**kwargs):
        # Separate model config kwargs from CLI kwargs
        model_kwargs = {}
        data_path = kwargs.pop("data_path")
        queue = kwargs.pop("queue", "gpu_h100")
        project = kwargs.pop("project", None)
        server_check = kwargs.pop("server_check", False)

        # Process kwargs for the model config
        for key, value in kwargs.items():
            if value is not None:
                model_kwargs[key] = value

        # Process constructor args (handle list/tuple conversions)
        processed_kwargs = process_constructor_args(config_class, model_kwargs)

        # Create model config instance
        try:
            model_config = config_class(**processed_kwargs)
        except TypeError as e:
            logger.error(f"Error creating {config_class.__name__}: {e}")
            logger.error(f"Provided arguments: {processed_kwargs}")
            sys.exit(1)

        # Append scale to data_path if present in model config
        final_data_path = data_path
        if hasattr(model_config, 'scale') and model_config.scale:
            final_data_path = os.path.join(data_path, model_config.scale)

        # Run server check or full inference
        if server_check:
            server = CellMapFlowServer(final_data_path, model_config)
            server._chunk_impl(None, None, 2, 2, 2, None)
            click.echo("Server check passed")
        else:
            command = f"{SERVER_COMMAND} {model_config.command} -d {final_data_path}"
            logger.info(f"Executing command: {command}")
            base_name = getattr(model_config, "name", None) or cli_name
            start_hosts(command, queue, project, base_name)
            neuroglancer_url = generate_neuroglancer_url(final_data_path)
            click.echo(f"Neuroglancer URL: {neuroglancer_url}")

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
        "-q", "--queue", default="gpu_h100", type=str, help="Queue for job submission"
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

    # Add model-specific options based on constructor parameters
    for param_name, param_info in reversed(list(sig.parameters.items())):
        option_config = create_click_option_from_param(
            param_name, param_info, used_short_names
        )
        if option_config:
            command_func = click.option(
                *option_config.pop("param_decls"), **option_config
            )(command_func)

    # Register as a command
    command_func = cli.command(name=cli_name)(command_func)

    return command_func


def register_all_model_commands():
    """
    Discover and register all ModelConfig subclasses as CLI commands.
    """
    model_configs = get_all_model_configs()

    for cli_name, config_class in model_configs.items():
        try:
            create_dynamic_command(cli_name, config_class)
            logger.debug(f"Registered command: {cli_name}")
        except Exception as e:
            logger.warning(f"Failed to register command for {cli_name}: {e}")


# Load user plugins before registering model commands
load_plugins()

# Register all commands at module load time
register_all_model_commands()


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
