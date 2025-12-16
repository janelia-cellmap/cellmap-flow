"""
Dynamic server CLI generator that automatically detects ModelConfig subclasses
and creates server commands based on their __init__ parameters.
"""

import click
import logging
import inspect
import sys
from typing import Type, Dict, get_type_hints

from cellmap_flow.image_data_interface import ImageDataInterface
from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.cli_utils import (
    get_all_subclasses,
    create_click_option_from_param,
    process_constructor_args,
    get_all_model_configs,
    print_available_models,
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
    help="Set the logging level"
)
def cli(log_level):
    """
    CellMap Flow Server - Dynamic CLI for running inference servers.
    
    Automatically generates server commands for all available ModelConfig subclasses.
    
    Examples:
        cellmap_flow_server dacapo -r my_run -i 100 -d /path/to/data
        cellmap_flow_server script -s /path/to/script.py -d /path/to/data
        cellmap_flow_server cellmap-model -f /path/to/model -n mymodel -d /path/to/data
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command(name="list-models")
def list_models():
    """List all available model configurations."""
    print_available_models("cellmap_flow_server")


def run_server(
    model_config, data_path, debug=False, port=0, certfile=None, keyfile=None
):
    """Run the CellMapFlow server with the given configuration."""
    server = CellMapFlowServer(data_path, model_config)
    server.run(
        debug=debug,
        port=port,
        certfile=certfile,
        keyfile=keyfile,
    )


def create_dynamic_server_command(cli_name: str, config_class: Type[ModelConfig]):
    """
    Dynamically create a Click command for a ModelConfig subclass server.
    """
    # Get constructor signature
    sig = inspect.signature(config_class.__init__)
    
    # Get type hints if available
    try:
        type_hints = get_type_hints(config_class.__init__)
    except:
        type_hints = {}
    
    # Create the command function
    def command_func(**kwargs):
        # Separate model config kwargs from server kwargs
        model_kwargs = {}
        data_path = kwargs.pop('data_path')
        debug = kwargs.pop('debug', False)
        port = kwargs.pop('port', 0)
        certfile = kwargs.pop('certfile', None)
        keyfile = kwargs.pop('keyfile', None)
        
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
        
        # Run the server
        run_server(model_config, data_path, debug, port, certfile, keyfile)
    
    # Add docstring
    command_func.__doc__ = f"""
    Run CellMapFlow server using {config_class.__name__}.
    
    Model parameters are auto-generated from the class constructor.
    """
    
    # Add common server options
    command_func = click.option(
        "-d", "--data-path", 
        required=True, 
        type=str, 
        help="Path to the dataset"
    )(command_func)
    
    command_func = click.option(
        "--debug",
        is_flag=True,
        help="Run in debug mode"
    )(command_func)
    
    command_func = click.option(
        "-p", "--port",
        default=0,
        type=int,
        help="Port to listen on"
    )(command_func)
    
    command_func = click.option(
        "--certfile",
        default=None,
        type=str,
        help="Path to SSL certificate file"
    )(command_func)
    
    command_func = click.option(
        "--keyfile",
        default=None,
        type=str,
        help="Path to SSL private key file"
    )(command_func)
    
    # Add model-specific options based on constructor parameters
    for param_name, param_info in reversed(list(sig.parameters.items())):
        option_config = create_click_option_from_param(param_name, param_info)
        if option_config:
            command_func = click.option(*option_config.pop('param_decls'), **option_config)(command_func)
    
    # Register as a command
    command_func = cli.command(name=cli_name)(command_func)
    
    return command_func


def register_all_server_commands():
    """
    Discover and register all ModelConfig subclasses as server CLI commands.
    """
    model_configs = get_all_model_configs()
    
    for cli_name, config_class in model_configs.items():
        try:
            create_dynamic_server_command(cli_name, config_class)
            logger.debug(f"Registered server command: {cli_name}")
        except Exception as e:
            logger.warning(f"Failed to register server command for {cli_name}: {e}")


# Register all commands at module load time
register_all_server_commands()


@cli.command()
@click.option(
    "-n", "--neuroglancer-url", required=True, type=str, help="Neuroglancer viewer URL."
)
@click.option(
    "-i", "--inference-host", required=True, type=str, help="Inference host(s)."
)
def run_ui_server(neuroglancer_url, inference_host):
    """Run the dashboard UI server."""
    create_and_run_app(neuroglancer_url, inference_host)


def main():
    """Entry point for the server CLI."""
    cli()


if __name__ == "__main__":
    main()
