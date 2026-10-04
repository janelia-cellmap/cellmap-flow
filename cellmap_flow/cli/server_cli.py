"""The inference server: serve one model's predictions, on the node an
inference job runs on.

- ``cellmap_flow serve --model <entry> -d <data path>`` (``serve``), what
  the launchers run since 0.3.0 (``serving.launch``). The entry is
  ``ModelConfig.launch_entry`` as JSON, rebuilt as a YAML's model entry is
  (``registry.build_model``).
- ``cellmap_flow_server``, the server's program before 0.3.0, which goes in
  the release after it. It takes ``--model`` too, for a deployment whose
  CELLMAP_FLOW_SERVER_COMMAND still names it; and it keeps a command for
  each model type, which a dashboard from before 0.3.0 launches: built
  when click asks for it, with the type's constructor arguments as options
  (``registry.click_options``, the ``ModelConfig.command`` form).
"""

import click
import json
import logging
import sys
from typing import Type

from cellmap_flow.cli.common import ModelTypeGroup, deprecation_notice, log_level_option, resample_option
from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.models import registry
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.plugins import load_plugins


logger = logging.getLogger(__name__)


def run_server(
    model_config, data_path, debug=False, port=0, certfile=None, keyfile=None, resample=True
):
    """Run the inference server with the given configuration; ``resample``
    as CellMapFlowServer takes it, the engine ``CELLMAP_FLOW_ENGINE``'s
    (serving.engine)."""
    from cellmap_flow.serving.engine import make_server

    server = make_server(data_path, model_config, resample=resample)
    server.run(
        debug=debug,
        port=port,
        certfile=certfile,
        keyfile=keyfile,
    )


def serve_entry(model_json, data_path, debug=False, port=0, certfile=None, keyfile=None, resample=True):
    """Build the model ``model_json`` describes, and serve it.

    ``model_json`` is a model entry (``ModelConfig.launch_entry``) as JSON.
    One that is not is a usage error; a model that cannot be built, or a
    server that crashes, exits 1 with the traceback in the job's log.
    """
    try:
        entry = json.loads(model_json)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"not JSON ({e})", param_hint="'--model'")
    if not isinstance(entry, dict):
        raise click.BadParameter(f"not a model entry: {model_json}", param_hint="'--model'")
    try:
        # The entry's own name, or none: the type's default, as launched.
        model_config = registry.build_model(entry, entry.get("name"))
    except ConfigError as e:
        raise click.BadParameter(str(e), param_hint="'--model'")
    except Exception:
        logger.exception(
            f"Failed to create the model {model_json} (likely a missing/mismatched "
            "dependency for this model's framework)"
        )
        sys.exit(1)

    try:
        run_server(model_config, data_path, debug, port, certfile, keyfile, resample)
    except Exception:
        logger.exception(f"Server for {type(model_config).__name__} crashed")
        sys.exit(1)


def _serve_options(required):
    """``--model`` and the server's own options, which ``serve`` requires
    and ``cellmap_flow_server`` takes instead of a type's command."""
    options = [
        click.option("--model", "model_json", required=required,
                     help="The model: its launch entry (ModelConfig.launch_entry), as JSON."),
        click.option("-d", "--data-path", required=required, help="Path to the dataset"),
        click.option("-p", "--port", default=0, type=int, help="Port to listen on"),
        click.option("--debug", is_flag=True, help="Run in debug mode"),
        click.option("--certfile", default=None, help="Path to SSL certificate file"),
        click.option("--keyfile", default=None, help="Path to SSL private key file"),
        # The launchers pass it on (serving.launch.server_argv).
        resample_option(),
    ]

    def decorate(command_func):
        for option in reversed(options):
            command_func = option(command_func)
        return command_func

    return decorate


@click.command()
@_serve_options(required=True)
def serve(model_json, data_path, port, debug, certfile, keyfile, resample):
    """Serve one model's predictions, as an inference job does on its node.

    The launchers (`infer`, `yaml`, the dashboard) run this for you, as
    CELLMAP_FLOW_SERVER_COMMAND (by default `cellmap_flow serve`). The
    model is a YAML model entry, as JSON:

    \b
      cellmap_flow serve -d /path/to/data.zarr/raw \\
        --model '{"type": "script", "script_path": "/path/to/model.py"}'
    """
    serve_entry(model_json, data_path, debug, port, certfile, keyfile, resample)


def create_dynamic_server_command(cli_name: str, config_class: Type[ModelConfig]):
    """
    Dynamically create a Click command for a ModelConfig subclass server.
    """
    # Create the command function
    def command_func(**kwargs):
        # Separate model config kwargs from server kwargs
        model_kwargs = {}
        data_path = kwargs.pop("data_path")
        debug = kwargs.pop("debug", False)
        port = kwargs.pop("port", 0)
        certfile = kwargs.pop("certfile", None)
        keyfile = kwargs.pop("keyfile", None)

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
        except Exception:
            logger.exception(
                f"Failed to create {config_class.__name__} with arguments "
                f"{processed_kwargs} (likely a missing/mismatched dependency "
                "for this model's framework)"
            )
            sys.exit(1)

        # Run the server
        try:
            run_server(model_config, data_path, debug, port, certfile, keyfile)
        except Exception:
            logger.exception(f"Server for {config_class.__name__} crashed")
            sys.exit(1)

    # Add docstring
    command_func.__doc__ = f"""
    Run CellMapFlow server using {config_class.__name__}.
    
    Model parameters are auto-generated from the class constructor.
    """

    # Add common server options
    command_func = click.option(
        "-d", "--data-path", required=True, type=str, help="Path to the dataset"
    )(command_func)

    command_func = click.option("--debug", is_flag=True, help="Run in debug mode")(
        command_func
    )

    command_func = click.option(
        "-p", "--port", default=0, type=int, help="Port to listen on"
    )(command_func)

    command_func = click.option(
        "--certfile", default=None, type=str, help="Path to SSL certificate file"
    )(command_func)

    command_func = click.option(
        "--keyfile", default=None, type=str, help="Path to SSL private key file"
    )(command_func)

    # Add model-specific options based on constructor parameters; -d and -p
    # are the command's own.
    # Applied last to first, because each decorator puts its option before
    # the ones already applied.
    for option_config in reversed(registry.click_options(config_class, {"-d", "-p"})):
        command_func = click.option(
            *option_config.pop("param_decls"), **option_config
        )(command_func)

    return click.command(name=cli_name)(command_func)


@click.group(cls=ModelTypeGroup, make_command=create_dynamic_server_command, invoke_without_command=True)
@log_level_option(default="INFO")
@_serve_options(required=False)
@click.pass_context
def cli(ctx, model_json, data_path, port, debug, certfile, keyfile, resample):
    """The inference server before 0.3.0: deprecated, and goes in the release
    after it. Use `cellmap_flow serve`.

    With --model and -d it is `cellmap_flow serve`. Each model type's
    command is the form launchers used before 0.3.0:

    \b
        cellmap_flow_server dacapo -r my_run -i 100 -d /path/to/data
        cellmap_flow_server script -s /path/to/script.py -d /path/to/data
        cellmap_flow_server cellmap -f /path/to/model -n mymodel -d /path/to/data
    """
    if ctx.invoked_subcommand == "list-models":
        deprecation_notice("cellmap_flow_server list-models", "cellmap_flow models")
        return
    if ctx.invoked_subcommand is not None:
        if model_json is not None:
            raise click.UsageError("--model is the model; give it or a model type's command, not both")
        deprecation_notice(f"cellmap_flow_server {ctx.invoked_subcommand}", "cellmap_flow serve --model")
        return
    if model_json is None:
        click.echo(ctx.get_help())
        return
    if data_path is None:
        raise click.MissingParameter(param_hint="'-d' / '--data-path'", param_type="option")
    deprecation_notice("cellmap_flow_server", "cellmap_flow serve")
    serve_entry(model_json, data_path, debug, port, certfile, keyfile, resample)


@cli.command(name="list-models")
def list_models():
    """List all available model configurations."""
    registry.print_available_models("cellmap_flow_server")


def main():
    """The ``cellmap_flow_server`` console script: load the plugins, whose
    model types have commands too, then run the command."""
    load_plugins()
    cli(prog_name="cellmap_flow_server")


if __name__ == "__main__":
    main()
