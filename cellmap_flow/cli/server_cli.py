import click
import logging

from cellmap_flow.utils.data import ScriptModelConfig, DaCapoModelConfig, BioModelConfig
from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.web_utils import get_free_port


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    """
    Command-line interface for the Cellmap flo application.

    Args:
        log_level (str): The desired log level for the application.
    Examples:
        To use Dacapo run the following commands:
        ```
        cellmap_flow_server dacapo -r my_run -i iteration -d data_path
        ```

        To use custom script
        ```
        cellmap_flow_server script -s script_path -d data_path
        ```

        To use bioimage-io model
        ```
        cellmap_flow_server bioimage -m model_path -d data_path
        ```
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


logger = logging.getLogger(__name__)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
@click.option(
    "-i",
    "--iteration",
    required=False,
    type=int,
    help="The iteration at which to train the run.",
    default=0,
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
@click.option("--debug", is_flag=True, help="Run in debug mode.")
@click.option("-p", "--port", default=0, type=int, help="Port to listen on.")
@click.option("--certfile", default=None, help="Path to SSL certificate file.")
@click.option("--keyfile", default=None, help="Path to SSL private key file.")
def dacapo(run_name, iteration, data_path, debug, port, certfile, keyfile):
    """Run the CellMapFlow server with a DaCapo model."""
    model_config = DaCapoModelConfig(run_name=run_name, iteration=iteration)
    run_server(model_config, data_path, debug, port, certfile, keyfile)


@cli.command()
@click.option(
    "-s",
    "--script_path",
    required=True,
    type=str,
    help="The path to the script to run.",
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
@click.option("--debug", is_flag=True, help="Run in debug mode.")
@click.option("-p", "--port", default=0, type=int, help="Port to listen on.")
@click.option("--certfile", default=None, help="Path to SSL certificate file.")
@click.option("--keyfile", default=None, help="Path to SSL private key file.")
def script(script_path, data_path, debug, port, certfile, keyfile):
    """Run the CellMapFlow server with a custom script."""
    model_config = ScriptModelConfig(script_path=script_path)
    run_server(model_config, data_path, debug, port, certfile, keyfile)


@cli.command()
@click.option(
    "-m", "--model_path", required=True, type=str, help="The path to the model."
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
@click.option("--debug", is_flag=True, help="Run in debug mode.")
@click.option("-p", "--port", default=0, type=int, help="Port to listen on.")
@click.option("--certfile", default=None, help="Path to SSL certificate file.")
@click.option("--keyfile", default=None, help="Path to SSL private key file.")
def bioimage(model_path, data_path, debug, port, certfile, keyfile):
    """Run the CellMapFlow server with a bioimage-io model."""
    model_config = BioModelConfig(model_name=model_path)
    run_server(model_config, data_path, debug, port, certfile, keyfile)


def run_server(model_config, data_path, debug, port, certfile, keyfile):
    server = CellMapFlowServer(data_path, model_config)
    if port == 0:
        port = get_free_port()

    server.run(
        debug=debug,
        port=port,
        certfile=certfile,
        keyfile=keyfile,
    )
