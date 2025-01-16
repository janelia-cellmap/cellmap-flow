import click
import logging

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
        cellmap_flow dacapo -r my_run -i iteration -d data_path
        ```

        To use custom script 
        ```
        cellmap_flow script -s script_path -d data_path
        ```

        To use bioimage-io model
        ```
        cellmap_flow bioimage -m model_path -d data_path
        ```
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


logger = logging.getLogger(__name__)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
@click.option(
    "-i","--iteration", required=False, type=int, help="The iteration at which to train the run."
    default=0
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
def dacapo(run_name, iteration, data_path):
    raise NotImplementedError("This command is not yet implemented.")

@cli.command()
@click.option(
    "-s", "--script_path", required=True, type=str, help="The path to the script to run."
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
def script(script_path, data_path):
    raise NotImplementedError("This command is not yet implemented.")

@cli.command()
@click.option(
    "-m", "--model_path", required=True, type=str, help="The path to the model."
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
def bioimage(model_path, data_path):
    raise NotImplementedError("This command is not yet implemented.")

