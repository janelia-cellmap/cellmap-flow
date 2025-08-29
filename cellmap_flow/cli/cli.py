import click
import logging
import click

from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.data import ScriptModelConfig
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url


logging.basicConfig()

logger = logging.getLogger(__name__)


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
@click.option(
    "-q",
    "--queue",
    required=False,
    type=str,
    help="The queue to use when submitting",
    default="gpu_h100",
)
@click.option(
    "-P",
    "--charge_group",
    required=False,
    type=str,
    help="The chargeback group to use when submitting",
    default=None,
)
def dacapo(run_name, iteration, data_path, queue, charge_group):
    command = f"{SERVER_COMMAND} dacapo -r {run_name} -i {iteration} -d {data_path}"
    run(command, data_path, queue, charge_group, run_name)
    raise NotImplementedError("This command is not yet implemented.")


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
@click.option(
    "-q",
    "--queue",
    required=False,
    type=str,
    help="The queue to use when submitting",
    default="gpu_h100",
)
@click.option(
    "-P",
    "--charge_group",
    required=False,
    type=str,
    help="The chargeback group to use when submitting",
    default=None,
)
def script(script_path, data_path, queue, charge_group):
    command = f"{SERVER_COMMAND} script -s {script_path} -d {data_path}"
    base_name = script_path.split("/")[-1].split(".")[0]
    run(command, data_path, queue, charge_group, base_name)


@cli.command()
@click.option(
    "-m", "--model_path", required=True, type=str, help="The path to the model."
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
@click.option(
    "-e",
    "--edge_length_to_process",
    required=False,
    type=int,
    help="For 2D models, the desired edge length of the chunk to process; batch size (z) will be adjusted to match as close as possible.",
)
@click.option(
    "-q",
    "--queue",
    required=False,
    type=str,
    help="The queue to use when submitting",
    default="gpu_h100",
)
@click.option(
    "-P",
    "--charge_group",
    required=False,
    type=str,
    help="The chargeback group to use when submitting",
    default=None,
)
def bioimage(model_path, data_path, edge_length_to_process, queue, charge_group):
    command = f"{SERVER_COMMAND} bioimage -m {model_path} -d {data_path} -e {edge_length_to_process}"
    base_name = model_path.split("/")[-1].split(".")[0]
    run(command, data_path, queue, charge_group, base_name)


@cli.command()
@click.option(
    "-f", "--config_folder", required=True, type=str, help="Path to the model folder"
)
@click.option("-n", "--name", required=True, type=str, help="Name of the model")
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
@click.option(
    "-q",
    "--queue",
    required=False,
    type=str,
    help="The queue to use when submitting",
    default="gpu_h100",
)
@click.option(
    "-P",
    "--charge_group",
    required=False,
    type=str,
    help="The chargeback group to use when submitting",
    default=None,
)
def cellmap_model(config_folder, name, data_path, queue, charge_group):
    """Run the CellMapFlow with a CellMap model."""
    command = (
        f"{SERVER_COMMAND} cellmap-model -f {config_folder} -n {name} -d {data_path}"
    )
    run(command, data_path, queue, charge_group, name)


@cli.command()
@click.option(
    "--script_path",
    "-s",
    type=str,
    help="Path to the Python script containing model specification",
)
@click.option("--dataset", "-d", type=str, help="Path to the dataset")
def script_server_check(script_path, dataset):
    model_config = ScriptModelConfig(script_path=script_path)
    server = CellMapFlowServer(dataset, model_config)
    chunk_x = 2
    chunk_y = 2
    chunk_z = 2

    server._chunk_impl(None, None, chunk_x, chunk_y, chunk_z, None)

    print("Server check passed")


def run(command, dataset_path, queue, charge_group, name):

    start_hosts(command, queue, charge_group, name)

    neuroglancer_url = generate_neuroglancer_url(dataset_path)
    while True:
        pass
