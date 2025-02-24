import click
import logging
import click

from cellmap_flow.server import CellMapFlowServer
from cellmap_flow.utils.bsub_utils import run_locally, start_hosts
from cellmap_flow.utils.data import ScriptModelConfig
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url


logging.basicConfig()

logger = logging.getLogger(__name__)


SERVER_COMMAND = "cellmap_flow_server"


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
    run(
        command,
        data_path,
        queue,
        charge_group,
    )
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
    run(command, data_path, queue, charge_group)


@cli.command()
@click.option(
    "-m", "--model_path", required=True, type=str, help="The path to the model."
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
def bioimage(model_path, data_path, queue, charge_group):
    command = f"{SERVER_COMMAND} bioimage -m {model_path} -d {data_path}"
    run(command, data_path, queue, charge_group)


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


def run(
    command,
    dataset_path,
    queue,
    charge_group,
):
    host = start_hosts(command, queue, charge_group)
    if host is None:
        raise Exception("Could not start host")

    inference_dict = {host: "prediction"}
    neuroglancer_url = generate_neuroglancer_url(dataset_path, inference_dict)
    ui_host = run_locally(
        f"cellmap_flow_server run-ui-server -n {neuroglancer_url} -i {host}"
    )
    print(host)
    while True:
        pass
