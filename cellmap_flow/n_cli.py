import click
import logging

import neuroglancer
import os
import sys
import signal
import select
import itertools
import click

from cellmap_flow.utils.bsub_utils import is_bsub_available, submit_bsub_job, parse_bpeek_output, run_locally, job_ids, security
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_link

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
def dacapo(run_name, iteration, data_path):
    command = f"{SERVER_COMMAND} dacapo -r {run_name} -i {iteration} -d {data_path}"
    run(command,data_path)
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
def script(script_path, data_path):
    command = f"{SERVER_COMMAND} script -s {script_path} -d {data_path}"
    run(command,data_path)
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



def run(command,dataset_path):

    host = start_hosts(command)
    if host is None:
        raise Exception("Could not start host")

    inference_dict = {host:"prediction"}

    generate_neuroglancer_link(dataset_path, inference_dict)

def start_hosts(command):
    if security == "https":
        command = f"{command} --certfile=host.cert --keyfile=host.key"


    if is_bsub_available():
        result = submit_bsub_job(command, job_name="example_job")
        job_id = result.stdout.split()[1][1:-1]
        job_ids.append(job_id)
        host = parse_bpeek_output(job_id)
    else:
        host= run_locally(command)

    return host
