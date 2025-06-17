import click
import logging

from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


logger = logging.getLogger(__name__)


@cli.command()
@click.option(
    "-y",
    "--yaml_config",
    required=True,
    type=click.Path(exists=True),
    help="The path to the YAML file.",
)
@click.option(
    "-c",
    "--client",
    is_flag=True,
    default=False,
    help="Run as client if this flag is set.",
)
def run(yaml_config, client):
    is_server = not client
    process = CellMapFlowBlockwiseProcessor(yaml_config, create=is_server)
    if is_server:
        process.run()
    else:
        process.client()
