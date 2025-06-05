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
    "-s",
    "--is_server",
    required=False,
    type=bool,
    default=False,
    help="The path to the output.",
)
def run(yaml_config, is_server):
    process = CellMapFlowBlockwiseProcessor(yaml_config, create=is_server)
    if is_server:
        process.run()
    else:
        process.client()
