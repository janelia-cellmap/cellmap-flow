import click
import logging

from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor


@click.command()
@click.argument("yaml_config", type=click.Path(exists=True))
@click.option(
    "-c",
    "--client",
    is_flag=True,
    default=False,
    help="Run as client if this flag is set.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(yaml_config, client, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    is_server = not client
    process = CellMapFlowBlockwiseProcessor(yaml_config, create=is_server)
    if is_server:
        process.run()
    else:
        process.client()


logger = logging.getLogger(__name__)
