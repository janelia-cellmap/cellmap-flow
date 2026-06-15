import click
import logging
import logging
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
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
    "-t",
    "--test",
    is_flag=True,
    default=False,
    help="Run process_fn on a single block end-to-end (no daisy scheduling) "
         "for quick YAML validation.",
)
@click.option(
    "--test-offset",
    type=str,
    default=None,
    help="Comma-separated write_roi offset in nm for --test "
         "(e.g. '8000,8000,30000'). Defaults to volume origin + context.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(yaml_config, client, test, test_offset, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    if test:
        offset = None
        if test_offset:
            offset = tuple(int(x.strip()) for x in test_offset.split(","))
        process = CellMapFlowBlockwiseProcessor(yaml_config, create=True)
        process.test_block(offset=offset)
        return

    is_server = not client
    process = CellMapFlowBlockwiseProcessor(yaml_config, create=is_server)
    if is_server:
        process.run()
    else:
        process.client()


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    cli()
