import click
import logging
from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.argument("yaml_configs", nargs=-1, required=True, type=click.Path(exists=True))
def cli(yaml_configs: tuple) -> None:
    """Process multiple YAML configuration files."""
    for yaml_config in yaml_configs:
        logger.info(f"Processing: {yaml_config}")
        process = CellMapFlowBlockwiseProcessor(yaml_config, create=True)
        process.run()


if __name__ == "__main__":
    cli()
