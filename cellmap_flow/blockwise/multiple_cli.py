import click
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.argument("yaml_configs", nargs=-1, required=True, type=click.Path(exists=True))
def cli(yaml_configs: tuple) -> None:
    """Process multiple YAML configuration files."""
    from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor
    from cellmap_flow.utils.config_utils import ConfigError

    for yaml_config in yaml_configs:
        logger.info(f"Processing: {yaml_config}")
        try:
            process = CellMapFlowBlockwiseProcessor(yaml_config, create=True)
        except ConfigError as e:
            raise click.ClickException(f"{yaml_config}: {e}")
        process.run()


if __name__ == "__main__":
    cli()
