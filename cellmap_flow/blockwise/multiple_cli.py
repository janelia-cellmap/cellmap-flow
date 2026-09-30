import click
import logging
from cellmap_flow.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

@click.command()
@click.argument("yaml_configs", nargs=-1, required=True, type=click.Path(exists=True))
def cli(yaml_configs: tuple) -> None:
    """Process multiple YAML configuration files."""
    configure_logging(logging.INFO)
    from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor
    from cellmap_flow.config.yaml import ConfigError

    incomplete = []
    for yaml_config in yaml_configs:
        logger.info(f"Processing: {yaml_config}")
        try:
            process = CellMapFlowBlockwiseProcessor(yaml_config, create=True)
        except ConfigError as e:
            raise click.ClickException(f"{yaml_config}: {e}")
        # Later configs are independent of this one, so carry on.
        if not process.run():
            incomplete.append(yaml_config)

    if incomplete:
        raise click.ClickException(
            f"Some blocks were not processed for: {', '.join(incomplete)}"
        )


if __name__ == "__main__":
    cli()
