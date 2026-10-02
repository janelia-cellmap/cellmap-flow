import click
import logging

from cellmap_flow.cli.common import log_level_option

logger = logging.getLogger(__name__)


@click.command(name="blockwise")
@click.argument("yaml_configs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-c",
    "--client",
    is_flag=True,
    default=False,
    help="Run as client if this flag is set.",
)
@log_level_option()
def cli(yaml_configs, client):
    """Run the model a blockwise YAML describes over the whole volume, block
    by block, writing its predictions to disk.

    Several YAMLs run one after the other. One that leaves blocks
    unprocessed does not stop the rest; the command fails at the end,
    naming them. --client runs one of a run's workers (the run starts them
    itself), so it takes one YAML.

    \b
      cellmap_flow blockwise config.yaml
      cellmap_flow blockwise mito.yaml er.yaml
    """
    # Imported inside the command so --help and argument errors do not
    # pay for the whole inference stack (~16s before this).
    from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor

    from cellmap_flow.config.yaml import ConfigError

    if client and len(yaml_configs) != 1:
        raise click.UsageError("--client runs one worker, of one YAML's run")

    incomplete = []
    for yaml_config in yaml_configs:
        if len(yaml_configs) > 1:
            logger.info(f"Processing: {yaml_config}")
        try:
            process = CellMapFlowBlockwiseProcessor(yaml_config, create=not client)
        except ConfigError as e:
            raise click.ClickException(f"{yaml_config}: {e}")
        if client:
            process.client()
        # Later configs are independent of this one, so carry on.
        elif not process.run():
            incomplete.append(yaml_config)

    if incomplete:
        raise click.ClickException(f"{', '.join(incomplete)}: some blocks were not processed")


if __name__ == "__main__":
    from cellmap_flow.plugins import load_plugins

    load_plugins()
    cli()
