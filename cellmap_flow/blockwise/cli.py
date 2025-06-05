import click
import logging

from cellmap_flow.utils.data import FlyModelConfig

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
    """
    Command-line interface for the Cellmap flo application.

    Args:
        log_level (str): The desired log level for the application.
    Examples:
        To use Dacapo run the following commands:
        ```
        cellmap_flow_server dacapo -r my_run -i iteration -d data_path
        ```

        To use custom script
        ```
        cellmap_flow_server script -s script_path -d data_path
        ```

        To use bioimage-io model
        ```
        cellmap_flow_server bioimage -m model_path -d data_path
        ```
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


logger = logging.getLogger(__name__)


@cli.command()
@click.option(
    "-c", "--checkpoint", required=True, type=str, help="The path to the checkpoint."
)
@click.option(
    "-ch",
    "--channels",
    required=True,
    type=str,
    help="The channels of the model. split by comma.",
)
@click.option(
    "-ivs",
    "--input_voxel_size",
    required=True,
    type=str,
    help="The input voxel size of the model. split by comma.",
)
@click.option(
    "-ovs",
    "--output_voxel_size",
    required=True,
    type=str,
    help="The output voxel size of the model. split by comma.",
)
@click.option(
    "-d", "--data_path", required=True, type=str, help="The path to the data."
)
@click.option(
    "-o", "--output_path", required=True, type=str, help="The path to the output."
)
@click.option(
    "-s", "--is_server", required=False, type=bool, default=False, help="The path to the output."
)
@click.option(
    "-outc", "--output_channels", required=False, type=str, default=None,help="The path to the output."
)
@click.option(
    "-json", "--json_data", required=False, type=str, default=None,help="The path to the output."
)
def run(checkpoint, channels, input_voxel_size, output_voxel_size, data_path, output_path,is_server, output_channels,json_data):
    """Run the CellMapFlow server with a Fly model."""
    channels = channels.split(",")
    input_voxel_size = tuple(map(int, input_voxel_size.split(",")))
    output_voxel_size = tuple(map(int, output_voxel_size.split(",")))
    output_channels = output_channels.split(",")
    
    model_config = FlyModelConfig(
        chpoint_path=checkpoint,
        channels=channels,
        input_voxel_size=input_voxel_size,
        output_voxel_size=output_voxel_size,
    )
    process = CellMapFlowBlockwiseProcessor(data_path,model_config, output_path,create=is_server, output_channels=output_channels,json_data=json_data)
    if is_server:
        process.run()
    else:
        process.client()
