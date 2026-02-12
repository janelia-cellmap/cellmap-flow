"""
Simple CLI for viewing datasets with CellMap Flow without requiring model configs.
"""

import click
import logging
import neuroglancer
from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.globals import g
from cellmap_flow.utils.scale_pyramid import get_raw_layer

logging.basicConfig()
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-d",
    "--dataset",
    required=True,
    type=str,
    help="Path to the dataset (zarr or n5)",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level",
)
def main(dataset, log_level):
    """
    Start CellMap Flow viewer with a dataset.

    Example:
        cellmap_flow_viewer -d /path/to/dataset.zarr
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    logger.info(f"Starting CellMap Flow viewer with dataset: {dataset}")

    # Set up neuroglancer server
    neuroglancer.set_server_bind_address("0.0.0.0")

    # Create viewer
    viewer = neuroglancer.Viewer()

    # Set dataset path in globals
    g.dataset_path = dataset
    g.viewer = viewer

    # Add dataset layer to viewer
    with viewer.txn() as s:
        # Set coordinate space
        s.dimensions = neuroglancer.CoordinateSpace(
            names=["z", "y", "x"],
            units="nm",
            scales=[8, 8, 8],
        )

        # Add data layer
        s.layers["data"] = get_raw_layer(dataset)

    # Print viewer URL
    logger.info(f"Neuroglancer viewer URL: {viewer}")
    print(f"\n{'='*80}")
    print(f"Neuroglancer viewer: {viewer}")
    print(f"Dataset: {dataset}")
    print(f"{'='*80}\n")

    # Start the dashboard app
    create_and_run_app(neuroglancer_url=str(viewer), inference_servers=None)


if __name__ == "__main__":
    main()
