"""
Simple CLI for viewing datasets with CellMap Flow without requiring model configs.
"""

import os

import click
import logging
from cellmap_flow.logging_setup import configure_logging
from cellmap_flow.globals import g

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
    "-P",
    "--project",
    default=None,
    help="Charge group (LSF project) billed for the models launched from the dashboard",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level",
)
def main(dataset, project, log_level):
    """
    Start CellMap Flow viewer with a dataset.

    Opens neuroglancer on the raw data and starts the dashboard, where models
    can be picked and submitted interactively. Use cellmap_flow_yaml instead to
    launch models from a config file.

    Example:

    \b
      cellmap_flow_view -d /path/to/dataset.zarr
    """
    # Imported inside the command so --help and argument errors do not
    # pay for the whole inference stack (~16s before this).
    import neuroglancer

    from cellmap_flow.dashboard.app import create_and_run_app
    from cellmap_flow.jobs.launch import install_cleanup_handlers
    from cellmap_flow.viewer.raw import get_raw_layer

    configure_logging(getattr(logging, log_level.upper()))
    # Models picked in the dashboard are jobs too; kill them on the way out.
    install_cleanup_handlers()

    logger.info(f"Starting CellMap Flow viewer with dataset: {dataset}")

    # Set up neuroglancer server
    neuroglancer.set_server_bind_address("0.0.0.0")

    # Create viewer
    viewer = neuroglancer.Viewer()

    # Fileglancer runs the viewer as an LSF job, and the models picked in the
    # dashboard should be billed where that job is: LSB_PROJECT_NAME is the
    # job's project. An explicit -P wins over it.
    if os.environ.get("LSB_PROJECT_NAME"):
        g.charge_group = os.environ["LSB_PROJECT_NAME"]
    if project:
        g.charge_group = project

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
    create_and_run_app(neuroglancer_url=str(viewer))


if __name__ == "__main__":
    main()
