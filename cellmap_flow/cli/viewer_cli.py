"""
Simple CLI for viewing datasets with CellMap Flow without requiring model configs.
"""

import os

import click
import logging
from cellmap_flow.cli.common import log_level_option, resample_option

logger = logging.getLogger(__name__)


@click.command(name="view")
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
@resample_option()
@log_level_option()
def main(dataset, project, resample):
    """
    Start CellMap Flow viewer with a dataset.

    Opens neuroglancer on the raw data and starts the dashboard, where models
    can be picked and submitted interactively. Use `cellmap_flow yaml` instead
    to launch models from a config file.

    The dashboard's Resample box starts ticked, so the models launched from
    it read the data resampled to their voxel size when it has no level at
    it; ``--no-resample`` starts it unticked.

    Example:

    \b
      cellmap_flow view -d /path/to/dataset.zarr
    """
    # Imported inside the command so --help and argument errors do not
    # pay for the whole inference stack (~16s before this).
    from cellmap_flow.dashboard.app import create_and_run_app
    from cellmap_flow.dashboard.services.launch import watch_layer_deletes
    from cellmap_flow.dashboard.state import get_session
    from cellmap_flow.jobs.launch import install_cleanup_handlers
    from cellmap_flow.jobs.settings import launcher_settings
    from cellmap_flow.viewer.bootstrap import new_viewer, viewer_url

    # Models picked in the dashboard are jobs too; kill them on the way out.
    install_cleanup_handlers()

    logger.info(f"Starting CellMap Flow viewer with dataset: {dataset}")

    # Fileglancer runs the viewer as an LSF job, and the models picked in the
    # dashboard should be billed where that job is: LSB_PROJECT_NAME is the
    # job's project. An explicit -P wins over it.
    # For this process only, not saved: it is this job's billing, not a
    # default for the next dashboard.
    if os.environ.get("LSB_PROJECT_NAME"):
        launcher_settings().charge_group = os.environ["LSB_PROJECT_NAME"]
    if project:
        launcher_settings().charge_group = project

    session = get_session()
    session.dataset_path = dataset
    session.resample = resample
    # At 8 nm whatever the data's voxel size, as this command has always shown it.
    viewer = new_viewer(dataset, scales=(8, 8, 8))
    session.viewer = viewer
    watch_layer_deletes(viewer)

    # Print viewer URL
    url = viewer_url(viewer)
    logger.info(f"Neuroglancer viewer URL: {url}")
    print(f"\n{'='*80}")
    print(f"Neuroglancer viewer: {url}")
    print(f"Dataset: {dataset}")
    print(f"{'='*80}\n")

    # Start the dashboard app
    create_and_run_app(neuroglancer_url=url)


if __name__ == "__main__":
    from cellmap_flow.plugins import load_plugins

    load_plugins()
    main()
