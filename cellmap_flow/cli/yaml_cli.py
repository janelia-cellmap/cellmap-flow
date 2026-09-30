"""
YAML-based CLI for running multiple models.
Uses YAML configuration files for batch processing.

This dynamically discovers ModelConfig subclasses just like cellmap_flow,
making it easy to add new model types without modifying this file.
"""

import sys
import logging
from cellmap_flow.logging_setup import configure_logging
import click
from typing import TYPE_CHECKING, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from cellmap_flow.jobs.launch import install_cleanup_handlers, start_hosts
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.serving.launch import server_command
from cellmap_flow.config.yaml import ConfigError, load_config, resolve_data_path
from cellmap_flow.globals import g

if TYPE_CHECKING:  # ModelConfig is only needed for the annotation below
    from cellmap_flow.models.models_config import ModelConfig

logger = logging.getLogger(__name__)

EXTRA_LAYER_TYPES = ("image", "segmentation")


def extra_layer_entries(config) -> list:
    """The YAML's ``extra_layers``, checked; [] when there are none.

    Each entry is ``{name, path, layer_type?, shader?, blend?,
    disable_meshes?}``: a volume shown beside the raw data under ``name``.

    Raises:
        ConfigError: an entry lacks a name or path, repeats a name or uses
            the raw layer's ("data"), or has an unknown layer_type.
    """
    entries = config.get("extra_layers") or []
    if not isinstance(entries, list):
        raise ConfigError("YAML 'extra_layers' must be a list")
    names = {"data"}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("path"):
            raise ConfigError(f"Each extra_layers entry needs a name and a path: {entry!r}")
        if entry["name"] in names:
            raise ConfigError(f"extra_layers name {entry['name']!r} is taken")
        names.add(entry["name"])
        if entry.get("layer_type", "image") not in EXTRA_LAYER_TYPES:
            raise ConfigError(
                f"extra_layers {entry['name']!r}: layer_type must be one of {EXTRA_LAYER_TYPES}"
            )
    return entries


def build_extra_layers(entries) -> dict:
    """``{name: neuroglancer layer}`` for extra_layers entries.

    Read as stored, without the input normalizers. An image entry's
    ``shader`` and ``blend`` are applied to its layer; a segmentation colours
    its ids itself, and has no blend. An entry whose volume cannot be opened
    is logged and left out, rather than stopping the dashboard.
    """
    from cellmap_flow.viewer.raw import get_raw_layer

    layers = {}
    for entry in entries:
        name = entry["name"]
        segmentation = entry.get("layer_type", "image") == "segmentation"
        try:
            layer = get_raw_layer(
                entry["path"],
                normalize=False,
                segmentation=segmentation,
                disable_meshes=bool(entry.get("disable_meshes", False)),
            )
        except Exception as e:
            logger.error(f"Could not open extra layer {name!r} ({entry['path']}): {e}")
            continue
        if segmentation:
            if entry.get("shader") or entry.get("blend"):
                logger.warning(f"Extra layer {name!r} is a segmentation: ignoring shader and blend")
        else:
            if entry.get("shader"):
                layer.shader = entry["shader"]
            if entry.get("blend"):
                layer.blend = entry["blend"]
        layers[name] = layer
    return layers


def run_multiple(
    models: List["ModelConfig"], dataset_path: str, charge_group: str, queue: str, wrap_raw: bool = True
) -> None:
    """
    Submit multiple model inference jobs.

    Args:
        models: List of ModelConfig instances to run
        dataset_path: Base path to the dataset
        charge_group: Billing/chargeback group
        queue: Job queue name
    """
    g.queue = queue
    g.charge_group = charge_group

    def _submit_model(model):
        current_data_path = resolve_data_path(dataset_path, getattr(model, "scale", None))
        if current_data_path != dataset_path:
            logger.info(
                f"Model {getattr(model, 'name', type(model).__name__)} specifies "
                f"scale {model.scale}; reading {current_data_path}"
            )

        command = server_command(model, current_data_path)
        model_name = getattr(model, "name", None) or type(model).__name__

        logger.info(f"Submitting job for model: {model_name}")
        logger.warning(f"Executing command: {command}")
        start_hosts(
            command, job_name=model_name, queue=queue, charge_group=charge_group
        )
        return model_name

    failed = []
    if models:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(_submit_model, model): model for model in models}
            for future in as_completed(futures):
                try:
                    name = future.result()
                    logger.info(f"Job for {name} is ready")
                except Exception as e:
                    model = futures[future]
                    model_name = getattr(model, "name", None) or type(model).__name__
                    logger.error(f"Failed to start job for {model_name}: {e}")
                    failed.append(model_name)

    # Some models starting is worth a dashboard; none of them is a failed
    # run, not an empty viewer to leave running.
    if models and len(failed) == len(models):
        raise JobStartError(f"No model server started ({', '.join(failed)})")

    # Imported here so --help and config errors do not pay for the viewer
    # stack (~16s before this).
    from cellmap_flow.dashboard.services.startup import generate_neuroglancer_url

    # Serves the dashboard; does not return.
    generate_neuroglancer_url(dataset_path,wrap_raw=wrap_raw)


@click.command()
@click.argument("config_path", type=click.Path(exists=True), required=False)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level",
)
@click.option("--list-types", is_flag=True, help="List available model types and exit")
@click.option(
    "--validate-only",
    is_flag=True,
    help="Validate YAML configuration without running jobs",
)
def main(config_path: str, log_level: str, list_types: bool, validate_only: bool):
    """
    Run multiple model inference jobs from a YAML configuration file.

    The YAML file should have the following structure:

    \b
    data_path: /path/to/data
    charge_group: my_group
    queue: gpu_h100        # optional, defaults to gpu_h100
    walltime: "08:00"      # optional; LSF run limit, "HH:MM" or minutes.
                           # Without it the queue's own default applies,
                           # which is 2 hours on the Janelia GPU queues.
    cycle_gpu_queues: true # optional; false pins the job to `queue` above
                           # instead of falling back to a queue with capacity.
    wrap_raw: true         # optional; false serves raw straight from the file
    extra_layers:          # optional; more volumes to show beside the raw data
      - name: mito_pred
        path: /path/to/pred.zarr/mito
        shader: "..."      # optional, image layers only
        blend: additive    # optional, image layers only
      - name: instances
        path: /path/to/instances.zarr/s0
        layer_type: segmentation  # default: image
        disable_meshes: true      # optional; no meshes computed on a pick
    json_data:             # optional; normalization and postprocessing
      input_norm:
        MinMaxNormalizer: {min_value: 0, max_value: 255}
        LambdaNormalizer: {expression: "x*2-1"}
      postprocess:
        SigmoidPostprocessor: {}
    models:
      - type: dacapo
        name: my_model
        run_name: my_run
        iteration: 100
      - type: fly
        name: fly_model
        checkpoint: /path/to/checkpoint.ts
        classes: [mito, er, nucleus]
        resolution: [4, 4, 4]

    Models may also be given as a mapping, where each key is the model name:

    \b
    models:
      my_model:
        type: dacapo
        run_name: my_run

    json_data is an inline mapping (or a JSON string), not a path to a file.

    Model types are automatically discovered from ModelConfig subclasses.
    Use --list-types to see all available types.

    Examples:

    \b
        cellmap_flow_yaml config.yaml
        cellmap_flow_yaml config.yaml --log-level DEBUG
        cellmap_flow_yaml --list-types
        cellmap_flow_yaml config.yaml --validate-only
    """
    configure_logging(getattr(logging, log_level.upper()))

    # List available model types
    if list_types:
        from cellmap_flow.models import registry

        model_types = registry.model_types()
        click.echo("Available model types:\n")
        for type_name, config_class in sorted(model_types.items()):
            click.echo(f"  {type_name:20s} - {config_class.__name__}")

            required = registry.required_params(config_class)
            if required:
                click.echo(f"                       Required: {', '.join(required)}")

        click.echo("\nSee example YAML configuration in the docstring with --help")
        return

    # Ensure config_path is provided when not listing types
    if not config_path:
        click.echo("Error: Missing argument 'CONFIG_PATH'.")
        click.echo("Try 'cellmap_flow_yaml --help' for help.")
        sys.exit(1)

    # Load and validate configuration
    logger.info(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
        extra_layers = extra_layer_entries(config)
    except ConfigError as e:
        raise click.ClickException(str(e))

    # Handle optional json_data for normalization/postprocessing
    if "json_data" in config:
        json_data = config["json_data"]
        logger.info(f"Loading normalization/postprocessing from: {json_data}")
        from cellmap_flow.pipeline_spec import PipelineSpec

        g.input_norms, g.postprocess = PipelineSpec.from_json_data(json_data, strict=True).build()
    else:
        logger.info("Using default normalization and postprocessing")

    data_path = config["data_path"]
    charge_group = config["charge_group"]
    queue = config["queue"]
    wrap_raw = config.get("wrap_raw", True)
    # Optional; falls back to the cached dashboard setting, then to the
    # site's default_walltime (jobs/site.py). Accepts "08:00" or plain minutes.
    walltime = config.get("walltime")
    # Optional; None means "leave whatever the dashboard setting is". Only an
    # explicit false pins submissions to `queue`.
    cycle_gpu_queues = config.get("cycle_gpu_queues")

    # Update globals; they are saved to the cache below, once this is a real
    # run rather than a --validate-only check.
    g.queue = queue
    g.charge_group = charge_group
    if walltime:
        g.walltime = walltime
    if cycle_gpu_queues is not None:
        g.cycle_gpu_queues = bool(cycle_gpu_queues)

    logger.info(f"Data path: {data_path}")
    logger.info(f"Charge group: {charge_group}")
    logger.info(f"Queue: {queue}")
    if not getattr(g, "cycle_gpu_queues", True):
        logger.info("GPU queue cycling: off (jobs wait for the queue above)")

    # Build model configuration objects dynamically
    logger.info("Building model configurations...")
    if config["models"]:
        from cellmap_flow.models.registry import build_models

        try:
            g.models_config = build_models(config["models"])
        except ConfigError as e:
            raise click.ClickException(str(e))
    else:
        g.models_config = []
        logger.info("No models configured — starting dashboard for interactive use")

    logger.info(f"Configured {len(g.models_config)} model(s):")
    for i, model in enumerate(g.models_config, 1):
        model_name = getattr(model, "name", None) or type(model).__name__
        logger.info(f"  {i}. {model_name} ({type(model).__name__})")

    # Validation mode - exit without running
    if validate_only:
        click.echo("\n✓ Configuration is valid!")
        click.echo(f"  - Models: {len(g.models_config)}")
        click.echo(f"  - Data path: {data_path}")
        click.echo(f"  - Queue: {queue}")
        if extra_layers:
            click.echo(f"  - Extra layers: {len(extra_layers)}")
        return

    g.save_server_config()
    g.extra_layers = build_extra_layers(extra_layers)

    # Run the models; Ctrl+C or SIGTERM from here on kills what was started.
    install_cleanup_handlers()
    try:
        run_multiple(g.models_config, data_path, charge_group, queue,wrap_raw=wrap_raw)
    except JobStartError as e:
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
