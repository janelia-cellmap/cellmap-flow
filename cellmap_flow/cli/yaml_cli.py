"""
YAML-based CLI for running multiple models.
Similar to cli_v2 but uses YAML configuration files for batch processing.

This dynamically discovers ModelConfig subclasses just like cli_v2,
making it easy to add new model types without modifying this file.
"""

import os
import sys
import logging
import click
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url
from cellmap_flow.utils.config_utils import (
    load_config,
    build_models,
    get_model_type_mapping,
)
from cellmap_flow.utils.serilization_utils import get_process_dataset
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ModelConfig

logger = logging.getLogger(__name__)


def _patch_neuroglancer_cache_control() -> None:
    # Chunk URLs are content-addressed by viewer/volume token, so within a
    # session the body for a given URL cannot change. Adding `immutable` lets
    # the browser serve repeats from disk cache with no revalidation
    # round-trip. Patch 34 (d446769).
    import neuroglancer.server

    if getattr(neuroglancer.server.SubvolumeHandler, "_cmf_cache_control_patched", False):
        return
    _orig_get = neuroglancer.server.SubvolumeHandler.get

    async def _patched_get(self, *args, **kwargs):
        self.set_header("Cache-Control", "public, max-age=31536000, immutable")
        return await _orig_get(self, *args, **kwargs)

    neuroglancer.server.SubvolumeHandler.get = _patched_get
    neuroglancer.server.SubvolumeHandler._cmf_cache_control_patched = True


_patch_neuroglancer_cache_control()


def run_multiple(
    models: List[ModelConfig], dataset_path: str, charge_group: str, queue: str, wrap_raw: bool = True
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
        current_data_path = dataset_path
        if hasattr(model, "scale") and model.scale:
            logger.warning(f"Model {getattr(model, 'name', type(model).__name__)} specifies scale {model.scale}, adjusting dataset path accordingly")
            current_data_path = os.path.join(dataset_path, model.scale)

        # Pre-assign a port so we know the host URL immediately (no waiting).
        # Patch b8c22bf: instead of waiting up to 120s for the subprocess to
        # print its address, get_free_port() picks an unused port and we pass
        # -p {port}. wait_for_host=False short-circuits the parent's
        # output-monitoring loop; the model loads in the background and
        # predictions appear in NG once ready (~60-90s) without blocking
        # dashboard startup. Patch a3d1cd9: job.host uses localhost (not
        # get_public_ip) so the SSH-tunneled browser can reach the URL.
        # Phase 8 amendment: job.host is proxy-aware. bootstrap_dashboard.sh
        # exports CMFLOW_PROXY_MODE; under spine, browser reaches the
        # subprocess via spine's nginx /inf-{port}/ forward instead of
        # localhost.
        from cellmap_flow.utils.web_utils import get_free_port
        server_port = get_free_port()
        command = f"{SERVER_COMMAND} {model.command} -d {current_data_path} -p {server_port}"
        model_name = getattr(model, "name", None) or type(model).__name__

        logger.info(f"Submitting job for model: {model_name}")
        logger.warning(f"Executing command: {command}")
        job = start_hosts(
            command, job_name=model_name, queue=queue, charge_group=charge_group,
            wait_for_host=False,
        )
        proxy_mode = os.environ.get("CMFLOW_PROXY_MODE", "direct-ssh")
        if proxy_mode == "spine":
            spine_url = os.environ.get(
                "CMFLOW_SPINE_URL", "https://spine.med.uvm.edu"
            ).rstrip("/")
            job.host = f"{spine_url}/inf-{server_port}"
        else:
            job.host = f"http://localhost:{server_port}"
        logger.info(f"Pre-assigned inference server {model_name} at {job.host}")
        return model_name

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

    generate_neuroglancer_url(dataset_path,wrap_raw=wrap_raw)

    logger.info("All jobs submitted. Monitoring...")
    # Prevent script from exiting immediately:
    while True:
        pass


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
    queue: gpu_h100  # optional, defaults to gpu_h100
    json_data: /path/to/config.json  # optional
    models:
      my_model:
        type: dacapo
        run_name: my_run
        iteration: 100
      fly_model:
        type: fly
        checkpoint: /path/to/checkpoint.ts
        classes: [mito, er, nucleus]
        resolution: [4, 4, 4]

    The model keys (my_model, fly_model) become the model names.

    Model types are automatically discovered from ModelConfig subclasses.
    Use --list-types to see all available types.

    Examples:

    \b
        cellmap_flow_yaml config.yaml
        cellmap_flow_yaml config.yaml --log-level DEBUG
        cellmap_flow_yaml --list-types
        cellmap_flow_yaml config.yaml --validate-only
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # List available model types
    if list_types:
        model_types = get_model_type_mapping()
        click.echo("Available model types:\n")
        for type_name, config_class in sorted(model_types.items()):
            click.echo(f"  {type_name:20s} - {config_class.__name__}")

            # Show required parameters
            import inspect

            sig = inspect.signature(config_class.__init__)
            required = [
                p
                for p, info in sig.parameters.items()
                if p != "self"
                and info.default is inspect.Parameter.empty
                and p not in ["name", "scale"]
            ]
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
    config = load_config(config_path)

    # Handle optional json_data for normalization/postprocessing
    if "json_data" in config:
        json_data = config["json_data"]
        logger.info(f"Loading normalization/postprocessing from: {json_data}")
        g.input_norms, g.postprocess = get_process_dataset(json_data)
    else:
        logger.info("Using default normalization and postprocessing")

    data_path = config["data_path"]
    charge_group = config["charge_group"]
    queue = config["queue"]
    wrap_raw = config.get("wrap_raw", True)
    # Update globals and save to cache
    g.queue = queue
    g.charge_group = charge_group
    g.save_server_config()

    logger.info(f"Data path: {data_path}")
    logger.info(f"Charge group: {charge_group}")
    logger.info(f"Queue: {queue}")

    # Build model configuration objects dynamically
    logger.info("Building model configurations...")
    if config["models"]:
        g.models_config = build_models(config["models"])
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
        return

    # Pre-build additional zarr layers (loaded at startup alongside EM).
    # Each entry stored as (layer, shader, blend) so generate_neuroglancer_url
    # can apply the YAML shader + blend mode on the live viewer; without this
    # the layer falls back to get_raw_layer's hardcoded white [-1,1] shader.
    extra_layers = config.get("extra_layers", [])
    if extra_layers:
        from cellmap_flow.utils.scale_pyramid import get_raw_layer
        g._extra_startup_layers = {}
        for layer_cfg in extra_layers:
            lpath = layer_cfg["path"]
            lname = layer_cfg["name"]
            lshader = layer_cfg.get("shader")
            lblend = layer_cfg.get("blend")
            # layer_type: "image" (default) or "segmentation". When
            # segmentation, get_raw_layer returns a SegmentationLayer and
            # NG renders categorical IDs with its built-in palette.
            ltype = layer_cfg.get("layer_type", "image")
            is_seg = ltype == "segmentation"
            # Per-layer disable_meshes (segmentation only): suppress NG's
            # auto-mesh subsource so segment-pick gestures don't trigger
            # marching-cubes mesh generation. Defaults to False — current
            # behavior is unchanged unless explicitly opted-in.
            ldisable_meshes = bool(layer_cfg.get("disable_meshes", False))
            logger.info(
                f"Pre-loading extra zarr layer: {lname} -> {lpath} "
                f"(type={ltype}, disable_meshes={ldisable_meshes})"
            )
            try:
                layer = get_raw_layer(
                    lpath, normalize=False, segmentation=is_seg,
                    disable_meshes=ldisable_meshes,
                )
                g._extra_startup_layers[lname] = (layer, lshader, lblend)
            except Exception as e:
                import traceback
                logger.error(f"Failed to load extra layer {lname}: {e}\n{traceback.format_exc()}")

    # Run the models
    run_multiple(g.models_config, data_path, charge_group, queue,wrap_raw=wrap_raw)


if __name__ == "__main__":
    main()
