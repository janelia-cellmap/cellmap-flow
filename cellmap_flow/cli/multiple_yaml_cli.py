import sys
import logging

from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url
from cellmap_flow.utils.config_utils import load_config, build_models
from cellmap_flow.globals import g


logger = logging.getLogger(__name__)


def run_multiple(models, dataset_path, charge_group, queue):
    g.queue = queue
    g.charge_group = charge_group

    for model in models:
        current_data_path = dataset_path
        if hasattr(model, "scale") and model.scale:
            # Derive a scaled data path if 'scale' is provided
            base_parts = dataset_path.rstrip("/").split("/")
            scaled_dir = f"{base_parts[-1].rsplit('/', 1)[-1]}/{model.scale}"
            # Reconstruct the path to the same parent directory with scale subfolder
            current_data_path = "/".join(base_parts[:-1] + [model.scale])

        command = f"{SERVER_COMMAND} {model.command} -d {current_data_path}"
        start_hosts(
            command, job_name=model.name, queue=queue, charge_group=charge_group
        )

    generate_neuroglancer_url(dataset_path)

    # Prevent script from exiting immediately:
    while True:
        pass


def main():
    """
    Usage:
        python cellmap_flow_with_yaml.py /path/to/config.yaml
    """
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(main.__doc__)
        sys.exit(0)

    config_path = sys.argv[1]
    config = load_config(config_path)

    data_path = config["data_path"]
    charge_group = config["charge_group"]
    queue = config["queue"]

    print("Data path:", data_path)

    # Build model configuration objects
    models = build_models(config["models"])

    # For debugging, print each model config
    for model in models:
        print(model)

    run_multiple(models, data_path, charge_group, queue)


if __name__ == "__main__":
    main()
