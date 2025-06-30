import sys
import yaml
import logging

from cellmap_flow.utils.data import (
    DaCapoModelConfig,
    BioModelConfig,
    ScriptModelConfig,
    CellMapModelConfig,
)


DEFAULT_SERVER_QUEUE = "gpu_h100"

logger = logging.getLogger(__name__)


def load_config(path):
    """
    Load and validate the YAML configuration.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Required top-level fields
    if "data_path" not in config:
        logger.error("Missing required field in YAML: data_path")
        sys.exit(1)
    if "charge_group" not in config:
        logger.error("Missing required field in YAML: charge_group")
        sys.exit(1)

    # If queue is missing, set default
    if "queue" not in config or not config["queue"]:
        logger.warning(
            f"Missing 'queue' in YAML, using default: {DEFAULT_SERVER_QUEUE}"
        )
        config["queue"] = DEFAULT_SERVER_QUEUE

    # Models must be a non-empty list
    if (
        "models" not in config
        or not isinstance(config["models"], list)
        or not config["models"]
    ):
        logger.error("YAML must contain a non-empty 'models' list")
        sys.exit(1)

    return config


def build_models(model_entries):
    """
    Given a list of model entries from YAML, instantiate the correct ModelConfig objects.
    """
    models = []
    for idx, entry in enumerate(model_entries):
        mtype = entry.get("type")
        if not mtype:
            logger.error(f"Model entry #{idx + 1} missing 'type' field")
            sys.exit(1)

        mtype_lower = mtype.lower()
        if mtype_lower == "dacapo":
            run_name = entry.get("run_name")
            iteration = entry.get("iteration", 0)
            name = entry.get("name", None)

            if not run_name:
                logger.error(f"Model entry #{idx + 1} (dacapo) missing 'run_name'")
                sys.exit(1)

            models.append(DaCapoModelConfig(run_name, iteration, name=name))

        elif mtype_lower in ("cellmap-model", "cellmap_model", "cellmapmodel"):
            config_folder = entry.get("config_folder")
            name = entry.get("name", None)
            scale = entry.get("scale", None)

            if not config_folder:
                logger.error(
                    f"Model entry #{idx + 1} (cellmap-model) missing 'config_folder'"
                )
                sys.exit(1)

            models.append(CellMapModelConfig(config_folder, name=name, scale=scale))

        elif mtype_lower == "script":
            script_path = entry.get("script_path")
            name = entry.get("name", None)
            scale = entry.get("scale", None)

            if not script_path:
                logger.error(f"Model entry #{idx + 1} (script) missing 'script_path'")
                sys.exit(1)

            models.append(ScriptModelConfig(script_path, name=name, scale=scale))

        elif mtype_lower in ("bioimage", "bio-image", "bio_image"):
            model_path = entry.get("model_path")
            name = entry.get("name", None)

            if not model_path:
                logger.error(f"Model entry #{idx + 1} (bioimage) missing 'model_path'")
                sys.exit(1)

            models.append(BioModelConfig(model_path, name=name))

        else:
            logger.error(
                f"Model entry #{idx + 1} has unrecognized type '{mtype}'. "
                "Valid types are: dacapo, cellmap-model, script, bioimage."
            )
            sys.exit(1)

    return models
