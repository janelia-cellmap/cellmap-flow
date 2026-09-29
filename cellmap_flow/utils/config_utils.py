"""
Smart YAML configuration utilities that dynamically discover and instantiate
ModelConfig subclasses, similar to the CLI v2 approach.
"""

import json
import os
import yaml
import logging
from typing import List, Dict, Any, Optional

from cellmap_flow.models import registry
from cellmap_flow.models.models_config import ModelConfig

DEFAULT_SERVER_QUEUE = "gpu_h100"

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """The YAML configuration is not usable; the message says why.

    Raised rather than calling sys.exit: this code also runs inside the
    dashboard (the blockwise precheck, a finetuned model's base config),
    where SystemExit slips past ``except Exception`` and the request thread
    dies without sending a response. The CLIs catch it and exit non-zero.
    """


def _node_kind(path) -> Optional[str]:
    """"array" or "group" for a local zarr (v2 or v3) or N5 node, else None.

    None also covers remote URLs and paths that do not exist yet: nothing
    here can tell what they are without opening them.
    """
    local = str(path)
    if local.startswith("file://"):
        local = local[len("file://"):]
    elif "://" in local:
        return None
    if os.path.isfile(os.path.join(local, ".zarray")):
        return "array"
    if os.path.isfile(os.path.join(local, ".zgroup")):
        return "group"
    for name, key, is_array in (
        ("zarr.json", "node_type", lambda v: v == "array"),
        ("attributes.json", "dimensions", lambda v: v is not None),  # N5
    ):
        meta_path = os.path.join(local, name)
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                return None
            return "array" if is_array(meta.get(key)) else "group"
    return None


def resolve_data_path(data_path: str, scale: Optional[str]) -> str:
    """The dataset a model reads: ``data_path``, with ``scale`` applied.

    The one rule every launcher uses (cellmap_flow, cellmap_flow_yaml and
    blockwise), so the same YAML reads the same data in each:

    - ``data_path`` is an array: it is used as it is. A ``scale`` naming a
      different level is ignored, with a warning.
    - otherwise (a multiscale group, or a path that cannot be inspected
      here, such as a URL): ``scale`` selects the level under it.
    """
    if not scale:
        return data_path
    scale = str(scale).strip("/")
    if _node_kind(data_path) == "array":
        tail = str(data_path).rstrip("/")
        if tail != scale and not tail.endswith("/" + scale):
            logger.warning(
                f"data_path {data_path} is an array, so it is used as is; "
                f"scale {scale!r}, which names a different level, is ignored"
            )
        return data_path
    return os.path.join(data_path, scale)


def get_model_type_mapping() -> Dict[str, type]:
    """Every model type by the name YAML ``type:`` uses: ``registry.model_types()``."""
    return registry.model_types()


def load_config(path: str) -> Dict[str, Any]:
    """
    Load and validate the YAML configuration.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Validated configuration dictionary

    Raises:
        ConfigError: a required field is missing or malformed
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")

    from cellmap_flow.globals import load_server_config_cache, SERVER_CONFIG_DEFAULTS

    # Required top-level fields
    if "data_path" not in config:
        raise ConfigError("Missing required field in YAML: data_path")

    # Fall back to cache then defaults for charge_group and queue
    cached = load_server_config_cache() or {}

    if "charge_group" not in config or not config["charge_group"]:
        fallback = cached.get("charge_group", SERVER_CONFIG_DEFAULTS.get("charge_group"))
        if fallback:
            logger.warning(f"Missing 'charge_group' in YAML, using cached value: {fallback}")
            config["charge_group"] = fallback
        else:
            raise ConfigError(
                "Missing required field in YAML: charge_group (no cache available)"
            )

    if "queue" not in config or not config["queue"]:
        fallback = cached.get("queue", DEFAULT_SERVER_QUEUE)
        logger.warning(f"Missing 'queue' in YAML, using: {fallback}")
        config["queue"] = fallback

    # Models field: must be a dict, list, or empty/missing (for dashboard-only mode)
    if "models" not in config or config["models"] is None:
        config["models"] = {}

    if not isinstance(config["models"], (dict, list)):
        raise ConfigError("YAML 'models' must be either a dict or list")

    return config


def build_model_from_entry(entry: Dict[str, Any], model_name: str) -> ModelConfig:
    """A model config from a YAML model entry: ``registry.build_model``.

    Raises:
        ConfigError: the entry does not describe a model that can be built
    """
    return registry.build_model(entry, model_name)


def build_models(model_entries: Dict[str, Dict[str, Any]]) -> List[ModelConfig]:
    """
    Given model entries from YAML, instantiate the correct ModelConfig objects.
    Uses dynamic discovery like the cellmap_flow CLI instead of hardcoded if/else chains.
    
    YAML format:
    models:
      my_model_1:
        type: cellmap
        checkpoint_path: /path/to/checkpoint
      my_model_2:
        type: dacapo
        run_name: my_run
        iteration: 50000
    
    Args:
        model_entries: Dictionary mapping model names to their configurations
        
    Returns:
        List of instantiated ModelConfig objects
    """
    models = []
    
    if isinstance(model_entries, list):
        entries = {}
        for entry in model_entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ConfigError("Each model entry in the list must have a 'name' field.")
            entries[entry["name"]] = entry
        model_entries = entries

    
    for model_name, entry in model_entries.items():
        model = build_model_from_entry(entry, model_name=model_name)
        models.append(model)
    
    return models
