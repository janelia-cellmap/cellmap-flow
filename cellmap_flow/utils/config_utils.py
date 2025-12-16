"""
Smart YAML configuration utilities that dynamically discover and instantiate
ModelConfig subclasses, similar to the CLI v2 approach.
"""

import sys
import yaml
import logging
import inspect
from typing import List, Dict, Any

from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.utils.cli_utils import get_all_subclasses, process_constructor_args

DEFAULT_SERVER_QUEUE = "gpu_h100"

logger = logging.getLogger(__name__)


def get_model_type_mapping() -> Dict[str, type]:
    """
    Get mapping of CLI-friendly names to ModelConfig classes.
    Uses the same logic as cli_v2 for consistency.
    
    Returns:
        Dictionary mapping model type names to ModelConfig classes
    """
    return get_all_subclasses(ModelConfig)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load and validate the YAML configuration.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Validated configuration dictionary
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

    # Models must be present and non-empty (can be dict or list for backward compatibility)
    if "models" not in config:
        logger.error("YAML must contain 'models' field")
        sys.exit(1)
    
    if isinstance(config["models"], dict):
        if not config["models"]:
            logger.error("YAML 'models' dict is empty")
            sys.exit(1)
    elif isinstance(config["models"], list):
        if not config["models"]:
            logger.error("YAML 'models' list is empty")
            sys.exit(1)
        logger.warning("Using deprecated list format for models. Consider using dict format with model names as keys.")
    else:
        logger.error("YAML 'models' must be either a dict or list")
        sys.exit(1)

    return config


def build_model_from_entry(entry: Dict[str, Any], model_name: str) -> ModelConfig:
    """
    Build a single ModelConfig instance from a YAML entry.
    Dynamically discovers the appropriate class and validates parameters.
    
    Args:
        entry: Dictionary containing model configuration from YAML
        model_name: Name/key of the model from YAML (used as the model's name)
        
    Returns:
        Instantiated ModelConfig subclass
    """
    mtype = entry.get("type")
    if not mtype:
        logger.error(f"Model '{model_name}' missing 'type' field")
        sys.exit(1)

    # Get available model types
    model_type_mapping = get_model_type_mapping()
    
    # Normalize the type name (handle different separators)
    mtype_normalized = mtype.lower().replace("_", "-")
    
    # Find matching model class
    config_class = None
    for type_name, cls in model_type_mapping.items():
        if type_name == mtype_normalized or mtype.lower() == type_name.replace("-", ""):
            config_class = cls
            break
    
    if config_class is None:
        available_types = ", ".join(sorted(model_type_mapping.keys()))
        logger.error(
            f"Model '{model_name}' has unrecognized type '{mtype}'. "
            f"Valid types are: {available_types}"
        )
        sys.exit(1)
    
    # Get constructor signature
    sig = inspect.signature(config_class.__init__)
    
    # Map YAML keys to constructor parameters
    # Handle common YAML naming conventions vs Python parameter names
    param_mapping = {
        # Common aliases for parameters
        "checkpoint": "checkpoint_path",
        "classes": "channels",
        "resolution": "input_voxel_size",
        "output_resolution": "output_voxel_size",
        "config_folder": "folder_path",
        "model_path": "model_name",
    }
    
    # Build kwargs from YAML entry
    kwargs = {}
    for yaml_key, yaml_value in entry.items():
        if yaml_key == "type":
            continue  # Skip the type field
        
        # Map YAML key to parameter name
        param_name = param_mapping.get(yaml_key, yaml_key)
        
        # Handle list/tuple conversions for resolution
        if param_name in ["input_voxel_size", "output_voxel_size"] and isinstance(yaml_value, int):
            yaml_value = (yaml_value, yaml_value, yaml_value)
        elif param_name in ["input_voxel_size", "output_voxel_size"] and isinstance(yaml_value, list):
            yaml_value = tuple(yaml_value)
        
        kwargs[param_name] = yaml_value
    
    # Use model_name as the name if not explicitly provided in YAML
    if 'name' not in kwargs:
        kwargs['name'] = model_name
    
    # Process constructor args (handles type conversions)
    processed_kwargs = process_constructor_args(config_class, kwargs)
    
    # Validate required parameters
    required_params = []
    for param_name, param_info in sig.parameters.items():
        if (param_name != 'self' and 
            param_info.default is inspect.Parameter.empty and
            param_name not in ['name', 'scale']):
            required_params.append(param_name)
            
            if param_name not in processed_kwargs:
                # Special case: if output_voxel_size is missing but input_voxel_size exists, use input_voxel_size
                if param_name == 'output_voxel_size' and 'input_voxel_size' in processed_kwargs:
                    processed_kwargs['output_voxel_size'] = processed_kwargs['input_voxel_size']
                    logger.warning(
                        f"Model '{model_name}' ({mtype}): 'output_voxel_size' not specified, "
                        f"using 'input_voxel_size' ({processed_kwargs['input_voxel_size']}) as default"
                    )
                    continue
                
                # Check if it exists under an alias
                found = False
                for yaml_key, mapped_param in param_mapping.items():
                    if mapped_param == param_name and yaml_key in entry:
                        found = True
                        break
                
                if not found:
                    logger.error(
                        f"Model '{model_name}' ({mtype}) missing required parameter '{param_name}'"
                    )
                    sys.exit(1)
    
    # Create model instance
    try:
        model = config_class(**processed_kwargs)
        logger.debug(f"Created model '{model_name}': {model}")
        return model
    except TypeError as e:
        logger.error(f"Error creating model '{model_name}' ({mtype}): {e}")
        logger.error(f"Provided parameters: {processed_kwargs}")
        logger.error(f"Required parameters: {required_params}")
        sys.exit(1)


def build_models(model_entries: Dict[str, Dict[str, Any]]) -> List[ModelConfig]:
    """
    Given model entries from YAML, instantiate the correct ModelConfig objects.
    Uses dynamic discovery like cli_v2 instead of hardcoded if/else chains.
    
    YAML format:
    models:
      my_model_1:
        type: cellmap-model
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
    
    for model_name, entry in model_entries.items():
        model = build_model_from_entry(entry, model_name=model_name)
        models.append(model)
    
    return models
