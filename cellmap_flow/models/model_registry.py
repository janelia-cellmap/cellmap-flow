"""Registry and introspection tools for ModelConfig subclasses."""

import json
import os
from typing import Dict, Any
from huggingface_hub import list_models, hf_hub_download
from cellmap_flow.models import registry
from cellmap_flow.models.models_config import (
    ScriptModelConfig,
    DaCapoModelConfig,
    FlyModelConfig,
    BioModelConfig,
    CellMapModelConfig,
    HuggingFaceModelConfig,
    FinetuneModelConfig,
)


# Registry of available model config classes
MODEL_CONFIG_CLASSES = {
    'ScriptModelConfig': ScriptModelConfig,
    'DaCapoModelConfig': DaCapoModelConfig,
    'FlyModelConfig': FlyModelConfig,
    'BioModelConfig': BioModelConfig,
    'CellMapModelConfig': CellMapModelConfig,
    'HuggingFaceModelConfig': HuggingFaceModelConfig,
    'FinetuneModelConfig': FinetuneModelConfig,
}

HUGGING_FACE_ORGS_NAME = "cellmap"
HF_CACHE_DIR = os.path.expanduser("~/.cellmap_flow/hugging_face")
HF_CACHE_FILE = os.path.join(HF_CACHE_DIR, "models_cache.json")

def get_parameter_info(cls) -> Dict[str, Any]:
    """Each constructor argument of ``cls``: ``registry.parameter_info``."""
    return registry.parameter_info(cls)


def get_all_model_configs() -> Dict[str, Dict[str, Any]]:
    """What the dashboard's model form offers, by class name: ``registry.describe_types``."""
    return registry.describe_types(MODEL_CONFIG_CLASSES)


def instantiate_model_config(class_name: str, params: Dict[str, Any]) -> Any:
    """
    Instantiate a ModelConfig subclass with the provided parameters.
    
    Args:
        class_name: Name of the ModelConfig subclass (e.g., 'ScriptModelConfig')
        params: The model form's values; see ``registry.coerce_form_params``
        
    Returns:
        An instance of the ModelConfig subclass
        
    Raises:
        ValueError: If class_name is not recognized or params are invalid
    """
    if class_name not in MODEL_CONFIG_CLASSES:
        raise ValueError(f"Unknown model config class: {class_name}")
    
    cls = MODEL_CONFIG_CLASSES[class_name]
    parsed_params = registry.coerce_form_params(cls, params)
    
    try:
        return cls(**parsed_params)
    except Exception as e:
        raise ValueError(f"Failed to instantiate {class_name}: {str(e)}")


def _fetch_huggingface_models(org_name: str = HUGGING_FACE_ORGS_NAME) -> Dict[str, Any]:
    """Fetch models from Hugging Face Hub and save to cache."""
    result = {}
    try:
        models = list_models(author=org_name)
        for m in models:
            try:
                path = hf_hub_download(m.id, "metadata.json")
                with open(path) as f:
                    metadata = json.load(f)
                result[m.id] = metadata
            except Exception as e:
                print(f"{m.id}: Could not load metadata.json ({e})")
    except Exception as e:
        print(f"Error fetching Hugging Face models: {str(e)}")
        return {}

    # Save to cache
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    with open(HF_CACHE_FILE, "w") as f:
        json.dump(result, f)

    return result


def list_huggingface_models(org_name: str = HUGGING_FACE_ORGS_NAME) -> Dict[str, Any]:
    """
    List available Hugging Face models, using cache if available.

    Args:
        org_name: Hugging Face organization name to filter models (default: "cellmap")

    Returns:
        A dict mapping model IDs to their metadata
    """
    if os.path.exists(HF_CACHE_FILE):
        with open(HF_CACHE_FILE) as f:
            return json.load(f)
    return _fetch_huggingface_models(org_name)


def refresh_huggingface_models(org_name: str = HUGGING_FACE_ORGS_NAME) -> Dict[str, Any]:
    """Force refresh the Hugging Face models cache."""
    return _fetch_huggingface_models(org_name)