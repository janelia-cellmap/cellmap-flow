"""The cellmap models published on the Hugging Face Hub, for the dashboard's picker.

``list_huggingface_models`` answers from a cache file under
``~/.cellmap_flow`` once there is one, so opening the dashboard does not
query the Hub each time; ``refresh_huggingface_models`` rebuilds it. Each
model is keyed by its repo id, with the ``metadata.json`` it publishes.
"""

import json
import os
from typing import Any, Dict

from huggingface_hub import hf_hub_download, list_models

HUGGING_FACE_ORGS_NAME = "cellmap"
HF_CACHE_DIR = os.path.expanduser("~/.cellmap_flow/hugging_face")
HF_CACHE_FILE = os.path.join(HF_CACHE_DIR, "models_cache.json")


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
