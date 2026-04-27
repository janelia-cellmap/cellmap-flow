import logging

from flask import Blueprint, request, jsonify

from cellmap_flow.globals import g, SERVER_CONFIG_KEYS
from cellmap_flow.models.run import update_run_models

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__)


@models_bp.route("/api/available-models")
def get_available_models():
    """Get available models from the model catalog"""
    models = {}

    # Build models from catalog
    if hasattr(g, 'model_catalog') and g.model_catalog:
        for category, category_models in g.model_catalog.items():
            if isinstance(category_models, dict):
                for model_name, model_path in category_models.items():
                    full_name = f"{category}/{model_name}"
                    models[full_name] = {
                        'name': full_name,
                        'category': category,
                        'model_name': model_name,
                        'path': model_path
                    }

    logger.info(f"Available models: {list(models.keys())}")
    return jsonify(models)


@models_bp.route("/api/model-config-types")
def get_model_config_types():
    """Get available ModelConfig subclasses and their parameter metadata"""
    from cellmap_flow.models.model_registry import get_all_model_configs

    try:
        config_types = get_all_model_configs()
        logger.info(f"Available model config types: {list(config_types.keys())}")
        return jsonify(config_types)
    except Exception as e:
        logger.error(f"Error getting model config types: {str(e)}")
        return jsonify({'error': str(e)}), 500


@models_bp.route("/api/create-model-config", methods=["POST"])
def create_model_config():
    """Create a ModelConfig instance from user-provided parameters"""
    from cellmap_flow.models.model_registry import instantiate_model_config

    try:
        data = request.get_json()
        class_name = data.get('class_name')
        params = data.get('params', {})

        if not class_name:
            return jsonify({'error': 'class_name is required'}), 400

        # Instantiate the model config
        model_config = instantiate_model_config(class_name, params)

        # Store it in g.models_config for use in pipeline
        if not hasattr(g, 'models_config'):
            g.models_config = []
        g.models_config.append(model_config)

        logger.info(f"Created {class_name}: {model_config.name}")
        return jsonify({
            'success': True,
            'message': f'Created {class_name}',
            'model_name': model_config.name,
            'config_dict': model_config.to_dict()
        })
    except Exception as e:
        logger.error(f"Error creating model config: {str(e)}")
        return jsonify({'error': str(e)}), 400


@models_bp.route("/api/huggingface-models")
def get_huggingface_models():
    """Get available models from Hugging Face (uses cache if available)"""
    from cellmap_flow.models.model_registry import list_huggingface_models

    try:
        hf_models = list_huggingface_models()
        logger.info(f"Hugging Face models: {list(hf_models.keys())}")
        return jsonify(hf_models)
    except Exception as e:
        logger.error(f"Error fetching Hugging Face models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@models_bp.route("/api/huggingface-models/refresh", methods=["POST"])
def refresh_huggingface_models_route():
    """Force refresh the Hugging Face models cache"""
    from cellmap_flow.models.model_registry import refresh_huggingface_models

    try:
        hf_models = refresh_huggingface_models()
        logger.info(f"Refreshed Hugging Face models: {list(hf_models.keys())}")
        return jsonify(hf_models)
    except Exception as e:
        logger.error(f"Error refreshing Hugging Face models: {str(e)}")
        return jsonify({'error': str(e)}), 500


@models_bp.route("/api/models", methods=["POST"])
def submit_models():
    data = request.get_json()
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    selected_models = data.get("selected_models", [])
    selected_hf_models = data.get("selected_hf_models", [])
    update_run_models(selected_models, selected_hf_models)
    logger.warning(f"Selected models: {selected_models}, HF models: {selected_hf_models}")
    return jsonify(
        {
            "message": "Data received successfully",
            "models": selected_models,
            "hf_models": selected_hf_models,
        }
    )


@models_bp.route("/api/server-config")
def get_server_config():
    """Get current server configuration."""
    config = {k: getattr(g, k) for k in SERVER_CONFIG_KEYS}
    config["cached"] = g._server_config_cached
    return jsonify(config)


@models_bp.route("/api/server-config", methods=["POST"])
def update_server_config():
    """Update server configuration and save to cache."""
    data = request.get_json()
    int_fields = {"nb_cores_master", "nb_cores_worker", "nb_workers"}
    for key in SERVER_CONFIG_KEYS:
        if key in data:
            value = int(data[key]) if key in int_fields else data[key]
            setattr(g, key, value)
    g.save_server_config()
    logger.info(f"Server config updated and cached: { {k: getattr(g, k) for k in SERVER_CONFIG_KEYS} }")
    return jsonify({"success": True, "config": {k: getattr(g, k) for k in SERVER_CONFIG_KEYS}})
