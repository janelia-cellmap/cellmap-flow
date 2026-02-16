import logging

from flask import Blueprint, request, jsonify

from cellmap_flow.globals import g
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


@models_bp.route("/api/models", methods=["POST"])
def submit_models():
    data = request.get_json()
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    selected_models = data.get("selected_models", [])
    update_run_models(selected_models)
    logger.warning(f"Selected models: {selected_models}")
    return jsonify(
        {
            "message": "Data received successfully",
            "models": selected_models,
        }
    )
