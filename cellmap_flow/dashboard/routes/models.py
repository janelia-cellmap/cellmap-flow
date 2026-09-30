import logging
from datetime import datetime

from flask import Blueprint, request, jsonify, Response

from cellmap_flow.dashboard.services.launch import update_run_models
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.globals import SERVER_CONFIG_KEYS

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__)


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

        # Configured, for the pipeline builder and blockwise.
        get_session().models_config.append(model_config)

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


@models_bp.route("/api/job-logs")
def job_logs():
    """The inference jobs' own output, so a failure can be read here.

    Without this, a server that dies on startup or 500s on every chunk says
    nothing in the dashboard -- the traceback is in the LSF job's output on a
    cluster node, and reading it means logging in and running bpeek.
    """
    jobs = []
    for job in get_session().jobs or []:
        try:
            status = job.get_status()
            text = job.peek()
        except Exception as e:
            status, text = None, f"Could not read job output: {e}"
        jobs.append(
            {
                "model_name": getattr(job, "model_name", None),
                "job_id": getattr(job, "job_id", None),
                "host": getattr(job, "host", None),
                "status": getattr(status, "value", None),
                # None means "no way to read this one" (a local job), which is
                # different from "read it and it was empty".
                "log": text,
            }
        )
    return jsonify({"success": True, "jobs": jobs})


@models_bp.route("/api/gpu-queues")
def gpu_queues():
    """Which GPU queues are open and how busy, for the queue picker.

    Polled about once a minute by the models tab; the underlying LSF query is
    cached server-side, so this is cheap to call.
    """
    from cellmap_flow.utils.lsf_queues import gpu_queue_availability

    return jsonify(gpu_queue_availability())


@models_bp.route("/api/server-config")
def get_server_config():
    """Get current server configuration."""
    session = get_session()
    return jsonify({**session.server_config, "cached": session.server_config_cached})


@models_bp.route("/api/export-config")
def export_config():
    """
    Export the dashboard's current live config (models, normalization,
    postprocessing, queue/charge_group) as a downloadable YAML file that can
    be reloaded later with `cellmap_flow_yaml`.
    """
    from cellmap_flow.finetune.finetuned_model_templates import (
        generate_current_config_yaml,
    )

    try:
        session = get_session()
        models = [m.to_dict() for m in (session.models_config or [])]
        yaml_text = generate_current_config_yaml(
            models=models,
            data_path=session.dataset_path or "",
            queue=session.queue,
            charge_group=session.charge_group,
            walltime=session.walltime,
            json_data=session.pipeline_spec.to_json_data(),
        )
    except Exception as e:
        logger.error(f"Error exporting config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cellmap_flow_config_{timestamp}.yaml"
    return Response(
        yaml_text,
        mimetype="text/yaml",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@models_bp.route("/api/server-config", methods=["POST"])
def update_server_config():
    """Update server configuration and save to cache."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "expected a JSON object"}), 400
    int_fields = {"nb_cores_master", "nb_cores_worker", "nb_workers"}
    # Validate everything first: a bad number is the client's mistake (400),
    # and must not leave half the settings applied.
    updates = {}
    for key in SERVER_CONFIG_KEYS:
        if key in data:
            value = data[key]
            if key in int_fields:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    return jsonify({
                        "success": False,
                        "error": f"{key} must be a whole number, got {value!r}",
                    }), 400
            updates[key] = value
    session = get_session()
    for key, value in updates.items():
        setattr(session, key, value)
    session.save_server_config()
    logger.info(f"Server config updated and cached: {session.server_config}")
    return jsonify({"success": True, "config": session.server_config})
