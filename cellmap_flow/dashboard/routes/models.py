import logging
from datetime import datetime

from flask import Blueprint, request, jsonify, Response

from cellmap_flow.dashboard.requests import CreateModelConfig, ServerConfigUpdate, SubmitModels, parse
from cellmap_flow.dashboard.services.launch import update_run_models
from cellmap_flow.dashboard.state import get_session

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__)


@models_bp.route("/api/model-config-types")
def get_model_config_types():
    """Get available ModelConfig subclasses and their parameter metadata"""
    from cellmap_flow.models.registry import describe_types

    try:
        config_types = describe_types()
        logger.info(f"Available model config types: {list(config_types.keys())}")
        return jsonify(config_types)
    except Exception as e:
        logger.error(f"Error getting model config types: {str(e)}")
        return jsonify({'error': str(e)}), 500


@models_bp.route("/api/create-model-config", methods=["POST"])
def create_model_config():
    """Create a ModelConfig instance from user-provided parameters"""
    from cellmap_flow.models.registry import instantiate_model_config

    body, error = parse(CreateModelConfig, request.get_json(silent=True))
    if error:
        return error
    class_name = body.class_name
    try:
        # The form's values are strings; the registry parses them.
        model_config = instantiate_model_config(class_name, body.params)

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
        return jsonify({'success': False, 'error': str(e)}), 400


@models_bp.route("/api/huggingface-models")
def get_huggingface_models():
    """Get available models from Hugging Face (uses cache if available)"""
    from cellmap_flow.models.hf_catalog import list_huggingface_models

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
    from cellmap_flow.models.hf_catalog import refresh_huggingface_models

    try:
        hf_models = refresh_huggingface_models()
        logger.info(f"Refreshed Hugging Face models: {list(hf_models.keys())}")
        return jsonify(hf_models)
    except Exception as e:
        logger.error(f"Error refreshing Hugging Face models: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _bioimage_answer(read):
    """``read()``'s zoo list as JSON, or its failure as {"error": ...}: a 502
    when the zoo's index could not be fetched, which the tab shows."""
    from cellmap_flow.models.bioimage_catalog import ZooIndexError

    try:
        return jsonify(read())
    except ZooIndexError as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        logger.error(f"Error listing BioImage Model Zoo models: {e}")
        return jsonify({"error": str(e)}), 500


@models_bp.route("/api/bioimage-models")
def get_bioimage_models():
    """The BioImage Model Zoo's models (cached): bioimage_catalog.list_bioimage_models."""
    from cellmap_flow.models.bioimage_catalog import list_bioimage_models

    return _bioimage_answer(list_bioimage_models)


@models_bp.route("/api/bioimage-models/refresh", methods=["POST"])
def refresh_bioimage_models_route():
    """Fetch the zoo's index again."""
    from cellmap_flow.models.bioimage_catalog import refresh_bioimage_models

    return _bioimage_answer(refresh_bioimage_models)


@models_bp.route("/api/models", methods=["POST"])
def submit_models():
    """Run the models the Models tab has ticked (a SubmitModels), and stop
    the others: services.launch.update_run_models. A zoo model that needs a
    voxel size and was given none, or a Cellpose model without one, is a
    400, and nothing changes."""
    body, error = parse(SubmitModels, request.get_json(silent=True))
    if error:
        return error
    selected_models, selected_hf_models = body.selected_models, body.selected_hf_models
    selected_bioimage = [s.model_dump() for s in body.selected_bioimage_models]
    selected_cellpose = [s.model_dump() for s in body.selected_cellpose_models]
    if body.resample is not None:
        get_session().resample = body.resample
    try:
        update_run_models(selected_models, selected_hf_models, selected_bioimage, selected_cellpose)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    logger.info(f"Selected models: {selected_models}, HF models: {selected_hf_models}, "
                f"bioimage models: {selected_bioimage}, cellpose models: {selected_cellpose}")
    return jsonify(
        {
            "message": "Data received successfully",
            "models": selected_models,
            "hf_models": selected_hf_models,
            "bioimage_models": selected_bioimage,
            "cellpose_models": selected_cellpose,
        }
    )


@models_bp.route("/api/job-logs")
def job_logs():
    """The inference jobs' own output, so a failure can be read here.

    Without this, a server that dies on startup or 500s on every chunk says
    nothing in the dashboard -- the traceback is in the LSF job's output on a
    cluster node, and reading it means logging in and running bpeek.
    """
    from cellmap_flow.jobs.launch import failed_starts, starting_jobs

    jobs = []
    # The starting ones too: from Submit until a server answers, which can be
    # minutes in a queue or installing an environment, the page said no job
    # had been submitted. And the ones that failed to start: their log is the
    # one being read, and it vanished from the page when the job died.
    starting = starting_jobs()
    failed = failed_starts()
    for job in list(get_session().jobs or []) + starting + failed:
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
                "status": ("starting" if job in starting
                           else "failed to start" if job in failed
                           else getattr(status, "value", None)),
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
    from cellmap_flow.jobs.queues import gpu_queue_availability

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
    be reloaded later with `cellmap_flow yaml`.
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
            resample=session.resample,
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
    # Validated whole first: a bad number is the client's mistake (400), and
    # must not leave half the settings applied.
    body, error = parse(ServerConfigUpdate, request.get_json(silent=True))
    if error:
        return error
    session = get_session()
    for key, value in body.updates().items():
        setattr(session, key, value)
    session.save_server_config()
    logger.info(f"Server config updated and cached: {session.server_config}")
    return jsonify({"success": True, "config": session.server_config})
