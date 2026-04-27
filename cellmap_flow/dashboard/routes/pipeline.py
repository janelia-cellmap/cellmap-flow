import os
import logging
import time
from datetime import datetime

import neuroglancer
import numpy as np
from flask import Blueprint, request, jsonify

from cellmap_flow.globals import g
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
from cellmap_flow.utils.load_py import load_safe_config
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.web_utils import encode_to_str, ARGS_KEY

logger = logging.getLogger(__name__)

pipeline_bp = Blueprint("pipeline", __name__)


def _save_shaders_from_viewer() -> None:
    """Read current shader and shaderControls from the neuroglancer viewer and persist them in globals."""
    if g.viewer is None:
        return
    try:
        state = g.viewer.state
        for layer in state.layers:
            shader = getattr(layer, "shader", None)
            if shader:
                g.shaders[layer.name] = shader
            shader_controls = getattr(layer, "shaderControls", None) or getattr(layer, "shader_controls", None)
            if shader_controls:
                g.shader_controls[layer.name] = shader_controls
    except Exception as exc:
        logger.warning(f"Could not save shaders from viewer: {exc}")


def is_output_segmentation():
    if len(g.postprocess) == 0:
        return False

    for postprocess in g.postprocess[::-1]:
        if postprocess.is_segmentation is not None:
            return postprocess.is_segmentation


def validate_pipeline_config(config):
    """Helper function to validate pipeline configuration"""
    try:
        normalizer_names = [n.get("name") for n in config.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        # Extract just the normalizer names from the list of dicts
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return {"valid": False, "error": f"Unknown normalizer: {norm_name}"}

        processor_names = [p.get("name") for p in config.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        # Extract just the postprocessor names from the list of dicts
        available_proc_names = [proc["name"] for proc in available_procs]
        for proc_name in processor_names:
            if proc_name not in available_proc_names:
                return {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}

        return {"valid": True}

    except Exception as e:
        return {"valid": False, "error": str(e)}


@pipeline_bp.route("/update/equivalences", methods=["POST"])
def update_equivalences():
    equivalences_info = request.get_json()
    dataset = equivalences_info["dataset"]
    equivalences_str = equivalences_info["equivalences"]
    equivalences = [
        [np.uint64(item) for item in sublist] for sublist in equivalences_str
    ]

    with g.viewer.txn() as s:
        for layer in s.layers:
            if layer.source[0].url.endswith(dataset):
                layer.equivalences = equivalences
                break
    return jsonify({"message": "Equivalences updated successfully"})


@pipeline_bp.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()

    # add dashboard url to data so we can update the state from the server
    data["dashboard_url"] = request.host_url

    # we want to set the time such that each request is unique
    data["time"] = time.time()

    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    custom_code = data.get("custom_code", None)
    if "custom_code" in data:
        del data["custom_code"]
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    g.input_norms = get_normalizations(data["input_norm"])
    g.postprocess = get_postprocessors(data["postprocess"])

    # Save current shader state from viewer before refreshing layers
    _save_shaders_from_viewer()

    with g.viewer.txn() as s:
        g.raw = get_raw_layer(g.dataset_path)
        s.layers["data"] = g.raw
        for job in g.jobs:
            model = job.model_name
            host = job.host
            st_data = encode_to_str(data)
            shader = g.shaders.get(model)

            if is_output_segmentation():
                s.layers[model] = neuroglancer.SegmentationLayer(
                    source=f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )
            else:
                kwargs = {"source": f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}"}
                if shader:
                    kwargs["shader"] = shader
                shader_controls = g.shader_controls.get(model)
                if shader_controls:
                    kwargs["shaderControls"] = shader_controls
                s.layers[model] = neuroglancer.ImageLayer(**kwargs)

    logger.warning(f"Input normalizers: {g.input_norms}")

    if custom_code:
        try:
            # Save custom code to a file with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_code_{timestamp}.py"
            filepath = os.path.join(g.CUSTOM_CODE_FOLDER, filename)

            with open(filepath, "w") as file:
                file.write(custom_code)

            config = load_safe_config(filepath)
            logger.warning(f"Custom code loaded successfully: {config}")

            logger.warning(get_input_normalizers())

        except Exception as e:
            logger.warning(f"Error executing custom code: {e}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
            "found_custom_normalizer": get_input_normalizers(),
        }
    )


@pipeline_bp.route("/api/pipeline/validate", methods=["POST"])
def validate_pipeline():
    """Validate a pipeline configuration"""
    try:
        data = request.get_json()

        # Validate normalizers
        normalizer_names = [n.get("name") for n in data.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return jsonify(
                    {"valid": False, "error": f"Unknown normalizer: {norm_name}"}
                ), 400

        # Validate postprocessors
        processor_names = [p.get("name") for p in data.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        available_proc_names = [proc["name"] for proc in available_procs]
        for proc_name in processor_names:
            if proc_name not in available_proc_names:
                return jsonify(
                    {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}
                ), 400

        return jsonify({"valid": True, "message": "Pipeline is valid"})

    except Exception as e:
        logger.error(f"Error validating pipeline: {e}")
        return jsonify({"valid": False, "error": str(e)}), 500


@pipeline_bp.route("/api/dataset-path", methods=["GET", "POST"])
def dataset_path_api():
    """Get or set the dataset path in globals"""
    if request.method == "GET":
        dataset_path = getattr(g, 'dataset_path', None) or ''
        return jsonify({'dataset_path': dataset_path})
    elif request.method == "POST":
        data = request.get_json()
        dataset_path = data.get('dataset_path', '')
        g.dataset_path = dataset_path
        logger.warning(f"Dataset path updated to: {dataset_path}")
        return jsonify({'success': True, 'dataset_path': g.dataset_path})


@pipeline_bp.route("/api/blockwise-config", methods=["GET", "POST"])
def blockwise_config_api():
    """Get or set blockwise configuration in globals"""
    if request.method == "GET":
        return jsonify({
            'queue': g.queue,
            'charge_group': g.charge_group,
            'nb_cores_master': g.nb_cores_master,
            'nb_cores_worker': g.nb_cores_worker,
            'nb_workers': g.nb_workers,
            'tmp_dir': g.tmp_dir,
            'blockwise_tasks_dir': g.blockwise_tasks_dir
        })
    elif request.method == "POST":
        data = request.get_json()
        g.queue = data.get('queue')
        g.charge_group = data.get('charge_group')
        g.nb_cores_master = int(data.get('nb_cores_master'))
        g.nb_cores_worker = int(data.get('nb_cores_worker'))
        g.nb_workers = int(data.get('nb_workers'))
        g.tmp_dir = data.get('tmp_dir')
        g.blockwise_tasks_dir = data.get('blockwise_tasks_dir')
        logger.warning(f"Blockwise config updated: queue={g.queue}, charge_group={g.charge_group}, cores_master={g.nb_cores_master}, cores_worker={g.nb_cores_worker}, workers={g.nb_workers}, tmp_dir={g.tmp_dir}, blockwise_tasks_dir={g.blockwise_tasks_dir}")
        return jsonify({'success': True, 'config': {
            'queue': g.queue,
            'charge_group': g.charge_group,
            'nb_cores_master': g.nb_cores_master,
            'nb_cores_worker': g.nb_cores_worker,
            'nb_workers': g.nb_workers,
            'tmp_dir': g.tmp_dir,
            'blockwise_tasks_dir': g.blockwise_tasks_dir
        }})


@pipeline_bp.route("/api/pipeline/apply", methods=["POST"])
def apply_pipeline():
    """Apply a pipeline configuration to the current inference"""
    try:
        data = request.get_json()
        logger.warning(f"\n{'='*80}")
        logger.warning(f"APPLY PIPELINE - Received data:")
        logger.warning(f"  Input normalizers: {data.get('input_normalizers', [])}")
        logger.warning(f"  Postprocessors: {data.get('postprocessors', [])}")

        # Validate first
        validation = validate_pipeline_config(data)
        if not validation["valid"]:
            return jsonify(validation), 400

        # Apply normalizers
        input_norms_config = {
            n["name"]: n.get("params", {}) for n in data.get("input_normalizers", [])
        }
        logger.warning(f"\nNormalizers config dict: {input_norms_config}")
        g.input_norms = get_normalizations(input_norms_config)

        # Apply postprocessors
        postprocs_config = {
            p["name"]: p.get("params", {}) for p in data.get("postprocessors", [])
        }
        logger.warning(f"Postprocessors config dict: {postprocs_config}")
        g.postprocess = get_postprocessors(postprocs_config)

        # Save complete pipeline visual state to globals
        g.pipeline_inputs = data.get("inputs", [])
        g.pipeline_outputs = data.get("outputs", [])
        g.pipeline_edges = data.get("edges", [])
        g.pipeline_normalizers = data.get("input_normalizers", [])
        g.pipeline_models = data.get("models", [])
        g.pipeline_postprocessors = data.get("postprocessors", [])

        # Also save model configs separately for easier access
        if not hasattr(g, 'pipeline_model_configs'):
            g.pipeline_model_configs = {}
        for model in data.get("models", []):
            if 'config' in model and model['config']:
                g.pipeline_model_configs[model['name']] = model['config']

        # Log the updated globals state
        logger.warning(f"\n{'='*80}")
        logger.warning(f"UPDATED GLOBALS (g) STATE:")
        logger.warning(f"{'='*80}")
        logger.warning(f"\ng.input_norms ({len(g.input_norms)} items):")
        for idx, norm in enumerate(g.input_norms):
            logger.warning(f"  [{idx}] {norm}")

        logger.warning(f"\ng.postprocess ({len(g.postprocess)} items):")
        for idx, post in enumerate(g.postprocess):
            logger.warning(f"  [{idx}] {post}")

        logger.warning(f"\ng.jobs ({len(g.jobs)} items):")
        for idx, job in enumerate(g.jobs):
            logger.warning(f"  [{idx}] model_name={getattr(job, 'model_name', 'N/A')}, host={getattr(job, 'host', 'N/A')}")

        logger.warning(f"\ng.pipeline_inputs ({len(g.pipeline_inputs)} items): {g.pipeline_inputs}")
        logger.warning(f"\ng.pipeline_outputs ({len(g.pipeline_outputs)} items): {g.pipeline_outputs}")
        logger.warning(f"\ng.pipeline_edges ({len(g.pipeline_edges)} items): {g.pipeline_edges}")
        logger.warning(f"\ng.pipeline_normalizers ({len(g.pipeline_normalizers)} items): {g.pipeline_normalizers}")
        logger.warning(f"\ng.pipeline_models ({len(g.pipeline_models)} items): {g.pipeline_models}")
        logger.warning(f"\ng.pipeline_postprocessors ({len(g.pipeline_postprocessors)} items): {g.pipeline_postprocessors}")

        logger.warning(f"{'='*80}\n")

        return jsonify({
            "message": "Pipeline applied successfully",
            "normalizers_applied": len(g.input_norms),
            "postprocessors_applied": len(g.postprocess),
        })

    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return jsonify({"error": str(e)}), 500


@pipeline_bp.route("/api/shaders", methods=["GET", "POST"])
def shaders_api():
    """Get or update stored shader strings and shaderControls.

    GET  -> returns current g.shaders and g.shader_controls
    POST -> merges incoming {"shaders": {...}, "shader_controls": {...}} into globals
           (also accepts flat {layer_name: shader_str} for backwards compat)
    """
    if request.method == "GET":
        # Also sync from viewer if available
        _save_shaders_from_viewer()
        return jsonify({"shaders": g.shaders, "shader_controls": g.shader_controls})

    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object"}), 400

    # Support both structured and flat formats
    if "shaders" in data or "shader_controls" in data:
        if "shaders" in data:
            g.shaders.update(data["shaders"])
        if "shader_controls" in data:
            g.shader_controls.update(data["shader_controls"])
    else:
        # Flat dict — treat as shaders only (backwards compat)
        g.shaders.update(data)

    logger.info(f"Shaders updated for layers: {list(g.shaders.keys())}")
    logger.info(f"ShaderControls updated for layers: {list(g.shader_controls.keys())}")
    return jsonify({"message": "Shaders updated", "shaders": g.shaders, "shader_controls": g.shader_controls})
