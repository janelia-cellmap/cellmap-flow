import json
import os
import socket
import neuroglancer
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import subprocess
import yaml
import tempfile
import re
from cellmap_flow.utils.web_utils import get_free_port
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
from cellmap_flow.models.model_merger import get_model_mergers_list
from cellmap_flow.utils.load_py import load_safe_config
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.web_utils import (
    encode_to_str,
    decode_to_json,
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
)
from cellmap_flow.models.run import update_run_models
from cellmap_flow.globals import g
import numpy as np
import time

logger = logging.getLogger(__name__)
# Explicitly set template and static folder paths for package installation
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)
NEUROGLANCER_URL = None
INFERENCE_SERVER = None

# Blockwise task directory will be set from globals or use default
def get_blockwise_tasks_dir():
    tasks_dir = getattr(g, 'blockwise_tasks_dir', None) or os.path.expanduser("~/.cellmap_flow/blockwise_tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir
CUSTOM_CODE_FOLDER = os.path.expanduser(
    os.environ.get(
        "CUSTOM_CODE_FOLDER",
        "~/Desktop/cellmap/cellmap-flow/example/example_norm",
    )
)


@app.route("/")
def index():
    # Render the main page with tabs
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()
    model_mergers = get_model_mergers_list()
    model_catalog = g.model_catalog
    model_catalog["User"] = {j.model_name: "" for j in g.jobs}
    default_post_process = {d.to_dict()["name"]: d.to_dict() for d in g.postprocess}
    default_input_norm = {d.to_dict()["name"]: d.to_dict() for d in g.input_norms}
    logger.warning(f"Model catalog: {model_catalog}")
    logger.warning(f"Default postprocess: {default_post_process}")
    logger.warning(f"Default input norm: {default_input_norm}")

    return render_template(
        "index.html",
        neuroglancer_url=NEUROGLANCER_URL,
        inference_servers=INFERENCE_SERVER,
        input_normalizers=input_norms,
        output_postprocessors=output_postprocessors,
        model_mergers=model_mergers,
        default_post_process=default_post_process,
        default_input_norm=default_input_norm,
        model_catalog=model_catalog,
        default_models=[j.model_name for j in g.jobs],
    )


@app.route("/pipeline-builder")
def pipeline_builder():
    """Render the drag-and-drop pipeline builder interface with current state from globals"""
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()

    # Get available models from models_config (not from catalog)
    available_models = {}
    # if hasattr(g, 'models_config') and g.models_config:
    #     for model_config in g.models_config:
    #         model_dict = model_config.to_dict()
    #         available_models[model_config.name] = model_dict
    
    logger.warning(f"\n{'='*80}")
    logger.warning(f"AVAILABLE MODELS DEBUG:")
    logger.warning(f"  Initial available_models keys: {list(available_models.keys())}")
    logger.warning(f"  g.models_config: {g.models_config if hasattr(g, 'models_config') else 'NOT SET'}")
    logger.warning(f"  Sample model with config:")
    for model_name, model_data in list(available_models.items())[:1]:
        logger.warning(f"    {model_name}: {model_data}")
    models_with_config = {}
    for model_name in available_models.keys():
        # Find matching config (strip _server suffix for matching)
        model_name_stripped = model_name.replace('_server', '')
        for model_config in g.models_config:
            config_name = getattr(model_config, 'name', '').replace('_server', '')
            if config_name == model_name_stripped:
                if hasattr(model_config, 'to_dict'):
                    models_with_config[model_name] = {
                        'name': model_name,
                        'config': model_config.to_dict()
                    }
                break
        # If no config found, just use the name
        if model_name not in models_with_config:
            models_with_config[model_name] = {'name': model_name}
    available_models = models_with_config

    # Check if we have stored pipeline state from previous apply
    if hasattr(g, 'pipeline_normalizers') and len(g.pipeline_normalizers) > 0:
        # Use stored pipeline state (includes IDs, positions, params)
        current_normalizers = g.pipeline_normalizers
        current_postprocessors = g.pipeline_postprocessors
        current_models = g.pipeline_models
        # Enrich current_models with config from g.models_config if available
        if hasattr(g, 'models_config') and g.models_config:
            for model_dict in current_models:
                if 'config' not in model_dict:
                    # Strip _server suffix for matching
                    model_name = model_dict['name'].replace('_server', '')
                    for model_config in g.models_config:
                        config_name = getattr(model_config, 'name', '').replace('_server', '')
                        if config_name == model_name:
                            if hasattr(model_config, 'to_dict'):
                                model_dict['config'] = model_config.to_dict()
                            break
        current_inputs = g.pipeline_inputs
        current_outputs = g.pipeline_outputs
        current_edges = g.pipeline_edges
    else:
        # Fall back to converting from globals.input_norms and globals.postprocess
        current_normalizers = []
        for idx, norm in enumerate(g.input_norms):
            norm_dict = norm.to_dict() if hasattr(norm, 'to_dict') else {'name': str(norm)}
            norm_name = norm_dict.get('name', str(norm))
            # Extract params: all dict items except 'name'
            params = {k: v for k, v in norm_dict.items() if k != 'name'}
            current_normalizers.append({
                'id': f'norm-{idx}-{int(time.time()*1000)}',
                'name': norm_name,
                'params': params
            })

        # Current models (from jobs and models_config)
        current_models = []
        logger.warning(f"\n{'='*80}")
        logger.warning(f"Building current_models from g.jobs:")
        logger.warning(f"  g.jobs count: {len(g.jobs)}")
        logger.warning(f"  g.models_config exists: {hasattr(g, 'models_config')}")
        if hasattr(g, 'models_config'):
            logger.warning(f"  g.models_config count: {len(g.models_config) if g.models_config else 0}")
            logger.warning(f"  g.models_config type: {type(g.models_config)}")
            logger.warning(f"  g.models_config value: {g.models_config}")
            if g.models_config:
                logger.warning(f"  g.models_config names: {[getattr(mc, 'name', 'NO_NAME') for mc in g.models_config]}")
                for mc in g.models_config:
                    logger.warning(f"    Config object: {mc}, has to_dict: {hasattr(mc, 'to_dict')}")
        
        # If models_config is empty but we have jobs, try to get configs from model_catalog
        if (not hasattr(g, 'models_config') or not g.models_config) and hasattr(g, 'model_catalog'):
            logger.warning(f"  models_config is empty, checking model_catalog for configs...")
            # Check if available_models dict has configs
            if available_models:
                logger.warning(f"  available_models has {len(available_models)} entries with potential configs")

        for idx, job in enumerate(g.jobs):
            if hasattr(job, 'model_name'):
                logger.warning(f"\n  Processing job {idx}: model_name={job.model_name}")
                model_dict = {'id': f'model-{idx}-{int(time.time()*1000)}', 'name': job.model_name, 'params': {}}
                # Try to find the corresponding ModelConfig to get full configuration
                config_found = False
                
                # First try g.models_config
                if hasattr(g, 'models_config') and g.models_config:
                    # Strip _server suffix for matching
                    job_model_name = job.model_name.replace('_server', '')
                    for model_config in g.models_config:
                        model_config_name = getattr(model_config, 'name', None)
                        config_name_stripped = model_config_name.replace('_server', '') if model_config_name else None
                        logger.warning(f"    Checking model_config: {model_config_name} (stripped: {config_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})")
                        if config_name_stripped and config_name_stripped == job_model_name:
                            # Export the full model config using to_dict()
                            if hasattr(model_config, 'to_dict'):
                                model_dict['config'] = model_config.to_dict()
                                logger.warning(f"    ✓ Config attached from models_config: {model_dict['config']}")
                                config_found = True
                            break
                
                # Fallback: check available_models dict (which was enriched earlier)
                if not config_found and available_models:
                    job_model_name = job.model_name.replace('_server', '')
                    for model_name, model_data in available_models.items():
                        model_name_stripped = model_name.replace('_server', '')
                        logger.warning(f"    Checking available_models: {model_name} (stripped: {model_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})")
                        if model_name_stripped == job_model_name and isinstance(model_data, dict) and 'config' in model_data:
                            model_dict['config'] = model_data['config']
                            logger.warning(f"    ✓ Config attached from available_models: {model_dict['config']}")
                            config_found = True
                            break
                
                # Second fallback: check previously saved pipeline_model_configs
                if not config_found and hasattr(g, 'pipeline_model_configs'):
                    job_model_name = job.model_name.replace('_server', '')
                    for saved_name, saved_config in g.pipeline_model_configs.items():
                        saved_name_stripped = saved_name.replace('_server', '')
                        logger.warning(f"    Checking pipeline_model_configs: {saved_name} (stripped: {saved_name_stripped}) vs job: {job.model_name} (stripped: {job_model_name})")
                        if saved_name_stripped == job_model_name:
                            model_dict['config'] = saved_config
                            logger.warning(f"    ✓ Config attached from pipeline_model_configs: {model_dict['config']}")
                            config_found = True
                            break
                
                if not config_found:
                    logger.warning(f"    ✗ No matching config found for {job.model_name}")
                    logger.warning(f"       TIP: Import a YAML with full model configs to populate g.pipeline_model_configs")
                current_models.append(model_dict)
        logger.warning(f"{'='*80}\n")

        current_postprocessors = []
        for idx, post in enumerate(g.postprocess):
            post_dict = post.to_dict() if hasattr(post, 'to_dict') else {'name': str(post)}
            post_name = post_dict.get('name', str(post))
            # Extract params: all dict items except 'name'
            params = {k: v for k, v in post_dict.items() if k != 'name'}
            current_postprocessors.append({
                'id': f'post-{idx}-{int(time.time()*1000)}',
                'name': post_name,
                'params': params
            })

        current_inputs = []
        current_outputs = []
        current_edges = []

    # Get current dataset_path from globals
    dataset_path = getattr(g, 'dataset_path', None) or ''
    
    # Get available model mergers
    model_mergers = get_model_mergers_list()

    return render_template(
        "pipeline_builder_v2.html",
        input_normalizers=input_norms or {},
        available_models=available_models or {},
        output_postprocessors=output_postprocessors or {},
        model_mergers=model_mergers or {},
        current_normalizers=current_normalizers,
        current_models=current_models,
        current_postprocessors=current_postprocessors,
        current_inputs=current_inputs,
        current_outputs=current_outputs,
        current_edges=current_edges,
        dataset_path=dataset_path,
    )


def is_output_segmentation():
    if len(g.postprocess) == 0:
        return False

    for postprocess in g.postprocess[::-1]:
        if postprocess.is_segmentation is not None:
            return postprocess.is_segmentation


@app.route("/update/equivalences", methods=["POST"])
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


@app.route("/api/models", methods=["POST"])
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


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()

    # add dashboard url to data so we can update the state from the server
    data["dashboard_url"] = request.host_url

    # we wanmt to set the time such that each request is unique
    data["time"] = time.time()

    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    custom_code = data.get("custom_code", None)
    if "custom_code" in data:
        del data["custom_code"]
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    g.input_norms = get_normalizations(data["input_norm"])
    g.postprocess = get_postprocessors(data["postprocess"])

    with g.viewer.txn() as s:
        # g.raw.invalidate()
        g.raw = get_raw_layer(g.dataset_path)
        s.layers["data"] = g.raw
        for job in g.jobs:
            model = job.model_name
            host = job.host
            # response = requests.post(f"{host}/input_normalize", json=data)
            # print(f"Response from {host}: {response.json()}")
            st_data = encode_to_str(data)

            if is_output_segmentation():
                s.layers[model] = neuroglancer.SegmentationLayer(
                    source=f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )
            else:
                s.layers[model] = neuroglancer.ImageLayer(
                    source=f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )

    logger.warning(f"Input normalizers: {g.input_norms}")

    if custom_code:

        try:
            # Save custom code to a file with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_code_{timestamp}.py"
            filepath = os.path.join(CUSTOM_CODE_FOLDER, filename)

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


@app.route("/api/pipeline/validate", methods=["POST"])
def validate_pipeline():
    """Validate a pipeline configuration"""
    try:
        data = request.get_json()

        # Validate normalizers
        normalizer_names = [n.get("name") for n in data.get("input_normalizers", [])]
        available_norms = get_input_normalizers()
        # Extract just the normalizer names from the list of dicts
        available_norm_names = [norm["name"] for norm in available_norms]
        for norm_name in normalizer_names:
            if norm_name not in available_norm_names:
                return jsonify(
                    {"valid": False, "error": f"Unknown normalizer: {norm_name}"}
                ), 400

        # Validate postprocessors
        processor_names = [p.get("name") for p in data.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        # Extract just the postprocessor names from the list of dicts
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


@app.route("/api/dataset-path", methods=["GET", "POST"])
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


@app.route("/api/blockwise-config", methods=["GET", "POST"])
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


@app.route("/api/pipeline/apply", methods=["POST"])
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


@app.route("/api/blockwise/validate", methods=["POST"])
def validate_blockwise():
    """Validate if pipeline is ready for blockwise processing"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})
        
        # Check required components
        if not pipeline.get("inputs") or len(pipeline["inputs"]) == 0:
            return {"valid": False, "error": "No input nodes defined"}
        
        if not pipeline.get("outputs") or len(pipeline["outputs"]) == 0:
            return {"valid": False, "error": "No output nodes defined"}
        
        if not pipeline.get("models") or len(pipeline["models"]) == 0:
            return {"valid": False, "error": "No models defined"}
        
        # Check blockwise config
        if not pipeline.get("blockwise_config") or len(pipeline["blockwise_config"]) == 0:
            return {"valid": False, "error": "No blockwise configuration defined"}
        
        # Check input has dataset_path
        input_node = pipeline["inputs"][0]
        if not input_node.get("params", {}).get("dataset_path"):
            return {"valid": False, "error": "Input node missing dataset_path"}
        
        # Check output has dataset_path
        output_node = pipeline["outputs"][0]
        if not output_node.get("params", {}).get("dataset_path"):
            return {"valid": False, "error": "Output node missing dataset_path"}
        
        logger.info("Pipeline validation passed")
        return {"valid": True, "message": "Pipeline is ready for blockwise processing"}
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"valid": False, "error": str(e)}


@app.route("/api/blockwise/generate", methods=["POST"])
def generate_blockwise_task():
    """Generate blockwise task YAML files"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})
        
        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}
        
        # Get blockwise config
        blockwise_config = pipeline["blockwise_config"][0]
        input_node = pipeline["inputs"][0]
        output_node = pipeline["outputs"][0]
        
        # Create task YAML content
        task_name = f"cellmap_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_yaml = {
            "data_path": input_node["params"]["dataset_path"],
            "output_path": output_node["params"]["dataset_path"],
            "task_name": task_name,
            "charge_group": blockwise_config["params"]["charge_group"],
            "queue": blockwise_config["params"]["queue"],
            "workers": blockwise_config["params"]["nb_workers"],
            "cpu_workers": blockwise_config["params"]["nb_cores_worker"],
            "tmp_dir": blockwise_config["params"]["tmp_dir"],
            "models": [],
            "input_normalizers": [],
            "postprocessors": []
        }
        
        # Add model_mode if multiple models are present and a merge mode is selected
        model_count = len(pipeline.get("models", []))
        model_mode = pipeline.get("model_mode", "")
        if model_count > 1 and model_mode:
            task_yaml["model_mode"] = model_mode
            logger.info(f"Adding model_mode: {model_mode} for {model_count} models")
        
        # Add models with full config
        for model in pipeline.get("models", []):
            model_entry = {
                "name": model.get("name"),
                **model.get("params", model.get("config", {}))
            }
            # Parse string representations of lists/tuples back to actual lists for specific fields
            import ast
            import re
            for field in ["channels", "input_size", "output_size", "input_voxel_size", "output_voxel_size"]:
                if field in model_entry:
                    value = model_entry[field]
                    # If it's already a list, keep it
                    if isinstance(value, (list, tuple)):
                        model_entry[field] = list(value)
                        logger.info(f"Field {field} is already a list: {model_entry[field]}")
                    # If it's a string that looks like a list/tuple, parse it
                    elif isinstance(value, str):
                        value_stripped = value.strip().strip("'\"")  # Remove outer quotes
                        if (value_stripped.startswith('[') or value_stripped.startswith('(')) and \
                           (value_stripped.endswith(']') or value_stripped.endswith(')')):
                            try:
                                # Fix unquoted identifiers: convert [mito] to ['mito']
                                # Replace word characters not inside quotes with quoted versions
                                fixed_value = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', r"'\1'", value_stripped)
                                # Remove duplicate quotes: ''mito'' -> 'mito'
                                fixed_value = re.sub(r"''+", "'", fixed_value)
                                logger.info(f"Fixing {field}: {value_stripped!r} -> {fixed_value!r}")
                                
                                parsed = ast.literal_eval(fixed_value)
                                if isinstance(parsed, (list, tuple)):
                                    model_entry[field] = list(parsed)
                                    logger.info(f"Parsed {field} from string {value!r} to list {model_entry[field]}")
                            except Exception as e:
                                logger.warning(f"Failed to parse {field}: {value}, error: {e}")
                    
            task_yaml["models"].append(model_entry)
        
        # Add normalizers
        for norm in pipeline.get("normalizers", []):
            norm_entry = {
                "name": norm.get("name"),
                "params": norm.get("params", {})
            }
            task_yaml["input_normalizers"].append(norm_entry)
        
        # Add postprocessors
        for post in pipeline.get("postprocessors", []):
            post_entry = {
                "name": post.get("name"),
                "params": post.get("params", {})
            }
            task_yaml["postprocessors"].append(post_entry)
        
        # Add output_channels from OUTPUT node if configured
        output_channels = output_node.get("params", {}).get("output_channels", [])
        if output_channels and isinstance(output_channels, list) and len(output_channels) > 0:
            task_yaml["output_channels"] = output_channels
            logger.info(f"Adding output_channels to YAML: {output_channels}")
        
        # Convert to YAML format with proper list handling
        yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
        
        # Save to file
        yaml_filename = f"{task_name}.yaml"
        tasks_dir = get_blockwise_tasks_dir()
        yaml_path = os.path.join(tasks_dir, yaml_filename)
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Generated blockwise task YAML at: {yaml_path}")
        logger.info(f"Task YAML content:\n{yaml_content}")
        
        return {
            "success": True,
            "task_yaml": yaml_content,
            "task_config": task_yaml,
            "task_path": yaml_path,
            "task_name": task_name,
            "message": "Blockwise task generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Task generation error: {str(e)}")
        return {"success": False, "error": str(e)}


@app.route("/api/blockwise/precheck", methods=["POST"])
def precheck_blockwise_task():
    """Precheck blockwise task configuration"""
    try:
        from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor
        
        data = request.get_json()
        pipeline = data.get("pipeline", {})
        
        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}
        
        # Generate task YAML first
        gen_result = generate_blockwise_task()
        if not gen_result.get("success"):
            return {"success": False, "error": gen_result.get("error")}
        
        yaml_path = gen_result.get("task_path")
        
        # Try to instantiate the processor to validate configuration
        try:
            _ = CellMapFlowBlockwiseProcessor(yaml_path, create=True)
            logger.info(f"Blockwise precheck passed for: {yaml_path}")
            return {
                "success": True,
                "message": "success"
            }
        except Exception as e:
            logger.error(f"Blockwise precheck failed: {str(e)}")
            return {"success": False, "error": str(e)}
        
    except Exception as e:
        logger.error(f"Precheck error: {str(e)}")
        return {"success": False, "error": str(e)}


@app.route("/api/blockwise/submit", methods=["POST"])
def submit_blockwise_task():
    """Submit blockwise task to LSF"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})
        job_name = data.get("job_name", f"cellmap_flow_{int(time.time())}")
        
        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}
        
        # Generate task YAML
        gen_result = generate_blockwise_task()
        if not gen_result.get("success"):
            return {"success": False, "error": gen_result.get("error")}
        
        yaml_path = gen_result["task_path"]
        blockwise_config = pipeline["blockwise_config"][0]
        
        # Build bsub command
        cores_master = blockwise_config["params"]["nb_cores_master"]
        charge_group = blockwise_config["params"]["charge_group"]
        queue = blockwise_config["params"]["queue"]
        
        bsub_cmd = [
            "bsub",
            "-J", job_name,
            "-n", str(cores_master),
            "-P", charge_group,
            # "-q", queue,
            "python", "-m", "cellmap_flow.blockwise.cli",
            yaml_path
        ]
        
        logger.info(f"Submitting LSF job: {' '.join(bsub_cmd)}")
        
        # Submit job - use same environment as parent process
        result = subprocess.run(bsub_cmd, capture_output=True, text=True, env=os.environ)
        
        if result.returncode == 0:
            output = result.stdout.strip()
            logger.info(f"Job submitted successfully: {output}")
            
            # Extract job ID from bsub output (format: "Job <12345> is submitted")
            match = re.search(r'<(\d+)>', output)
            job_id = match.group(1) if match else "unknown"
            
            return {
                "success": True,
                "job_id": job_id,
                "task_path": yaml_path,
                "command": " ".join(bsub_cmd),
                "message": f"Task submitted as job {job_id}"
                }
        else:
            error_msg = result.stderr or result.stdout
            logger.error(f"LSF submission failed: {error_msg}")
            return {"success": False, "error": f"LSF error: {error_msg}"}
        
    except Exception as e:
        logger.error(f"Submission error: {str(e)}")
        return {"success": False, "error": str(e)}


# Global state for BBX generator
bbx_generator_state = {
    "dataset_path": None,
    "num_boxes": 0,
    "bounding_boxes": [],
    "viewer": None,
    "viewer_process": None,
    "viewer_url": None,
    "viewer_state": None
}


@app.route("/api/bbx-generator", methods=["POST"])
def start_bbx_generator():
    """Start the Neuroglancer viewer for creating bounding boxes"""
    try:
        # Set Neuroglancer server to bind to 0.0.0.0 for external access
        neuroglancer.set_server_bind_address("0.0.0.0")
        
        data = request.json
        dataset_path = data.get("dataset_path", "")
        num_boxes = data.get("num_boxes", 1)
        
        if not dataset_path:
            return jsonify({"error": "Dataset path is required"}), 400
        
        # Create Neuroglancer viewer
        viewer = neuroglancer.Viewer()
        
        with viewer.txn() as s:
            # Set coordinate space
            s.dimensions = neuroglancer.CoordinateSpace(
                names=["z", "y", "x"],
                units="nm",
                scales=[8, 8, 8],
            )
            
            # Add image layer
            s.layers["fibsem"] = get_raw_layer(dataset_path)
            
            # Add LOCAL annotation layer for bounding boxes
            s.layers["annotations"] = neuroglancer.LocalAnnotationLayer(
                dimensions=neuroglancer.CoordinateSpace(
                    names=["z", "y", "x"],
                    units="nm",
                    scales=[1, 1, 1],
                ),
            )
        
        # Store state
        bbx_generator_state["dataset_path"] = dataset_path
        bbx_generator_state["num_boxes"] = num_boxes
        bbx_generator_state["bounding_boxes"] = []
        bbx_generator_state["viewer"] = viewer
        
        # Get the viewer URL and fix localhost reference
        viewer_url = str(viewer)
        
        # Replace localhost with the actual request host for external access
        # Parse the URL and replace localhost with the client's host
        if "localhost" in viewer_url:
            # Get the client's host from the request
            client_host = request.host.split(":")[0]  # Get just the host part without port
            viewer_url = viewer_url.replace("localhost", client_host)
            logger.info(f"Replaced localhost with {client_host} in viewer URL")
        
        bbx_generator_state["viewer_url"] = viewer_url
        bbx_generator_state["viewer_state"] = viewer.state
        
        logger.info(f"Starting BBX generator with viewer URL: {viewer_url}")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Target boxes: {num_boxes}")
        
        # For iframe access, we need to return the raw viewer URL
        # Neuroglancer server should be accessible at the returned URL
        return jsonify({
            "success": True,
            "viewer_url": viewer_url,
            "dataset_path": dataset_path,
            "num_boxes": num_boxes
        })
    
    except Exception as e:
        logger.error(f"Error starting BBX generator: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bbx-generator/status", methods=["GET"])
def get_bbx_generator_status():
    """Get current status of bounding box generation"""
    try:
        # Extract bounding boxes from viewer if it exists
        bboxes = []
        if bbx_generator_state.get("viewer"):
            viewer = bbx_generator_state["viewer"]
            try:
                with viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if hasattr(annotations_layer, 'annotations'):
                            for ann in annotations_layer.annotations:
                                # Check if this is a bounding box annotation
                                if type(ann).__name__ == "AxisAlignedBoundingBoxAnnotation":
                                    point_a = ann.point_a
                                    point_b = ann.point_b
                                    
                                    # Ensure point_a is the min and point_b is the max
                                    offset = [min(point_a[j], point_b[j]) for j in range(3)]
                                    max_point = [max(point_a[j], point_b[j]) for j in range(3)]
                                    shape = [int(max_point[j] - offset[j]) for j in range(3)]
                                    offset = [int(x) for x in offset]
                                    
                                    bboxes.append({
                                        "offset": offset,
                                        "shape": shape,
                                    })
                    except KeyError:
                        logger.warning("Annotations layer not found in viewer")
            except Exception as e:
                logger.warning(f"Error extracting bboxes from viewer: {str(e)}")
        
        bbx_generator_state["bounding_boxes"] = bboxes
        
        return jsonify({
            "dataset_path": bbx_generator_state.get("dataset_path"),
            "num_boxes": bbx_generator_state.get("num_boxes"),
            "bounding_boxes": bboxes,
            "count": len(bboxes)
        })
    
    except Exception as e:
        logger.error(f"Error getting BBX status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bbx-generator/finalize", methods=["POST"])
def finalize_bbx_generation():
    """Finalize bounding box generation and return results"""
    try:
        # Extract final bounding boxes from viewer
        bboxes = []
        if bbx_generator_state.get("viewer"):
            viewer = bbx_generator_state["viewer"]
            try:
                with viewer.txn() as s:
                    try:
                        annotations_layer = s.layers["annotations"]
                        if hasattr(annotations_layer, 'annotations'):
                            for ann in annotations_layer.annotations:
                                # Check if this is a bounding box annotation
                                if type(ann).__name__ == "AxisAlignedBoundingBoxAnnotation":
                                    point_a = ann.point_a
                                    point_b = ann.point_b
                                    
                                    # Ensure point_a is the min and point_b is the max
                                    offset = [min(point_a[j], point_b[j]) for j in range(3)]
                                    max_point = [max(point_a[j], point_b[j]) for j in range(3)]
                                    shape = [int(max_point[j] - offset[j]) for j in range(3)]
                                    offset = [int(x) for x in offset]
                                    
                                    bboxes.append({
                                        "offset": offset,
                                        "shape": shape,
                                    })
                    except KeyError:
                        logger.warning("Annotations layer not found in viewer")
            except Exception as e:
                logger.warning(f"Error extracting final bboxes: {str(e)}")
        
        # Reset state
        bbx_generator_state["dataset_path"] = None
        bbx_generator_state["num_boxes"] = 0
        bbx_generator_state["bounding_boxes"] = []
        bbx_generator_state["viewer_url"] = None
        bbx_generator_state["viewer"] = None
        
        return jsonify({
            "success": True,
            "bounding_boxes": bboxes,
            "count": len(bboxes)
        })
    
    except Exception as e:
        logger.error(f"Error finalizing BBX generation: {str(e)}")
        return jsonify({"error": str(e)}), 500


def create_and_run_app(neuroglancer_url=None, inference_servers=None):
    global NEUROGLANCER_URL, INFERENCE_SERVER
    NEUROGLANCER_URL = neuroglancer_url
    INFERENCE_SERVER = inference_servers
    hostname = socket.gethostname()
    port = 0
    logger.warning(f"Host name: {hostname}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    # app.run(debug=True)
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
