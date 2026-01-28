import json
import os
import socket
import neuroglancer
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from cellmap_flow.utils.web_utils import get_free_port
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
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
    
    # Convert current pipeline state from globals to dict format with unique IDs
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
    
    return render_template(
        "pipeline_builder.html",
        input_normalizers=input_norms or {},
        output_postprocessors=output_postprocessors or {},
        current_normalizers=current_normalizers,
        current_postprocessors=current_postprocessors,
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
        for norm_name in normalizer_names:
            if norm_name not in available_norms:
                return jsonify(
                    {"valid": False, "error": f"Unknown normalizer: {norm_name}"}
                ), 400
        
        # Validate postprocessors
        processor_names = [p.get("name") for p in data.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        for proc_name in processor_names:
            if proc_name not in available_procs:
                return jsonify(
                    {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}
                ), 400
        
        return jsonify({"valid": True, "message": "Pipeline is valid"})
    
    except Exception as e:
        logger.error(f"Error validating pipeline: {e}")
        return jsonify({"valid": False, "error": str(e)}), 500


@app.route("/api/pipeline/apply", methods=["POST"])
def apply_pipeline():
    """Apply a pipeline configuration to the current inference"""
    try:
        data = request.get_json()
        
        # Validate first
        validation = validate_pipeline_config(data)
        if not validation["valid"]:
            return jsonify(validation), 400
        
        # Apply normalizers
        input_norms_config = {
            n["name"]: n.get("params", {}) for n in data.get("input_normalizers", [])
        }
        g.input_norms = get_normalizations(input_norms_config)
        
        # Apply postprocessors
        postprocs_config = {
            p["name"]: p.get("params", {}) for p in data.get("postprocessors", [])
        }
        g.postprocess = get_postprocessors(postprocs_config)
        
        logger.warning(f"Pipeline applied: {len(g.input_norms)} normalizers, {len(g.postprocess)} postprocessors")
        
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
        for norm_name in normalizer_names:
            if norm_name not in available_norms:
                return {"valid": False, "error": f"Unknown normalizer: {norm_name}"}
        
        processor_names = [p.get("name") for p in config.get("postprocessors", [])]
        available_procs = get_postprocessors_list()
        for proc_name in processor_names:
            if proc_name not in available_procs:
                return {"valid": False, "error": f"Unknown postprocessor: {proc_name}"}
        
        return {"valid": True}
    
    except Exception as e:
        return {"valid": False, "error": str(e)}


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
