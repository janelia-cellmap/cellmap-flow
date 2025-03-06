import json
import socket
import neuroglancer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
from cellmap_flow.utils.load_py import load_custom_code_str
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.web_utils import (
    encode_to_str,
    ARGS_KEY,
)
from cellmap_flow.models.run import update_run_models
import cellmap_flow.globals as g
import numpy as np

logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)
NEUROGLANCER_URL = None
INFERENCE_SERVER = None


@app.route("/")
def index():
    
    g.model_catalog["User"] = {j.model_name:"" for j in g.jobs}
    default_post_process = {d.to_dict()["name"]: d.to_dict() for d in g.postprocess}
    default_input_norm = {d.to_dict()["name"]: d.to_dict() for d in g.input_norms}
    logger.warning(f"Model catalog: {g.model_catalog}")
    logger.warning(f"Default postprocess: {default_post_process}")
    logger.warning(f"Default input norm: {default_input_norm}")

    return render_template(
        "index.html",
        neuroglancer_url=NEUROGLANCER_URL,
        inference_servers=INFERENCE_SERVER,
        input_normalizers=g.input_norms_functions,
        output_postprocessors=g.output_postprocessors_functions,
        default_post_process=default_post_process,
        default_input_norm=default_input_norm,
        model_catalog=g.model_catalog,
        default_models=[j.model_name for j in g.jobs],
    )


def is_output_segmentation():
    if len(g.postprocess) > 0 and g.postprocess[-1].is_segmentation:
        return True
    return False


@app.route("/update/equivalences", methods=["POST"])
def update_equivalences():
    equivalences = [
        [np.uint64(item) for item in sublist]
        for sublist in json.loads(request.get_json())
    ]
    with g.viewer.txn() as s:
        s.layers[-1].equivalences = equivalences
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
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")


    with g.viewer.txn() as s:
        # g.raw.invalidate()
        g.raw = get_raw_layer(g.dataset_path)
        s.layers["raw"] = g.raw
        for job in g.jobs:
            model = job.model_name
            host = job.host
            # response = requests.post(f"{host}/input_normalize", json=data)
            # print(f"Response from {host}: {response.json()}")
            st_data = encode_to_str(data)

            if is_output_segmentation():
                s.layers[model] = neuroglancer.SegmentationLayer(
                    source=f"n5://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )
            else:
                s.layers[model] = neuroglancer.ImageLayer(
                    source=f"n5://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                )

    custom_code = None
    for kk in ["input_norm", "postprocess"]:
        if kk in data:
            if "custom_code" in data[kk]:
                custom_code = data[kk].get("custom_code", None)
                del data[kk]["custom_code"]
    logger.warning(f"Data received: {type(data)} - {data.keys()} -{data}")
    g.input_norms = get_normalizations(data["input_norm"])
    g.postprocess = get_postprocessors(data["postprocess"])

    logger.warning(f"Input normalizers: {g.input_norms}")

    if custom_code:

        try:
            load_custom_code_str(custom_code)
            logger.warning(get_input_normalizers())

            g.input_norms_functions = get_input_normalizers()
            g.output_postprocessors_functions = get_postprocessors_list()
            

        except Exception as e:
            return jsonify(
                {
                    "message": "Error loading custom code",
                    "error": str(e),
                }
            )

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
            "found_custom_code": True if custom_code else False,
        }
    )


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
