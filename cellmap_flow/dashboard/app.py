import os
import socket
import neuroglancer
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from cellmap_flow.utils.web_utils import get_free_port
from cellmap_flow.norm.input_normalize import (
    get_input_normalizers,
    get_normalizations,
)
from cellmap_flow.post.postprocessors import get_postprocessors_list, get_postprocessors
from cellmap_flow.utils.load_py import load_safe_config
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.web_utils import encode_to_str, decode_to_json, ARGS_KEY, INPUT_NORM_DICT_KEY, POSTPROCESS_DICT_KEY
import cellmap_flow.globals as g

app = Flask(__name__)
CORS(app)
NEUROGLANCER_URL = None
INFERENCE_SERVER = None

CustomCodeFolder = "/Users/zouinkhim/Desktop/cellmap/cellmap-flow/example/example_norm"

@app.route("/")
def index():
    # Render the main page with tabs
    input_norms = get_input_normalizers()
    output_postprocessors = get_postprocessors_list()

    return render_template(
        "index.html",
        neuroglancer_url=NEUROGLANCER_URL,
        inference_servers=INFERENCE_SERVER,
        input_normalizers=input_norms,
        output_postprocessors = output_postprocessors,
    )


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    print(f"Data received: {type(data)} - {data.keys()} -{data}", flush=True)
    custom_code = data.get("custom_code", None)
    if "custom_code" in data:
        del data["custom_code"]
    print(f"Data received: {type(data)} - {data.keys()} -{data}", flush=True)
    g.input_norms = get_normalizations(data["input_norm"])
    g.postprocess = get_postprocessors(data["postprocess"])



    with g.viewer.txn() as s:
        # g.raw.invalidate()
        g.raw = get_raw_layer(g.dataset_path)
        s.layers["raw"] = g.raw
        for model,host in g.models_host.items():
            # response = requests.post(f"{host}/input_normalize", json=data)
            # print(f"Response from {host}: {response.json()}")
            st_data = encode_to_str(data)
            s.layers[model] = neuroglancer.ImageLayer(
                source=f"n5://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
            )

    print(f"Input normalizers: {g.input_norms}", flush=True)

    if custom_code:

        try:
            # Save custom code to a file with date and time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_code_{timestamp}.py"
            filepath = os.path.join(CustomCodeFolder, filename)

            with open(filepath, "w") as file:
                file.write(custom_code)

            config = load_safe_config(filepath)
            print(f"Custom code loaded successfully: {config}")

            print(get_input_normalizers())

        except Exception as e:
            print(f"Error executing custom code: {e}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
            "found_custom_normalizer": get_input_normalizers(),
        }
    )


def create_and_run_app(neuroglancer_url=None, inference_servers=None):
    global NEUROGLANCER_URL, INFERENCE_SERVER
    NEUROGLANCER_URL = neuroglancer_url
    INFERENCE_SERVER = inference_servers
    hostname = socket.gethostname()
    port = get_free_port()
    print(f"Host name: {hostname}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    # app.run(debug=True)
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
