import socket
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cellmap_flow.utils.web_utils import get_free_port
from cellmap_flow.norm.input_normalize import (
    get_normalizers,
    InputNormalizer,
    get_normalizations,
)
import os
from cellmap_flow.utils.load_py import load_safe_config
from datetime import datetime
import cellmap_flow.globals as g

app = Flask(__name__)
CORS(app)
NEUROGLANCER_URL = None
INFERENCE_SERVER = None

CustomCodeFolder = "/Users/zouinkhim/Desktop/cellmap/cellmap-flow/example/example_norm"


@app.route("/")
def index():
    # Render the main page with tabs
    input_norms = get_normalizers()

    return render_template(
        "index.html",
        neuroglancer_url=NEUROGLANCER_URL,
        inference_servers=INFERENCE_SERVER,
        input_normalizers=input_norms,
    )


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    custom_code = data.get("custom_code", None)
    if "custom_code" in data:
        del data["custom_code"]
    print(f"Data received: {type(data)} - {data.keys()} -{data}", flush=True)
    g.input_norms = get_normalizations(data)
    g.raw.invalidate()

    # 1) Extract user code from the payload, if present

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

            print(get_normalizers())

        except Exception as e:
            print(f"Error executing custom code: {e}")

    return jsonify(
        {
            "message": "Data received successfully",
            "received_data": data,
            "found_custom_normalizer": get_normalizers(),
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
