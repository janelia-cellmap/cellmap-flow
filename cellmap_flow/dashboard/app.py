import socket
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cellmap_flow.utils.web_utils import get_free_port

app = Flask(__name__)
CORS(app)
NEUROGLANCER_URL = None
INFERENCE_SERVER = None


@app.route("/")
def index():
    # Render the main page with tabs
    return render_template(
        "index.html",
        neuroglancer_url=NEUROGLANCER_URL,
        inference_servers=INFERENCE_SERVER,
    )


@app.route("/api/process", methods=["POST"])
def process():
    """
    Example endpoint to receive JSON data from the form submission.
    Just echoes back the received data in this example.
    """

    data = request.get_json()
    # Here you could add logic to process the data, e.g.:
    # - if "method" in data: handle input normalization
    # - if "outputChannel" in data: handle output settings
    # - etc.

    return jsonify({"message": "Data received successfully", "received_data": data})


def create_and_run_app(neuroglancer_url=None, inference_servers=None):
    global NEUROGLANCER_URL, INFERENCE_SERVER
    NEUROGLANCER_URL = neuroglancer_url
    INFERENCE_SERVER = inference_servers
    hostname = socket.gethostname()
    port = get_free_port()
    print(f"Host name: {hostname}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    # app.run(debug=True)
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
