from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def index():
    # Render the main page with tabs
    return render_template("index.html")


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


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
