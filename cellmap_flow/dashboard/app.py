import os
import socket
import logging

from flask import Flask
from flask_cors import CORS

from cellmap_flow.globals import g, LogHandler
from cellmap_flow.utils.logging_setup import LOG_DATEFMT, LOG_FORMAT
from cellmap_flow.dashboard.routes.logging_routes import logging_bp
from cellmap_flow.dashboard.routes.index_page import index_bp
from cellmap_flow.dashboard.routes.pipeline_builder_page import pipeline_builder_bp
from cellmap_flow.dashboard.routes.models import models_bp
from cellmap_flow.dashboard.routes.pipeline import pipeline_bp
from cellmap_flow.dashboard.routes.model_advice import model_advice_bp
from cellmap_flow.dashboard.routes.blockwise import blockwise_bp
from cellmap_flow.dashboard.routes.bbx_generator import bbx_bp
from cellmap_flow.dashboard.routes.finetune import finetune_bp

logger = logging.getLogger(__name__)

# Explicitly set template and static folder paths for package installation
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Feed the dashboard's log panel.
#
# This was attached to logging.getLogger(__name__), so the panel only ever
# received records emitted by app.py itself -- which is almost nothing.
# Everything from finetune_utils, bsub_utils and the route modules went to
# the terminal and nowhere else, so reading a failure meant having shell
# access to whatever machine the dashboard happened to land on.
#
# Attach to the package logger instead: every cellmap_flow.* record
# propagates up to it, while werkzeug's per-request lines -- which would
# swamp the panel -- do not.
package_logger = logging.getLogger("cellmap_flow")
if not any(isinstance(h, LogHandler) for h in package_logger.handlers):
    log_handler = LogHandler()
    log_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    package_logger.addHandler(log_handler)
package_logger.setLevel(logging.INFO)

# Register all blueprints
app.register_blueprint(logging_bp)
app.register_blueprint(index_bp)
app.register_blueprint(pipeline_builder_bp)
app.register_blueprint(models_bp)
app.register_blueprint(pipeline_bp)
app.register_blueprint(blockwise_bp)
app.register_blueprint(bbx_bp)
app.register_blueprint(finetune_bp)
app.register_blueprint(model_advice_bp)


def create_and_run_app(neuroglancer_url=None, inference_servers=None):
    g.NEUROGLANCER_URL = neuroglancer_url
    g.INFERENCE_SERVER = inference_servers
    hostname = socket.gethostname()
    port = 0

    from werkzeug.serving import make_server

    # threaded=True is not optional here. make_server defaults to
    # threaded=False, processes=1 -- one request at a time -- whereas the
    # app.run() this replaced defaulted to threaded=True. Under a single
    # worker a long POST blocks everything behind it: a yaml crop load ran
    # for three minutes while every /load-crops-progress poll sat queued, so
    # the dialog froze on "Starting..." and only unblocked once the work had
    # already finished, releasing ~40 queued polls in one second. The SSE log
    # streams are worse -- an open stream would hold the only thread forever.
    server = make_server("0.0.0.0", port, app, threaded=True)
    actual_port = server.socket.getsockname()[1]
    url = f"http://{hostname}:{actual_port}"
    logger.info(f"Dashboard running at: {url}")
    print(f"\n * Dashboard URL: {url}\n")
    try:
        service_url_path = os.environ.get("SERVICE_URL_PATH")
        if service_url_path:
            with open(service_url_path, "w") as f:
                f.write(url)
    except Exception as e:
        logger.warning(f"Failed to write service URL to {service_url_path}: {e}")
    server.serve_forever()


if __name__ == "__main__":
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
