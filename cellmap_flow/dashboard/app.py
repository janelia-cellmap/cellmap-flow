import os
import socket
import logging

from flask import Flask

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
from cellmap_flow.dashboard.routes.review_routes import review_bp

logger = logging.getLogger(__name__)

# Explicitly set template and static folder paths for package installation
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# No CORS: the dashboard's pages call it from its own origin, and it can
# submit LSF jobs and read files, so other sites must not be able to script
# it through the user's browser. The inference servers keep CORS open because
# Neuroglancer fetches their chunks cross-origin.
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Feed the dashboard's log panel.
#
# This was attached to logging.getLogger(__name__), so the panel only ever
# received records emitted by app.py itself -- which is almost nothing.
# Everything from finetune_utils, the job launcher and the route modules went to
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
app.register_blueprint(review_bp)


def _announce_service_url(url):
    """Tell whoever launched the dashboard where it is.

    Fileglancer runs the dashboard as a service job and reads its URL from
    the file named by SERVICE_URL_PATH; without the variable this is a no-op.
    Never raises: a bad path must not take the dashboard down with it.
    """
    path = os.environ.get("SERVICE_URL_PATH")
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(url)
    except OSError as e:
        logger.warning(f"Could not write the dashboard URL to {path}: {e}")


def create_and_run_app(neuroglancer_url=None):
    from werkzeug.serving import make_server

    g.NEUROGLANCER_URL = neuroglancer_url
    hostname = socket.gethostname()
    # threaded=True is not optional. make_server defaults to one request at a
    # time, and the dashboard has long POSTs (a YAML crop load ran for three
    # minutes) and open SSE log streams, each of which would then block every
    # other request. app.run() used to default to threaded=True; this keeps it.
    server = make_server("0.0.0.0", 0, app, threaded=True)
    url = f"http://{hostname}:{server.socket.getsockname()[1]}"
    logger.info(f"Dashboard running at: {url}")
    print(f"\n * Dashboard URL: {url}\n", flush=True)
    _announce_service_url(url)
    server.serve_forever()


if __name__ == "__main__":
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
