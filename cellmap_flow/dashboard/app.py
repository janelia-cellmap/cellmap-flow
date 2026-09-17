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
    logger.debug(f"Host name: {hostname}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
