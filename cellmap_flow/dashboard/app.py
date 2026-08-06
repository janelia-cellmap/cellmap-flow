import os
import socket
import logging
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from cellmap_flow.globals import g, LogHandler
from cellmap_flow.dashboard.routes.logging_routes import logging_bp
from cellmap_flow.dashboard.routes.index_page import index_bp
from cellmap_flow.dashboard.routes.pipeline_builder_page import pipeline_builder_bp
from cellmap_flow.dashboard.routes.models import models_bp
from cellmap_flow.dashboard.routes.pipeline import pipeline_bp
from cellmap_flow.dashboard.routes.blockwise import blockwise_bp
from cellmap_flow.dashboard.routes.bbx_generator import bbx_bp
from cellmap_flow.dashboard.routes.finetune import finetune_bp

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load a .env file from the repo root if present (e.g. GOOGLE_CLOUD_PROJECT
    for the AI-annotate Gemini/Vertex AI integration). Existing env vars win.
    """
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.is_file():
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

# Explicitly set template and static folder paths for package installation
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Add custom log handler to logger
log_handler = LogHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

# Register all blueprints
app.register_blueprint(logging_bp)
app.register_blueprint(index_bp)
app.register_blueprint(pipeline_builder_bp)
app.register_blueprint(models_bp)
app.register_blueprint(pipeline_bp)
app.register_blueprint(blockwise_bp)
app.register_blueprint(bbx_bp)
app.register_blueprint(finetune_bp)


def create_and_run_app(neuroglancer_url=None, inference_servers=None):
    g.NEUROGLANCER_URL = neuroglancer_url
    g.INFERENCE_SERVER = inference_servers
    hostname = socket.gethostname()
    port = 0
    logger.warning(f"Host name: {hostname}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    create_and_run_app(neuroglancer_url="https://neuroglancer-demo.appspot.com/")
