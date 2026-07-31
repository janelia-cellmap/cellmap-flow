import logging
import sys
import subprocess

from flask import Blueprint, request, jsonify

from cellmap_flow.utils.import_utils import INSTALLABLE_PACKAGES

logger = logging.getLogger(__name__)

dependencies_bp = Blueprint("dependencies", __name__)


@dependencies_bp.route("/api/dependencies/install", methods=["POST"])
def install_dependency():
    """Install missing optional package(s), restricted to a fixed allow-list.

    The client can only ask for packages by their import name; the actual pip
    install spec always comes from INSTALLABLE_PACKAGES on the server, never
    from the request body, since this dashboard has no authentication.
    """
    data = request.get_json() or {}
    packages = data.get("packages", [])

    if not packages:
        return jsonify({"success": False, "error": "No packages specified"}), 400

    unknown = [p for p in packages if p not in INSTALLABLE_PACKAGES]
    if unknown:
        logger.warning(f"Rejected install request for non-allow-listed package(s): {unknown}")
        return jsonify(
            {"success": False, "error": f"Not on install allow-list: {', '.join(unknown)}"}
        ), 400

    specs = [INSTALLABLE_PACKAGES[p] for p in packages]
    logger.warning(f"Installing dependencies: {specs}")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *specs],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Install timed out after 5 minutes"}), 504

    log = (result.stdout + result.stderr)[-4000:]
    if result.returncode != 0:
        logger.error(f"pip install failed for {specs}:\n{log}")

    return jsonify({"success": result.returncode == 0, "log": log})
