import os
import queue
import logging

from flask import Blueprint, Response

from cellmap_flow.dashboard.state import log_buffer, log_clients

logger = logging.getLogger(__name__)

logging_bp = Blueprint("logging", __name__)


@logging_bp.route("/api/logs/stream")
def stream_logs():
    """Stream logs via Server-Sent Events (SSE)"""
    def generate():
        # Send existing log buffer first
        for log_line in log_buffer:
            yield f"data: {log_line}\n\n"

        # Create a queue for this client
        client_queue = queue.Queue(maxsize=100)
        log_clients.append(client_queue)

        try:
            while True:
                try:
                    log_line = client_queue.get(timeout=30)
                    yield f"data: {log_line}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            # Clean up when client disconnects
            if client_queue in log_clients:
                log_clients.remove(client_queue)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })


@logging_bp.route("/api/templates/bbox-json")
def get_bbox_json_template():
    """Serve the bounding box JSON format template"""
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "templates",
        "bbox_json_template.html"
    )
    try:
        with open(template_path, 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return "<p>Template not found</p>", 404
