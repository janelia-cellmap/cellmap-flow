import os
import queue
import logging

from flask import Blueprint, Response

from cellmap_flow.dashboard.state import get_session

logger = logging.getLogger(__name__)

logging_bp = Blueprint("logging", __name__)


class LogHandler(logging.Handler):
    """Feeds the log panel: each record into the session's log_buffer, and
    to every open stream's queue.

    It runs on whichever thread logged, often one with no app context (a
    launch, a job monitor), so it asks for the session at emit time.
    """

    def emit(self, record):
        log_entry = self.format(record)
        session = get_session()
        session.log_buffer.append(log_entry)
        # A copy: a stream opening or closing on another thread changes the list.
        for client_queue in list(session.log_clients):
            try:
                client_queue.put_nowait(log_entry)
            except queue.Full:
                pass


def _event(log_entry):
    """One SSE message holding ``log_entry``: a "data:" line per line of it.
    A traceback's later lines, sent bare, are not data, and the browser
    dropped them."""
    return "".join(f"data: {line}\n" for line in log_entry.splitlines() or [""]) + "\n"


@logging_bp.route("/api/logs/stream")
def stream_logs():
    """The log panel's records as Server-Sent Events: the buffer's, then
    each new one as it is logged, with a keepalive comment every 30 s."""
    session = get_session()

    def generate():
        # The queue first, so that nothing logged while the buffer is sent is
        # lost (a record logged just then may come twice), and a copy of the
        # buffer: iterating the buffer itself across the yields died with
        # "deque mutated during iteration" when another thread logged.
        client_queue = queue.Queue(maxsize=100)
        session.log_clients.append(client_queue)
        try:
            for log_line in list(session.log_buffer):
                yield _event(log_line)
            while True:
                try:
                    log_line = client_queue.get(timeout=30)
                    yield _event(log_line)
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            # Clean up when client disconnects
            if client_queue in session.log_clients:
                session.log_clients.remove(client_queue)

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
