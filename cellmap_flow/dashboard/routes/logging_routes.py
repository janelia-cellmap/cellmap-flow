import os
import queue
import logging

from flask import Blueprint, Response, request

from cellmap_flow.dashboard.state import get_session

logger = logging.getLogger(__name__)

logging_bp = Blueprint("logging", __name__)


class LogHandler(logging.Handler):
    """Feeds the log panel: each record, numbered, into the session's
    log_buffer, and to every open stream's queue.

    It runs on whichever thread logged, often one with no app context (a
    launch, a job monitor), so it asks for the session at emit time.
    """

    def emit(self, record):
        session = get_session()
        # Handler.handle holds the handler's lock around emit, so the records
        # go into the buffer in the order they are numbered.
        numbered = (next(session.log_ids), self.format(record))
        session.log_buffer.append(numbered)
        # A copy: a stream opening or closing on another thread changes the list.
        for client_queue in list(session.log_clients):
            try:
                client_queue.put_nowait(numbered)
            except queue.Full:
                pass


def requested_offset(request):
    """Where a log stream's client wants it from, in the dashboard's log
    protocol (static/js/lib/sse.js): the SSE Last-Event-ID, else ?offset=."""
    for raw in (request.headers.get("Last-Event-ID"), request.args.get("offset")):
        try:
            if raw not in (None, ""):
                return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return 0


def _event(record_id, log_entry):
    """One SSE message holding ``log_entry``, with its number as the id: a
    "data:" line per line of it. A traceback's later lines, sent bare, are
    not data, and the browser dropped them."""
    data = "".join(f"data: {line}\n" for line in log_entry.splitlines() or [""])
    return f"id: {record_id}\n{data}\n"


@logging_bp.route("/api/logs/stream")
def stream_logs():
    """The log panel's records as Server-Sent Events: those in the buffer
    after the one the client asks from (requested_offset; a reconnect asks
    from the last it was sent), then each new one as it is logged, with a
    keepalive comment every 30 s."""
    session = get_session()
    after = requested_offset(request)

    def generate():
        # The queue first, so that nothing logged while the buffer is sent is
        # lost; a record that comes both ways is sent once. A copy of the
        # buffer: iterating the buffer itself across the yields died with
        # "deque mutated during iteration" when another thread logged.
        client_queue = queue.Queue(maxsize=100)
        session.log_clients.append(client_queue)
        sent = after
        try:
            for record_id, log_line in list(session.log_buffer):
                if record_id > sent:
                    sent = record_id
                    yield _event(record_id, log_line)
            while True:
                try:
                    record_id, log_line = client_queue.get(timeout=30)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if record_id > sent:
                    sent = record_id
                    yield _event(record_id, log_line)
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
