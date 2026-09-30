"""HTTP routes for the instance-review workflow (the dashboard's Review tab).

Thin Flask wrappers over cellmap_flow.review helpers. The index opened by
/api/review/open, and the last instance picked in the viewer, live in the
dashboard session's ``review`` (a ReviewSession; dashboard.state). Each
request opens its own SQLite connection, read-only except for verdict and
undo (no pooling — write rate is one row per user click, read rate is one
row per GET /review/next).

Navigation on /review/next is server-side: a viewer.txn() propagates to the
browser via neuroglancer's WebSocket. It is best-effort — if there is no
viewer yet, navigation is silently skipped and the instance record is still
returned.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from flask import Blueprint, Response, jsonify, request, stream_with_context

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.io.metadata import nm_per_unit
from cellmap_flow.review import (
    count_instances,
    get_instance,
    get_next,
    get_progress,
    open_db,
    queues,
    record_verdict,
    resolve_db_path,
    undo_verdict,
)

logger = logging.getLogger(__name__)

review_bp = Blueprint("review", __name__)

_REVIEW_PICK_ACTION = "review-pick"
_REVIEW_PICK_KEYBINDING = "keyt"  # press 't' over a segment to pick it

# A pick stream with nothing new sends a heartbeat this often. Writing it is
# also how the server notices the browser tab has gone and frees the thread.
PICK_STREAM_HEARTBEAT_S = 25.0


class PickBoard:
    """The last instance picked with the t key, numbered by ``seq``.

    The neuroglancer action posts to it. Each pick stream remembers the last
    ``seq`` it sent and waits for a different one. Nothing is ever cleared,
    so one stream waking up cannot hide a pick from another, as a shared
    Event that each woken stream cleared did.
    """

    def __init__(self):
        self._changed = threading.Condition()
        self.seq = 0
        self.label_id: Optional[int] = None
        self.at: Optional[float] = None  # time.monotonic() of the pick
        self.closed = False

    def post(self, label_id: int) -> int:
        with self._changed:
            self.seq += 1
            self.label_id = label_id
            self.at = time.monotonic()
            self._changed.notify_all()
            return self.seq

    def latest(self):
        """``(seq, label_id, at)`` of the last pick."""
        with self._changed:
            return self.seq, self.label_id, self.at

    def wait_past(self, seq: int, timeout: float):
        """``(seq, label_id)`` once there is a pick after ``seq``, or as they
        are after ``timeout`` seconds or a close()."""
        with self._changed:
            self._changed.wait_for(lambda: self.seq != seq or self.closed, timeout)
            return self.seq, self.label_id

    def close(self) -> None:
        """Wake every stream waiting here so it ends: another index was opened."""
        with self._changed:
            self.closed = True
            self._changed.notify_all()


@dataclass
class ReviewSession:
    """The index /api/review/open selected, kept as the dashboard session's ``review``."""

    db_path: str  # resolved by review.resolve_db_path
    reviewer: str
    segmentation_layer: Optional[str] = None
    picks: PickBoard = field(default_factory=PickBoard)


def _no_session():
    return jsonify({"error": "no review index open; POST /api/review/open first"}), 409


def _json_body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _optional_int(name: str) -> Optional[int]:
    """Query argument ``name`` as an int, None when absent; ValueError otherwise."""
    raw = request.args.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _read_instance(db_path: str, instance_id: int) -> Optional[dict]:
    conn = open_db(db_path)
    try:
        return get_instance(conn, instance_id)
    finally:
        conn.close()


def _on_review_pick(action_state) -> None:
    """The 'review-pick' neuroglancer action: pick the segment under the cursor.

    Reads the label_id under the cursor at action-fire time via NG's
    `ActionState.selected_values[<seg_layer>]` (a clean, atomic snapshot
    populated by NG-JS at click/keypress time — distinct from
    `state.position`, which is hover-polluted and unreliable for this
    purpose) and posts it to the open session's PickBoard. The instance
    lookup happens in the Flask routes that read the board.

    Runs on Neuroglancer's Tornado event loop, which is single-threaded and
    shared with every HTTP-request dispatch (including subvolume chunk
    fetches). Keep this handler O(1) — no DB I/O, nothing that can yield.

    One module-level function: neuroglancer keeps a set of handlers per
    action, so registering it again on every /open adds nothing.
    """
    session = get_session().review
    if session is None or not session.segmentation_layer:
        return
    try:
        sv_map = action_state.selected_values
        entry = sv_map.get(session.segmentation_layer) if sv_map is not None else None
        if entry is None or entry.value is None:
            return
        raw = entry.value
        try:
            label_id = int(raw)
        except (TypeError, ValueError):
            key = getattr(raw, "key", None)
            value = getattr(raw, "value", None)
            label_id = int(key if key is not None else value)
    except Exception as e:
        logger.debug(f"review: could not read the picked segment: {e}")
        return
    if label_id == 0:
        return
    seq = session.picks.post(label_id)
    # Timestamps measure end-to-end latency: keypress → action arrival
    # (this line) → the stream or poll that serves it (PICK_POLL_ARRIVAL).
    logger.debug(
        f"PICK_HANDLER_ENTRY seq={seq} label={label_id} t_mono={time.monotonic():.3f}"
    )


def _register_pick_action(viewer) -> None:
    """Bind the t key in ``viewer`` to the 'review-pick' action."""
    viewer.actions.add(_REVIEW_PICK_ACTION, _on_review_pick)
    with viewer.config_state.txn() as cs:
        cs.input_event_bindings.viewer[_REVIEW_PICK_KEYBINDING] = _REVIEW_PICK_ACTION


def _navigate_viewer(instance: dict) -> bool:
    """Best-effort: move the viewer to the instance's nm centroid.

    Neuroglancer's s.position is expressed in **voxels of the viewer's
    coordinate space**, not in nm, and neuroglancer reports that space's
    scales in SI units (8 nm reads back as 8e-9 m). So each nm centroid is
    divided by its axis's scale in nm. Axes are matched by name (z, y, x),
    falling back to the first three dimensions; other dimensions keep
    their position. A viewer with no dimensions yet gets the nm values.

    Returns True if navigation happened, False if silently skipped
    (no viewer attached). Never raises — viewer errors are logged but
    the HTTP response continues with the instance payload.
    """
    viewer = get_session().viewer
    if viewer is None:
        logger.info("review: no viewer yet; skipping navigation")
        return False
    try:
        with viewer.txn() as s:
            dims = s.dimensions
            if dims is not None and len(dims.names) >= 3:
                names = list(dims.names)
                scales_nm = [
                    float(scale) * nm_per_unit(unit)
                    for scale, unit in zip(dims.scales, dims.units)
                ]
            else:
                names, scales_nm = ["z", "y", "x"], [1.0, 1.0, 1.0]
            position = (
                [float(p) for p in s.position]
                if s.position is not None and len(s.position) == len(names)
                else [0.0] * len(names)
            )
            for fallback, (axis, key) in enumerate(
                (("z", "cz_nm"), ("y", "cy_nm"), ("x", "cx_nm"))
            ):
                i = names.index(axis) if axis in names else fallback
                position[i] = float(instance[key]) / scales_nm[i]
            s.position = position
            logger.debug(
                f"review: navigated to instance {instance['id']} "
                f"(scales_nm={scales_nm}, position={position})"
            )
        return True
    except Exception as e:
        logger.warning(f"review: viewer navigation failed: {e}")
        return False


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@review_bp.route("/api/review/open", methods=["POST"])
def review_open():
    """Select the active review index for this dashboard session.

    Body: {"db_path": "....sqlite",
           "reviewer": "davi",             # optional
           "segmentation_layer": "labels"  # optional; enables the t-key pick
          }

    Only a ``.sqlite`` or ``.db`` file (after resolving symlinks) is
    opened, and the name is checked before existence.
    """
    data = _json_body()
    db_path = data.get("db_path")
    reviewer = data.get("reviewer") or os.environ.get("USER", "")
    seg_layer = data.get("segmentation_layer") or None

    if not db_path or not isinstance(db_path, str):
        return jsonify({"success": False, "error": "db_path is required"}), 400
    if not isinstance(reviewer, str) or not isinstance(seg_layer, (str, type(None))):
        return jsonify({"success": False,
                        "error": "reviewer and segmentation_layer must be strings"}), 400
    try:
        real_path = resolve_db_path(db_path)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except FileNotFoundError:
        return jsonify({"success": False,
                        "error": f"review index not found: {db_path}"}), 404

    try:
        conn = open_db(real_path)
        try:
            n = count_instances(conn)
        finally:
            conn.close()
    except (ValueError, sqlite3.Error) as e:
        return jsonify({"success": False,
                        "error": f"could not open review index: {e}"}), 400

    previous = get_session().review
    get_session().review = ReviewSession(real_path, reviewer, seg_layer)
    if previous is not None:
        previous.picks.close()
    if seg_layer is not None and get_session().viewer is not None:
        try:
            _register_pick_action(get_session().viewer)
            logger.info(f"review: registered 'review-pick' action on key 't' for layer {seg_layer!r}")
        except Exception as e:
            logger.warning(f"review: failed to register pick action: {e}")
    logger.info(
        f"review: opened db={real_path} reviewer={reviewer!r} "
        f"seg_layer={seg_layer!r} n_instances={n}"
    )
    return jsonify({
        "success": True,
        "db_path": real_path,
        "reviewer": reviewer,
        "segmentation_layer": seg_layer,
        "n_instances": n,
    })


@review_bp.route("/api/review/next", methods=["GET"])
def review_next():
    """Return the next unreviewed instance in the chosen queue.

    Query: ?order=<queue>&min_vox=100&skip_rank=12, where the queues are
    the index's rank_<queue> columns; the first one when order is absent.

    Side effect: navigates the viewer to the instance's centroid (if
    viewer exists).
    """
    session = get_session().review
    if session is None:
        return _no_session()

    order = request.args.get("order")
    try:
        min_vox = _optional_int("min_vox")
        skip_rank = _optional_int("skip_rank")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    conn = open_db(session.db_path)
    try:
        if order is None:
            order = next(iter(queues(conn)), "")
        inst = get_next(conn, order, min_vox, skip_rank)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    if inst is None:
        return jsonify({"done": True,
                        "message": "no more unreviewed instances matching filters"})

    navigated = _navigate_viewer(inst)
    inst["done"] = False
    inst["navigated"] = navigated
    return jsonify(inst)


@review_bp.route("/api/review/verdict", methods=["POST"])
def review_verdict():
    """Record a verdict for an instance.

    Body: {"id": 123,
           "verdict": "blessed" | "edited" | "erased",
           "edit_details": {...},        # optional, only for edited
           "entry_method": "next" | "show" | "pick",  # optional
          }
    """
    session = get_session().review
    if session is None:
        return _no_session()

    data = _json_body()
    try:
        instance_id = int(data["id"])
        verdict = str(data["verdict"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"id and verdict required: {e}"}), 400

    edit_details = data.get("edit_details")
    if edit_details is not None and not isinstance(edit_details, dict):
        return jsonify({"error": "edit_details must be a JSON object"}), 400

    entry_method = data.get("entry_method")
    if entry_method is not None:
        if not isinstance(entry_method, str):
            return jsonify({"error": "entry_method must be a string"}), 400

    conn = open_db(session.db_path, write=True)
    try:
        row = record_verdict(conn, instance_id, verdict, session.reviewer,
                             edit_details, entry_method=entry_method)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    return jsonify({"success": True, "ledger": row})


@review_bp.route("/api/review/undo", methods=["POST"])
def review_undo():
    """Clear the ledger row for an instance (un-bless / un-edit).

    Body: {"id": 123}
    """
    session = get_session().review
    if session is None:
        return _no_session()

    data = _json_body()
    try:
        instance_id = int(data["id"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"id required: {e}"}), 400

    conn = open_db(session.db_path, write=True)
    try:
        row = undo_verdict(conn, instance_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    return jsonify({"success": True, "ledger": row})


@review_bp.route("/api/review/progress", methods=["GET"])
def review_progress():
    """Aggregate review progress."""
    session = get_session().review
    if session is None:
        return _no_session()

    conn = open_db(session.db_path)
    try:
        p = get_progress(conn)
    finally:
        conn.close()
    p["db_path"] = session.db_path
    return jsonify(p)


@review_bp.route("/api/review/show/<int:instance_id>", methods=["GET"])
def review_show(instance_id: int):
    """Full instance record + ledger state for a specific id.

    Side effect: navigates the viewer to the instance's centroid (if a
    viewer is attached), the reliable way to go to a known ID.
    """
    session = get_session().review
    if session is None:
        return _no_session()

    inst = _read_instance(session.db_path, instance_id)
    if inst is None:
        return jsonify({"error": f"instance {instance_id} not in index"}), 404
    _navigate_viewer(inst)
    return jsonify(inst)


@review_bp.route("/api/review/current_pick", methods=["GET"])
def review_current_pick():
    """Return the most-recent instance picked via the 'review-pick' NG action.

    Two-stage to keep NG's Tornado event loop unblocked:
      - the action handler posts `label_id` only (no DB I/O on the loop)
      - this endpoint, served by Flask on its own thread pool, does the
        catalog lookup at poll time

    Response:
      200 with {"pick": instance_record, "seq": int}
          when a pick has been recorded since /api/review/open
      204 No Content when no pick yet
    """
    session = get_session().review
    if session is None:
        return _no_session()
    seq, label_id, picked_at = session.picks.latest()
    if label_id is None:
        return ("", 204)
    t_poll = time.monotonic()
    inst = _read_instance(session.db_path, label_id)
    logger.debug(
        f"PICK_POLL_ARRIVAL seq={seq} label={label_id} "
        f"age_since_handler_ms={(t_poll - picked_at) * 1000.0:.1f} "
        f"flask_db_ms={(time.monotonic() - t_poll) * 1000.0:.1f}"
    )
    if inst is None:
        return jsonify({
            "error": f"label_id {label_id} not in catalog",
            "seq": seq,
        }), 404
    inst["label_id"] = int(inst["id"])
    return jsonify({"pick": inst, "seq": seq})


@review_bp.route("/api/review/pick_stream", methods=["GET"])
def review_pick_stream():
    """Server-Sent Events stream of pick updates.

    Single long-lived HTTP/1.1 connection — much cheaper than polling,
    and it does not compete with NG chunk fetches for fresh connection
    slots. The action handler (Tornado side) posts to the session's
    PickBoard; this generator (Werkzeug worker thread) waits on it and
    emits one SSE event per new pick. Every stream tracks its own last
    sequence number, so each open tab sees every pick.

    Heartbeats every PICK_STREAM_HEARTBEAT_S keep proxies from
    idle-closing the connection. The stream ends when another index is
    opened; the browser's EventSource then reconnects to the new one.
    """
    session = get_session().review
    if session is None:
        return _no_session()
    picks = session.picks

    def lookup(label_id):
        try:
            inst = _read_instance(session.db_path, label_id)
        except Exception as e:
            logger.warning(f"review: pick_stream db lookup failed: {e}")
            return None
        if inst is not None:
            inst["label_id"] = int(inst["id"])
        return inst

    def event(payload):
        return f"data: {json.dumps(payload)}\n\n"

    def stream():
        # First yield is a real `data:` event with the current pick (if any)
        # so EventSource clients see something the moment they open. Comment
        # lines (": ...") are dropped by some buffering proxies and don't
        # trigger onmessage on the client.
        sent, label_id, _ = picks.latest()
        yield event({
            "seq": sent,
            "pick": lookup(label_id) if label_id is not None else None,
        })

        while get_session().review is session and not picks.closed:
            seq, label_id = picks.wait_past(sent, PICK_STREAM_HEARTBEAT_S)
            if get_session().review is not session or picks.closed:
                break
            if seq != sent:
                sent = seq
                inst = lookup(label_id)
                if inst is not None:
                    yield event({"pick": inst, "seq": seq})
                    continue
            # No new pick (timeout) → heartbeat as a `data:` event with no
            # pick change (kind="heartbeat") so client knows we're alive.
            yield event({"kind": "heartbeat", "seq": seq})

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@review_bp.route("/api/review/status", methods=["GET"])
def review_status():
    """Current review-session state (is an index open? which reviewer?)."""
    session = get_session().review
    return jsonify({
        "db_path": session.db_path if session else None,
        "reviewer": session.reviewer if session else None,
        "segmentation_layer": session.segmentation_layer if session else None,
        "viewer_attached": get_session().viewer is not None,
    })
