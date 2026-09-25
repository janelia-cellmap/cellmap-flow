"""HTTP routes for the instance-review workflow (Phase 2).

Thin Flask wrappers over cellmap_flow.review helpers. Per-request SQLite
connections (no pooling — write rate is one row per user click, read
rate is one row per GET /review/next).

Navigation on /review/next is server-side: mutating g.viewer.txn()
propagates to the browser via neuroglancer's WebSocket. If
g.review_segmentation_layer is set, also sets the highlighted segment
set on that layer. Both are best-effort — if g.viewer is None (no
viewer is attached yet), navigation is silently skipped and the
instance record is still returned.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from flask import Blueprint, Response, jsonify, request, stream_with_context

from cellmap_flow.globals import g
from cellmap_flow.review import (
    ORDER_COL,
    count_instances,
    find_instance_at_voxel,
    get_instance,
    get_meta,
    get_next,
    get_progress,
    open_db,
    record_verdict,
    undo_verdict,
)
import json as _json

logger = logging.getLogger(__name__)

review_bp = Blueprint("review", __name__)


# Neuroglancer serializes CoordinateSpace to SI base units internally.
# A viewer configured with scales=[6,6,6] units="nm" reads back as
# scales=[6e-9,6e-9,6e-9] units="m". Table converts the reported unit
# to nm so we can combine it with the nm-unit centroids we store in
# the review SQLite.
_UNIT_TO_NM = {
    "m": 1e9,
    "mm": 1e6,
    "um": 1e3,
    "µm": 1e3,
    "nm": 1.0,
    "": 1.0,
}


def _require_db() -> Optional[str]:
    """Return the active db path or None (caller should respond 409)."""
    return g.review_db_path if getattr(g, "review_db_path", None) else None


_REVIEW_PICK_KEYBINDING = "keyt"  # press 't' over a segment to pick it

# Wake-up signal for the SSE pick stream. The NG action handler sets
# this when a new pick lands; the streaming generator in
# /api/review/pick_stream wait()s on it.
_pick_event = threading.Event()


def _register_pick_action(viewer, seg_layer_name: str) -> None:
    """Register the 'review-pick' Neuroglancer action.

    Reads the label_id under the cursor at action-fire time via NG's
    `ActionState.selected_values[<seg_layer>]` (a clean, atomic snapshot
    populated by NG-JS at click/keypress time — distinct from
    `state.position`, which is hover-polluted and unreliable for this
    purpose). The label_id is looked up in the active review db; the
    found instance is stashed in `g.review_last_pick` for the dashboard
    JS to poll via /api/review/current_pick.

    Idempotent: re-calling reattaches the handler and rebinds the key.
    """
    def handler(action_state):
        # Runs on Neuroglancer's Tornado event loop, which is single-threaded
        # and shared with every HTTP-request dispatch (including subvolume
        # chunk fetches). Keep this handler O(1) — no DB I/O, nothing that
        # can yield. Flask /api/review/current_pick does the DB lookup
        # off-loop.
        #
        # Instrumentation: ONE logger.warning at action-arrival, ONE
        # time.monotonic() write. Timestamps let us measure end-to-end
        # latency: keypress (user clock) → action-arrival here (this log)
        # → poll-arrival in Flask (separate log) → UI update.
        try:
            t0 = time.monotonic()
            sv_map = action_state.selected_values
            entry = sv_map.get(seg_layer_name) if sv_map is not None else None
            if entry is None or entry.value is None:
                return
            raw = entry.value
            try:
                label_id = int(raw)
            except (TypeError, ValueError):
                key = getattr(raw, "key", None)
                value = getattr(raw, "value", None)
                label_id = int(key if key is not None else value)
            if label_id == 0:
                return
            g.review_last_pick_label_id = label_id
            g.review_last_pick_ts = t0
            g.review_pick_seq = (getattr(g, "review_pick_seq", 0) or 0) + 1
            _pick_event.set()  # wake the SSE stream
            logger.warning(
                f"PICK_HANDLER_ENTRY seq={g.review_pick_seq} "
                f"label={label_id} t_mono={t0:.3f}"
            )
        except Exception:
            return

    viewer.actions.add("review-pick", handler)
    with viewer.config_state.txn() as cs:
        cs.input_event_bindings.viewer[_REVIEW_PICK_KEYBINDING] = "review-pick"


def _navigate_viewer(instance: dict) -> bool:
    """Best-effort: move g.viewer to the instance's nm centroid and
    highlight the segment on g.review_segmentation_layer if configured.

    Neuroglancer's s.position is expressed in **voxels of the viewer's
    coordinate space**, not in nm. To convert our nm centroid we divide
    by the per-axis scale from s.dimensions (which is nm-per-voxel for
    each named axis). Assumes the viewer's dimensions are declared in
    (z, y, x) order, matching bbx_generator.py and the dashboard
    configs. Fallback: if dimensions are missing or shaped
    unexpectedly, send the nm value directly (best-effort still lands
    somewhere in-volume even if scale is off).

    Returns True if navigation happened, False if silently skipped
    (no viewer attached). Never raises — viewer errors are logged but
    the HTTP response continues with the instance payload.
    """
    viewer = getattr(g, "viewer", None)
    if viewer is None:
        logger.info("review: g.viewer is None; skipping navigation")
        return False
    try:
        with viewer.txn() as s:
            # Reconstruct the per-axis nm-per-voxel factor. Neuroglancer
            # normalizes to SI meters internally, so `scales` alone can
            # be 10^9 smaller than the user-facing "6nm" — must multiply
            # by the unit factor.
            scales_nm = [1.0, 1.0, 1.0]
            try:
                dims = s.dimensions
                if dims is not None:
                    raw_scales = list(dims.scales) if dims.scales is not None else []
                    raw_units = list(dims.units) if dims.units is not None else []
                    for i in range(min(3, len(raw_scales))):
                        unit = str(raw_units[i]).strip().lower() if i < len(raw_units) else "nm"
                        factor = _UNIT_TO_NM.get(unit, 1.0)
                        scales_nm[i] = float(raw_scales[i]) * factor
                    logger.warning(
                        f"review: viewer scales_nm={scales_nm} "
                        f"(raw_scales={raw_scales} raw_units={raw_units})"
                    )
            except Exception as e:
                logger.warning(
                    f"review: could not read s.dimensions: {e}"
                )

            s.position = [
                float(instance["cz_nm"]) / scales_nm[0],
                float(instance["cy_nm"]) / scales_nm[1],
                float(instance["cx_nm"]) / scales_nm[2],
            ]

            seg_layer = getattr(g, "review_segmentation_layer", None)
            if seg_layer:
                seg_id = int(instance["id"])
                logger.warning(
                    f"review: navigated to instance {seg_id} "
                    f"(layer {seg_layer!r}, no segment-level highlight)"
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
    """Select the active review SQLite for this dashboard session.

    Body: {"db_path": "...",
           "reviewer": "davi",             # optional
           "segmentation_layer": "labels"  # optional; layer to highlight
          }
    """
    data = request.get_json(silent=True) or {}
    db_path = data.get("db_path")
    reviewer = data.get("reviewer") or os.environ.get("USER", "")
    seg_layer = data.get("segmentation_layer")

    if not db_path:
        return jsonify({"success": False, "error": "db_path is required"}), 400
    if not os.path.exists(db_path):
        return jsonify({"success": False,
                        "error": f"db not found: {db_path}"}), 404

    try:
        conn = open_db(db_path)
        n = count_instances(conn)
        conn.close()
    except Exception as e:
        return jsonify({"success": False,
                        "error": f"could not open db: {e}"}), 500

    g.review_db_path = db_path
    g.review_reviewer = reviewer
    g.review_segmentation_layer = seg_layer
    g.review_last_pick_label_id = None
    g.review_pick_seq = 0
    if seg_layer is not None and getattr(g, "viewer", None) is not None:
        try:
            _register_pick_action(g.viewer, seg_layer)
            logger.info(f"review: registered 'review-pick' action on key 't' for layer {seg_layer!r}")
        except Exception as e:
            logger.warning(f"review: failed to register pick action: {e}")
    logger.info(
        f"review: opened db={db_path} reviewer={reviewer!r} "
        f"seg_layer={seg_layer!r} n_instances={n}"
    )
    return jsonify({
        "success": True,
        "db_path": db_path,
        "reviewer": reviewer,
        "segmentation_layer": seg_layer,
        "n_instances": n,
    })


@review_bp.route("/api/review/next", methods=["GET"])
def review_next():
    """Return the next unreviewed instance in the chosen queue.

    Query: ?order=fm|smallest|random&min_vox=100

    Side effect: navigates g.viewer to the instance's centroid (if
    viewer exists) and highlights it on the configured segmentation
    layer (if set).
    """
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open; POST /api/review/open first"}), 409

    order = request.args.get("order", "fm")
    if order not in ORDER_COL:
        return jsonify({"error": f"order must be one of {list(ORDER_COL)}"}), 400

    min_vox_raw = request.args.get("min_vox")
    min_vox = int(min_vox_raw) if min_vox_raw is not None else None
    skip_rank_raw = request.args.get("skip_rank")
    skip_rank = int(skip_rank_raw) if skip_rank_raw is not None else None

    conn = open_db(db_path)
    try:
        inst = get_next(conn, order, min_vox, skip_rank)
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
           "entry_method": "next" | "select_at" | "show",  # optional
          }
    """
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open; POST /api/review/open first"}), 409

    data = request.get_json(silent=True) or {}
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

    reviewer = getattr(g, "review_reviewer", None) or os.environ.get("USER", "")

    conn = open_db(db_path)
    try:
        row = record_verdict(conn, instance_id, verdict, reviewer,
                             edit_details, entry_method=entry_method)
    except ValueError as e:
        conn.close()
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    return jsonify({"success": True, "ledger": row})


@review_bp.route("/api/review/undo", methods=["POST"])
def review_undo():
    """Clear the ledger row for an instance (un-bless / un-edit).

    Body: {"id": 123}
    """
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open; POST /api/review/open first"}), 409

    data = request.get_json(silent=True) or {}
    try:
        instance_id = int(data["id"])
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"id required: {e}"}), 400

    conn = open_db(db_path)
    try:
        row = undo_verdict(conn, instance_id)
    except ValueError as e:
        conn.close()
        return jsonify({"error": str(e)}), 400
    finally:
        conn.close()

    return jsonify({"success": True, "ledger": row})


@review_bp.route("/api/review/progress", methods=["GET"])
def review_progress():
    """Aggregate review progress."""
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open; POST /api/review/open first"}), 409

    conn = open_db(db_path)
    try:
        p = get_progress(conn)
    finally:
        conn.close()
    p["db_path"] = db_path
    return jsonify(p)


@review_bp.route("/api/review/show/<int:instance_id>", methods=["GET"])
def review_show(instance_id: int):
    """Full instance record + ledger state for a specific id.

    Side effect: navigates g.viewer to the instance's centroid (if a
    viewer is attached). Bypasses the cursor-state-read path of
    select_at, so it's the reliable way to navigate to a known ID
    when select_at gives stale cursor reads.
    """
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open; POST /api/review/open first"}), 409

    conn = open_db(db_path)
    try:
        inst = get_instance(conn, instance_id)
    finally:
        conn.close()

    if inst is None:
        return jsonify({"error": f"instance {instance_id} not in index"}), 404
    _navigate_viewer(inst)
    return jsonify(inst)


@review_bp.route("/api/review/debug/viewer_state", methods=["GET"])
def review_debug_viewer_state():
    """Dump the server-side cached neuroglancer viewer state.

    Diagnostic for the "select_at returns stale label voxel" bug. Cross-
    reference the returned `position` against what the browser shows in
    its URL fragment after a position-bar paste — if they differ, the
    server's cached view of state.position is lagging or partial.
    """
    viewer = getattr(g, "viewer", None)
    if viewer is None:
        return jsonify({"error": "g.viewer is not attached"}), 409
    try:
        state = viewer.state
        def _coerce(v):
            try:
                return [float(x) for x in v]
            except Exception:
                return list(v) if v is not None else None
        out = {
            "position": _coerce(state.position) if state.position is not None else None,
            "dimensions": {
                "names": list(state.dimensions.names) if state.dimensions is not None else None,
                "scales": _coerce(state.dimensions.scales) if state.dimensions is not None else None,
                "units": [str(u) for u in state.dimensions.units] if state.dimensions is not None else None,
            } if state.dimensions is not None else None,
            "cross_section_scale": float(state.cross_section_scale) if getattr(state, "cross_section_scale", None) is not None else None,
        }
        try:
            import json as _json2
            out["state_json"] = _json2.loads(_json2.dumps(state.to_json(), default=str))
        except Exception as e:
            out["state_json_error"] = str(e)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": f"could not read viewer state: {e}"}), 500


@review_bp.route("/api/review/current_pick", methods=["GET"])
def review_current_pick():
    """Return the most-recent instance picked via the 'review-pick' NG action.

    Two-stage to keep NG's Tornado event loop unblocked:
      - the action handler stashes `label_id` only (no DB I/O on the loop)
      - this endpoint, served by Flask on its own thread pool, does the
        catalog lookup at poll time

    Response:
      200 with {"pick": instance_record, "seq": int}
          when a pick has been recorded since /api/review/open
      204 No Content when no pick yet (or last pick was cleared)
    """
    label_id = getattr(g, "review_last_pick_label_id", None)
    if label_id is None:
        return ("", 204)
    t_poll = time.monotonic()
    seq = getattr(g, "review_pick_seq", 0)
    pick_ts = getattr(g, "review_last_pick_ts", t_poll)
    age_ms = (t_poll - pick_ts) * 1000.0
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open"}), 409
    conn = open_db(db_path)
    try:
        inst = get_instance(conn, label_id)
    finally:
        conn.close()
    t_done = time.monotonic()
    db_ms = (t_done - t_poll) * 1000.0
    # Log only on first observation of a new seq, so polls that see no
    # change don't spam the log.
    last_logged_seq = getattr(g, "review_last_logged_pick_seq", -1)
    if seq != last_logged_seq:
        g.review_last_logged_pick_seq = seq
        logger.warning(
            f"PICK_POLL_ARRIVAL seq={seq} label={label_id} "
            f"age_since_handler_ms={age_ms:.1f} flask_db_ms={db_ms:.1f}"
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

    Single long-lived HTTP/1.1 connection — much cheaper than 700ms-
    interval polling, and crucially does NOT compete with NG chunk
    fetches for fresh connection slots over the SSH tunnel. The
    action handler (Tornado side) sets `_pick_event`; this generator
    (Werkzeug worker thread) wait()s on it and emits one SSE event
    per new pick. Heartbeats every 25s keep proxies / load balancers
    from idle-closing the connection.
    """
    db_path = _require_db()
    if db_path is None:
        return jsonify({"error": "no review db open"}), 409

    def stream():
        last_seq_emitted = -1
        # First yield is a real `data:` event with the current pick (if any)
        # so EventSource clients see something the moment they open. Comment
        # lines (": ...") are dropped by some buffering proxies and don't
        # trigger onmessage on the client.
        first_seq = getattr(g, "review_pick_seq", 0) or 0
        first_label = getattr(g, "review_last_pick_label_id", None)
        first_payload = {"seq": first_seq, "pick": None}
        if first_seq > 0 and first_label is not None:
            try:
                conn = open_db(db_path)
                try:
                    inst = get_instance(conn, first_label)
                finally:
                    conn.close()
                if inst is not None:
                    inst["label_id"] = int(inst["id"])
                    first_payload["pick"] = inst
                    last_seq_emitted = first_seq
            except Exception as e:
                logger.warning(f"review: pick_stream initial db lookup failed: {e}")
        yield f"data: {_json.dumps(first_payload)}\n\n"

        while True:
            woke = _pick_event.wait(timeout=25.0)
            _pick_event.clear()
            current_seq = getattr(g, "review_pick_seq", 0) or 0
            if woke and current_seq != last_seq_emitted and current_seq > 0:
                label_id = getattr(g, "review_last_pick_label_id", None)
                if label_id is not None:
                    try:
                        conn = open_db(db_path)
                        try:
                            inst = get_instance(conn, label_id)
                        finally:
                            conn.close()
                    except Exception as e:
                        logger.warning(f"review: pick_stream db lookup failed: {e}")
                        inst = None
                    if inst is not None:
                        inst["label_id"] = int(inst["id"])
                        last_seq_emitted = current_seq
                        payload = {"pick": inst, "seq": current_seq}
                        yield f"data: {_json.dumps(payload)}\n\n"
                        continue
            # No new pick (timeout) → heartbeat as a `data:` event with no
            # pick change (kind="heartbeat") so client knows we're alive.
            yield f"data: {_json.dumps({'kind':'heartbeat','seq':current_seq})}\n\n"

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
    """Current review-session state (is a db open? which reviewer?)."""
    return jsonify({
        "db_path": getattr(g, "review_db_path", None),
        "reviewer": getattr(g, "review_reviewer", None),
        "segmentation_layer": getattr(g, "review_segmentation_layer", None),
        "viewer_attached": getattr(g, "viewer", None) is not None,
    })
