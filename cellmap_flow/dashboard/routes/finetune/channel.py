"""The channel of a model of several that the viewer shows, in server mode:
the browser's route (web/src/backend/routes/), answered by the dashboard,
so the Finetune tab's pick (Use this model, the training form's Channel)
shows the same in both.

POST /api/finetune/channel {model_name, select_channel} records the channel
(an index into the model's output channels, as select_channel takes it, or
null to forget the pick) as the one to show of that model
(Session.shown_channels), and moves the model's layer to it, if the viewer
has one: fly_organelles' layer opened on all_mem, its first of eight, while
the paint and the training were for mito. Every layer built for the model
after (a redraw, a start, viewer.layers prediction_layer), and for a
finetune of it unless one was picked for that (Session.shown_channel),
opens on it. A model with no channel axis, or under a chain that changes
the number of its channels, shows no pick (viewer.layers display_channel
refuses it). Forgetting the pick leaves the layer where it is, where the
user may have moved it since. Answers ``{success, shown}``, ``shown``
whether a layer was moved.
"""

import logging

from flask import jsonify, request

from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.serving.client import fetch_model_info, running_job_host
from cellmap_flow.viewer.layers import channel_position

logger = logging.getLogger(__name__)


@finetune_bp.route("/api/finetune/channel", methods=["POST"])
def show_channel():
    data = request.get_json(silent=True) or {}
    name = str(data.get("model_name") or "").strip()
    channel = data.get("select_channel")
    if not name:
        return jsonify({"success": False, "error": "model_name is required"}), 400
    if channel is not None and (isinstance(channel, bool) or not isinstance(channel, int) or channel < 0):
        return jsonify({"success": False, "error": "select_channel is a channel's index (0 and up), or null"}), 400
    session = get_session()
    if channel is None:
        session.shown_channels.pop(name, None)
        return jsonify({"success": True, "shown": False})
    session.shown_channels[name] = channel
    shown = _move_layer(name, channel)
    logger.info(f"{name}'s layer shows channel {channel}" if shown
                else f"{name}'s layers will open on channel {channel}")
    return jsonify({"success": True, "shown": shown})


def _move_layer(name, channel):
    """Whether ``name``'s image layer was there to move to ``channel``, and
    was moved: its local position, as prediction_layer opens one
    (channel_position, over its server's model_info and the session's
    chain). Nothing else of the layer changes, so the user's shader,
    opacity and blending stay."""
    session = get_session()
    host = running_job_host(name)
    if session.viewer is None or not host:
        return False
    # Asked before the transaction: a server's answer can take seconds, and
    # the viewer's state is locked for the length of one.
    position = channel_position(fetch_model_info(host), session.postprocess, channel)
    if position is None:
        return False
    with session.viewer.txn() as s:
        if name not in s.layers or s.layers[name].type != "image":
            return False
        s.layers[name].local_position = position
    return True
