"""``POST /api/models/resolve``: a model entry from whatever reference the user
pasted; ``POST /api/models/add``: run such an entry.

The body is ``{"ref": ..., "name": ..., "voxel_size": ...}`` (only ``ref``
is required; ``voxel_size`` is one number, one per axis, or "8,8,8"). The
answer is ``models.resolve``'s ``Resolved``::

    {"success": true, "type": "fly", "params": {"checkpoint_path": ...},
     "name": "run07_432000", "env": "fly", "how": "a fly_organelles training checkpoint",
     "needs": [], "notes": [...]}

``type``, ``params`` and ``name`` are the model entry, which the model form
submits once the user has given what ``needs`` lists. Nothing is built or
launched here. A reference that resolves to nothing, or to a model
cellmap-flow cannot serve, is a 400 with ``{"success": false, "error": ...}``.
"""

import logging
from typing import Annotated, Any, Optional

from flask import Blueprint, jsonify, request
from pydantic import BaseModel, Field

from cellmap_flow.dashboard.requests import _required, parse

logger = logging.getLogger(__name__)

model_resolve_bp = Blueprint("model_resolve", __name__)


class ResolveModel(BaseModel):
    """What the user pasted, and optionally the model's name and voxel size."""

    ref: Annotated[str, _required("ref", strip=True)] = Field(None, validate_default=True)
    name: Optional[str] = None
    voxel_size: Any = None


@model_resolve_bp.route("/api/models/resolve", methods=["POST"])
def resolve_model():
    from cellmap_flow.models.resolve import resolve

    body, error = parse(ResolveModel, request.get_json(silent=True))
    if error:
        return error
    try:
        resolved = resolve(body.ref, name=(body.name or "").strip() or None, voxel_size=body.voxel_size)
    except ValueError as e:
        logger.info(f"Could not resolve model {body.ref!r}: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    return jsonify({"success": True, **resolved.to_json()})


class AddModel(BaseModel):
    """A model entry to run: ``type``, its parameters and ``name``, as resolve
    answers them with what ``needs`` asked filled in."""

    entry: dict


@model_resolve_bp.route("/api/models/add", methods=["POST"])
def add_model():
    """Build the entry and start its server, in its environment.

    The model joins the session's models and jobs like one Submit started, so
    it stays running while it is ticked on the Models tab (the page adds a
    ticked box for it) and is stopped when unticked. A 400 names what is
    wrong with the entry; a 409, a model of that name already running.
    """
    from cellmap_flow.config.yaml import ConfigError
    from cellmap_flow.dashboard.services.launch import _launch, run_model_config, running_names
    from cellmap_flow.dashboard.state import get_session
    from cellmap_flow.models.registry import build_model
    from cellmap_flow.pipeline_spec import PipelineSpec

    body, error = parse(AddModel, request.get_json(silent=True))
    if error:
        return error
    entry = dict(body.entry)
    name = str(entry.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "The model needs a name"}), 400
    session = get_session()
    if name in running_names():
        return jsonify({"success": False, "error": f"A model named {name} is already running"}), 409
    try:
        model_config = build_model(entry, name)
    except (ConfigError, ValueError, TypeError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    session.models_config = [mc for mc in session.models_config if getattr(mc, "name", None) != name]
    session.models_config.append(model_config)
    st_data = PipelineSpec.from_steps(session.input_norms, session.postprocess).to_url_blob()
    _launch(name, run_model_config, model_config, st_data, daemon=True)
    logger.info(f"Adding model {name} ({entry.get('type')})")
    return jsonify({"success": True, "name": name})
