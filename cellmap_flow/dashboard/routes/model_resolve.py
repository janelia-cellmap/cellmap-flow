"""``POST /api/models/resolve``: a model entry from whatever reference the user pasted.

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
