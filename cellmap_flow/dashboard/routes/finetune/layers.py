"""Add, remove and rename layers of the running viewer over HTTP.

For scripts and the command line; the dashboard's own pages do not call
these. Each is a POST with a JSON body:

- ``/api/viewer/add-image-layer``
- ``/api/viewer/add-segmentation-layer``
- ``/api/viewer/remove-layer``
- ``/api/viewer/rename-layer``

Idempotency: add-* replaces a same-named layer; remove-layer is a no-op
when the name is absent; rename-layer answers 409 rather than overwrite
another layer.

The layer's per-name bookkeeping -- the session's ``shaders``,
``shader_controls`` and ``extra_layers`` (dashboard.state) -- follows a
remove or a rename, so a later layer of the same name starts fresh.

``path`` is opened as ``/api/set-data`` opens a dataset: any zarr, n5 or
precomputed store the dashboard's user can read.
"""

import logging

from flask import jsonify, request

from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.state import get_session

logger = logging.getLogger(__name__)

_BOOKKEEPING = ("shaders", "shader_controls", "extra_layers")


def _error(message, status=400):
    return jsonify({"success": False, "error": message}), status


def _add_layer(name, layer, layer_type, path):
    with get_session().viewer.txn() as s:
        if name in s.layers:
            logger.info(f"Replacing existing layer {name}")
            del s.layers[name]
        s.layers[name] = layer
    logger.info(f"Added {layer_type} layer: {name} -> {path}")
    return jsonify(
        {
            "success": True,
            "layer_name": name,
            "layer_type": layer_type,
            "reload_page": True,
        }
    )


@finetune_bp.route("/api/viewer/add-segmentation-layer", methods=["POST"])
def add_segmentation_layer_to_viewer():
    """Register a static segmentation zarr on the running NG viewer.

    Required: path, name. Optional: blend, disable_meshes.
    """
    data = request.get_json() or {}
    try:
        path = data.get("path")
        name = data.get("name")
        if not path or not name:
            return _error("Missing path or name")
        session = get_session()
        if session.viewer is None:
            return _error("viewer not initialized")

        from cellmap_flow.viewer.raw import get_raw_layer

        layer = get_raw_layer(
            path, segmentation=True, disable_meshes=bool(data.get("disable_meshes", False))
        )
        if data.get("blend"):
            layer.blend = data["blend"]
        return _add_layer(name, layer, "segmentation", path)

    except Exception as e:
        logger.error(f"Error adding segmentation layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/viewer/add-image-layer", methods=["POST"])
def add_image_layer_to_viewer():
    """Register a static image zarr on the running NG viewer.

    Required: path, name. Optional: shader, blend.
    """
    data = request.get_json() or {}
    try:
        path = data.get("path")
        name = data.get("name")
        if not path or not name:
            return _error("Missing path or name")
        session = get_session()
        if session.viewer is None:
            return _error("viewer not initialized")

        from cellmap_flow.viewer.raw import get_raw_layer

        layer = get_raw_layer(path, normalize=False)
        if data.get("shader"):
            layer.shader = data["shader"]
        if data.get("blend"):
            layer.blend = data["blend"]
        return _add_layer(name, layer, "image", path)

    except Exception as e:
        logger.error(f"Error adding image layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/viewer/remove-layer", methods=["POST"])
def remove_layer_from_viewer():
    """Drop a layer from the running NG viewer state by name.

    Idempotent: returns success with removed=false if the name is absent.
    """
    data = request.get_json() or {}
    try:
        name = data.get("name")
        if not name:
            return _error("Missing name")
        session = get_session()
        if session.viewer is None:
            return _error("viewer not initialized")

        # Read first: txn() pushes the whole state back even when nothing
        # changed.
        removed = name in session.viewer.state.layers
        if removed:
            with session.viewer.txn() as s:
                if name in s.layers:
                    del s.layers[name]
        for attr in _BOOKKEEPING:
            getattr(session, attr).pop(name, None)

        logger.info(f"Removed layer: {name} (was_present={removed})")
        return jsonify(
            {
                "success": True,
                "layer_name": name,
                "removed": removed,
                "reload_page": removed,
            }
        )

    except Exception as e:
        logger.error(f"Error removing layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/viewer/rename-layer", methods=["POST"])
def rename_layer_in_viewer():
    """Rename a layer in the running NG viewer state.

    Required: old_name, new_name. 404s if old_name is absent; 409s if
    new_name already exists (refuses to silently overwrite). The layer
    keeps its place in the layer list and its visibility.
    """
    data = request.get_json() or {}
    try:
        old_name = data.get("old_name")
        new_name = data.get("new_name")
        if not old_name or not new_name:
            return _error("Missing old_name or new_name")
        session = get_session()
        if session.viewer is None:
            return _error("viewer not initialized")

        if old_name == new_name:
            return jsonify(
                {
                    "success": True,
                    "renamed": False,
                    "old_name": old_name,
                    "new_name": new_name,
                    "reload_page": False,
                }
            )

        layers = session.viewer.state.layers
        if old_name not in layers:
            return _error(f"Layer not found: {old_name}", 404)
        if new_name in layers:
            return _error(f"Target name already exists: {new_name}", 409)
        with session.viewer.txn() as s:
            s.layers[old_name].name = new_name

        for attr in _BOOKKEEPING:
            bookkeeping = getattr(session, attr)
            if old_name in bookkeeping:
                bookkeeping[new_name] = bookkeeping.pop(old_name)

        logger.info(f"Renamed layer: {old_name} -> {new_name}")
        return jsonify(
            {
                "success": True,
                "renamed": True,
                "old_name": old_name,
                "new_name": new_name,
                "reload_page": True,
            }
        )

    except Exception as e:
        logger.error(f"Error renaming layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
