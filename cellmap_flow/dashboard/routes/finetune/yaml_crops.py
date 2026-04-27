"""Endpoint for bulk-loading externally annotated crops via a YAML manifest."""

import logging

import numpy as np
from flask import jsonify
from pydantic import ValidationError

from cellmap_flow.dashboard.routes.finetune.annotation_core import _get_selected_model_config
from cellmap_flow.dashboard.routes.finetune.common import ensure_corrections_storage
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.finetune.crop_loader import load_crops, parse_crops_yaml
from cellmap_flow.globals import g

logger = logging.getLogger(__name__)


def load_crops_from_yaml_response(data):
    """Materialize a batch of YAML-described crops as ``_chunk_*.zarr`` entries.

    Request JSON:
        - ``model_name``: required, used to derive input/output shape/voxel size
        - ``output_path``: optional, base path for the session corrections dir
        - ``yaml``: required, the YAML text (or path to a YAML file)
    """
    try:
        model_name = data.get("model_name")
        output_path = data.get("output_path")
        yaml_input = data.get("yaml")

        if not yaml_input:
            return jsonify({"success": False, "error": "Missing 'yaml' field"}), 400
        if not model_name:
            return jsonify({"success": False, "error": "Missing 'model_name' field"}), 400

        try:
            crops_config = parse_crops_yaml(yaml_input)
        except ValidationError as e:
            return (
                jsonify({"success": False, "error": "YAML validation failed", "details": e.errors()}),
                400,
            )
        except Exception as e:
            return jsonify({"success": False, "error": f"YAML parse error: {e}"}), 400

        if not crops_config.crops:
            return jsonify({"success": False, "error": "No crops listed in YAML"}), 400

        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        config = model_config.config
        read_shape = np.array(config.read_shape)
        write_shape = np.array(config.write_shape)
        claimed_input_voxel_size = np.array(config.input_voxel_size)
        claimed_output_voxel_size = np.array(config.output_voxel_size)
        input_size = (read_shape / claimed_input_voxel_size).astype(int)
        output_size = (write_shape / claimed_output_voxel_size).astype(int)

        raw_dataset_path = getattr(g, "dataset_path", None)
        if not raw_dataset_path:
            return jsonify({"success": False, "error": "No raw dataset path configured"}), 400

        _, corrections_dir = ensure_corrections_storage(output_path)

        result = load_crops(
            crops_config,
            raw_dataset_path=raw_dataset_path,
            corrections_dir=corrections_dir,
            input_size=input_size,
            output_size=output_size,
            claimed_input_voxel_size=claimed_input_voxel_size,
            claimed_output_voxel_size=claimed_output_voxel_size,
            model_name=model_name,
        )

        try:
            refresh_annotated_regions_layer()
        except Exception as e:
            logger.warning(f"refresh_annotated_regions_layer failed: {e}")

        return jsonify(
            {
                "success": True,
                "n_crops_requested": len(crops_config.crops),
                "n_chunks_created": len(result["created"]),
                "n_errors": len(result["errors"]),
                "errors": result["errors"],
            }
        )
    except Exception as e:
        logger.exception("load_crops_from_yaml_response failed")
        return jsonify({"success": False, "error": str(e)}), 500
