"""
API routes for managing training data (raw + ground truth crop pairs).

Provides import/export from YAML and path validation.
"""

import logging
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

training_data_bp = Blueprint("training_data", __name__)


@training_data_bp.route("/api/finetune/training-data/import", methods=["GET"])
def import_training_data():
    """
    Import training data configuration from a YAML file.

    Query params:
        path: Absolute path to the YAML file

    Returns:
        JSON with parsed datasets list
    """
    yaml_path = request.args.get("path", "").strip()
    if not yaml_path:
        return jsonify({"success": False, "error": "path parameter is required"}), 400

    p = Path(yaml_path)
    if not p.exists():
        return jsonify({"success": False, "error": f"File not found: {yaml_path}"}), 404
    if not p.is_file():
        return jsonify({"success": False, "error": f"Not a file: {yaml_path}"}), 400

    try:
        with open(p) as f:
            config = yaml.safe_load(f)

        datasets_raw = config.get("datasets", {})
        datasets = []
        for name, info in datasets_raw.items():
            datasets.append({
                "name": name,
                "raw": info.get("raw", ""),
                "crops": info.get("crops", {}),
            })

        return jsonify({"success": True, "datasets": datasets, "source_path": yaml_path})

    except Exception as e:
        logger.error(f"Error importing YAML {yaml_path}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@training_data_bp.route("/api/finetune/training-data/export", methods=["POST"])
def export_training_data():
    """
    Export training data configuration to a YAML file.

    Body:
        { "path": "/output/path.yaml", "datasets": [ { "name": ..., "raw": ..., "crops": {...} } ] }
    """
    data = request.get_json()
    output_path = data.get("path", "").strip()
    datasets_list = data.get("datasets", [])

    if not output_path:
        return jsonify({"success": False, "error": "path is required"}), 400
    if not datasets_list:
        return jsonify({"success": False, "error": "No datasets to export"}), 400

    try:
        # Build YAML structure
        yaml_data = {"datasets": {}}
        for ds in datasets_list:
            name = ds.get("name", "").strip()
            if not name:
                continue
            entry = {}
            raw = ds.get("raw", "").strip()
            if raw:
                entry["raw"] = raw
            crops = ds.get("crops", {})
            if crops:
                entry["crops"] = {k: v for k, v in crops.items() if k.strip() and v.strip()}
            yaml_data["datasets"][name] = entry

        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported training data YAML to {output_path}")
        return jsonify({"success": True, "path": output_path})

    except Exception as e:
        logger.error(f"Error exporting YAML to {output_path}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@training_data_bp.route("/api/finetune/training-data/validate", methods=["POST"])
def validate_training_data():
    """
    Validate that all raw and crop paths exist and are readable zarr arrays.

    Body:
        { "datasets": [ { "name": ..., "raw": ..., "crops": {...} } ] }

    Returns:
        Validation results per dataset with per-path status.
    """
    data = request.get_json()
    datasets_list = data.get("datasets", [])

    if not datasets_list:
        return jsonify({"success": False, "error": "No datasets to validate"}), 400

    results = []
    all_valid = True

    for ds in datasets_list:
        name = ds.get("name", "unknown")
        raw_path = ds.get("raw", "").strip()
        crops = ds.get("crops", {})

        # Validate raw path
        raw_valid = False
        raw_error = None
        if not raw_path:
            raw_error = "Empty path"
        else:
            rp = Path(raw_path)
            if not rp.exists():
                raw_error = "Path does not exist"
            elif not (rp / ".zarray").exists() and not (rp / ".zattrs").exists():
                raw_error = "Not a valid zarr array (no .zarray or .zattrs)"
            else:
                raw_valid = True

        if not raw_valid:
            all_valid = False

        # Validate each crop
        crop_results = {}
        for crop_name, crop_path in crops.items():
            crop_path = crop_path.strip() if crop_path else ""
            if not crop_path:
                crop_results[crop_name] = {"valid": False, "error": "Empty path"}
                all_valid = False
                continue
            cp = Path(crop_path)
            if not cp.exists():
                crop_results[crop_name] = {"valid": False, "error": "Path does not exist"}
                all_valid = False
            elif not (cp / ".zarray").exists() and not (cp / ".zattrs").exists():
                crop_results[crop_name] = {"valid": False, "error": "Not a valid zarr array"}
                all_valid = False
            else:
                crop_results[crop_name] = {"valid": True}

        results.append({
            "name": name,
            "raw_valid": raw_valid,
            "raw_error": raw_error,
            "crops": crop_results,
        })

    return jsonify({"success": True, "all_valid": all_valid, "results": results})
