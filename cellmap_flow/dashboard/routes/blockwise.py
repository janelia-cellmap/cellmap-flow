import os
import re
import ast
import logging
import subprocess
import time
from datetime import datetime

import yaml
from flask import Blueprint, request

from cellmap_flow.globals import g
from cellmap_flow.utils.web_utils import INPUT_NORM_DICT_KEY, POSTPROCESS_DICT_KEY
from cellmap_flow.globals import get_blockwise_tasks_dir

logger = logging.getLogger(__name__)

blockwise_bp = Blueprint("blockwise", __name__)


@blockwise_bp.route("/api/blockwise/validate", methods=["POST"])
def validate_blockwise():
    """Validate if pipeline is ready for blockwise processing"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})

        # Check required components
        if not pipeline.get("inputs") or len(pipeline["inputs"]) == 0:
            return {"valid": False, "error": "No input nodes defined"}

        if not pipeline.get("outputs") or len(pipeline["outputs"]) == 0:
            return {"valid": False, "error": "No output nodes defined"}

        if not pipeline.get("models") or len(pipeline["models"]) == 0:
            return {"valid": False, "error": "No models defined"}

        # Check blockwise config
        if not pipeline.get("blockwise_config") or len(pipeline["blockwise_config"]) == 0:
            return {"valid": False, "error": "No blockwise configuration defined"}

        # Check input has dataset_path
        input_node = pipeline["inputs"][0]
        if not input_node.get("params", {}).get("dataset_path"):
            return {"valid": False, "error": "Input node missing dataset_path"}

        # Check output has dataset_path
        output_node = pipeline["outputs"][0]
        if not output_node.get("params", {}).get("dataset_path"):
            return {"valid": False, "error": "Output node missing dataset_path"}

        logger.info("Pipeline validation passed")
        return {"valid": True, "message": "Pipeline is ready for blockwise processing"}

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {"valid": False, "error": str(e)}


@blockwise_bp.route("/api/blockwise/generate", methods=["POST"])
def generate_blockwise_task():
    """Generate blockwise task YAML files"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})

        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}

        # Get blockwise config
        blockwise_config = pipeline["blockwise_config"][0]
        input_node = pipeline["inputs"][0]
        output_node = pipeline["outputs"][0]

        # Get output path and ensure it ends with .zarr
        output_path = output_node["params"]["dataset_path"]
        if output_path:
            # Remove trailing slashes
            output_path = output_path.rstrip('/\\')
            # Add .zarr if not already present
            if '.zarr' not in output_path:
                output_path = output_path + '.zarr'

        # Create task YAML content
        task_name = f"cellmap_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task_yaml = {
            "data_path": input_node["params"]["dataset_path"],
            "output_path": output_path,
            "task_name": task_name,
            "charge_group": blockwise_config["params"]["charge_group"],
            "queue": blockwise_config["params"]["queue"],
            "workers": blockwise_config["params"]["nb_workers"],
            "cpu_workers": blockwise_config["params"]["nb_cores_worker"],
            "tmp_dir": blockwise_config["params"]["tmp_dir"],
            "models": []
        }

        # Add bounding_boxes from INPUT node if they exist
        bounding_boxes = input_node.get("params", {}).get("bounding_boxes", [])
        if bounding_boxes and isinstance(bounding_boxes, list) and len(bounding_boxes) > 0:
            task_yaml["bounding_boxes"] = bounding_boxes
            logger.info(f"Adding bounding_boxes to YAML: {len(bounding_boxes)} box(es)")

        # Add separate_bounding_boxes_zarrs flag from INPUT node if set
        separate_zarrs = input_node.get("params", {}).get("separate_bounding_boxes_zarrs", False)
        if separate_zarrs:
            task_yaml["separate_bounding_boxes_zarrs"] = True
            logger.info("Adding separate_bounding_boxes_zarrs: True")

        # Add model_mode if multiple models are present and a merge mode is selected
        model_count = len(pipeline.get("models", []))
        model_mode = pipeline.get("model_mode", "")
        if model_count > 1 and model_mode:
            task_yaml["model_mode"] = model_mode
            logger.info(f"Adding model_mode: {model_mode} for {model_count} models")

        # Add models with full config
        for model in pipeline.get("models", []):
            model_entry = {
                "name": model.get("name"),
                **model.get("params", model.get("config", {}))
            }
            # Parse string representations of lists/tuples back to actual lists for specific fields
            for field in ["channels", "input_size", "output_size", "input_voxel_size", "output_voxel_size"]:
                if field in model_entry:
                    value = model_entry[field]
                    # If it's already a list, keep it
                    if isinstance(value, (list, tuple)):
                        model_entry[field] = list(value)
                        logger.info(f"Field {field} is already a list: {model_entry[field]}")
                    # If it's a string that looks like a list/tuple, parse it
                    elif isinstance(value, str):
                        value_stripped = value.strip().strip("'\"")  # Remove outer quotes
                        if (value_stripped.startswith('[') or value_stripped.startswith('(')) and \
                           (value_stripped.endswith(']') or value_stripped.endswith(')')):
                            try:
                                # Fix unquoted identifiers: convert [mito] to ['mito']
                                fixed_value = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', r"'\1'", value_stripped)
                                # Remove duplicate quotes: ''mito'' -> 'mito'
                                fixed_value = re.sub(r"''+", "'", fixed_value)
                                logger.info(f"Fixing {field}: {value_stripped!r} -> {fixed_value!r}")

                                parsed = ast.literal_eval(fixed_value)
                                if isinstance(parsed, (list, tuple)):
                                    model_entry[field] = list(parsed)
                                    logger.info(f"Parsed {field} from string {value!r} to list {model_entry[field]}")
                            except Exception as e:
                                logger.warning(f"Failed to parse {field}: {value}, error: {e}")

            task_yaml["models"].append(model_entry)

        # Serialize normalizers and postprocessors to json_data format
        normalizers_list = pipeline.get("normalizers", [])
        postprocessors_list = pipeline.get("postprocessors", [])

        # Create json_data for blockwise processor - maintain order by using list iteration order
        if normalizers_list or postprocessors_list:
            try:
                # Build normalizers dict - preserve insertion order from normalizers_list
                norm_fns = {}
                for norm in normalizers_list:
                    if isinstance(norm, dict):
                        norm_name = norm.get("name")
                        norm_params = norm.get("params", {})
                    else:
                        continue
                    if norm_name:
                        norm_fns[norm_name] = norm_params

                # Build postprocessors dict - preserve insertion order from postprocessors_list
                post_fns = {}
                for post in postprocessors_list:
                    if isinstance(post, dict):
                        post_name = post.get("name")
                        post_params = post.get("params", {})
                    else:
                        continue
                    if post_name:
                        post_fns[post_name] = post_params

                # Create json_data as dict (not JSON string) using the correct key constants
                json_data_dict = {
                    INPUT_NORM_DICT_KEY: norm_fns,
                    POSTPROCESS_DICT_KEY: post_fns
                }
                # Store as dict (YAML will handle it properly)
                task_yaml["json_data"] = json_data_dict
                logger.info(f"Added json_data as dict with {len(normalizers_list)} normalizers and {len(postprocessors_list)} postprocessors")
            except Exception as e:
                logger.warning(f"Failed to create json_data: {e}")

        # Add output_channels from OUTPUT node if configured
        output_channels = output_node.get("params", {}).get("output_channels", [])
        if output_channels and isinstance(output_channels, list) and len(output_channels) > 0:
            task_yaml["output_channels"] = output_channels
            logger.info(f"Adding output_channels to YAML: {output_channels}")

        # Convert to YAML format with proper list handling
        yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Save to file
        yaml_filename = f"{task_name}.yaml"
        tasks_dir = get_blockwise_tasks_dir()
        yaml_path = os.path.join(tasks_dir, yaml_filename)

        # Check if we need to generate multiple YAMLs (one per bbox with separate output paths)
        output_base_path = output_path
        yaml_paths = []

        if separate_zarrs and bounding_boxes and len(bounding_boxes) > 0:
            # Generate separate YAML for each bounding box
            logger.info(f"Generating separate YAMLs for {len(bounding_boxes)} bounding box(es)")
            for bbox_idx, bbox in enumerate(bounding_boxes):
                # Create a copy of task_yaml for this bbox
                bbox_task_yaml = task_yaml.copy()

                # Keep only this bbox in bounding_boxes
                bbox_task_yaml["bounding_boxes"] = [bbox]

                # Set output path to box_X subdirectory
                bbox_output_path = os.path.join(output_base_path, f"box_{bbox_idx + 1}")
                bbox_task_yaml["output_path"] = bbox_output_path

                # Update task name to include bbox index
                bbox_task_name = f"{task_name}_box{bbox_idx + 1}"
                bbox_task_yaml["task_name"] = bbox_task_name

                # Convert to YAML
                bbox_yaml_content = yaml.dump(bbox_task_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False)

                # Save bbox YAML
                bbox_yaml_filename = f"{bbox_task_name}.yaml"
                bbox_yaml_path = os.path.join(tasks_dir, bbox_yaml_filename)
                with open(bbox_yaml_path, 'w') as f:
                    f.write(bbox_yaml_content)

                yaml_paths.append(bbox_yaml_path)
                logger.info(f"Generated bbox {bbox_idx + 1} YAML at: {bbox_yaml_path}")
        else:
            # Single YAML for all bboxes
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)
            yaml_paths = [yaml_path]
            logger.info(f"Generated blockwise task YAML at: {yaml_path}")

        logger.info(f"Task YAML content:\n{yaml_content}")

        return {
            "success": True,
            "task_yaml": yaml_content,
            "task_config": task_yaml,
            "task_paths": yaml_paths,
            "task_name": task_name,
            "message": "Blockwise task generated successfully"
        }

    except Exception as e:
        logger.error(f"Task generation error: {str(e)}")
        return {"success": False, "error": str(e)}


@blockwise_bp.route("/api/blockwise/precheck", methods=["POST"])
def precheck_blockwise_task():
    """Precheck blockwise task configuration using already-generated YAML"""
    try:
        from cellmap_flow.blockwise.blockwise_processor import CellMapFlowBlockwiseProcessor

        data = request.get_json()
        yaml_paths = data.get("yaml_paths", [])

        if not yaml_paths:
            return {"success": False, "error": "No YAML paths provided. Please generate task first."}

        # Try to instantiate the processor to validate configuration with the first YAML
        try:
            _ = CellMapFlowBlockwiseProcessor(yaml_paths[0], create=True)
            logger.info(f"Blockwise precheck passed for: {yaml_paths[0]}")
            return {
                "success": True,
                "message": "success"
            }
        except Exception as e:
            logger.error(f"Blockwise precheck failed: {str(e)}")
            return {"success": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Precheck error: {str(e)}")
        return {"success": False, "error": str(e)}


@blockwise_bp.route("/api/blockwise/submit", methods=["POST"])
def submit_blockwise_task():
    """Submit blockwise task to LSF"""
    try:
        data = request.get_json()
        pipeline = data.get("pipeline", {})
        job_name = data.get("job_name", f"cellmap_flow_{int(time.time())}")

        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}

        # Generate task YAML
        gen_result = generate_blockwise_task()
        if not gen_result.get("success"):
            return {"success": False, "error": gen_result.get("error")}

        yaml_paths = gen_result.get("task_paths", [gen_result.get("task_path")])
        blockwise_config = pipeline["blockwise_config"][0]

        # Build bsub command
        cores_master = blockwise_config["params"]["nb_cores_master"]
        charge_group = blockwise_config["params"]["charge_group"]
        queue = blockwise_config["params"]["queue"]

        bsub_cmd = [
            "bsub",
            "-J", job_name,
            "-n", str(cores_master),
            "-P", charge_group,
            # "-q", queue,
            "python", "-m", "cellmap_flow.blockwise.multiple_cli",
        ] + yaml_paths

        logger.info(f"Submitting LSF job: {' '.join(bsub_cmd)}")

        # Submit job - use same environment as parent process
        result = subprocess.run(bsub_cmd, capture_output=True, text=True, env=os.environ)

        if result.returncode == 0:
            output = result.stdout.strip()
            logger.info(f"Job submitted successfully: {output}")

            # Extract job ID from bsub output (format: "Job <12345> is submitted")
            match = re.search(r'<(\d+)>', output)
            job_id = match.group(1) if match else "unknown"

            return {
                "success": True,
                "job_id": job_id,
                "task_paths": yaml_paths,
                "command": " ".join(bsub_cmd),
                "message": f"Task submitted as job {job_id}"
                }
        else:
            error_msg = result.stderr or result.stdout
            logger.error(f"LSF submission failed: {error_msg}")
            return {"success": False, "error": f"LSF error: {error_msg}"}

    except Exception as e:
        logger.error(f"Submission error: {str(e)}")
        return {"success": False, "error": str(e)}
