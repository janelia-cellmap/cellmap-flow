import os
import re
import ast
import logging
import subprocess
import sys
import time
from datetime import datetime

import yaml
from flask import Blueprint, request

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.spec import JobSpec
from cellmap_flow.utils.bsub_utils import DEFAULT_WALLTIME
from cellmap_flow.utils.web_utils import INPUT_NORM_DICT_KEY, POSTPROCESS_DICT_KEY
from cellmap_flow.globals import get_blockwise_tasks_dir

logger = logging.getLogger(__name__)

blockwise_bp = Blueprint("blockwise", __name__)


def _task_walltime():
    """The LSF run limit for a blockwise task's master and its workers.

    The dashboard's walltime setting, as for inference servers. Without -W a
    GPU worker is killed at the queue's two-hour default.
    """
    return get_session().walltime or DEFAULT_WALLTIME


def _sanitize_job_name(name) -> str:
    """Reduce a user-typed job name to something safe for an LSF -J value, a
    YAML filename, a daisy task id and a log name: keep [A-Za-z0-9_.-],
    collapse everything else into single underscores."""
    if not name:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def _make_task_name(requested_name: str, timestamp: str) -> str:
    """Single source of truth for the blockwise run's identifier. It names the
    generated YAML(s), the master LSF job (-J), the daisy task and therefore
    the worker LSF jobs and their logs. The timestamp keeps it unique so
    re-using a name never overwrites the YAML a running master's workers are
    still reading."""
    base = _sanitize_job_name(requested_name) or "cellmap_flow"
    return f"{base}_{timestamp}"


def _existing_task_paths(paths):
    """``paths`` if it is a non-empty list of existing files, else None."""
    if not isinstance(paths, list) or not paths:
        return None
    if not all(isinstance(p, str) and os.path.isfile(p) for p in paths):
        return None
    return list(paths)


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

        # The job name the user typed, if any, names the task: the YAML, the
        # master job, the daisy task, the workers and the logs.
        task_name = _make_task_name(
            data.get("job_name", ""), datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        task_yaml = {
            "data_path": input_node["params"]["dataset_path"],
            "output_path": output_path,
            "task_name": task_name,
            "charge_group": blockwise_config["params"]["charge_group"],
            "queue": blockwise_config["params"]["queue"],
            "workers": blockwise_config["params"]["nb_workers"],
            "cpu_workers": blockwise_config["params"]["nb_cores_worker"],
            "tmp_dir": blockwise_config["params"]["tmp_dir"],
            # Each worker's -W; the master gets the same (see submit).
            "walltime": _task_walltime(),
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

        # Create json_data for the blockwise processor in the ordered list form
        # [{name, **params}]. A dict keyed by name kept only the last of two
        # steps with the same name, silently changing the chain.
        if normalizers_list or postprocessors_list:
            try:
                def chain_steps(steps):
                    return [
                        {"name": step["name"], **(step.get("params") or {})}
                        for step in steps
                        if isinstance(step, dict) and step.get("name")
                    ]

                json_data_dict = {
                    INPUT_NORM_DICT_KEY: chain_steps(normalizers_list),
                    POSTPROCESS_DICT_KEY: chain_steps(postprocessors_list),
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
        # precheck() rather than constructing the processor: that created the
        # output arrays, loaded every model into this process, and replaced
        # the dashboard's live g.input_norms and g.postprocess.
        from cellmap_flow.blockwise.blockwise_processor import precheck

        data = request.get_json()
        yaml_paths = data.get("yaml_paths", [])

        if not yaml_paths:
            return {"success": False, "error": "No YAML paths provided. Please generate task first."}

        try:
            for yaml_path in yaml_paths:
                precheck(yaml_path)
            logger.info(f"Blockwise precheck passed for: {', '.join(yaml_paths)}")
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

        # First validate
        validation = validate_blockwise()
        if not validation.get("valid"):
            return {"success": False, "error": validation.get("error")}

        # Submit the YAMLs that /api/blockwise/generate wrote and
        # /api/blockwise/precheck checked, when the client sends them back.
        # Regenerating writes new files under a new task name, so what ran
        # was not what had been checked.
        requested = data.get("yaml_paths")
        yaml_paths = _existing_task_paths(requested)
        # The name generate gave the YAMLs; the page sends it back with them.
        task_name = data.get("task_name")
        if yaml_paths is not None:
            logger.info(f"Submitting the given task YAML(s): {', '.join(yaml_paths)}")
        else:
            if requested:
                logger.warning(
                    f"Not every given task YAML exists ({requested}); generating new ones"
                )
            gen_result = generate_blockwise_task()
            if not gen_result.get("success"):
                return {"success": False, "error": gen_result.get("error")}

            yaml_paths = gen_result.get("task_paths", [gen_result.get("task_path")])
            task_name = gen_result.get("task_name")
        blockwise_config = pipeline["blockwise_config"][0]

        # The master carries the task's name, so `bjobs -J <task>` is the
        # master and `bjobs -J "predict_*_<task>*"` are its workers.
        job_name = _sanitize_job_name(task_name or data.get("job_name")) or (
            f"cellmap_flow_{int(time.time())}"
        )

        # The master is a CPU job on the default queue (no -q, no -gpu); the
        # configured queue is for the workers and travels in the YAML.
        spec = JobSpec(
            name=job_name,
            # This interpreter, not whatever "python" is first on the PATH
            # the job inherits: that is the environment cellmap_flow is in.
            # Run as it stands: nothing in it needs a shell.
            argv=(sys.executable, "-m", "cellmap_flow.blockwise.multiple_cli", *yaml_paths),
            queue=None,
            gpus=0,
            cpus=blockwise_config["params"]["nb_cores_master"],
            charge_group=blockwise_config["params"]["charge_group"],
            walltime=_task_walltime(),
            log_dir=get_blockwise_tasks_dir(),
        )
        bsub_cmd = jobs_lsf.bsub_argv(spec)
        log_pattern = str(jobs_lsf.log_pattern(spec))

        try:
            # No timeout: an over-ratio request is held for minutes before
            # bsub answers, and the job still lands.
            job_id = jobs_lsf.submit(spec, bsub_timeout=None).job_id
        except jobs_lsf.JobIdMissingError as e:
            logger.warning(f"Submitted, but bsub gave no job id: {e.output}")
            job_id = "unknown"
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout
            logger.error(f"LSF submission failed: {error_msg}")
            return {"success": False, "error": f"LSF error: {error_msg}"}

        return {
            "success": True,
            "job_id": job_id,
            "task_name": job_name,
            "task_paths": yaml_paths,
            "log_path": log_pattern.replace("%J", job_id),
            "command": " ".join(bsub_cmd),
            "message": f"Task {job_name} submitted as job {job_id}"
        }

    except Exception as e:
        logger.error(f"Submission error: {str(e)}")
        return {"success": False, "error": str(e)}
