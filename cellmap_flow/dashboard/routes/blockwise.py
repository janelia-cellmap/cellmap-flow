"""The pipeline builder's blockwise steps: validate, generate (the task YAML
the blockwise CLI runs), precheck, and submit (the task's master, to LSF).

Each answers 200 whatever happens, with "valid" (validate) or "success"
false beside an "error" when it cannot go on: the builder reads that flag at
each step and shows the error. A body a step cannot take is refused before
anything is done (the requests.Blockwise* models); a failure after that,
writing the YAML, a precheck or bsub, is answered the same way.
"""

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

from cellmap_flow.dashboard.requests import (
    BlockwiseGenerate,
    BlockwisePrecheck,
    BlockwiseSubmit,
    BlockwiseValidate,
    check,
)
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import JobSpec
from cellmap_flow.serving.protocol import INPUT_NORM_KEY, POSTPROCESS_KEY

logger = logging.getLogger(__name__)

blockwise_bp = Blueprint("blockwise", __name__)


def _task_walltime():
    """The LSF run limit for a blockwise task's master and its workers.

    The dashboard's walltime setting, as for inference servers. Without -W a
    GPU worker is killed at the queue's two-hour default.
    """
    return get_session().walltime or current_site().default_walltime


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
    """``paths`` if there are some and each is a file, else None."""
    if not paths or not all(os.path.isfile(p) for p in paths):
        return None
    return list(paths)


def _read(model, flag):
    """``(the request's body as model, None)``, or ``(None, the answer)`` to a
    body it is not: a 200 with ``flag`` false and what is wrong, which is how
    the builder learns a step failed (not requests.parse()'s 400)."""
    body, error = check(model, request.get_json(silent=True))
    return body, (None if error is None else {flag: False, "error": error})


@blockwise_bp.route("/api/blockwise/validate", methods=["POST"])
def validate_blockwise():
    """Whether the builder's pipeline is ready for blockwise processing, as
    requests.BlockwisePipeline describes it: {"valid": True, "message"}, or
    {"valid": False, "error"}."""
    _, refused = _read(BlockwiseValidate, "valid")
    if refused:
        return refused
    logger.info("Pipeline validation passed")
    return {"valid": True, "message": "Pipeline is ready for blockwise processing"}


# A model's fields that are lists, which the builder's text fields may send
# as the text typed.
_LIST_FIELDS = ("channels", "input_size", "output_size", "input_voxel_size", "output_voxel_size")


def _list_from_text(value):
    """``value`` as a list when it is a list or tuple, or text that reads as
    one: "[mito, er]" or "(1, 2)", perhaps in quotes, bare words taken as
    strings. Anything else, and text that does not parse, is kept as it is."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if not isinstance(value, str):
        return value
    text = value.strip().strip("'\"")
    if not (text.startswith(("[", "(")) and text.endswith(("]", ")"))):
        return value
    # Quote the bare words ([mito] -> ['mito']), then undo what that does to
    # words already quoted (''mito'' -> 'mito').
    text = re.sub(r"''+", "'", re.sub(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", r"'\1'", text))
    try:
        parsed = ast.literal_eval(text)
    except Exception as e:
        logger.warning(f"Could not read {value!r} as a list: {e}")
        return value
    return list(parsed) if isinstance(parsed, (list, tuple)) else value


def _model_entry(model):
    """A model node's entry in the task YAML: its name, then its params (or,
    without params, the config it was defined with), the list fields as lists."""
    entry = {"name": model.name, **model.settings()}
    for field in _LIST_FIELDS:
        if field in entry:
            entry[field] = _list_from_text(entry[field])
    return entry


def _chain_steps(steps):
    """A chain in the task YAML's ordered form, ``[{name, **params}]``. (A dict
    keyed by name kept only the last of two steps with the same name.) A node
    without a name is left out."""
    return [{"name": step.name, **(step.params or {})} for step in steps if step.name]


def _task_text(task):
    """The task YAML: keys in the order they were added, lists in block style."""
    return yaml.dump(task, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _write_task(tasks_dir, task):
    """Write ``task`` to <tasks_dir>/<its task_name>.yaml; the path."""
    path = os.path.join(tasks_dir, f"{task['task_name']}.yaml")
    with open(path, "w") as f:
        f.write(_task_text(task))
    logger.info(f"Generated blockwise task YAML at: {path}")
    return path


def _generate(pipeline, job_name):
    """Write the task YAML(s) for ``pipeline`` (a requests.BlockwisePipeline);
    generate's answer. A task named after ``job_name`` and the time."""
    settings = pipeline.blockwise_config[0].params
    input_params = pipeline.inputs[0].params
    output_params = pipeline.outputs[0].params

    # The output is a zarr: without a trailing slash, and with .zarr added
    # when the path has none.
    output_path = output_params["dataset_path"].rstrip("/\\")
    if ".zarr" not in output_path:
        output_path += ".zarr"

    # The job name the user typed, if any, names the task: the YAML, the
    # master job, the daisy task, the workers and the logs.
    task_name = _make_task_name(job_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    task = {
        "data_path": input_params["dataset_path"],
        "output_path": output_path,
        "task_name": task_name,
        "charge_group": settings.charge_group,
        "queue": settings.queue,
        "workers": settings.nb_workers,
        "cpu_workers": settings.nb_cores_worker,
        "tmp_dir": settings.tmp_dir,
        # Each worker's -W; the master gets the same (see submit).
        "walltime": _task_walltime(),
        "models": [_model_entry(model) for model in pipeline.models],
    }

    bounding_boxes = input_params.get("bounding_boxes", [])
    if bounding_boxes and isinstance(bounding_boxes, list):
        task["bounding_boxes"] = bounding_boxes
    separate_zarrs = input_params.get("separate_bounding_boxes_zarrs", False)
    if separate_zarrs:
        task["separate_bounding_boxes_zarrs"] = True
    if len(pipeline.models) > 1 and pipeline.model_mode:
        task["model_mode"] = pipeline.model_mode
    if pipeline.normalizers or pipeline.postprocessors:
        task["json_data"] = {
            INPUT_NORM_KEY: _chain_steps(pipeline.normalizers),
            POSTPROCESS_KEY: _chain_steps(pipeline.postprocessors),
        }
    output_channels = output_params.get("output_channels", [])
    if output_channels and isinstance(output_channels, list):
        task["output_channels"] = output_channels

    tasks_dir = get_session().tasks_dir()
    if separate_zarrs and bounding_boxes:
        # One task per box, each writing its own box_<n> zarr in the output.
        task_paths = [
            _write_task(tasks_dir, {
                **task,
                "bounding_boxes": [bbox],
                "output_path": os.path.join(output_path, f"box_{n}"),
                "task_name": f"{task_name}_box{n}",
            })
            for n, bbox in enumerate(bounding_boxes, start=1)
        ]
    else:
        task_paths = [_write_task(tasks_dir, task)]

    return {
        "success": True,
        # The task before it is split per box, for the builder's console.
        "task_yaml": _task_text(task),
        "task_config": task,
        "task_paths": task_paths,
        "task_name": task_name,
        "message": "Blockwise task generated successfully"
    }


@blockwise_bp.route("/api/blockwise/generate", methods=["POST"])
def generate_blockwise_task():
    """Write the task YAML the blockwise CLI runs, for the builder's pipeline;
    one per bounding box when each box gets its own zarr.

    {"success": True, "task_paths", "task_name", "task_yaml", "task_config",
    "message"}, or {"success": False, "error"}.
    """
    body, refused = _read(BlockwiseGenerate, "success")
    if refused:
        return refused
    try:
        return _generate(body.pipeline, body.job_name)
    except Exception as e:
        logger.error(f"Task generation error: {str(e)}")
        return {"success": False, "error": str(e)}


@blockwise_bp.route("/api/blockwise/precheck", methods=["POST"])
def precheck_blockwise_task():
    """Check the task YAMLs generate wrote, as blockwise_processor.precheck
    does: {"success": True, "message": "success"}, or {"success": False,
    "error"} with the first one's problem."""
    body, refused = _read(BlockwisePrecheck, "success")
    if refused:
        return refused
    try:
        # precheck() rather than constructing the processor: that created the
        # output arrays, loaded every model into this process, and replaced
        # the dashboard's live chain.
        from cellmap_flow.blockwise.blockwise_processor import precheck

        for yaml_path in body.yaml_paths:
            precheck(yaml_path)
        logger.info(f"Blockwise precheck passed for: {', '.join(body.yaml_paths)}")
        return {"success": True, "message": "success"}

    except Exception as e:
        logger.error(f"Blockwise precheck failed: {str(e)}")
        return {"success": False, "error": str(e)}


@blockwise_bp.route("/api/blockwise/submit", methods=["POST"])
def submit_blockwise_task():
    """Submit the task's master to LSF, which runs the task's workers.

    {"success": True, "job_id", "task_name", "task_paths", "log_path",
    "command", "message"}, or {"success": False, "error"}; "job_id" is
    "unknown" when bsub accepted the job without naming it.
    """
    body, refused = _read(BlockwiseSubmit, "success")
    if refused:
        return refused
    try:
        # Submit the YAMLs that /api/blockwise/generate wrote and
        # /api/blockwise/precheck checked, when the client sends them back.
        # Regenerating writes new files under a new task name, so what ran
        # was not what had been checked.
        requested = body.yaml_paths
        yaml_paths = _existing_task_paths(requested)
        # The name generate gave the YAMLs; the page sends it back with them.
        task_name = body.task_name
        if yaml_paths is not None:
            logger.info(f"Submitting the given task YAML(s): {', '.join(yaml_paths)}")
        else:
            if requested:
                logger.warning(
                    f"Not every given task YAML exists ({requested}); generating new ones"
                )
            generated = _generate(body.pipeline, body.job_name)
            yaml_paths, task_name = generated["task_paths"], generated["task_name"]
        settings = body.pipeline.blockwise_config[0].params

        # The master carries the task's name, so `bjobs -J <task>` is the
        # master and `bjobs -J "predict_*_<task>*"` are its workers.
        job_name = _sanitize_job_name(task_name or body.job_name) or (
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
            cpus=settings.nb_cores_master,
            charge_group=settings.charge_group,
            walltime=_task_walltime(),
            log_dir=get_session().tasks_dir(),
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
