"""The pipeline builder's blockwise steps: validate, generate (the task YAML
the blockwise CLI runs), precheck, and submit (the task's master, to LSF).

Each answers 200 whatever happens, with "valid" (validate) or "success"
false beside an "error" when it cannot go on: the builder reads that flag at
each step and shows the error. A body a step cannot take is refused before
anything is done (the requests.Blockwise* models); a failure after that,
writing the YAML, a precheck or bsub, is answered the same way.
"""

import json
import os
import re
import ast
import logging
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import yaml
from flask import Blueprint, request

from cellmap_flow.blockwise import max_block_size_cache
from cellmap_flow.blockwise.max_block_size import MaxBlockSizeResult
from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.dashboard.requests import (
    BlockwiseGenerate,
    BlockwisePrecheck,
    BlockwiseSubmit,
    BlockwiseValidate,
    MaxBlockSizeCancel,
    MaxBlockSizeRequest,
    MaxBlockSizeStatus,
    OutputDtypeRequest,
    check,
)
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.queues import queue_list
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import JobSpec, JobStatus
from cellmap_flow.models import registry
from cellmap_flow.serving.protocol import INPUT_NORM_KEY, POSTPROCESS_KEY

logger = logging.getLogger(__name__)

blockwise_bp = Blueprint("blockwise", __name__)


_WALLTIME = re.compile(r"^\d+(:[0-5]\d)?$")


def _task_walltime(settings=None):
    """The LSF run limit for a blockwise task's master and its workers.

    The Blockwise Config node's Walltime, when it names one ("HH:MM" or
    minutes, as bsub -W takes them), else the dashboard's walltime setting,
    as for inference servers. Without -W a GPU worker is killed at the
    queue's two-hour default.

    Raises:
        ValueError: the node's walltime is not something -W would take.
    """
    typed = str(getattr(settings, "walltime", None) or "").strip()
    if not typed:
        return get_session().walltime or current_site().default_walltime
    if not _WALLTIME.match(typed):
        raise ValueError(f"Walltime {typed!r} is not HH:MM or a number of minutes (08:00, or 480)")
    return typed


def _tasks_dir(settings=None):
    """Where a task's YAML and its jobs' logs go: the Blockwise Config node's
    Tasks Directory when it names one (made if missing), else the
    dashboard's. (The node's field was shown but never read: every task
    landed in the dashboard's directory whatever it said.)"""
    typed = str(getattr(settings, "blockwise_tasks_dir", None) or "").strip()
    if not typed:
        return get_session().tasks_dir()
    tasks_dir = os.path.abspath(os.path.expanduser(typed))
    os.makedirs(tasks_dir, exist_ok=True)
    return tasks_dir


def _block_size_problem(entries, block_size):
    """Why ``block_size`` (the output's chunk shape, voxels) does not go
    with the models' tiles, or None.

    Blockwise's blocks are each model's output tile; block_size sets the
    output's chunks, which the blocks are laid out on so that each writes
    chunks of its own (blockwise_processor.py). So a tile must be a whole
    number of chunks on every axis: a 336 tile may be written in 84s (for
    readers that want small chunks), not in 100s, and not in 392s. Checked
    here for the entries that say their tile (``output_size``: fly, onnx);
    a script's or a CellMap export's is read off the model, which the
    dashboard does not build, so the run itself checks those
    (CellMapFlowBlockwiseProcessor.__init__).
    """
    block = list(block_size)
    if not all(isinstance(v, (int, float)) and v == int(v) and v > 0 for v in block):
        return f"Block Size {block_size} must be whole, positive voxel counts"
    for entry in entries:
        output = entry.get("output_size")
        if not isinstance(output, (list, tuple)):
            continue
        name = entry.get("name") or entry.get("type") or "model"
        if len(output) != len(block):
            return f"Block Size {block_size} has {len(block)} sizes; {name}'s output_size {list(output)} has {len(output)}"
        if any(int(o) % int(b) for o, b in zip(output, block)):
            return (
                f"Block Size {block_size} must divide {name}'s output_size {list(output)} on every axis: a block "
                f"is one model tile, laid on the output's chunk grid, and a chunk two blocks share is written "
                f"by both at once. Leave it blank for chunks the size of the tile."
            )
    return None


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
    body, refused = _read(BlockwiseValidate, "valid")
    if refused:
        return refused
    settings = body.pipeline.blockwise_config[0].params
    block_size = _list_from_text(settings.block_size)
    if block_size and isinstance(block_size, list):
        problem = _block_size_problem([_model_entry(model) for model in body.pipeline.models], block_size)
        if problem:
            return {"valid": False, "error": problem}
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
    """Write the task YAML for ``pipeline`` (a requests.BlockwisePipeline);
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
        "walltime": _task_walltime(settings),
        "models": [_model_entry(model) for model in pipeline.models],
    }

    block_size = _list_from_text(settings.block_size)
    if block_size and isinstance(block_size, list):
        problem = _block_size_problem(task["models"], block_size)
        if problem:
            raise ValueError(problem)
        task["block_size"] = block_size

    # The Output node's dtype, when one is picked; blank leaves it to the
    # task (blockwise_processor: the last postprocessor's that declares one,
    # else the model's own), as /api/blockwise/output-dtype shows it.
    dtype_text = str(output_params.get("dtype") or "").strip()
    if dtype_text:
        try:
            task["dtype"] = np.dtype(dtype_text).name
        except TypeError:
            raise ValueError(f"Output dtype {dtype_text!r} is not a numpy dtype (uint8, uint16, float32, ...)")

    bounding_boxes = input_params.get("bounding_boxes", [])
    if bounding_boxes and isinstance(bounding_boxes, list):
        task["bounding_boxes"] = bounding_boxes
    separate_zarrs = input_params.get("separate_bounding_boxes_zarrs", False)
    if separate_zarrs:
        task["separate_bounding_boxes_zarrs"] = True
    # The dashboard's Resample setting, as the models it serves use it; a
    # task resamples unless it says resample: false.
    if not get_session().resample:
        task["resample"] = False
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

    # One task, whatever the boxes: with separate_bounding_boxes_zarrs the
    # blockwise run writes each box to its own <output>/box_<n>. (It took
    # one box, so this wrote a task per box, each with its own output path.)
    task_paths = [_write_task(_tasks_dir(settings), task)]

    return {
        "success": True,
        "task_yaml": _task_text(task),
        "task_config": task,
        "task_paths": task_paths,
        "task_name": task_name,
        "message": "Blockwise task generated successfully",
    }


@blockwise_bp.route("/api/blockwise/generate", methods=["POST"])
def generate_blockwise_task():
    """Write the task YAML the blockwise CLI runs, for the builder's pipeline.

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
            # Run as it stands: nothing in it needs a shell. -P keeps the
            # job's working directory off sys.path, where another cellmap-flow
            # checkout would shadow the installed one.
            argv=(sys.executable, "-P", "-m", "cellmap_flow.blockwise.multiple_cli", *yaml_paths),
            queue=None,
            gpus=0,
            cpus=settings.nb_cores_master,
            charge_group=settings.charge_group,
            walltime=_task_walltime(settings),
            log_dir=_tasks_dir(settings),
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


# --- max block size: how big a block this model's forward pass fits on the
# blockwise-config node's queue, instead of guessing or leaving the model's
# own (often much smaller) native output tile. The probe needs a real GPU of
# the queue's type, so it runs as a short job on that queue
# (blockwise.max_block_size_cli), not in the dashboard's own process. ------


def _max_block_size_model_config(pipeline):
    """The pipeline's first model, built: what the probe and the cache key
    off. Raises ConfigError for a model entry that cannot be built."""
    entry = _model_entry(pipeline.models[0])
    return registry.build_model(entry), entry


def _primary_queue(queue):
    """The one queue to probe and cache a block size for, out of what the
    Blockwise Config node's Queue field names (always a list by the time it
    reaches here, since the dashboard's Queue picker ticks several: "try
    gpu_h100, then gpu_h200" is normal there). Unlike a worker's real
    placement (blockwise_processor.spawn_worker), this is deliberately NOT
    resolved through jobs.queues.candidates' freest-now fallback: a block
    size is tied to one specific GPU type, and resolving it through
    current LSF load would make the cache key -- and so the probed answer
    -- depend on when it happened to be asked, rather than naming one
    queue consistently. None if queue names none."""
    try:
        names = queue_list(queue)
    except ValueError:
        return None
    return names[0] if names else None


@blockwise_bp.route("/api/blockwise/max-block-size/check", methods=["POST"])
def check_max_block_size():
    """The cached max block size for the pipeline's first model on its
    blockwise-config queue (the first, if several are named), if there is
    one: {"cached": True, "result": {...}}, or {"cached": False, "error"}
    when there is not (including a model or queue problem -- /compute
    reports the same thing again, so the builder shows it either way)."""
    body, refused = _read(MaxBlockSizeRequest, "cached")
    if refused:
        return refused
    try:
        queue = _primary_queue(body.pipeline.blockwise_config[0].params.queue)
        if not queue:
            return {"cached": False, "error": "Set a queue in the Blockwise Config node first"}
        model_config, _ = _max_block_size_model_config(body.pipeline)
        cached = max_block_size_cache.load_cached(model_config, queue)
        return {"cached": True, "result": cached} if cached is not None else {"cached": False}
    except Exception as e:
        logger.error(f"Max block size cache check error: {e}")
        return {"cached": False, "error": str(e)}


@blockwise_bp.route("/api/blockwise/max-block-size/compute", methods=["POST"])
def compute_max_block_size():
    """Submit a short probe job to the pipeline's blockwise-config queue.

    {"success": True, "job_id", "result_path", "model_entry", "queue",
    "log_path", "message"} to poll (/status below), or {"success": False,
    "error"}.
    """
    body, refused = _read(MaxBlockSizeRequest, "success")
    if refused:
        return refused
    try:
        settings = body.pipeline.blockwise_config[0].params
        queue = _primary_queue(settings.queue)
        if not queue:
            return {"success": False, "error": "Set a queue in the Blockwise Config node first"}
        model_config, entry = _max_block_size_model_config(body.pipeline)

        submitted_at = time.time()
        job_name = _sanitize_job_name(
            f"max_block_size_{entry.get('name') or entry.get('type') or 'model'}_{int(submitted_at)}"
        )
        tasks_dir = _tasks_dir(settings)
        result_path = os.path.join(tasks_dir, f"{job_name}.json")
        spec = JobSpec(
            name=job_name,
            # This interpreter, not whatever "python" is first on PATH: the
            # same reasoning as submit's own master job, above.
            argv=(
                sys.executable, "-P", "-m", "cellmap_flow.cli.main", "max-block-size",
                "--model", json.dumps(entry), "--out", result_path, "--queue", queue,
            ),
            queue=queue,
            charge_group=settings.charge_group,
            gpus=1,
            cpus=4,
            walltime=_task_walltime(),  # the probe takes a minute: the dashboard's, not a task's long -W
            log_dir=tasks_dir,
        )
        try:
            job_id = jobs_lsf.submit(spec, bsub_timeout=None).job_id
        except jobs_lsf.JobIdMissingError as e:
            logger.warning(f"Max block size probe submitted, but bsub gave no job id: {e.output}")
            job_id = "unknown"

        return {
            "success": True,
            "job_id": job_id,
            "result_path": result_path,
            "model_entry": entry,
            "queue": queue,
            "log_path": str(jobs_lsf.log_pattern(spec)).replace("%J", job_id),
            "submitted_at": submitted_at,
            "message": f"Probing the max block size for {entry.get('name') or entry.get('type')} on {queue}",
        }
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout
        logger.error(f"Max block size probe submission failed: {error_msg}")
        return {"success": False, "error": f"LSF error: {error_msg}"}
    except Exception as e:
        logger.error(f"Max block size compute error: {e}")
        return {"success": False, "error": str(e)}


# How long after LSF says a probe job ended its result may still turn up
# here. The job writes the file on a compute node a few seconds before LSF
# reports it done, and this host's NFS client can keep answering "no such
# file" from its cache for up to a minute after that (a poll that found
# nothing is remembered): a DONE job's result, written at 11:35:22 and
# reported done at 11:35:25, was invisible to the 11:35:26 poll and called a
# failure. A job that EXITed wrote nothing in all likelihood, so less; one
# LSF has forgotten ended long ago, so none.
RESULT_GRACE_SECONDS = {JobStatus.COMPLETED: 90, JobStatus.FAILED: 20}
# job id -> when this dashboard first saw LSF report it ended.
_ENDED_AT = {}


def _result_visible(path) -> bool:
    """Whether ``path`` exists -- after listing its directory, which makes an
    NFS client revalidate it, where a plain stat answers from a cache that
    can be a minute behind another host's write."""
    try:
        os.listdir(os.path.dirname(path) or ".")
    except OSError:
        pass
    return os.path.isfile(path)


def _fresh_cached_result(body):
    """The result the probe job itself stored in the cache (max_block_size_cli
    --queue) since compute submitted it, or None: a second way to its
    answer, when its result file has not shown up here yet."""
    if not (body.model_entry and body.queue and body.submitted_at):
        return None
    try:
        model_config = registry.build_model(body.model_entry)
        when = max_block_size_cache.stored_at(model_config, body.queue)
        if when is not None and when >= body.submitted_at - 1:
            return max_block_size_cache.load_cached(model_config, body.queue)
    except Exception as e:
        logger.debug(f"Could not read the max block size cache for job {body.job_id}: {e}")
    return None


@blockwise_bp.route("/api/blockwise/max-block-size/status", methods=["POST"])
def max_block_size_status():
    """Poll a job /compute submitted: {"done": False} while it may still be
    running (with a "note" once LSF says it ended and its result is being
    waited for), {"done": True, "result": {...}} once its result is here
    (its file, or the cache entry it stored; cached for next time), or
    {"done": True, "error": "..."} if it ended -- however it ended -- and
    no result followed within RESULT_GRACE_SECONDS."""
    # "success", not "done": _read's generic refusal is {flag: False, "error"},
    # and False here would tell the builder to keep polling a request that
    # was never going to produce a result -- translated to "done" below.
    body, refused = _read(MaxBlockSizeStatus, "success")
    if refused:
        return {"done": True, "error": refused["error"]}
    try:
        if _result_visible(body.result_path):
            with open(body.result_path) as f:
                result = json.load(f)
            if body.model_entry and body.queue:
                try:
                    model_config = registry.build_model(body.model_entry)
                    max_block_size_cache.store(model_config, body.queue, MaxBlockSizeResult.from_dict(result))
                except Exception as e:
                    logger.warning(f"Could not cache max block size result: {e}")
            _ENDED_AT.pop(body.job_id, None)
            return {"done": True, "result": result}
        cached = _fresh_cached_result(body)
        if cached is not None:
            _ENDED_AT.pop(body.job_id, None)
            return {"done": True, "result": cached}

        reported = jobs_lsf.statuses([body.job_id])
        if body.job_id not in reported:
            # bjobs could not be run just now (statuses tells that apart from
            # "not found"): nothing is known either way, so keep waiting.
            return {"done": False}
        status = reported[body.job_id]
        if status in (JobStatus.RUNNING, JobStatus.PENDING):
            return {"done": False}
        # Ended. Its result may still be on its way here (see
        # RESULT_GRACE_SECONDS); past that, it never wrote one -- a DONE job
        # too, which the builder once polled forever.
        how = f"status: {status.name}" if status is not None else "LSF no longer knows it"
        grace = RESULT_GRACE_SECONDS.get(status, 0)
        # first_seen before the clock is read again: the other way round,
        # the first poll's "waited" was a few microseconds negative, which is
        # less than a grace of zero.
        first_seen = _ENDED_AT.setdefault(body.job_id, time.time())
        waited = time.time() - first_seen
        if waited < grace:
            return {
                "done": False,
                "note": f"Job {body.job_id} ended ({how}); waiting up to {grace:.0f} s for its result to show up here",
            }
        _ENDED_AT.pop(body.job_id, None)
        where = f"; see its log, {body.log_path}" if body.log_path else ""
        return {"done": True, "error": f"Job {body.job_id} ended without writing a result ({how}){where}"}
    except Exception as e:
        logger.error(f"Max block size status error: {e}")
        return {"done": True, "error": str(e)}


@blockwise_bp.route("/api/blockwise/max-block-size/cancel", methods=["POST"])
def cancel_max_block_size():
    """bkill a probe job /compute submitted -- the queue is full, or the user
    changed their mind: {"success": True} once LSF has it (or the job had
    already ended), else {"success": False, "error"}."""
    body, refused = _read(MaxBlockSizeCancel, "success")
    if refused:
        return refused
    error = jobs_lsf.kill(body.job_id)
    if error:
        logger.warning(f"Could not kill max block size probe {body.job_id}: {error}")
        return {"success": False, "error": f"Could not kill job {body.job_id}: {error}"}
    logger.info(f"Killed max block size probe {body.job_id}")
    return {"success": True}


@blockwise_bp.route("/api/blockwise/output-dtype", methods=["POST"])
def output_dtype():
    """The dtype a task's output would be written in, as blockwise_processor
    decides it when the task does not say: the last postprocessor's that
    declares one (pipeline_spec.chain_output_dtype), else the model's own.

    {"success": True, "dtype": "uint8", "source": "ThresholdPostprocessor"},
    or {"dtype": None, "source": "model"} when only the model decides: its
    dtype is read off the built model, which the dashboard does not build
    (float32, for nearly every type).
    """
    from cellmap_flow.pipeline_spec import PipelineSpec, chain_output_dtype

    body, refused = _read(OutputDtypeRequest, "success")
    if refused:
        return refused
    try:
        steps = _chain_steps(body.pipeline.postprocessors)
        postprocess = PipelineSpec(postprocess=steps).build()[1] if steps else []
        dtype = chain_output_dtype(postprocess, None)
        if dtype is None:
            return {"success": True, "dtype": None, "source": "model"}
        # The last step that, on its own, answers a dtype is the one that set it.
        source = next(
            (type(step).__name__ for step in reversed(postprocess) if chain_output_dtype([step], None) is not None),
            "model",
        )
        return {"success": True, "dtype": np.dtype(dtype).name, "source": source}
    except Exception as e:
        logger.error(f"Output dtype error: {e}")
        return {"success": False, "error": str(e)}
