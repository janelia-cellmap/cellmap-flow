"""Start the models picked on the Models tab, and show each in the viewer.

- ``update_run_models()``: stop and forget the models no longer picked,
  start the ones newly picked, each in its own thread. A finetune job's
  server is the Finetune tab's to stop, never this one's.
- ``run_model()`` / ``run_hf_model()``: start one catalog or Hugging Face
  model's inference server (``start_hosts``, which records the job) and add
  its layer, the one Submit would give it. Each is served from its type's
  ``default_env``, when the type sets one (``serving.launch.server_argv_for``).
"""

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.serving.launch import server_command_for
from cellmap_flow.jobs.launch import start_hosts
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.viewer.raw import PREDICTION_COLORS
from cellmap_flow.models.models_config import HuggingFaceModelConfig
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.viewer.layers import prediction_layer
import threading
from typing import List
import re
import logging

logger = logging.getLogger(__name__)


def _sanitize_job_name(name: str) -> str:
    """A Hugging Face model's name: its repo's last part, spaces and hyphens
    made underscores. It names the bsub job, the viewer layer and the
    model's config. (routes/blockwise has a _sanitize_job_name of its own,
    which keeps hyphens in a task's name: making the two one would rename
    either the Hugging Face models or the blockwise tasks.)"""
    return re.sub(r"[\s\-]+", "_", name)


def _start(command, name):
    """start_hosts(), or None after logging why the job did not start.

    These run in the dashboard's launch threads: an uncaught JobStartError
    (including a bsub timeout) only reached stderr, never the log panel.
    """
    session = get_session()
    try:
        return start_hosts(
            command, job_name=name, queue=session.queue, charge_group=session.charge_group
        )
    except JobStartError as e:
        logger.error(f"Could not start model '{name}': {e}")
        return None


def _show(job, st_data):
    """Add the started model's layer, the one Submit would give it."""
    session = get_session()
    names = [j.model_name for j in session.jobs]
    index = names.index(job.model_name) if job.model_name in names else 0
    layer = prediction_layer(
        job.model_name, job.host, st_data, dataset_path=session.dataset_path, postprocess=session.postprocess,
        shader=session.shaders.get(job.model_name), shader_controls=session.shader_controls.get(job.model_name),
        color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)],
    )
    with session.viewer.txn() as s:
        s.layers[job.model_name] = layer


def run_model(model_path, name, st_data):
    if model_path is None or model_path == "":
        logger.error(f"Model path is empty for {name}")
        return
    session = get_session()
    command = server_command_for("cellmap", {"folder_path": model_path, "name": name}, session.dataset_path, session.resample)
    logger.info(f"To be submitted command : {command}")
    job = _start(command, name)
    if job is not None:
        _show(job, st_data)


def run_hf_model(repo, name, st_data):
    """Run a Hugging Face model by repo ID."""
    name = _sanitize_job_name(name)
    session = get_session()
    command = server_command_for("huggingface", {"repo": repo, "name": name}, session.dataset_path, session.resample)
    logger.info(f"To be submitted HF command : {command}")
    job = _start(command, name)
    if job is not None:
        _show(job, st_data)


def kill_n_remove_from_neuroglancer(jobs, s):
    """Kill ``jobs`` and drop their layers from the viewer state ``s``."""
    for job in jobs:
        if job.model_name in s.layers:
            del s.layers[job.model_name]
        job.kill()


def update_run_models(names: List[str], hf_repos: List[str] = None):
    session = get_session()
    if hf_repos is None:
        hf_repos = []

    all_names = names + [_sanitize_job_name(repo.split("/")[-1]) for repo in hf_repos]
    # Not a finetune job's server (finetune_layers marks it): it is the
    # training job itself, and its name, new with each iteration, has no
    # box on a Models tab rendered before it, so every Submit bkilled it.
    to_be_killed = [
        j for j in session.jobs if j.model_name not in all_names and not getattr(j, "owned_by_finetune", False)
    ]
    names_running = [j.model_name for j in session.jobs]

    threads = []
    st_data = PipelineSpec.from_steps(session.input_norms, session.postprocess).to_url_blob()

    print(f"Current catalog: {session.model_catalog}")
    with session.viewer.txn() as s:
        kill_n_remove_from_neuroglancer(to_be_killed, s)
        # Forget them too: a killed job left in the jobs still counts as
        # running, so selecting that model again did nothing, and each
        # PUT /api/pipeline rebuilt a layer pointing at its dead host.
        session.jobs = [j for j in session.jobs if j not in to_be_killed]
        # Launch local catalog models
        for _, group in session.model_catalog.items():
            for name, model_path in group.items():
                if name in names and name not in names_running:
                    logger.info(f"To be submitted model : {model_path}")
                    thread = threading.Thread(
                        target=run_model, args=(model_path, name, st_data)
                    )
                    thread.start()
                    threads.append(thread)

        # Launch Hugging Face models
        for repo in hf_repos:
            hf_name = _sanitize_job_name(repo.split("/")[-1])
            if hf_name not in names_running:
                logger.info(f"To be submitted HF model : {repo}")
                # Create and store HuggingFaceModelConfig for pipeline builder
                hf_config = HuggingFaceModelConfig(repo=repo, name=hf_name)
                existing_names = [getattr(mc, 'name', None) for mc in session.models_config]
                if hf_name not in existing_names:
                    session.models_config.append(hf_config)
                thread = threading.Thread(
                    target=run_hf_model, args=(repo, hf_name, st_data)
                )
                thread.start()
                threads.append(thread)
    # for thread in threads:
    #     thread.join()
