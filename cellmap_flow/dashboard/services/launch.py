"""Start the models picked on the Models tab, and show each in the viewer.

- ``update_run_models()``: stop and forget the models no longer picked,
  start the ones newly picked, each in its own thread.
- ``run_model()`` / ``run_hf_model()``: start one catalog or Hugging Face
  model's inference server (``start_hosts``, which records the job) and add
  its layer, the one Submit would give it.
"""

from cellmap_flow.globals import g
from cellmap_flow.serving.launch import server_argv_for
from cellmap_flow.utils.bsub_utils import JobStartError, start_hosts
from cellmap_flow.utils.scale_pyramid import PREDICTION_COLORS
from cellmap_flow.utils.web_utils import (
    kill_n_remove_from_neuroglancer,
    get_norms_post_args,
)
from cellmap_flow.models.models_config import HuggingFaceModelConfig
from cellmap_flow.viewer.layers import prediction_layer
import threading
from typing import List
import re
import shlex
import logging

logger = logging.getLogger(__name__)


def _sanitize_job_name(name: str) -> str:
    """Replace spaces and hyphens with underscores for bsub job names."""
    return re.sub(r"[\s\-]+", "_", name)


def _start(command, name):
    """start_hosts(), or None after logging why the job did not start.

    These run in the dashboard's launch threads: an uncaught JobStartError
    (including a bsub timeout) only reached stderr, never the log panel.
    """
    try:
        return start_hosts(
            command, job_name=name, queue=g.queue, charge_group=g.charge_group
        )
    except JobStartError as e:
        logger.error(f"Could not start model '{name}': {e}")
        return None


def _show(job, st_data):
    """Add the started model's layer, the one Submit would give it."""
    names = [j.model_name for j in g.jobs]
    index = names.index(job.model_name) if job.model_name in names else 0
    layer = prediction_layer(
        job.model_name, job.host, st_data, dataset_path=g.dataset_path, postprocess=g.postprocess,
        shader=g.shaders.get(job.model_name), shader_controls=g.shader_controls.get(job.model_name),
        color=PREDICTION_COLORS[index % len(PREDICTION_COLORS)],
    )
    with g.viewer.txn() as s:
        s.layers[job.model_name] = layer


def run_model(model_path, name, st_data):
    if model_path is None or model_path == "":
        logger.error(f"Model path is empty for {name}")
        return
    command = shlex.join(
        server_argv_for("cellmap", {"folder_path": model_path, "name": name}, g.dataset_path)
    )
    logger.info(f"To be submitted command : {command}")
    job = _start(command, name)
    if job is not None:
        _show(job, st_data)


def run_hf_model(repo, name, st_data):
    """Run a Hugging Face model by repo ID."""
    name = _sanitize_job_name(name)
    command = shlex.join(
        server_argv_for("huggingface", {"repo": repo, "name": name}, g.dataset_path)
    )
    logger.info(f"To be submitted HF command : {command}")
    job = _start(command, name)
    if job is not None:
        _show(job, st_data)


def update_run_models(names: List[str], hf_repos: List[str] = None):

    if hf_repos is None:
        hf_repos = []

    all_names = names + [_sanitize_job_name(repo.split("/")[-1]) for repo in hf_repos]
    to_be_killed = [j for j in g.jobs if j.model_name not in all_names]
    names_running = [j.model_name for j in g.jobs]

    threads = []
    st_data = get_norms_post_args(g.input_norms, g.postprocess)

    print(f"Current catalog: {g.model_catalog}")
    with g.viewer.txn() as s:
        kill_n_remove_from_neuroglancer(to_be_killed, s)
        # Forget them too: a killed job left in g.jobs still counts as
        # running, so selecting that model again did nothing, and
        # /api/process kept rebuilding layers pointing at its dead host.
        g.jobs = [j for j in g.jobs if j not in to_be_killed]
        # Launch local catalog models
        for _, group in g.model_catalog.items():
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
                existing_names = [getattr(mc, 'name', None) for mc in g.models_config]
                if hf_name not in existing_names:
                    g.models_config.append(hf_config)
                thread = threading.Thread(
                    target=run_hf_model, args=(repo, hf_name, st_data)
                )
                thread.start()
                threads.append(thread)
    # for thread in threads:
    #     thread.join()
