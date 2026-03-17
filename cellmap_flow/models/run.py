from cellmap_flow.globals import g


from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    kill_n_remove_from_neuroglancer,
    get_norms_post_args,
)
from cellmap_flow.models.models_config import HuggingFaceModelConfig
import neuroglancer
import threading
from typing import List
import re
import logging

logger = logging.getLogger(__name__)


def _sanitize_job_name(name: str) -> str:
    """Replace spaces and hyphens with underscores for bsub job names."""
    return re.sub(r"[\s\-]+", "_", name)


def run_model(model_path, name, st_data):
    if model_path is None or model_path == "":
        logger.error(f"Model path is empty for {name}")
        return
    command = (
        f"{SERVER_COMMAND} cellmap --folder-path {model_path} --name {name} -d {g.dataset_path}"
    )
    logger.error(f"To be submitted command : {command}")
    job = start_hosts(
        command, job_name=name, queue=g.queue, charge_group=g.charge_group
    )
    with g.viewer.txn() as s:
        s.layers[job.model_name] = neuroglancer.ImageLayer(
            source=f"zarr://{job.host}/{job.model_name}{ARGS_KEY}{st_data}{ARGS_KEY}",
            shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){{emitRGB(color * normalized());}}""",
        )


def run_hf_model(repo, name, st_data):
    """Run a Hugging Face model by repo ID."""
    name = _sanitize_job_name(name)
    command = (
        f"{SERVER_COMMAND} huggingface --repo {repo} --name {name} -d {g.dataset_path}"
    )
    logger.error(f"To be submitted HF command : {command}")
    job = start_hosts(
        command, job_name=name, queue=g.queue, charge_group=g.charge_group
    )
    with g.viewer.txn() as s:
        s.layers[job.model_name] = neuroglancer.ImageLayer(
            source=f"zarr://{job.host}/{job.model_name}{ARGS_KEY}{st_data}{ARGS_KEY}",
            shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){{emitRGB(color * normalized());}}""",
        )


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
        # Launch local catalog models
        for _, group in g.model_catalog.items():
            for name, model_path in group.items():
                if name in names and name not in names_running:
                    logger.error(f"To be submitted model : {model_path}")
                    thread = threading.Thread(
                        target=run_model, args=(model_path, name, st_data)
                    )
                    thread.start()
                    threads.append(thread)

        # Launch Hugging Face models
        for repo in hf_repos:
            hf_name = _sanitize_job_name(repo.split("/")[-1])
            if hf_name not in names_running:
                logger.error(f"To be submitted HF model : {repo}")
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
