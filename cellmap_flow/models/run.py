from cellmap_flow.globals import g

from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    kill_n_remove_from_neuroglancer,
    get_norms_post_args,
)
import neuroglancer
import threading
from typing import List
import logging

logger = logging.getLogger(__name__)


def run_model(model_path, name, st_data):
    if model_path is None or model_path == "":
        logger.error(f"Model path is empty for {name}")
        return
    command = (
        f"{SERVER_COMMAND} cellmap-model -f {model_path} -n {name} -d {g.dataset_path}"
    )
    logger.error(f"To be submitted command : {command}")
    job = start_hosts(
        command, job_name=name, queue=g.queue, charge_group=g.charge_group
    )
    with g.viewer.txn() as s:
        s.layers[job.model_name] = neuroglancer.ImageLayer(
            source=f"n5://{job.host}/{job.model_name}{ARGS_KEY}{st_data}{ARGS_KEY}",
            shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){{emitRGB(color * normalized());}}""",
        )


def update_run_models(names: List[str]):

    to_be_killed = [j for j in g.jobs if j.model_name not in names]
    names_running = [j.model_name for j in g.jobs]

    threads = []
    st_data = get_norms_post_args(g.input_norms, g.postprocess)

    print(f"Current catalog: {g.model_catalog}")
    with g.viewer.txn() as s:
        kill_n_remove_from_neuroglancer(to_be_killed, s)
        for _, group in g.model_catalog.items():
            for name, model_path in group.items():
                if name in names and name not in names_running:
                    logger.error(f"To be submitted model : {model_path}")
                    thread = threading.Thread(
                        target=run_model, args=(model_path, name, st_data)
                    )
                    thread.start()
                    threads.append(thread)
    # for thread in threads:
    #     thread.join()
