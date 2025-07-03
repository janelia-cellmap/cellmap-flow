queue = "gpu_a100"
charge_group = "cellmap"

import os
import yaml


from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.ds import find_target_scale
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url
import threading
from cellmap_flow.globals import Flow
from cellmap_flow.utils.data import FlyModelConfig


import sys


def main():
    g = Flow()
    args = sys.argv[1:]
    yaml_file = args[0]
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    zarr_grp_path = data["input"]
    g.queue = queue
    g.charge_group = charge_group
    threads = []
    for run_name, run_items in data["runs"].items():
        print(run_name)
        print(run_items)
        res = run_items["res"]
        res = (res, res, res)
        print(res)
        scale = run_items.get("scale", None)
        if scale is None:
            scale, _, _ = find_target_scale(zarr_grp_path, res)
        print(scale)
        data_path = os.path.join(zarr_grp_path, scale)
        model_config = FlyModelConfig(
            checkpoint_path=run_items["checkpoint"],
            channels=run_items["classes"],
            input_voxel_size=res,
            output_voxel_size=res,
            name=run_name,
        )
        model_command = f"fly -c {model_config.checkpoint_path} -ch {','.join(model_config.channels)} -ivs {','.join(map(str,model_config.input_voxel_size))} -ovs {','.join(map(str,model_config.output_voxel_size))}"
        command = f"{SERVER_COMMAND} {model_command} -d {data_path}"
        thread = threading.Thread(
            target=start_hosts, args=(command, queue, charge_group, model_config.name)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    generate_neuroglancer_url(data_path)
    while True:
        pass
