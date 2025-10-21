"""
CLI for running Fly models via YAML configuration.

This module provides a command-line interface for running multiple Fly models
based on a YAML configuration file. It supports custom input/output sizes,
normalization, and postprocessing.

Usage:
    cellmap_flow_fly /path/to/config.yaml

YAML Configuration Format:
    input: /path/to/input.zarr
    queue: gpu_h100
    charge_group: cellmap  # Project/billing group
    json_data: /path/to/norm_postprocess.json  # Optional
    input_size: [178, 178, 178]  # Optional, default shown
    output_size: [56, 56, 56]  # Optional, default shown
    runs:
      model_name:
        checkpoint: /path/to/checkpoint.pth
        classes: [channel1, channel2]
        res: 8  # Resolution in nm
        scale: s1  # Optional scale level
"""

import os
import yaml


from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.ds import find_target_scale
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url
import threading
from cellmap_flow.globals import Flow
from cellmap_flow.utils.data import FlyModelConfig
from cellmap_flow.utils.serilization_utils import get_process_dataset
from cellmap_flow.globals import g

import sys


def main():
    g = Flow()
    args = sys.argv[1:]
    yaml_file = args[0]
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    zarr_grp_path = data["input"]
    json_data = data.get("json_data", None)

    if json_data:
        g.input_norms, g.postprocess = get_process_dataset(json_data)

    queue = data.get("queue", "gpu_h100")
    if "charge_group" not in data:
        raise ValueError("charge_group is required in the YAML file")
    charge_group = data["charge_group"]
    input_size = tuple(data.get("input_size", (178, 178, 178)))
    output_size = tuple(data.get("output_size", (56, 56, 56)))
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
            input_size=input_size,
            output_size=output_size,
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
