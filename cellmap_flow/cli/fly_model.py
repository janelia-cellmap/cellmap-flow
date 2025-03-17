#%%
yaml_file = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/flow/generate_flow/nuc.yaml"
#%%
queue = "gpu_short"
charge_group = "cellmap"
#%%
import argparse
import os
import yaml
import zarr 

from funlib.geometry.coordinate import Coordinate

from cellmap_flow.utils.bsub_utils import start_hosts, SERVER_COMMAND
from cellmap_flow.utils.neuroglancer_utils import generate_neuroglancer_url
import threading
import cellmap_flow.globals as g
from cellmap_flow.utils.data import FlyModelConfig
def get_scale_info(zarr_grp):
    attrs = zarr_grp.attrs
    resolutions = {}
    offsets = {}
    shapes = {}
    for scale in attrs["multiscales"][0]["datasets"]:
        resolutions[scale["path"]] = scale["coordinateTransformations"][0]["scale"]
        offsets[scale["path"]] = scale["coordinateTransformations"][1]["translation"]
        shapes[scale["path"]] = zarr_grp[scale["path"]].shape
    return offsets, resolutions, shapes


def find_target_scale(zarr_grp_path, target_resolution):
    zarr_grp = zarr.open(zarr_grp_path,mode="r")
    offsets, resolutions, shapes = get_scale_info(zarr_grp)
    target_scale = None
    for scale, res in resolutions.items():
        if Coordinate(res) == Coordinate(target_resolution):
            target_scale = scale
            break
    if target_scale is None:
        msg = f"Zarr {zarr_grp.store.path}, {zarr_grp.path} does not contain array with sampling {target_resolution}"
        raise ValueError(msg)
    return target_scale, offsets[target_scale], shapes[target_scale]


def main(yaml_file):
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
        scale,_,_ = find_target_scale(zarr_grp_path, res)
        print(scale)
        data_path = os.path.join(zarr_grp_path, scale)
        model_config = FlyModelConfig(
            chpoint_path=run_items["checkpoint"],
            channels=run_items["classes"],
            input_voxel_size=res,
            output_voxel_size=res,
            name=run_name,
        )
        model_command = f"fly -c {model_config.chpoint_path} -ch {','.join(model_config.channels)} -ivs {','.join(map(str,model_config.input_voxel_size))} -ovs {','.join(map(str,model_config.output_voxel_size))}"
        command = f"{SERVER_COMMAND} {model_command} -d {data_path}"
        thread = threading.Thread(target=start_hosts, args=(command, queue, charge_group, model_config.name))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    generate_neuroglancer_url(data_path)
    while True:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fly model")
    parser.add_argument("yaml_file", type=str, help="Path to the YAML file")
    args = parser.parse_args()
    main(args.yaml_file)