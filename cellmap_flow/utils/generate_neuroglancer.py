import os
import zarr
import yaml
import neuroglancer
from cellmap_flow.utils.scale_pyramid import get_raw_layer

neuroglancer.set_server_bind_address("0.0.0.0")

def generate_neuroglancer(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    predictions = [f for f in os.listdir(config["output_path"]) if os.path.isdir(os.path.join(config["output_path"], f))]
    output_path = os.path.join(config["output_path"])
    raw_path = config["data_path"]

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.layers["raw"] = get_raw_layer(raw_path)
        for pred in predictions:
            s.layers[pred] = get_raw_layer(os.path.join(output_path, pred))

    viewer_url = str(viewer)
    print("viewer", viewer_url)
    input("Press Enter to continue...")
    return viewer_url


def fix_all(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    predictions = [os.path.join(config["output_path"], f) for f in os.listdir(config["output_path"]) if os.path.isdir(os.path.join(config["output_path"], f))]
    for pred_path in predictions:
        c_path = pred_path
        while ".zarr" in c_path:
            print("Fixing ", c_path)
            zarr.open(c_path, mode="a") 
            c_path = os.path.dirname(c_path)

    