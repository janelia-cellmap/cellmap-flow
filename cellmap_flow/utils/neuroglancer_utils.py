import neuroglancer
import itertools
import logging

from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.utils.scale_pyramid import get_raw_layer
import cellmap_flow.globals as g
from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    INPUT_NORM_DICT_KEY,
    POSTPROCESS_DICT_KEY,
    encode_to_str,
    )

logger = logging.getLogger(__name__)

neuroglancer.set_server_bind_address("0.0.0.0")

def list_cls_to_dict(ll):
    args = {}
    norms = {}
    for n in ll:
        name = n.name()
        elms = n.to_dict()
        elms.pop("name")
        elms = {k: str(v) for k, v in elms.items()}
        norms[name] = elms

    return norms

def get_norms_post_args():
    args = {}
    
    args[INPUT_NORM_DICT_KEY] = list_cls_to_dict(g.input_norms)
    args[POSTPROCESS_DICT_KEY] = list_cls_to_dict(g.postprocess)
    st_data = encode_to_str(args)
    return st_data

def generate_neuroglancer_url(dataset_path, inference_dict):
    g.viewer = neuroglancer.Viewer()
    g.dataset_path = dataset_path
    st_data = get_norms_post_args()

    # Add a layer to the viewer
    with g.viewer.txn() as s:
        if dataset_path.startswith("/"):
            g.raw = get_raw_layer(dataset_path)
            s.layers["raw"] = g.raw
        else:
            if ".zarr" in dataset_path:
                filetype = "zarr"
            elif ".n5" in dataset_path:
                filetype = "n5"
            else:
                filetype = "precomputed"
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{filetype}://{dataset_path}",
            )
        colors = [
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "magenta",
        ]
        color_cycle = itertools.cycle(colors)
        for host, model in inference_dict.items():
            g.models_host[model] = host
            color = next(color_cycle)
            s.layers[model] = neuroglancer.ImageLayer(
                source=f"n5://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
    #uicontrol vec3 color color(default="{color}");
    void main(){{emitRGB(color * normalized());}}""",
            )
    # show(viewer)
    viewer_url = str(g.viewer)
    # .replace("zouinkhim-lm1", "192.168.1.167")
    print("viewer", viewer_url)
    url = create_and_run_app(neuroglancer_url=viewer_url)
    show(url)
    return url


def show(viewer):
    print()
    print()
    print("**********************************************")
    print("LINK:")
    print(viewer)
    print("**********************************************")
    print()
    print()
