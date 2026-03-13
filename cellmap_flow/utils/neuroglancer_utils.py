import neuroglancer
import itertools
import logging

from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.globals import g
import os

from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    get_norms_post_args,
)


logger = logging.getLogger(__name__)

neuroglancer.set_server_bind_address("0.0.0.0")


def generate_neuroglancer_url(dataset_path,wrap_raw=True):
    g.viewer = neuroglancer.Viewer()
    g.dataset_path = dataset_path
    st_data = get_norms_post_args(g.input_norms, g.postprocess)

    # Add a layer to the viewer
    with g.viewer.txn() as s:
        g.raw = get_raw_layer(dataset_path, wrap_raw=wrap_raw)
        s.layers["data"] = g.raw
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
        for job in g.jobs:
            model = job.model_name
            host = job.host
            color = next(color_cycle)
            default_shader = f"""#uicontrol invlerp normalized(range=[0.5, 0.5], window=[0, 1]);
    #uicontrol vec3 color color(default="{color}");
    void main(){{emitRGB(color * normalized());}}"""
            shader = g.shaders.get(model, default_shader)
            if model not in g.shaders:
                g.shaders[model] = default_shader
            layer_kwargs = {
                "source": f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}",
                "shader": shader,
            }
            shader_controls = g.shader_controls.get(model)
            if shader_controls:
                layer_kwargs["shaderControls"] = shader_controls
            s.layers[model] = neuroglancer.ImageLayer(**layer_kwargs)
    # show(viewer)
    viewer_url = str(g.viewer)
    # .replace("zouinkhim-lm1", "192.168.1.167")
    print("viewer", viewer_url)
    url = create_and_run_app(neuroglancer_url=viewer_url)
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
