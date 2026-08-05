import neuroglancer
import itertools
import logging

from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.ds import find_closest_scale, get_scale_info, _open_zarr
from cellmap_flow.globals import g

from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    get_norms_post_args,
)


logger = logging.getLogger(__name__)

neuroglancer.set_server_bind_address("0.0.0.0")


def get_raw_closest_scale(dataset_path, target_resolution):
    """Return the raw multiscale scale (as a tuple of nm) closest to the
    model's target resolution, or None if it can't be determined."""
    try:
        zarr_grp = _open_zarr(dataset_path, mode="r")
        _, resolutions, _ = get_scale_info(zarr_grp)
        target_scale, _, _ = find_closest_scale(dataset_path, target_resolution)
        return tuple(resolutions[target_scale])
    except Exception as e:
        logger.warning(
            f"Could not determine closest raw scale for {dataset_path} at "
            f"target_resolution={target_resolution}: {e}"
        )
        return None


def build_prediction_source(host, model, st_data, override_scales):
    """Build a source spec for the prediction zarr that overrides the
    source dimensions' scales so the layer overlays the raw at its native
    resolution (e.g. claim a 16nm model output is actually at 12nm).

    The prediction zarr is 4D (z, y, x, c). We override the spatial scales
    and leave the channel dim as a unitless dimension.
    """
    url = f"zarr://{host}/{model}{ARGS_KEY}{st_data}{ARGS_KEY}"
    if override_scales is None:
        return url
    sx, sy, sz = override_scales[0], override_scales[1], override_scales[2]
    # Use a dict form so we can supply matching input/output dimensions
    # of the same rank (neuroglancer requires equal rank on both sides).
    return {
        "url": url,
        "transform": {
            "outputDimensions": {
                "z": [sz * 1e-9, "m"],
                "y": [sy * 1e-9, "m"],
                "x": [sx * 1e-9, "m"],
                "c^": [1, ""],
            },
            "inputDimensions": {
                "z": [sz * 1e-9, "m"],
                "y": [sy * 1e-9, "m"],
                "x": [sx * 1e-9, "m"],
                "c^": [1, ""],
            },
        },
    }


def generate_neuroglancer_url(dataset_path,wrap_raw=True):
    g.viewer = neuroglancer.Viewer()
    g.dataset_path = dataset_path
    st_data = get_norms_post_args(g.input_norms, g.postprocess)

    # Map model name -> ModelConfig for voxel-size lookups
    model_configs_by_name = {}
    for mc in getattr(g, "models_config", []) or []:
        model_configs_by_name[mc.name] = mc

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

            # Lie about the prediction's voxel size so it overlays the raw
            # at the closest available scale (model trained at 16nm but raw
            # is multiscale 6/12/24/...; we tell neuroglancer "treat the
            # output as 12nm" so it lines up).
            override_scales = None
            mc = model_configs_by_name.get(model)
            if mc is not None:
                try:
                    output_voxel_size = tuple(mc.lightweight_info()["output_voxel_size"])
                    closest = get_raw_closest_scale(dataset_path, output_voxel_size)
                    if closest is not None and tuple(closest) != output_voxel_size:
                        override_scales = closest
                        logger.info(
                            f"Model '{model}' output_voxel_size={output_voxel_size} "
                            f"overridden to closest raw scale {closest} for viewer overlay"
                        )
                except Exception as e:
                    logger.warning(f"Could not compute override scales for '{model}': {e}")

            source = build_prediction_source(host, model, st_data, override_scales)
            layer_kwargs = {
                "source": source,
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
