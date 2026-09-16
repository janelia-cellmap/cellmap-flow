import neuroglancer
import itertools
import logging

from cellmap_flow.dashboard.app import create_and_run_app
from cellmap_flow.utils.output_probe import output_display_range
from cellmap_flow.utils.scale_pyramid import (
    PREDICTION_COLORS,
    get_raw_layer,
    prediction_shader,
)
from cellmap_flow.utils.server_info import fetch_model_info
from cellmap_flow.utils.ds import find_closest_scale, get_scale_info, _open_zarr
from cellmap_flow.utils import zarr_v3
from cellmap_flow.globals import g
import os

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
        v3_container = zarr_v3.find_v3_container(dataset_path)
        if v3_container is not None and zarr_v3.multiscales_from_group(v3_container) is not None:
            _, resolutions, _ = zarr_v3.get_scale_info_v3(v3_container)
            target_scale, _, _ = zarr_v3.find_closest_scale_v3(v3_container, target_resolution)
            return tuple(resolutions[target_scale])
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
        color_cycle = itertools.cycle(PREDICTION_COLORS)
        for job in g.jobs:
            model = job.model_name
            host = job.host
            color = next(color_cycle)
            # Over the range the postprocessing chain actually produces. The
            # previous default was range=[0.5, 0.5]: lo == hi turns invlerp
            # into a step at 0.5, so after a DefaultPostprocessor (0-255) the
            # whole prediction rendered as solid colour.
            # One round trip, used for both the contrast range and the voxel
            # size below.
            info = fetch_model_info(host)
            try:
                steps = [
                    p.to_dict() for p in (g.postprocess or []) if hasattr(p, "to_dict")
                ]
                value_range = output_display_range(steps, info.get("output_class"))
            except Exception as e:
                logger.debug(f"Could not compute a display range for {model}: {e}")
                value_range = None
            default_shader = prediction_shader(color, value_range)
            shader = g.shaders.get(model, default_shader)
            if model not in g.shaders:
                g.shaders[model] = default_shader

            # Lie about the prediction's voxel size so it overlays the raw
            # at the closest available scale (model trained at 16nm but raw
            # is multiscale 6/12/24/...; we tell neuroglancer "treat the
            # output as 12nm" so it lines up).
            override_scales = None
            try:
                # Prefer the running server's answer. mc.config would build the
                # model here just to read a voxel size, which for a script model
                # means downloading weights and taking a CUDA context -- it
                # throws on a node without a free one, and the exception was
                # swallowed, silently leaving the overlay misaligned.
                output_voxel_size = info.get("output_voxel_size")
                if not output_voxel_size:
                    mc = model_configs_by_name.get(model)
                    if mc is not None:
                        output_voxel_size = mc.config.output_voxel_size
                if output_voxel_size:
                    output_voxel_size = tuple(output_voxel_size)
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
