import neuroglancer
import itertools
import logging

from cellmap_flow.utils.scale_pyramid import get_raw_layer
from cellmap_flow.utils.ds import (
    _is_zarr_group,
    _join_path,
    _open_zarr,
    find_closest_scale,
    get_scale_info,
)
from cellmap_flow.globals import g

from cellmap_flow.utils.web_utils import (
    ARGS_KEY,
    get_norms_post_args,
)


logger = logging.getLogger(__name__)

neuroglancer.set_server_bind_address("0.0.0.0")


def _read_data_reference_scale(dataset_path):
    # Read the data layer's first multiscale CoordinateSpace from disk so the
    # caller can pin viewer.state.dimensions before any layers are added.
    # Returns None for non-zarr-multiscale sources (precomputed, single arrays).
    if dataset_path.startswith("precomputed://"):
        return None
    try:
        from cellmap_flow.image_data_interface import ImageDataInterface

        grp = _open_zarr(dataset_path, mode="r")
        if not _is_zarr_group(grp):
            return None

        multiscales = grp.attrs.get("multiscales", None)
        if multiscales and multiscales[0].get("datasets"):
            scale_path = multiscales[0]["datasets"][0].get("path")
        else:
            scales = sorted(
                [k for k in grp.keys() if k.startswith("s") and k[1:].isdigit()],
                key=lambda x: int(x[1:]),
            )
            if not scales:
                return None
            scale_path = scales[0]

        if scale_path in (None, "", "."):
            ref_path = dataset_path
        else:
            ref_path = _join_path(dataset_path, scale_path)
        ref = ImageDataInterface(ref_path, normalize=False)
        return neuroglancer.CoordinateSpace(
            names=list(ref.axes_names), units="nm", scales=list(ref.voxel_size),
        )
    except Exception as e:
        logger.warning(f"Could not read reference scale from {dataset_path}: {e}")
        return None


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
    if not host:
        logger.warning(f"No host known yet for model '{model}' -- skipping its layer")
        return None
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

    # Pin the viewer's global coordinate space to the data layer's smallest
    # scale BEFORE adding any layers. See _read_data_reference_scale.
    _ref_dims = _read_data_reference_scale(dataset_path)

    # Add a layer to the viewer
    with g.viewer.txn() as s:
        if _ref_dims is not None:
            s.dimensions = _ref_dims
            logger.info(f"Pinned viewer dimensions to {_ref_dims.to_json()}")

        # Cap NG client-side download concurrency at 32. Note: NG-Python's
        # `s.concurrent_downloads` writes chunkQueueManager.capacities.download
        # .itemLimit — the TOTAL queue cap (queued + in-flight), not just
        # in-flight. cap=8 starves the FOV; cap=32 fills cleanly without
        # OOM on a base-killed H100/H200. Tunable: 16 if co-tenanted with
        # base + multiple LoRA serves; 32 typical; 64-100 solo. The attr
        # bakes into the viewer-state JSON URL hash so the cap survives
        # tab reloads (unlike runtime overrides from the JS console).
        s.concurrent_downloads = 32

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
            default_shader = f"""#uicontrol invlerp normalized(range=[0, 0.5])
#uicontrol vec3 color color(default="{color}")
void main() {{
  float v = normalized();
  if (v <= 0.0)
    emitRGB(color * v);
//    emitTransparent();
  else emitRGB(color * v);
}}"""
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
                    output_voxel_size = tuple(mc.config.output_voxel_size)
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
            if source is None:
                continue
            layer_kwargs = {
                "source": source,
                "shader": shader,
                "blend": "additive",
            }
            shader_controls = g.shader_controls.get(model)
            if shader_controls:
                layer_kwargs["shaderControls"] = shader_controls
            s.layers[model] = neuroglancer.ImageLayer(**layer_kwargs)

        # Add extra startup layers (pre-built in yaml_cli from extra_layers
        # config). Entries are (layer, shader, blend) tuples; apply non-None
        # overrides to the layer object before adding it to the viewer.
        for lname, item in getattr(g, "_extra_startup_layers", {}).items():
            if isinstance(item, tuple):
                llayer, lshader, lblend = item
                # SegmentationLayer has no shader attribute — skip any
                # shader override for segmentation entries (they render via
                # NG's built-in categorical palette instead).
                is_seg = isinstance(llayer, neuroglancer.SegmentationLayer)
                if lshader and not is_seg:
                    llayer.shader = lshader
                if lblend:
                    llayer.blend = lblend
            else:
                # Backward compat: older _extra_startup_layers entries
                # were bare layers.
                llayer = item
            s.layers[lname] = llayer
    # viewer_url is neuroglancer.Viewer()'s own self-contained address (serves both
    # the web UI and data from one origin) -- correct to use as-is for direct/
    # non-tunneled access (the common case: user reaches the compute node's real
    # address directly, no reverse proxy in front). It's only wrong if the user is
    # SSH-tunneling (localhost:port -> node:port) -- there's no reliable way to
    # detect that server-side, so we no longer force-rewrite the host; if you're
    # tunneling and the printed link's host isn't reachable, swap it to localhost
    # (same port) yourself. Do NOT rewrite this to match the dashboard's own
    # X-Forwarded-* handling either -- that's a separate, genuinely-proxy-only
    # concern (see index_page.py's forwarded_host check).
    viewer_url = str(g.viewer)
    show(viewer_url)  # print prominently *before* the blocking create_and_run_app() call below
    logger.info(f"Neuroglancer viewer (open directly, no dashboard needed): {viewer_url}")
    from cellmap_flow.dashboard.app import create_and_run_app

    # create_and_run_app() calls app.run() and blocks forever -- nothing after this
    # line ever executes. Do not add more "print/return the final URL" logic here.
    create_and_run_app(neuroglancer_url=viewer_url, port=5000)


def show(viewer):
    print()
    print()
    print("**********************************************")
    print("LINK:")
    print(viewer)
    print("**********************************************")
    print()
    print()
