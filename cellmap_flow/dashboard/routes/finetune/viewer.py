import logging
from pathlib import Path

import neuroglancer
from flask import jsonify

from cellmap_flow.globals import g
from cellmap_flow.utils.load_py import load_safe_config

logger = logging.getLogger(__name__)


def add_finetuned_layer_to_viewer_response(data):
    try:
        from cellmap_flow.utils.bsub_utils import LSFJob
        from cellmap_flow.utils.neuroglancer_utils import (
            build_prediction_source,
            get_norms_post_args,
            get_raw_closest_scale,
        )

        server_url = data.get("server_url")
        model_name = data.get("model_name")
        model_script_path = data.get("model_script_path")
        custom_shader = data.get("shader")
        if not server_url or not model_name:
            return jsonify({"success": False, "error": "Missing server_url or model_name"}), 400

        base_model_name = model_name.rsplit("_finetuned_", 1)[0] if "_finetuned_" in model_name else model_name
        if model_script_path and Path(model_script_path).exists():
            try:
                model_config = load_safe_config(model_script_path)
                if not hasattr(g, "models_config"):
                    g.models_config = []
                g.models_config = [
                    mc
                    for mc in g.models_config
                    if not (hasattr(mc, "name") and mc.name.startswith(f"{base_model_name}_finetuned"))
                ]
                g.models_config.append(model_config)
            except Exception as e:
                logger.warning(f"Could not load model config: {e}")

        if not hasattr(g, "model_catalog"):
            g.model_catalog = {}
        if "Finetuned" not in g.model_catalog:
            g.model_catalog["Finetuned"] = {}
        g.model_catalog["Finetuned"] = {
            name: path
            for name, path in g.model_catalog["Finetuned"].items()
            if not name.startswith(f"{base_model_name}_finetuned")
        }
        g.model_catalog["Finetuned"][model_name] = model_script_path if model_script_path else ""

        finetune_job = None
        for ft_job in g.finetune_job_manager.jobs.values():
            if ft_job.finetuned_model_name == model_name:
                finetune_job = ft_job
                break

        if finetune_job and finetune_job.job_id:
            inference_job = LSFJob(job_id=finetune_job.job_id, model_name=model_name)
            inference_job.host = server_url
            inference_job.status = finetune_job.status
            g.jobs = [
                job
                for job in g.jobs
                if not (
                    hasattr(job, "model_name")
                    and job.model_name
                    and job.model_name.startswith(f"{base_model_name}_finetuned")
                )
            ]
            g.jobs.append(inference_job)
        else:
            logger.warning(f"Could not find finetune job for {model_name}, Job object not created")

        with g.viewer.txn() as s:
            if model_name in s.layers:
                del s.layers[model_name]

            st_data = get_norms_post_args(g.input_norms, g.postprocess)
            override_scales = None
            try:
                output_voxel_size = None
                if finetune_job is not None and finetune_job.params:
                    output_voxel_size = tuple(finetune_job.params.get("output_voxel_size") or ())
                if not output_voxel_size:
                    for mc in getattr(g, "models_config", []) or []:
                        if mc.name == model_name:
                            output_voxel_size = tuple(mc.config.output_voxel_size)
                            break
                dataset_path = getattr(g, "dataset_path", None)
                if output_voxel_size and dataset_path:
                    closest = get_raw_closest_scale(dataset_path, output_voxel_size)
                    if closest is not None and tuple(closest) != tuple(output_voxel_size):
                        override_scales = closest
            except Exception as e:
                logger.warning(f"Could not compute override scales for finetuned '{model_name}': {e}")

            default_shader = """#uicontrol invlerp normalized(range=[0, 0.5])
#uicontrol vec3 color color(default="red")
void main() {
  float v = normalized();
  if (v <= 0.0)
    emitRGB(color * v);
//    emitTransparent();
  else emitRGB(color * v);
}"""
            s.layers[model_name] = neuroglancer.ImageLayer(
                source=build_prediction_source(server_url, model_name, st_data, override_scales),
                shader=custom_shader if custom_shader else default_shader,
            )

        return jsonify(
            {
                "success": True,
                "layer_name": model_name,
                "model_name": model_name,
                "reload_page": True,
            }
        )
    except Exception as e:
        logger.error(f"Error adding finetuned layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Static-layer viewer-state CRUD (Phase 3 port from vacc-compat finetune_routes:1775+).
#
# These four handlers let the operator add or remove static zarr layers
# (image and segmentation) on a running dashboard without a restart.
# Pattern mirrors add_finetuned_layer_to_viewer_response above: take
# pre-parsed data dict -> mutate g.viewer.txn() -> return reload_page=true.
#
# Idempotency: add-* overwrites a same-named layer (matches the
# add-finetuned-layer convention); remove-layer is a no-op when the
# target name is absent; rename-layer 409s on a target-name collision
# rather than silently overwriting.
# ---------------------------------------------------------------------------

def add_segmentation_layer_to_viewer_response(data):
    """Register a static segmentation zarr on the running NG viewer.

    Mirrors yaml_cli's extra_layers loading for layer_type='segmentation'.
    Required: path, name. Optional: blend, disable_meshes.
    """
    try:
        path = data.get("path")
        name = data.get("name")
        blend = data.get("blend")
        disable_meshes = bool(data.get("disable_meshes", False))

        if not path or not name:
            return (
                jsonify({"success": False, "error": "Missing path or name"}),
                400,
            )

        from cellmap_flow.utils.scale_pyramid import get_raw_layer

        layer = get_raw_layer(
            path,
            normalize=False,
            segmentation=True,
            disable_meshes=disable_meshes,
        )
        if blend:
            layer.blend = blend

        with g.viewer.txn() as s:
            if name in s.layers:
                logger.info(f"Replacing existing layer {name}")
                del s.layers[name]
            s.layers[name] = layer

        logger.info(
            f"Added segmentation layer: {name} -> {path} "
            f"(disable_meshes={disable_meshes})"
        )
        return jsonify(
            {
                "success": True,
                "layer_name": name,
                "layer_type": "segmentation",
                "reload_page": True,
            }
        )

    except Exception as e:
        logger.error(f"Error adding segmentation layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500



def add_image_layer_to_viewer_response(data):
    """Register a static image zarr on the running NG viewer.

    Mirrors yaml_cli's extra_layers loading for layer_type='image'.
    Required: path, name. Optional: shader, blend.
    """
    try:
        path = data.get("path")
        name = data.get("name")
        shader = data.get("shader")
        blend = data.get("blend")

        if not path or not name:
            return (
                jsonify({"success": False, "error": "Missing path or name"}),
                400,
            )

        from cellmap_flow.utils.scale_pyramid import get_raw_layer

        layer = get_raw_layer(
            path,
            normalize=False,
            segmentation=False,
        )
        if shader:
            layer.shader = shader
        if blend:
            layer.blend = blend

        with g.viewer.txn() as s:
            if name in s.layers:
                logger.info(f"Replacing existing layer {name}")
                del s.layers[name]
            s.layers[name] = layer

        logger.info(f"Added image layer: {name} -> {path}")
        return jsonify(
            {
                "success": True,
                "layer_name": name,
                "layer_type": "image",
                "reload_page": True,
            }
        )

    except Exception as e:
        logger.error(f"Error adding image layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def remove_layer_from_viewer_response(data):
    """Drop a layer from the running NG viewer state by name.

    Idempotent: returns success with removed=false if the name is absent.
    Also cleans up companion bookkeeping (g.shaders, g._extra_startup_layers)
    so a subsequent re-add picks up fresh defaults.
    """
    try:
        name = data.get("name")

        if not name:
            return (
                jsonify({"success": False, "error": "Missing name"}),
                400,
            )

        with g.viewer.txn() as s:
            removed = name in s.layers
            if removed:
                del s.layers[name]

        # Companion bookkeeping — silently skip if the keys aren't present.
        if hasattr(g, "shaders") and name in g.shaders:
            del g.shaders[name]
        if hasattr(g, "shader_controls") and name in g.shader_controls:
            del g.shader_controls[name]
        if hasattr(g, "_extra_startup_layers") and name in g._extra_startup_layers:
            del g._extra_startup_layers[name]

        logger.info(f"Removed layer: {name} (was_present={removed})")
        return jsonify(
            {
                "success": True,
                "layer_name": name,
                "removed": removed,
                "reload_page": removed,
            }
        )

    except Exception as e:
        logger.error(f"Error removing layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def rename_layer_in_viewer_response(data):
    """Rename a layer in the running NG viewer state.

    Required: old_name, new_name. 404s if old_name is absent; 409s if
    new_name already exists (refuses to silently overwrite).

    Implementation note: NG's ManagedLayer doesn't expose a public rename
    primitive, so we move the underlying Layer object across keys inside a
    single txn. Companion shader/control bookkeeping is migrated too.
    """
    try:
        old_name = data.get("old_name")
        new_name = data.get("new_name")

        if not old_name or not new_name:
            return (
                jsonify(
                    {"success": False, "error": "Missing old_name or new_name"}
                ),
                400,
            )

        if old_name == new_name:
            return jsonify(
                {
                    "success": True,
                    "renamed": False,
                    "old_name": old_name,
                    "new_name": new_name,
                    "reload_page": False,
                }
            )

        with g.viewer.txn() as s:
            if old_name not in s.layers:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Layer not found: {old_name}",
                        }
                    ),
                    404,
                )
            if new_name in s.layers:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Target name already exists: {new_name}",
                        }
                    ),
                    409,
                )
            managed = s.layers[old_name]
            # ManagedLayer.layer is the underlying ImageLayer / SegmentationLayer / etc.
            inner_layer = getattr(managed, "layer", managed)
            del s.layers[old_name]
            s.layers[new_name] = inner_layer

        # Migrate companion bookkeeping so future accesses by new_name hit fresh state.
        for attr in ("shaders", "shader_controls", "_extra_startup_layers"):
            d = getattr(g, attr, None)
            if isinstance(d, dict) and old_name in d:
                d[new_name] = d.pop(old_name)

        logger.info(f"Renamed layer: {old_name} -> {new_name}")
        return jsonify(
            {
                "success": True,
                "renamed": True,
                "old_name": old_name,
                "new_name": new_name,
                "reload_page": True,
            }
        )

    except Exception as e:
        logger.error(f"Error renaming layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
