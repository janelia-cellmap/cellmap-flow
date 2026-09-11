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

            s.layers[model_name] = neuroglancer.ImageLayer(
                source=build_prediction_source(server_url, model_name, st_data, override_scales),
                shader="""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){emitRGB(color * normalized());}""",
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
