"""The models to annotate for, a new annotation volume, and the user's settings.

Routes: GET ``/api/finetune/models``, POST ``/api/finetune/create-volume``,
and GET and POST ``/api/finetune/user-prefs``.
"""

import logging
import os
import time

from flask import jsonify, request

from cellmap_flow.dashboard.finetune_utils import ensure_minio_serving
from cellmap_flow.dashboard.requests import CreateVolume, parse
from cellmap_flow.dashboard.routes.finetune.blueprint import finetune_bp
from cellmap_flow.dashboard.routes.finetune.common import (
    current_chain,
    ensure_corrections_storage,
    find_model_config,
    load_user_prefs,
    rewrite_minio_url_for_proxy,
    save_user_prefs,
    session_store,
    write_volume_manifest,
)
from cellmap_flow.dashboard.routes.finetune.overlay import refresh_annotated_regions_layer
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.session.volume import create_volume_zarr, new_volume_id, plan_volume
from cellmap_flow.models.geometry_cache import build_here, resolve_model_geometry
from cellmap_flow.serving.client import (
    fetch_model_info,
    model_geometry,
    running_job_host,
)

logger = logging.getLogger(__name__)


def serve_new_volume(geometry, corrections_dir, dataset_path, model_name):
    """Write a new volume with ``geometry`` into ``corrections_dir`` and serve it.

    The volume records the dashboard's current normalization and
    postprocessing chains, so the trainer can reproduce the inference-side
    normalization and the finetuned model's YAML the postprocessing. Returns
    ``(volume_id, zarr_path, url)``, the URL as the browser reaches MinIO.
    Registering the volume is the caller's.
    """
    volume_id = new_volume_id()
    zarr_path = os.path.join(corrections_dir, f"{volume_id}.zarr")
    input_norm, postprocess = current_chain()
    create_volume_zarr(
        zarr_path,
        geometry,
        dataset_path=dataset_path,
        model_name=model_name,
        input_norm=input_norm,
        postprocess=postprocess,
    )
    minio_url = ensure_minio_serving(zarr_path, volume_id, output_base_dir=corrections_dir)
    return volume_id, zarr_path, rewrite_minio_url_for_proxy(minio_url)


def _get_selected_model_config(model_name):
    if not get_session().models_config:
        return None, (jsonify({"success": False, "error": "No models loaded"}), 400)

    model_config = find_model_config(model_name)
    if model_config is None:
        return None, (
            jsonify({"success": False, "error": f"Model {model_name} not found"}),
            404,
        )

    return model_config, None


# ``model_config.config`` is far from free: for a script model it executes the
# config file, which downloads weights and runs torch.export before it can
# report a shape. ModelConfig caches the result, but only on success -- a
# failure leaves ``_config`` None, so the next access redoes the whole thing.
# The finetune tab polls this endpoint every two seconds while it waits for a
# model, which turns one failure (a busy GPU, say) into hundreds of full model
# loads. Remember failures briefly instead, short enough that a transient cause
# still recovers on its own.
_CONFIG_FAILURE_COOLDOWN_SECONDS = 60
_config_failure_until = {}


def _config_retry_blocked(name) -> bool:
    until = _config_failure_until.get(name)
    return until is not None and time.time() < until


def _geometry_from_server(name):
    """Geometry from the running inference server, which already has the model."""
    return model_geometry(fetch_model_info(running_job_host(name)))


def _geometry_from_saved_pipeline(name):
    cfg = get_session().builder_model_configs.get(name)
    if not cfg:
        return None
    return {
        "write_shape": cfg.get("write_shape", []),
        "output_voxel_size": cfg.get("output_voxel_size", []),
        "output_channels": cfg.get("output_channels", 1),
    }


def _geometry_from_local_load(name, model_config):
    """Last resort: build the model here just to read its shape.

    Only reached when no server is up and nothing was saved, because it is by
    far the most expensive and least reliable source -- see the note on
    _CONFIG_FAILURE_COOLDOWN_SECONDS above.
    """
    if model_config is None or _config_retry_blocked(name):
        return None
    try:
        # Refuses a model that runs in its own environment, with a message
        # that says so, rather than half-building it with this env's packages.
        config = build_here(model_config)
        _config_failure_until.pop(name, None)
        return {
            "write_shape": list(config.write_shape),
            "output_voxel_size": list(config.output_voxel_size),
            "output_channels": config.output_channels,
        }
    except Exception as e:
        _config_failure_until[name] = time.time() + _CONFIG_FAILURE_COOLDOWN_SECONDS
        logger.warning(
            f"Could not extract config for {name}: {e}. Not retrying for "
            f"{_CONFIG_FAILURE_COOLDOWN_SECONDS}s."
        )
        return None


def _finetune_modes(model_config):
    """What the model can be finetuned with ("lora", "full"), for the tab's
    LoRA rank options; None when there is no config to ask (a model a job
    serves without one), which leaves every option offered."""
    if model_config is None or not hasattr(model_config, "finetune_modes"):
        return None
    try:
        return list(model_config.finetune_modes())
    except Exception as e:
        logger.warning(f"Could not tell how {getattr(model_config, 'name', 'a model')} can be finetuned: {e}")
        return None


@finetune_bp.route("/api/finetune/models", methods=["GET"])
def get_finetune_models():
    try:
        models = []
        seen = set()

        # Every name we might report on: configured models first, then any job
        # running without a matching config (a yaml-launched model, say).
        session = get_session()
        configs_by_name = {}
        for mc in session.models_config or []:
            name = getattr(mc, "name", None)
            if name:
                configs_by_name[name] = mc
        names = list(configs_by_name)
        for job in session.jobs or []:
            job_name = getattr(job, "model_name", None)
            if job_name and job_name not in configs_by_name:
                names.append(job_name)

        for name in names:
            if name in seen:
                continue
            # Cheapest and most reliable source first. Asking the server costs
            # one HTTP round trip; loading the model locally costs a weight
            # download, a torch.export and a CUDA context.
            geometry = (
                _geometry_from_server(name)
                or _geometry_from_saved_pipeline(name)
                or _geometry_from_local_load(name, configs_by_name.get(name))
            )
            if geometry is None:
                logger.warning(f"No configuration available for model: {name}")
                continue
            seen.add(name)
            models.append({"name": name, **geometry,
                           "finetune_modes": _finetune_modes(configs_by_name.get(name))})

        selected = models[0]["name"] if len(models) == 1 else None
        return jsonify({"models": models, "selected_model": selected})
    except Exception as e:
        logger.error(f"Error getting finetune models: {e}")
        return jsonify({"error": str(e)}), 500


@finetune_bp.route("/api/finetune/create-volume", methods=["POST"])
def create_annotation_volume():
    body, refused = parse(CreateVolume, request.get_json() or {})
    if refused:
        return refused
    try:
        model_name = body.model_name
        output_path = body.output_path

        model_config, error_response = _get_selected_model_config(model_name)
        if error_response is not None:
            return error_response

        # Ask the running inference server for the geometry. It already has
        # the model; building it here instead costs a full load on the
        # dashboard's CPU -- 43s in one measured session, for shapes the
        # server can report in milliseconds -- and is what made "create
        # annotation volume" feel slow. model_config.config stays as the
        # fallback for when no server is up.
        config = resolve_model_geometry(model_name, model_config)

        session = get_session()
        dataset_path = session.dataset_path
        if not dataset_path:
            return jsonify({"success": False, "error": "No dataset path configured"}), 400

        geometry = plan_volume(dataset_path, config, resample=session.resample)
        _, corrections_dir = ensure_corrections_storage(output_path)
        volume_id, zarr_path, minio_url = serve_new_volume(
            geometry, corrections_dir, dataset_path, model_name
        )
        session_store().register_volume(
            volume_id,
            **geometry.record(
                zarr_path,
                dataset_path=dataset_path,
                model_name=model_name,
                corrections_dir=corrections_dir,
            ),
        )
        # The trainer finds the volume only through this manifest.
        write_volume_manifest(session.annotation_volumes[volume_id])
        refresh_annotated_regions_layer()

        return jsonify(
            {
                "success": True,
                "volume_id": volume_id,
                "zarr_path": zarr_path,
                "minio_url": minio_url,
                "neuroglancer_url": f"{minio_url}/annotation",
                "metadata": {
                    "dataset_shape_voxels": list(geometry.dataset_shape_voxels),
                    "chunk_size": list(geometry.chunk_size),
                    "output_voxel_size": list(geometry.output_voxel_size),
                    "claimed_output_voxel_size": list(geometry.claimed_output_voxel_size),
                    "dataset_offset_nm": list(geometry.dataset_offset_nm),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error creating annotation volume: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/user-prefs", methods=["GET"])
def get_user_prefs():
    return jsonify({"success": True, "prefs": load_user_prefs()})


@finetune_bp.route("/api/finetune/user-prefs", methods=["POST"])
def set_user_prefs():
    data = request.get_json() or {}
    try:
        prefs = load_user_prefs()
        prefs.update({key: value for key, value in data.items() if value is not None})
        save_user_prefs(prefs)
        return jsonify({"success": True, "prefs": prefs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
