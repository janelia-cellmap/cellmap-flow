import json
import logging
import re as _re
import subprocess
import time
import traceback
from pathlib import Path

from flask import Blueprint, jsonify, request, Response

from cellmap_flow.globals import g
from cellmap_flow.dashboard.state import finetune_job_manager, get_or_create_session_path

logger = logging.getLogger(__name__)

finetune_bp = Blueprint("finetune", __name__)


@finetune_bp.route("/api/finetune/models", methods=["GET"])
def get_finetune_models():
    """Get available models for finetuning with their configurations."""
    try:
        models = []

        # Extract from g.models_config
        if hasattr(g, "models_config") and g.models_config:
            for model_config in g.models_config:
                try:
                    config = model_config.config
                    models.append(
                        {
                            "name": model_config.name,
                            "write_shape": list(config.write_shape),
                            "output_voxel_size": list(config.output_voxel_size),
                            "output_channels": config.output_channels,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract config for {model_config.name}: {e}"
                    )

        # If no models found in g.models_config, try to get from running jobs
        # This handles the case where models were submitted via GUI after app started
        if len(models) == 0 and hasattr(g, "jobs") and g.jobs:
            logger.warning("No models in g.models_config, checking running jobs")
            # Try to get model configs from jobs
            for job in g.jobs:
                if hasattr(job, "model_name"):
                    job_model_name = job.model_name
                    # Look for config in pipeline_model_configs (if available)
                    if hasattr(g, "pipeline_model_configs") and job_model_name in g.pipeline_model_configs:
                        config_dict = g.pipeline_model_configs[job_model_name]
                        try:
                            models.append(
                                {
                                    "name": job_model_name,
                                    "write_shape": config_dict.get("write_shape", []),
                                    "output_voxel_size": config_dict.get("output_voxel_size", []),
                                    "output_channels": config_dict.get("output_channels", 1),
                                }
                            )
                            logger.info(f"Found config for {job_model_name} in pipeline_model_configs")
                        except Exception as e:
                            logger.warning(f"Could not extract config for {job_model_name}: {e}")
                    else:
                        logger.warning(f"No configuration found for running job: {job_model_name}")
                        logger.warning(f"  → Model needs write_shape, output_voxel_size, and output_channels for finetuning")
                        logger.warning(f"  → Consider restarting with a proper YAML configuration file")

        # Determine selected model
        selected = models[0]["name"] if len(models) == 1 else None

        return jsonify({"models": models, "selected_model": selected})

    except Exception as e:
        logger.error(f"Error getting finetune models: {e}")
        return jsonify({"error": str(e)}), 500


@finetune_bp.route("/api/finetune/view-center", methods=["GET"])
def get_view_center():
    """Get current view center position from Neuroglancer viewer."""
    try:
        if not hasattr(g, "viewer") or g.viewer is None:
            return jsonify({"success": False, "error": "Viewer not initialized"}), 400

        # Access viewer state using transaction
        with g.viewer.txn() as s:
            # Get the current view position (center of view)
            position = s.position

            # Get the viewer dimensions to extract scales
            dimensions = s.dimensions
            scales_nm = None

            if dimensions and hasattr(dimensions, "scales"):
                # CoordinateSpace has scales attribute directly
                scales_nm = list(dimensions.scales)
                logger.info(f"Viewer scales (raw): {scales_nm}")

                # Check units and convert if needed
                if hasattr(dimensions, "units"):
                    units = dimensions.units
                    # units can be a string (same for all axes) or list
                    if isinstance(units, str):
                        units = [units] * len(scales_nm)

                    # Convert to nm if needed
                    converted_scales = []
                    for scale, unit in zip(scales_nm, units):
                        if unit == "m":
                            converted_scales.append(scale * 1e9)  # meters to nanometers
                        elif unit == "nm":
                            converted_scales.append(scale)
                        else:
                            logger.warning(f"Unknown unit: {unit}, assuming nm")
                            converted_scales.append(scale)
                    scales_nm = converted_scales

                logger.info(f"Viewer scales (nm): {scales_nm}")
            else:
                logger.warning("Could not extract scales from viewer dimensions")

            # Convert to list if it's a numpy array or coordinate object
            if hasattr(position, "tolist"):
                position = position.tolist()
            elif hasattr(position, "__iter__"):
                position = list(position)

            logger.info(f"Got view center position: {position}")

            return jsonify(
                {"success": True, "position": position, "scales_nm": scales_nm}
            )

    except Exception as e:
        logger.error(f"Error getting view center position: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/submit", methods=["POST"])
def submit_finetuning():
    """
    Submit a finetuning job to the LSF cluster.

    Request body:
    {
        "model_name": "model_name",
        "lora_r": 8,
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 0.0001
    }
    """
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        corrections_path_str = data.get("corrections_path")
        lora_r = data.get("lora_r", 8)
        num_epochs = data.get("num_epochs", 10)
        batch_size = data.get("batch_size", 2)
        learning_rate = data.get("learning_rate", 1e-4)
        checkpoint_path_override = data.get("checkpoint_path")  # Optional override
        auto_serve = data.get("auto_serve", True)  # Auto-serve by default
        loss_type = data.get("loss_type", "mse")
        label_smoothing = data.get("label_smoothing", 0.1)
        distillation_lambda = data.get("distillation_lambda", 0.0)
        distillation_scope = data.get("distillation_scope", "unlabeled")
        margin = data.get("margin", 0.3)
        balance_classes = data.get("balance_classes", False)
        queue = data.get("queue", "gpu_h100")

        if not model_name:
            return jsonify({"success": False, "error": "model_name is required"}), 400

        if not corrections_path_str:
            return jsonify({"success": False, "error": "corrections_path is required. Please specify the output path where annotation crops are saved."}), 400

        # Find model config
        model_config = None
        for config in g.models_config:
            if config.name == model_name:
                model_config = config
                break

        if not model_config:
            return jsonify({"success": False, "error": f"Model {model_name} not found"}), 404

        # Check if corrections path is a YAML data config file
        base_corrections_path = Path(corrections_path_str)
        is_yaml_data = base_corrections_path.suffix.lower() in (".yaml", ".yml") and base_corrections_path.is_file()

        if is_yaml_data:
            # YAML dataset mode: pass the YAML file directly
            actual_corrections_path = base_corrections_path
            # Use the YAML file's parent directory as output base
            output_base = base_corrections_path.parent / "output"
            output_base.mkdir(parents=True, exist_ok=True)
            session_path = output_base
            logger.info(f"Using YAML data config: {actual_corrections_path}")
            logger.info(f"Output base: {output_base}")
        else:
            # Traditional corrections directory mode
            # Check if this looks like a session path with corrections subdirectory
            actual_corrections_path = None
            if base_corrections_path.name == "corrections" and base_corrections_path.exists():
                # User provided the full path including "/corrections"
                actual_corrections_path = base_corrections_path
                session_path = base_corrections_path.parent
            else:
                # User provided the base path - find the session with corrections
                session_path = get_or_create_session_path(str(base_corrections_path))
                actual_corrections_path = Path(session_path) / "corrections"

            if not actual_corrections_path.exists():
                return jsonify({
                    "success": False,
                    "error": f"Corrections path does not exist: {actual_corrections_path}. Please create annotation crops first."
                }), 400

            # Derive output base from session path for finetuning outputs
            output_base = Path(session_path)
            logger.info(f"Using session path for finetuning: {session_path}")
            logger.info(f"Corrections path: {actual_corrections_path}")
            logger.info(f"Output base: {output_base}")

        # Detect sparse annotations: check if any correction has source=sparse_volume
        has_sparse = False
        if not is_yaml_data:
            try:
                for p in actual_corrections_path.iterdir():
                    if p.suffix == ".zarr" and (p / ".zattrs").exists():
                        attrs = json.loads((p / ".zattrs").read_text())
                        if attrs.get("source") == "sparse_volume":
                            has_sparse = True
                            break
            except Exception as e:
                logger.warning(f"Error checking for sparse annotations: {e}")

        sparse_auto_switched = False
        if has_sparse:
            logger.info("Detected sparse annotations, will use mask_unannotated=True")
            # Auto-switch to better defaults for sparse scribbles
            if loss_type == "mse":  # only override if user hasn't explicitly chosen
                loss_type = "margin"
                distillation_lambda = 0.5
                sparse_auto_switched = True
                logger.info("Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations")

        # Submit job
        finetune_job = finetune_job_manager.submit_finetuning_job(
            model_config=model_config,
            corrections_path=actual_corrections_path,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_base=output_base,
            checkpoint_path_override=Path(checkpoint_path_override) if checkpoint_path_override else None,
            auto_serve=auto_serve,
            mask_unannotated=has_sparse,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            balance_classes=balance_classes,
            queue=queue,
        )

        logger.info(f"Submitted finetuning job: {finetune_job.job_id}")

        # Get LSF job ID or local PID
        lsf_job_id = None
        if finetune_job.lsf_job:
            if hasattr(finetune_job.lsf_job, 'job_id'):
                lsf_job_id = finetune_job.lsf_job.job_id
            elif hasattr(finetune_job.lsf_job, 'process'):
                lsf_job_id = f"PID:{finetune_job.lsf_job.process.pid}"

        response = {
            "success": True,
            "job_id": finetune_job.job_id,
            "lsf_job_id": lsf_job_id,
            "output_dir": str(finetune_job.output_dir),
            "message": "Finetuning job submitted successfully"
        }
        if sparse_auto_switched:
            response["note"] = "Auto-switched to margin loss + distillation (lambda=0.5) for sparse annotations"

        return jsonify(response)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error submitting finetuning job: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/jobs", methods=["GET"])
def get_finetuning_jobs():
    """Get list of all finetuning jobs."""
    try:
        jobs = finetune_job_manager.list_jobs()
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/status", methods=["GET"])
def get_job_status(job_id):
    """Get detailed status of a specific job."""
    try:
        status = finetune_job_manager.get_job_status(job_id)
        if status is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, **status})
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/logs", methods=["GET"])
def get_job_logs(job_id):
    """Get training logs for a specific job."""
    try:
        logs = finetune_job_manager.get_job_logs(job_id)
        if logs is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        logger.error(f"Error getting job logs: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/logs/stream", methods=["GET"])
def stream_job_logs(job_id):
    """Server-Sent Events stream for live training logs."""

    # Patterns to filter out of the log stream
    _log_filters = [
        _re.compile(r"^\s+base_model\.\S+\.lora_"),  # gradient norm lines
        _re.compile(r"^INFO:werkzeug:"),
        _re.compile(r"^Array metadata \(scale="),  # server chunk metadata
        _re.compile(r"^Host name:"),
        _re.compile(r"^DEBUG trainer:"),
    ]

    def _should_show(line):
        for pat in _log_filters:
            if pat.search(line):
                return False
        return True

    def _iter_visible_lines(text):
        for line in text.splitlines():
            if line and _should_show(line):
                yield line

    def _sse_data_block(lines):
        if not lines:
            return None
        payload = "\n".join(lines)
        # SSE multiline payload requires each line to be prefixed with "data: ".
        return "data: " + payload.replace("\n", "\ndata: ") + "\n\n"

    def _read_bpeek_content(lsf_job_id):
        try:
            result = subprocess.run(
                ["bpeek", str(lsf_job_id)],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            logger.debug(f"bpeek call failed for job {lsf_job_id}: {e}")
            return None

        output = (result.stdout or "")
        stderr = (result.stderr or "").strip()
        if stderr and "Not yet started" not in stderr:
            logger.debug(f"bpeek stderr for job {lsf_job_id}: {stderr}")
        # Empty output can happen while pending; treat as no-new-data rather than error.
        return output

    def generate():
        # Keepalive cadence for SSE (helps prevent proxy/browser buffering)
        heartbeat_interval_s = 1.0
        last_heartbeat = time.perf_counter()

        # Check if job exists
        if job_id not in finetune_job_manager.jobs:
            yield f"data: Job {job_id} not found\n\n"
            return

        finetune_job = finetune_job_manager.jobs[job_id]
        lsf_job_id = None
        if finetune_job.lsf_job and hasattr(finetune_job.lsf_job, "job_id"):
            lsf_job_id = finetune_job.lsf_job.job_id

        use_bpeek = lsf_job_id is not None
        last_bpeek_line_count = 0
        last_bpeek_poll = 0.0
        bpeek_poll_interval_s = 0.25

        # Send existing content first (bpeek preferred, file fallback)
        if use_bpeek:
            initial = _read_bpeek_content(lsf_job_id)
            if initial is None:
                use_bpeek = False
            else:
                initial_lines = initial.splitlines()
                last_bpeek_line_count = len(initial_lines)
                block = _sse_data_block(list(_iter_visible_lines(initial)))
                if block:
                    yield block

        if not use_bpeek and finetune_job.log_file.exists():
            try:
                with open(finetune_job.log_file, "r") as f:
                    existing_content = f.read()
                block = _sse_data_block(list(_iter_visible_lines(existing_content)))
                if block:
                    yield block
            except Exception as e:
                logger.error(f"Error reading log file: {e}")

        # Then tail for new content (bpeek preferred, file fallback)
        last_position = finetune_job.log_file.stat().st_size if finetune_job.log_file.exists() else 0

        while finetune_job.status.value in ["PENDING", "RUNNING"]:
            try:
                now = time.perf_counter()

                if use_bpeek and lsf_job_id and now - last_bpeek_poll >= bpeek_poll_interval_s:
                    last_bpeek_poll = now
                    content = _read_bpeek_content(lsf_job_id)
                    if content is None:
                        # Fall back to file tailing for this stream connection
                        use_bpeek = False
                    else:
                        current_lines = content.splitlines()
                        if len(current_lines) < last_bpeek_line_count:
                            # bpeek output rolled/restarted; resync from current full output.
                            delta_lines = current_lines
                        else:
                            delta_lines = current_lines[last_bpeek_line_count:]
                        last_bpeek_line_count = len(current_lines)
                        if delta_lines:
                            delta_text = "\n".join(delta_lines)
                            block = _sse_data_block(list(_iter_visible_lines(delta_text)))
                            if block:
                                yield block

                if not use_bpeek and finetune_job.log_file.exists():
                    with open(finetune_job.log_file, "r") as f:
                        f.seek(last_position)
                        new_content = f.read()
                        last_position = f.tell()
                    if new_content:
                        block = _sse_data_block(list(_iter_visible_lines(new_content)))
                        if block:
                            yield block

                # Emit heartbeat comments to force periodic flush and keep connection alive.
                if now - last_heartbeat >= heartbeat_interval_s:
                    yield ": ping\n\n"
                    last_heartbeat = now

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error streaming logs: {e}")
                break

        yield f"data: === Training {finetune_job.status.value} ===\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@finetune_bp.route("/api/finetune/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running finetuning job."""
    try:
        success = finetune_job_manager.cancel_job(job_id)

        if success:
            return jsonify({"success": True, "message": f"Job {job_id} cancelled"})
        else:
            return jsonify({"success": False, "error": "Failed to cancel job"}), 400

    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/inference-server", methods=["GET"])
def get_inference_server_status(job_id):
    """Get inference server status for a finetuning job."""
    try:
        job = finetune_job_manager.get_job(job_id)
        if not job:
            return jsonify({"success": False, "error": "Job not found"}), 404

        return jsonify({
            "success": True,
            "ready": job.inference_server_ready,
            "url": job.inference_server_url,
            "model_name": job.finetuned_model_name,
            "model_script_path": str(job.model_script_path) if job.model_script_path else None
        })

    except Exception as e:
        logger.error(f"Error getting inference server status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/viewer/add-finetuned-layer", methods=["POST"])
def add_finetuned_layer_to_viewer():
    """Add finetuned model layer to Neuroglancer viewer and register model in system."""
    try:
        data = request.get_json()
        server_url = data.get("server_url")
        model_name = data.get("model_name")
        model_script_path = data.get("model_script_path")

        if not server_url or not model_name:
            return jsonify({"success": False, "error": "Missing server_url or model_name"}), 400

        logger.info(f"Registering finetuned model: {model_name} at {server_url}")

        # 1. Load model config from script if provided
        if model_script_path and Path(model_script_path).exists():
            try:
                from cellmap_flow.utils.load_py import load_safe_config
                model_config = load_safe_config(model_script_path)

                # Add to models_config if not already there
                if not hasattr(g, 'models_config'):
                    g.models_config = []

                # Remove old finetuned model configs with same base name
                base_model_name = model_name.rsplit("_finetuned_", 1)[0] if "_finetuned_" in model_name else model_name
                g.models_config = [
                    mc for mc in g.models_config
                    if not (hasattr(mc, 'name') and mc.name.startswith(f"{base_model_name}_finetuned"))
                ]

                # Add new config
                g.models_config.append(model_config)
                logger.info(f"Loaded model config from {model_script_path}")
            except Exception as e:
                logger.warning(f"Could not load model config: {e}")

        # 2. Add to model_catalog under "Finetuned" group
        if not hasattr(g, 'model_catalog'):
            g.model_catalog = {}

        if "Finetuned" not in g.model_catalog:
            g.model_catalog["Finetuned"] = {}

        # Remove old finetuned models with same base name
        base_model_name = model_name.rsplit("_finetuned_", 1)[0] if "_finetuned_" in model_name else model_name
        g.model_catalog["Finetuned"] = {
            name: path for name, path in g.model_catalog["Finetuned"].items()
            if not name.startswith(f"{base_model_name}_finetuned")
        }

        # Add new finetuned model
        g.model_catalog["Finetuned"][model_name] = model_script_path if model_script_path else ""
        logger.info(f"Added to model catalog: Finetuned/{model_name}")

        # 3. Create a Job object for the running inference server
        from cellmap_flow.utils.bsub_utils import LSFJob

        # Find the corresponding finetune job
        finetune_job = None
        for job_id, ft_job in finetune_job_manager.jobs.items():
            if ft_job.finetuned_model_name == model_name:
                finetune_job = ft_job
                break

        if finetune_job and finetune_job.job_id:
            # Create Job object pointing to the running server
            inference_job = LSFJob(job_id=finetune_job.job_id, model_name=model_name)
            inference_job.host = server_url
            inference_job.status = finetune_job.status

            # Remove old jobs for this base model
            g.jobs = [
                j for j in g.jobs
                if not (hasattr(j, 'model_name') and j.model_name and j.model_name.startswith(f"{base_model_name}_finetuned"))
            ]

            # Add to jobs
            g.jobs.append(inference_job)
            logger.info(f"Created Job object for {model_name} with job_id {finetune_job.job_id}")
        else:
            logger.warning(f"Could not find finetune job for {model_name}, Job object not created")

        # 4. Add neuroglancer layer
        layer_name = model_name  # Use model name directly (not prefixed with "finetuned_")

        with g.viewer.txn() as s:
            # Remove old finetuned layer if it exists
            if layer_name in s.layers:
                logger.info(f"Removing old finetuned layer: {layer_name}")
                del s.layers[layer_name]

            # Add new layer pointing to inference server
            from cellmap_flow.utils.neuroglancer_utils import get_norms_post_args
            from cellmap_flow.utils.web_utils import ARGS_KEY

            st_data = get_norms_post_args(g.input_norms, g.postprocess)

            # Create image layer for finetuned model (same style as normal models)
            import neuroglancer
            layer_source = f"zarr://{server_url}/{model_name}{ARGS_KEY}{st_data}{ARGS_KEY}"
            s.layers[layer_name] = neuroglancer.ImageLayer(
                source=layer_source,
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){{emitRGB(color * normalized());}}""",
            )

        logger.info(f"Added neuroglancer layer: {layer_name} -> {server_url}")

        return jsonify({
            "success": True,
            "layer_name": layer_name,
            "model_name": model_name,
            "reload_page": True  # Signal frontend to reload to see new model in catalog
        })

    except Exception as e:
        logger.error(f"Error adding finetuned layer: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@finetune_bp.route("/api/finetune/job/<job_id>/restart", methods=["POST"])
def restart_finetuning_job(job_id):
    """Restart training on the same GPU via in-process control channel."""
    try:
        restart_t0 = time.perf_counter()
        data = request.get_json() or {}

        # Extract updated parameters
        updated_params = {}
        for key in ["num_epochs", "batch_size", "learning_rate", "loss_type", "distillation_lambda", "distillation_scope", "margin"]:
            if key in data and data[key] is not None:
                updated_params[key] = data[key]

        # Send restart request (same job, same GPU)
        signal_t0 = time.perf_counter()
        job = finetune_job_manager.restart_finetuning_job(
            job_id=job_id,
            updated_params=updated_params
        )
        signal_elapsed = time.perf_counter() - signal_t0
        total_elapsed = time.perf_counter() - restart_t0
        logger.info(
            f"Restart request processed for job {job_id}: "
            f"signal_write={signal_elapsed:.2f}s total={total_elapsed:.2f}s"
        )

        return jsonify({
            "success": True,
            "job_id": job.job_id,
            "message": "Restart request sent. Training will restart on the same GPU.",
        })

    except Exception as e:
        logger.error(f"Error restarting job: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
