"""What the dashboard shows of a finetune job's model as it trains.

The job manager (finetune.job_manager) knows nothing of the viewer:
it tells its listeners when a job's inference server is up and when an
iteration finishes (FinetuneJobListener). The dashboard's listener,
``FinetuneLayerListener``, answers each event with the dashboard's two
views of the model:

- ``add_finetuned_layer()``: its viewer layer, served by the job's server,
  replacing the previous iteration's; and its job among the session's jobs;
- ``register_finetuned_model()``: its FinetuneModelConfig among the
  session's models, so the pipeline builder offers it.

``follow_jobs(manager)`` has the listener told of a manager's jobs; the
routes that put jobs into the manager (submit, and finding the jobs of an
earlier dashboard) call it first.
"""

import logging
import threading

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.finetune.job_manager.listener import FinetuneJobListener
from cellmap_flow.finetune.job_manager.persistence import finetune_export_kwargs, recorded_model_entry
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.spec import JobStatus, public_server_url
from cellmap_flow.models.models_config import FinetuneModelConfig
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.serving.client import fetch_model_info
from cellmap_flow.viewer.layers import prediction_layer

logger = logging.getLogger(__name__)


def _is_finetuned_from(model_name, base_model_name):
    """Whether ``model_name`` is an iteration of a model finetuned from ``base_model_name``."""
    return bool(model_name) and model_name.startswith(f"{base_model_name}_finetuned")


def add_finetuned_layer(job, model_name):
    """Add (or replace) the layer of ``job``'s model, ``model_name``.

    The layer is the one every model gets (viewer.layers.prediction_layer),
    over the job's inference server, in red; the previous iteration's layer
    (``job.finetuned_model_name``) goes. The session's jobs get a job for
    the server in place of any earlier iteration's, so Submit and the
    pipeline builder see the model as running. Nothing is added before the
    server is up.
    """
    session = get_session()
    server_url = job.inference_server_url
    if not server_url:
        # The trainer prints its completion marker before it starts the
        # server, and with auto-serve off never starts one. A layer made
        # now had the source zarr://None/..., and came with a jobs entry
        # whose host was None -- permanently, without auto-serve.
        logger.info(f"No inference server for {model_name} yet; the layer is added once it is up.")
        return

    # A local run is a LocalJob, which has a process and no job_id.
    inference_job = LSFJob(job_id=getattr(job.lsf_job, "job_id", None) or "local", model_name=model_name)
    # The address viewers use (see jobs.spec.public_server_url); the
    # dashboard's own requests and the restart control keep server_url.
    inference_job.host = public_server_url(server_url)
    inference_job.status = JobStatus.RUNNING
    # The Finetune tab owns it: the Models tab's Submit must not kill it.
    inference_job.owned_by_finetune = True

    # Replace any old finetuned jobs for this base model. One assignment of
    # a new list, rather than filter-then-append on the shared one: this
    # runs on the monitor thread while request threads use the jobs.
    session.jobs = [
        j for j in list(session.jobs) if not _is_finetuned_from(getattr(j, "model_name", None), job.model_name)
    ] + [inference_job]
    logger.info(f"Added finetuned job to the session's jobs: {model_name}")

    # Get pre/post processing args (same hash as other models)
    st_data = PipelineSpec.from_steps(session.input_norms, session.postprocess).to_url_blob()

    viewer = session.viewer
    if viewer is None:
        logger.error("The viewer is None - neuroglancer not initialized yet")
        return

    # The server is asked at the address the dashboard reaches it by; the
    # layer's source is the viewers' address. The job's record of its voxel
    # size stands in for a server too old to report one.
    layer = prediction_layer(
        model_name, inference_job.host, st_data, dataset_path=session.dataset_path, postprocess=session.postprocess,
        color="red", info=fetch_model_info(server_url),
        fallback_output_voxel_size=job.params.get("output_voxel_size"),
    )
    logger.info(f"Adding neuroglancer layer: {model_name}")

    with viewer.txn() as s:
        # Remove old finetuned layer if it exists (exact name match)
        old_layer_name = job.finetuned_model_name
        if old_layer_name and old_layer_name in s.layers:
            logger.info(f"Removing old finetuned layer: {old_layer_name}")
            del s.layers[old_layer_name]

        # Also remove by current name in case of re-add
        if model_name in s.layers:
            del s.layers[model_name]

        s.layers[model_name] = layer

    logger.info(f"Successfully added neuroglancer layer: {model_name}")


def _yaml_model_entry(yaml_path):
    """The first models: entry of a serving YAML, or None if it cannot be read."""
    if not yaml_path:
        return None
    try:
        import yaml

        with open(yaml_path) as f:
            models = (yaml.safe_load(f) or {}).get("models") or []
    except Exception:
        return None
    entry = models[0] if models else None
    return entry if isinstance(entry, dict) and entry.get("base_model") else None


def register_finetuned_model(job, model_name):
    """Put a FinetuneModelConfig for ``job``'s model ``model_name`` among the
    session's models, in place of any earlier iteration's, so the pipeline
    builder offers it with its parameters filled in."""
    session = get_session()
    params = job.params

    # The trainer's own YAML for this iteration says exactly what it
    # exported and on which base; registering from it keeps the pipeline
    # builder's model identical to the one the YAML serves. Without one,
    # fall back to the run's latest export and the base model's entry.
    entry = _yaml_model_entry(job.model_yaml_path)
    if entry is not None:
        export = {k: entry[k] for k in ("lora_adapter_path", "weights_path") if entry.get(k)}
        base_model_dict = entry.get("base_model")
    else:
        export = finetune_export_kwargs(job.output_dir, params)
        base_model_dict = None

    # Find the base model's to_dict() among the session's models
    if base_model_dict is None:
        for mc in session.models_config or []:
            if getattr(mc, "name", None) == job.model_name:
                base_model_dict = mc.to_dict()
                break

    if base_model_dict is None:
        # The model the run's trainer was given, as its record has it: a
        # Fly, cellmap or finetune model's (job_manager.submit.model_entry).
        base_model_dict = recorded_model_entry(job.output_dir)

    if base_model_dict is None:
        # Last, rebuilt from the job's params as a Fly model. Its sizes are
        # not among them, so it is 178/56, which is what a Fly run's trainer
        # built before its entry was recorded.
        base_model_dict = {"type": "fly"}
        if params.get("model_checkpoint"):
            base_model_dict["checkpoint_path"] = params["model_checkpoint"]
        for key in ("channels", "input_voxel_size", "output_voxel_size"):
            if key in params:
                base_model_dict[key] = params[key]

    ft_config = FinetuneModelConfig(base_model=base_model_dict, name=model_name, scale=params.get("scale"), **export)

    # Remove any previous finetuned versions of the same base model
    session.models_config = [
        mc for mc in session.models_config if not _is_finetuned_from(getattr(mc, "name", None), job.model_name)
    ]
    session.models_config.append(ft_config)
    logger.info(f"Registered FinetuneModelConfig: {model_name}")


class FinetuneLayerListener(FinetuneJobListener):
    """The dashboard's listener: on each event, the model's layer and its
    pipeline-builder model. Either failing is logged, and does not stop the
    other."""

    def on_server_ready(self, job, url, model_name):
        self._show(job, model_name, "Failed to add finetuned model to neuroglancer")

    def on_iteration_complete(self, job, model_name):
        self._show(job, model_name, "Failed to update neuroglancer layer")

    @staticmethod
    def _show(job, model_name, failure):
        try:
            add_finetuned_layer(job, model_name)
        except Exception as e:
            logger.error(f"{failure}: {e}", exc_info=True)
        try:
            register_finetuned_model(job, model_name)
        except Exception as e:
            logger.error(f"Failed to register FinetuneModelConfig: {e}", exc_info=True)


_LISTENER = FinetuneLayerListener()
_FOLLOWING = threading.Lock()


def follow_jobs(manager):
    """Have the dashboard's listener told of ``manager``'s jobs, once however
    often this is called (request threads call it concurrently)."""
    with _FOLLOWING:
        manager.add_listener(_LISTENER)
