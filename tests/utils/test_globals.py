"""What the deprecated ``cellmap_flow.globals.g`` answers to."""

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.globals import g

# Every name g answers to. When g's state moves to its owners (K16), the
# deprecated g left in its place must still answer to each of these for one
# release, and to nothing else.
FLOW_ATTRIBUTES = [
    # the launcher settings, saved in ~/.cellmap_flow/server_config.yaml
    "queue", "charge_group", "walltime", "cycle_gpu_queues", "nb_cores_master", "nb_cores_worker", "nb_workers",
    "_server_config_cached",
    # the chain
    "input_norms", "postprocess", "input_norm_config", "postprocess_config", "pipeline_spec",
    # the servers this process started
    "jobs",
    # the dashboard's session
    "NEUROGLANCER_URL", "viewer", "dataset_path", "raw", "extra_layers", "shaders", "shader_controls",
    "models_config", "model_catalog", "tmp_dir", "blockwise_tasks_dir", "log_buffer", "log_clients",
    "bbx_generator_state", "review",
    # the pipeline builder's last apply
    "pipeline_inputs", "pipeline_outputs", "pipeline_edges", "pipeline_normalizers", "pipeline_models",
    "pipeline_postprocessors", "pipeline_model_configs",
    # the finetune tab
    "minio_state", "annotation_volumes", "output_sessions", "finetune_job_manager",
]
FLOW_METHODS = ["save_server_config", "set_pipeline", "get_output_dtype"]


def test_g_answers_to_these_names_and_no_others():
    get_session().builder_model_configs  # pipeline_model_configs is made on first use
    names = {name for name in dir(g) if not name.startswith("_")}
    names |= {"_server_config_cached"} & set(dir(g))
    assert names == set(FLOW_ATTRIBUTES) | set(FLOW_METHODS)

