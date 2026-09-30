"""What ``cellmap_flow.globals.g`` answers to, and the dashboard's saved
settings (~/.cellmap_flow/server_config.yaml).

SERVER_CONFIG_KEYS comes from SERVER_CONFIG_DEFAULTS, and the dashboard routes
and Flow.save_server_config() read every key with a bare getattr. When Flow
assigned each attribute by hand, adding "walltime" to the defaults gave
save_server_config() a key no instance had, and every cellmap_flow_yaml run
died on AttributeError at startup.
"""

import pytest
import yaml

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


@pytest.fixture
def G(tmp_path, monkeypatch):
    """globals with its config file under tmp_path and no Flow built yet. The
    shared instance comes back afterwards; reloading the module instead would
    leave every importer of `g` holding a different Flow."""
    import cellmap_flow.globals as G

    monkeypatch.setattr(G, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    monkeypatch.setattr(G.Flow, "_instance", None)
    return G


def test_without_a_saved_config_every_key_has_its_default(G):
    flow = G.Flow()
    assert {k: getattr(flow, k) for k in G.SERVER_CONFIG_KEYS} == G.SERVER_CONFIG_DEFAULTS
    assert flow._server_config_cached is False


def test_a_config_saved_before_a_key_existed_loads_and_saves_every_key(G, tmp_path):
    old = {k: v for k, v in G.SERVER_CONFIG_DEFAULTS.items() if k != "walltime"}
    (tmp_path / "server_config.yaml").write_text(yaml.safe_dump({**old, "queue": "gpu_a100"}))
    flow = G.Flow()
    assert (flow.queue, flow.walltime, flow._server_config_cached) == (
        "gpu_a100", G.SERVER_CONFIG_DEFAULTS["walltime"], True)

    flow.walltime = "06:30"
    flow.save_server_config()  # where it crashed
    saved = yaml.safe_load((tmp_path / "server_config.yaml").read_text())
    assert set(saved) == set(G.SERVER_CONFIG_KEYS) and saved["queue"] == "gpu_a100"
    G.Flow._instance = None
    assert G.Flow().walltime == "06:30"
