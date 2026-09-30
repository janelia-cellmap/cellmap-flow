"""The deprecated ``cellmap_flow.globals.g``: for one release, every name it
had still works, on its owner's state, and warns with what replaces it.

Scripts and plugins outside the package are what still use g; the documented
use is ``g.input_norms = [...]`` in a user's process_chunk script
(docs/source/scripts.rst).
"""

import re

import numpy as np
import pytest

from cellmap_flow.dashboard.state import get_session
from cellmap_flow.globals import Flow, g
from cellmap_flow.jobs import settings
from cellmap_flow.jobs.launch import started_jobs
from cellmap_flow.jobs.settings import launcher_settings
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.post.postprocessors import SigmoidPostprocessor, SimpleBlockwiseMerger
from cellmap_flow.process_chain import process_chain

SETTINGS = ("queue", "charge_group", "walltime", "cycle_gpu_queues", "nb_cores_master", "nb_cores_worker",
            "nb_workers")
CHAIN = ("input_norms", "postprocess", "input_norm_config", "postprocess_config")
BUILDER = ("inputs", "outputs", "edges", "normalizers", "models", "postprocessors")
SESSION = ("viewer", "dataset_path", "raw", "extra_layers", "shaders", "shader_controls", "models_config",
           "model_catalog", "tmp_dir", "blockwise_tasks_dir", "log_buffer", "log_clients", "bbx_generator_state",
           "review", "minio_state", "annotation_volumes", "output_sessions", "finetune_job_manager")

# Each name g had that can be set: (what its warning names instead, where the value lives now).
ATTRIBUTES = {
    **{key: (f"launcher_settings().{key}", lambda key=key: getattr(launcher_settings(), key)) for key in SETTINGS},
    "_server_config_cached": ("launcher_settings().cached", lambda: launcher_settings().cached),
    **{name: (f"process_chain().{name}", lambda name=name: getattr(process_chain(), name)) for name in CHAIN},
    "jobs": ("started_jobs()", started_jobs),
    "NEUROGLANCER_URL": ("get_session().neuroglancer_url", lambda: get_session().neuroglancer_url),
    **{f"pipeline_{key}": (f'get_session().builder_state["{key}"]', lambda key=key: get_session().builder_state[key])
       for key in BUILDER},
    "pipeline_model_configs": ("get_session().builder_model_configs", lambda: get_session().builder_model_configs),
    **{name: (f"get_session().{name}", lambda name=name: getattr(get_session(), name)) for name in SESSION},
}


def _warns(name, replacement):
    """pytest.warns for g's warning about ``name``, naming ``replacement``."""
    return pytest.warns(DeprecationWarning, match=f"{re.escape(name)} is deprecated.*; use .*{re.escape(replacement)}")


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / "server_config.yaml"
    monkeypatch.setattr(settings, "SERVER_CONFIG_PATH", str(path))
    return path


def test_g_answers_to_the_names_it_had_and_no_others():
    names = {name for name in dir(g) if not name.startswith("_")} | {"_server_config_cached"} & set(dir(g))
    assert names == {*ATTRIBUTES, "pipeline_spec", "save_server_config", "set_pipeline", "get_output_dtype"}
    with pytest.raises(AttributeError):
        g.no_such_name


@pytest.mark.parametrize("name", ATTRIBUTES)
def test_each_name_is_its_owners_and_warns_at_the_callers_line(name):
    """Written through g, the value is the owner's; read through g, it comes
    back. Each warning points at the line that used g, where the fix goes."""
    replacement, owner = ATTRIBUTES[name]
    value = [object()]
    with _warns(f"g.{name}", replacement) as written:
        setattr(g, name, value)
    assert owner() == value  # for jobs, the started list's contents
    with _warns(f"g.{name}", replacement) as read:
        assert getattr(g, name) == value
    assert {w.filename for w in [*written, *read]} == {__file__}


def test_the_methods_and_the_derived_spec_are_the_owners(config_file):
    merger = SimpleBlockwiseMerger()
    spec = PipelineSpec([], [{"name": "SimpleBlockwiseMerger"}])
    with _warns("g.set_pipeline()", "process_chain().set()"):
        g.set_pipeline(spec, built=([], [merger]))
    assert process_chain().postprocess == [merger] and process_chain().postprocess[0] is merger
    with _warns("g.pipeline_spec", "process_chain().spec"):
        assert g.pipeline_spec == process_chain().spec == spec
    with pytest.raises(AttributeError):
        g.pipeline_spec = PipelineSpec()  # derived from the chain

    process_chain().postprocess = [SigmoidPostprocessor()]
    with _warns("g.get_output_dtype()", "process_chain().output_dtype()"):
        assert g.get_output_dtype(np.uint8) == np.float32  # the chain's, when none is given

    launcher_settings().queue = "gpu_a100"
    with _warns("g.save_server_config()", "launcher_settings().save()"):
        g.save_server_config()
    assert launcher_settings().cached and "queue: gpu_a100\n" in config_file.read_text()


def test_flow_is_g():
    with _warns("Flow()", "Flow() returns g"):
        assert Flow() is g
