"""`cellmap_flow yaml`: the bundled examples, how a bad config is reported,
what it shows in the viewer, and what it does when servers fail to start."""

import glob
import logging
import os

import neuroglancer
import numpy as np
import pytest
import yaml
import zarr
from click.testing import CliRunner

from cellmap_flow.cli import yaml_cli
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.dashboard.services import startup
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import settings
from cellmap_flow.jobs.spec import JobStartError
from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.pipeline_spec import PipelineSpec
from cellmap_flow.process_chain import process_chain

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tests", "script_test", "fake_model_script.py")
RAW = os.path.join(ROOT, "tests", "script_test", "dummy.zarr", "raw")
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "example", "*.yaml")))
RUN_CONFIGS = [p for p in EXAMPLES if "data_path" in (yaml.safe_load(open(p)) or {})]


EXAMPLE_RUN_CONFIGS = [pytest.param(p, id=os.path.basename(p)) for p in RUN_CONFIGS]


def test_there_are_example_run_configs_to_check():
    assert len(RUN_CONFIGS) >= 3


@pytest.mark.parametrize("path", EXAMPLE_RUN_CONFIGS)
def test_the_example_run_configs_are_valid(path, placeholder_cellmap_folders):
    result = CliRunner().invoke(yaml_cli.main, [path, "--validate-only"])
    assert result.exit_code == 0 and "Configuration is valid" in result.output, result.output


def _array(path, dtype=np.uint8):
    array = zarr.open_group(str(path.parent), mode="a").create_dataset(path.name, data=np.zeros((4, 4, 4), dtype))
    array.attrs.update(resolution=[8, 8, 8], offset=[0, 0, 0])
    return str(path)


def _config(tmp_path, models=None, extra_layers=None, **settings):
    config = {"data_path": _array(tmp_path / "raw.zarr" / "raw"), "charge_group": "grp", "models": models or {},
              **settings}
    if extra_layers is not None:
        config["extra_layers"] = extra_layers
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(config))
    return str(tmp_path / "c.yaml")


@pytest.mark.parametrize("models, extra_layers, message", [
    pytest.param({"m": {"type": "nope"}}, None, "'nope'", id="unknown-type"),
    pytest.param({"m": {"type": "cellmap", "config_folder": "/no/such/folder"}}, None, "'m'", id="missing-model-folder"),
    pytest.param({}, [{"name": "data", "path": "/x.zarr"}], "extra_layers", id="extra-layer-named-data"),
])
def test_a_bad_config_is_reported_and_exits_non_zero(tmp_path, models, extra_layers, message):
    """The YAML loader called sys.exit(1); now the CLI turns ConfigError into a clean exit."""
    if "cellmap" in str(models):
        pytest.importorskip("cellmap_models.model_export.cellmap_model")
    result = CliRunner().invoke(yaml_cli.main, [_config(tmp_path, models, extra_layers), "--validate-only"])
    assert result.exit_code == 1 and message in result.output, result.output
    assert not isinstance(result.exception, ConfigError), "caught, not a traceback"
    assert get_session().extra_layers == {}


def test_extra_layers_are_shown_beside_the_raw_data(tmp_path, monkeypatch):
    from neuroglancer.viewer_base import ViewerBase

    monkeypatch.setattr(neuroglancer, "Viewer", ViewerBase)
    monkeypatch.setattr(startup, "create_and_run_app", lambda **k: "url")
    monkeypatch.setattr(yaml_cli, "install_cleanup_handlers", lambda: None)
    config = _config(tmp_path, extra_layers=[
        {"name": "pred", "path": _array(tmp_path / "pred.zarr" / "mito"),
         "shader": "void main() { emitGrayscale(1.0); }", "blend": "additive"},
        {"name": "ids", "path": _array(tmp_path / "ids.zarr" / "s0", np.uint64), "layer_type": "segmentation",
         "disable_meshes": True},
    ])
    result = CliRunner().invoke(yaml_cli.main, [config])
    assert result.exit_code == 0, result.output
    layers = get_session().viewer.state.layers
    assert [layer.name for layer in layers] == ["data", "pred", "ids"]
    pred, ids = layers["pred"].to_json(), layers["ids"].to_json()
    assert (pred["type"], pred["blend"], pred["shader"]) == ("image", "additive", "void main() { emitGrayscale(1.0); }")
    assert ids["type"] == "segmentation" and ids["source"][0]["subsources"] == {"meshes": False}


def test_a_run_leaves_the_chain_the_models_and_the_settings_for_the_dashboard(tmp_path, monkeypatch):
    """What the dashboard and the finetune manifest read after a YAML run:
    the session's models and extra layers, the process's chain, and the
    settings saved for the next dashboard."""
    ran = []
    monkeypatch.setattr(settings, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    monkeypatch.setattr(yaml_cli, "install_cleanup_handlers", lambda: None)
    monkeypatch.setattr(yaml_cli, "run_multiple", lambda *args, **kwargs: ran.append(args))
    config = _config(
        tmp_path,
        models={"m": {"type": "script", "script_path": SCRIPT}},
        extra_layers=[{"name": "pred", "path": _array(tmp_path / "pred.zarr" / "mito")}],
        queue="gpu_h200", walltime="36:00", cycle_gpu_queues=False, resample=True,
        json_data={"input_norm": {"MinMaxNormalizer": {"min_value": 0, "max_value": 255}},
                   "postprocess": {"ThresholdPostprocessor": {"threshold": 0.5}}},
    )
    result = CliRunner().invoke(yaml_cli.main, [config])
    assert result.exit_code == 0, result.output + repr(result.exception)

    session, chain = get_session(), process_chain()
    assert [(type(m).__name__, m.name) for m in session.models_config] == [("ScriptModelConfig", "m")]
    assert ran == [(session.models_config, yaml.safe_load(open(config))["data_path"], "grp", "gpu_h200")]
    assert list(session.extra_layers) == ["pred"]
    # For the models the dashboard launches later, not only the YAML's.
    assert session.resample is True
    assert ([type(s).__name__ for s in chain.input_norms], [type(s).__name__ for s in chain.postprocess]) == (
        ["MinMaxNormalizer"], ["ThresholdPostprocessor"])
    # The steps as the YAML gives them, as a chain submitted in the
    # dashboard is: the finetune manifest's text. (It was the live steps'
    # to_dict(), with every default: "invert": False here.)
    assert session.pipeline_spec == PipelineSpec(
        [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}],
        [{"name": "ThresholdPostprocessor", "threshold": 0.5}],
    )
    assert yaml.safe_load((tmp_path / "server_config.yaml").read_text()) == {
        **settings.SERVER_CONFIG_DEFAULTS,
        "queue": "gpu_h200", "charge_group": "grp", "walltime": "36:00", "cycle_gpu_queues": False,
    }


@pytest.mark.parametrize("failing, viewer_opened", [
    pytest.param({"bad"}, True, id="one-of-two"),
    pytest.param({"good", "bad"}, False, id="all"),
])
def test_only_the_servers_that_started_are_shown(monkeypatch, caplog, failing, viewer_opened):
    """A model whose server never came up was logged as ready, and the viewer
    got a zarr://None/... layer for it."""
    viewers, started = [], []

    def start_hosts(command, queue=None, charge_group=None, job_name=None, **_):
        if job_name in failing:
            raise JobStartError(f"{job_name} never reported a server address")
        started.append(job_name)

    monkeypatch.setattr("cellmap_flow.jobs.launch.start_hosts", start_hosts)
    monkeypatch.setattr(startup, "generate_neuroglancer_url", lambda path, wrap_raw=True: viewers.append(path))
    models = [ScriptModelConfig(script_path=SCRIPT, name=n) for n in ("good", "bad")]
    with caplog.at_level(logging.ERROR):
        if viewer_opened:
            yaml_cli.run_multiple(models, RAW, "grp", "gpu_h100")
        else:
            with pytest.raises(JobStartError):  # an error, not an empty dashboard
                yaml_cli.run_multiple(models, RAW, "grp", "gpu_h100")
    assert viewers == ([RAW] if viewer_opened else [])
    assert any("bad" in r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR), "the failure is said"

