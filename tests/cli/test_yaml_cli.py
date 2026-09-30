"""`cellmap_flow_yaml`: the bundled examples, how a bad config is reported,
what it shows in the viewer, and what it does when servers fail to start."""

import glob
import logging
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml
import zarr
from click.testing import CliRunner

from cellmap_flow.cli import yaml_cli
from cellmap_flow.globals import g
from cellmap_flow.models.models_config import ScriptModelConfig
from cellmap_flow.utils import neuroglancer_utils
from cellmap_flow.utils.bsub_utils import JobStartError
from cellmap_flow.utils.config_utils import ConfigError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tests", "script_test", "fake_model_script.py")
RAW = os.path.join(ROOT, "tests", "script_test", "dummy.zarr", "raw")
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "example", "*.yaml")))
RUN_CONFIGS = [p for p in EXAMPLES if "data_path" in (yaml.safe_load(open(p)) or {})]


@pytest.fixture
def placeholder_cellmap_folders(monkeypatch):
    """CellMapModelConfig reads its folder's metadata.json when constructed, and
    the examples use placeholder paths: what is checked is the YAML."""
    cellmap_model = pytest.importorskip("cellmap_models.model_export.cellmap_model")
    monkeypatch.setattr(cellmap_model, "CellmapModel", lambda folder_path: None)


EXAMPLE_RUN_CONFIGS = [pytest.param(p, id=os.path.basename(p)) for p in RUN_CONFIGS]


def test_there_are_example_run_configs_to_check():
    assert len(RUN_CONFIGS) >= 3


@pytest.mark.parametrize("path", EXAMPLE_RUN_CONFIGS)
def test_the_example_run_configs_are_valid(path, placeholder_cellmap_folders):
    result = CliRunner().invoke(yaml_cli.main, [path, "--validate-only"])
    assert result.exit_code == 0 and "Configuration is valid" in result.output, result.output


@pytest.mark.parametrize("path", EXAMPLE_RUN_CONFIGS)
def test_the_examples_commented_output_channels_are_lists(path):
    """Uncommented, "# output_channels:mito,ld" was one channel named "mito,ld",
    and the processor's index() raised."""
    for line in open(path).read().splitlines():
        if line.lstrip("# ").strip().startswith("output_channels"):
            assert isinstance(yaml.safe_load(line.lstrip("# ").strip())["output_channels"], list), line


def test_the_pipeline_example_names_real_steps():
    from cellmap_flow.norm.input_normalize import get_normalizations
    from cellmap_flow.post.postprocessors import get_postprocessors

    pipeline = yaml.safe_load(open(os.path.join(ROOT, "example", "example_pipeline.yaml")))

    def steps(listed):
        return [{"name": s["name"], **(s.get("params") or {})} for s in listed]

    # An unknown name is skipped, silently.
    assert len(get_normalizations(steps(pipeline["input_normalizers"]))) == len(pipeline["input_normalizers"])
    assert len(get_postprocessors(steps(pipeline["postprocessors"]))) == len(pipeline["postprocessors"])


def _array(path, dtype=np.uint8):
    array = zarr.open_group(str(path.parent), mode="a").create_dataset(path.name, data=np.zeros((4, 4, 4), dtype))
    array.attrs.update(resolution=[8, 8, 8], offset=[0, 0, 0])
    return str(path)


def _config(tmp_path, models=None, extra_layers=None):
    config = {"data_path": _array(tmp_path / "raw.zarr" / "raw"), "charge_group": "grp", "models": models or {}}
    if extra_layers is not None:
        config["extra_layers"] = extra_layers
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(config))
    return str(tmp_path / "c.yaml")


@pytest.mark.parametrize("models, extra_layers, message", [
    pytest.param({"m": {"type": "nope"}}, None, "unrecognized type 'nope'", id="unknown-type"),
    pytest.param({"m": {"type": "cellmap", "config_folder": "/no/such/folder"}}, None, "Error creating model 'm' (cellmap)",
                 id="missing-model-folder"),
    pytest.param({}, [{"path": "/x.zarr"}], "extra_layers", id="extra-layer-without-a-name"),
    pytest.param({}, [{"name": "data", "path": "/x.zarr"}], "extra_layers", id="extra-layer-named-data"),
    pytest.param({}, [{"name": "x", "path": "/x.zarr", "layer_type": "points"}], "extra_layers",
                 id="extra-layer-of-an-unknown-type"),
])
def test_a_bad_config_is_reported_and_exits_non_zero(tmp_path, models, extra_layers, message):
    """config_utils called sys.exit(1); now the CLI turns ConfigError into a clean exit."""
    if "cellmap" in str(models):
        pytest.importorskip("cellmap_models.model_export.cellmap_model")
    result = CliRunner().invoke(yaml_cli.main, [_config(tmp_path, models, extra_layers), "--validate-only"])
    assert result.exit_code == 1 and message in result.output, result.output
    assert not isinstance(result.exception, ConfigError), "caught, not a traceback"
    assert g.extra_layers == {}


def test_extra_layers_are_shown_beside_the_raw_data(tmp_path, monkeypatch):
    from neuroglancer.viewer_base import ViewerBase

    monkeypatch.setattr(neuroglancer_utils.neuroglancer, "Viewer", ViewerBase)
    monkeypatch.setattr(neuroglancer_utils, "create_and_run_app", lambda **k: "url")
    monkeypatch.setattr(yaml_cli, "install_cleanup_handlers", lambda: None)
    config = _config(tmp_path, extra_layers=[
        {"name": "pred", "path": _array(tmp_path / "pred.zarr" / "mito"),
         "shader": "void main() { emitGrayscale(1.0); }", "blend": "additive"},
        {"name": "ids", "path": _array(tmp_path / "ids.zarr" / "s0", np.uint64), "layer_type": "segmentation",
         "disable_meshes": True},
    ])
    result = CliRunner().invoke(yaml_cli.main, [config])
    assert result.exit_code == 0, result.output
    layers = g.viewer.state.layers
    assert [layer.name for layer in layers] == ["data", "pred", "ids"]
    pred, ids = layers["pred"].to_json(), layers["ids"].to_json()
    assert (pred["type"], pred["blend"], pred["shader"]) == ("image", "additive", "void main() { emitGrayscale(1.0); }")
    assert ids["type"] == "segmentation" and ids["source"][0]["subsources"] == {"meshes": False}


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

    monkeypatch.setattr(yaml_cli, "start_hosts", start_hosts)
    monkeypatch.setattr(neuroglancer_utils, "generate_neuroglancer_url", lambda path, wrap_raw=True: viewers.append(path))
    models = [ScriptModelConfig(script_path=SCRIPT, name=n) for n in ("good", "bad")]
    with caplog.at_level(logging.ERROR):
        if viewer_opened:
            yaml_cli.run_multiple(models, RAW, "grp", "gpu_h100")
        else:
            with pytest.raises(JobStartError):  # an error, not an empty dashboard
                yaml_cli.run_multiple(models, RAW, "grp", "gpu_h100")
    assert viewers == ([RAW] if viewer_opened else [])
    assert any("bad" in r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR), "the failure is said"


def test_validate_only_writes_nothing_to_the_home_directory(tmp_path):
    """--validate-only saved the YAML's queue and charge group as the
    defaults of the next dashboard. A fresh interpreter, HOME empty, so
    anything written under it is caught."""
    home = tmp_path / "home"
    home.mkdir()
    config = tmp_path / "c.yaml"
    config.write_text(f"data_path: /d.zarr\ncharge_group: someone_elses_group\nqueue: gpu_h200\n"
                      f"models:\n  m: {{type: script, script_path: {SCRIPT}}}\n")
    env = {**os.environ, "HOME": str(home), "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"}
    result = subprocess.run([sys.executable, "-m", "cellmap_flow.cli.yaml_cli", str(config), "--validate-only"],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0 and "Configuration is valid" in result.stdout, result.stderr[-2000:]
    assert list(home.iterdir()) == [], [str(p) for p in home.rglob("*")]
