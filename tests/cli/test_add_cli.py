"""`cellmap_flow add`: the entry it prints for a config's `models:`, and `--run`,
which serves it as `cellmap_flow yaml` would. Offline or with the Hub faked:
nothing here reaches the network."""

import os

import pytest
import yaml
from click.testing import CliRunner

from cellmap_flow.cli import main, yaml_cli
from cellmap_flow.dashboard.state import get_session
from cellmap_flow.jobs import launch, settings
from cellmap_flow.models import resolve as resolve_module

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tests", "script_test", "fake_model_script.py")
RAW = os.path.join(ROOT, "tests", "script_test", "dummy.zarr", "raw")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # Recorded rather than raised: resolve takes an unreachable Hub for a
    # reason to assume, and would swallow the error.
    asked = []
    monkeypatch.setattr(resolve_module, "_hf_files", lambda *args: asked.append(args) or [])
    monkeypatch.setattr(resolve_module, "_zoo_entries", lambda: asked.append("zoo") or [])
    yield
    assert asked == [], "looked up online"


def _add(*args):
    return CliRunner().invoke(main.cli, ["add", *args])


def test_it_prints_the_entry_as_yaml_for_a_config(monkeypatch):
    monkeypatch.setattr(resolve_module, "_hf_files", lambda repo, revision: ["metadata.json", "model.ts"])
    result = _add("cellmap/mito", "-n", "mito")
    assert result.exit_code == 0, result.output
    assert result.output == (
        "# huggingface: a cellmap-models export on Hugging Face\n"
        "# runs in this environment\n"
        "models:\n"
        "  mito:\n"
        "    type: huggingface\n"
        "    repo: cellmap/mito\n"
    )


def test_what_it_still_needs_is_said_and_lists_stay_on_one_line():
    needs = _add("cpsam", "--offline")
    assert "# runs in the cellpose4 environment (its type's default)\n" in needs.output
    assert "# still needs: voxel_size (add to the entry)\n" in needs.output
    given = _add("cpsam", "-v", "16,8,8")
    assert given.output.endswith("    voxel_size: [16, 8, 8]\n")
    assert yaml.safe_load(given.output) == {
        "models": {"cpsam": {"type": "cellpose", "pretrained_model": "cpsam", "voxel_size": [16, 8, 8]}}}


def test_a_reference_it_cannot_resolve_is_an_error():
    result = _add("unet", "--offline")
    assert result.exit_code == 1 and "Could not tell what model 'unet' is" in result.output


@pytest.fixture
def served(monkeypatch, tmp_path):
    """What --run hands `cellmap_flow yaml`'s run, which starts nothing here."""
    ran = []
    monkeypatch.setattr(settings, "SERVER_CONFIG_PATH", str(tmp_path / "server_config.yaml"))
    monkeypatch.setattr(launch, "install_cleanup_handlers", lambda: None)
    monkeypatch.setattr(yaml_cli, "run_multiple", lambda *args, **kwargs: ran.append((args, kwargs)))
    return ran


def test_run_serves_the_model_as_a_yaml_would(served):
    result = _add(SCRIPT, "--run", "-d", RAW, "-q", "gpu_h200", "-P", "grp", "--resample")
    assert result.exit_code == 0, result.output + repr(result.exception)
    (models, data_path, project, queue), kwargs = served[0]
    assert [(type(m).__name__, m.name) for m in models] == [("ScriptModelConfig", "fake_model_script")]
    assert (data_path, project, queue, kwargs) == (RAW, "grp", "gpu_h200", {"resample": True})
    assert get_session().models_config == models and get_session().resample is True
    assert settings.launcher_settings().queue == "gpu_h200"


@pytest.mark.parametrize("args, message", [
    (["cpsam", "--run"], "--run needs --data-path"),
    (["cpsam", "--run", "-d", RAW], "cpsam still needs voxel_size"),
])
def test_run_refuses_what_it_cannot_serve(served, args, message):
    result = _add(*args)
    assert result.exit_code != 0 and message in result.output
    assert served == []
