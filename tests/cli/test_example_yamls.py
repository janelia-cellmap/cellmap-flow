"""The YAML files shipped in example/ are valid for the tools they are for."""

import glob
import os

import pytest
import yaml
from click.testing import CliRunner

from cellmap_flow.cli.yaml_cli import main as yaml_main

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "example", "*.yaml")))
RUN_CONFIGS = [p for p in EXAMPLES if "data_path" in (yaml.safe_load(open(p)) or {})]


def test_there_are_examples_to_check():
    assert len(RUN_CONFIGS) >= 3


@pytest.fixture
def placeholder_cellmap_folders(monkeypatch):
    """Let a cellmap model entry name a folder that does not exist.

    CellMapModelConfig reads the folder's metadata.json as soon as it is
    constructed, and the examples use placeholder paths; what is being checked
    here is the YAML (types, required fields), not those files.
    """
    cellmap_model = pytest.importorskip("cellmap_models.model_export.cellmap_model")

    class Placeholder:
        def __init__(self, folder_path):
            self.folder_path = folder_path

    monkeypatch.setattr(cellmap_model, "CellmapModel", Placeholder)


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=os.path.basename)
def test_run_configs_pass_validate_only(path, placeholder_cellmap_folders):
    result = CliRunner().invoke(yaml_main, [path, "--validate-only"])
    assert result.exit_code == 0, result.output
    assert "Configuration is valid" in result.output


def test_a_missing_model_folder_is_reported_not_raised(tmp_path):
    pytest.importorskip("cellmap_models.model_export.cellmap_model")
    config = tmp_path / "c.yaml"
    config.write_text(
        "data_path: /d.zarr\ncharge_group: g\n"
        f"models:\n  m: {{type: cellmap, config_folder: {tmp_path / 'missing'}}}\n"
    )
    result = CliRunner().invoke(yaml_main, [str(config), "--validate-only"])
    assert result.exit_code == 1
    assert "Error creating model 'm' (cellmap)" in result.output


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=os.path.basename)
def test_output_channels_examples_are_lists(path):
    # The comment used to read "# output_channels:mito,ld"; uncommented, that
    # is one channel named "mito,ld", and the processor's index() raises.
    text = open(path).read()
    for line in text.splitlines():
        stripped = line.lstrip("# ").strip()
        if stripped.startswith("output_channels"):
            value = yaml.safe_load(stripped)
            assert isinstance(value, dict), stripped
            assert isinstance(value["output_channels"], list), stripped


def test_the_pipeline_example_names_real_steps():
    from cellmap_flow.norm.input_normalize import get_normalizations
    from cellmap_flow.post.postprocessors import get_postprocessors

    pipeline = yaml.safe_load(open(os.path.join(ROOT, "example", "example_pipeline.yaml")))

    def as_list(steps):
        return [{"name": s["name"], **(s.get("params") or {})} for s in steps]

    norms = get_normalizations(as_list(pipeline["input_normalizers"]))
    posts = get_postprocessors(as_list(pipeline["postprocessors"]))
    assert len(norms) == len(pipeline["input_normalizers"]), "an unknown normalizer is skipped"
    assert len(posts) == len(pipeline["postprocessors"]), "an unknown postprocessor is skipped"
