"""The onnx model type: its geometry from a manifest the file carries, from
the entry alone and from a Hugging Face folder's manifest (the Hub a fake),
what it feeds the session, YAML entries as the browser's Export Config writes
them, and a forward through the real onnxruntime on the browser tests'
head.onnx when it is installed (pixi's onnx environment has it; the test
environments do not, so that test skips there). Geometry runs against a
stand-in onnxruntime, so it is checked everywhere."""

import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from cellmap_flow.models import registry
from cellmap_flow.models.configs.onnx import MANIFEST_KEY
from cellmap_flow.models.models_config import OnnxModelConfig
from tests.utils.onnx_helpers import FINETUNED_MANIFEST, HEAD, with_metadata

ROOT = Path(__file__).resolve().parents[2]


# --- the browser's download --------------------------------------------------------------

@pytest.fixture
def finetuned(tmp_path):
    """head.onnx with the finetuned manifest written in, saved as the browser's download names it."""
    path = tmp_path / "mito_aff_setup53-2d_16nm-ft1.onnx"
    path.write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: json.dumps(FINETUNED_MANIFEST), "x": "y"}))
    return path


# --- a stand-in onnxruntime ----------------------------------------------------------------

class FakeSession:
    """An InferenceSession that has the graph's input and output, and runs
    zeros of the output's fixed sizes."""

    def __init__(self, inputs, outputs, providers):
        self._inputs, self._outputs, self.providers = inputs, outputs, providers
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        self.feeds.append(feed)
        return [np.zeros(self._outputs[0].shape, dtype=np.float32)]


def arg(name, shape, type_="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=type_)


@pytest.fixture
def ort(monkeypatch):
    """A stand-in onnxruntime: a session built has the graph ``ort.graph`` (inputs,
    outputs), by default the fixture's."""
    fake = types.ModuleType("onnxruntime")
    fake.__version__ = "0.0-fake"
    fake.graph = ([arg("input", [1, 1, 1, 8, 8])], [arg("output", [1, 2, 1, 6, 6])])
    fake.sessions = []
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake.disable_telemetry_events = lambda: None

    def session(path, providers=None):
        built = FakeSession(*fake.graph, providers)
        built.path = str(path)
        fake.sessions.append(built)
        return built

    fake.InferenceSession = session
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return fake


def _geometry(model):
    config = model.config
    return (tuple(config.input_voxel_size), tuple(config.output_voxel_size), tuple(config.read_shape),
            tuple(config.write_shape), config.channels, config.block_shape.tolist())


# --- geometry -----------------------------------------------------------------------------

def test_a_finetuned_file_says_its_geometry_itself(finetuned, ort):
    """What the browser downloads: a 2D student, its blocks one slice thick,
    served 8 slices a chunk, each its own call, as the browser feeds it."""
    model = OnnxModelConfig(path=str(finetuned))
    assert model.name == "mito_aff_setup53-2d_16nm-ft1"
    assert _geometry(model) == ((16, 16, 16), (16, 16, 16), (128, 128, 128), (128, 96, 96),
                                ["mito_aff_z", "mito_aff_y"], [8, 6, 6, 2])
    assert model.config.output_kind == "raw logits (apply sigmoid / threshold at 0)"
    assert model.geometry.context == (0, 16, 16)
    out = model.config.model(torch.zeros(1, 1, 8, 8, 8))
    assert tuple(out.shape) == (1, 2, 8, 6, 6) and out.dtype == torch.float32
    (session,) = ort.sessions
    assert [tuple(f["input"].shape) for f in session.feeds[-8:]] == [(1, 1, 1, 8, 8)] * 8
    assert all(f["input"].dtype == np.float32 for f in session.feeds)
    assert OnnxModelConfig(path=str(finetuned), slices_per_chunk=2).config.block_shape.tolist() == [2, 6, 6, 2]
    # The dashboard's model advice offers the manifest's normalization.
    assert model.declared_input_norm() == FINETUNED_MANIFEST["input_norm"]


# --- a Hugging Face folder ---------------------------------------------------------------------

SETUP_54 = {
    "model": "setup_54", "task": "mito_affinities", "channels": ["mito_aff_z", "mito_aff_y", "mito_aff_x"],
    "input_voxel_size_nm": [16, 16, 16], "output_voxel_size_nm": [16, 16, 16],
    "output": "raw logits (apply sigmoid / threshold at 0)",
    "input_norm": [{"name": "MinMaxNormalizer", "min_value": 0, "max_value": 255}],
    "io": "input 'input' float32 [1,1,z,y,x]; output 'output' float32 [1,C,z',y',x']",
    "variants": [{"variant": v, "input_size": n, "output_size": n - 88, "file": f"setup_54_in{n}_{v}.onnx"}
                 for n in (148, 132, 116) for v in ("fp32", "fp16", "int8w")],
}


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A stand-in huggingface_hub: setup_54's manifest, and each .onnx as an empty file; what was downloaded."""
    calls = []

    def hf_hub_download(repo, filename, revision=None):
        calls.append((repo, filename, revision))
        path = tmp_path / "hub" / repo / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("manifest.json"):
            path.write_text(json.dumps(SETUP_54))
        else:
            path.write_bytes(b"")
        return str(path)

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_a_hub_folder_serves_fp32_at_its_largest_block_by_default(hub, ort):
    ort.graph = ([arg("input", [1, 1, 148, 148, 148])], [arg("output", [1, 3, 60, 60, 60])])
    model = registry.build_model({"type": "onnx", "folder": "setup_54"}, "mito")
    # As the dashboard lists it before its server is up: from the manifest alone.
    assert model.declared_geometry() == {"write_shape": [960] * 3, "output_voxel_size": [16] * 3, "output_channels": 3}
    assert hub == [("cellmap/onnx-web-models", "setup_54/manifest.json", None)] and not ort.sessions
    assert _geometry(model) == ((16, 16, 16), (16, 16, 16), (2368,) * 3, (960,) * 3,
                                ["mito_aff_z", "mito_aff_y", "mito_aff_x"], [60, 60, 60, 3])
    assert hub == [("cellmap/onnx-web-models", "setup_54/manifest.json", None),
                   ("cellmap/onnx-web-models", "setup_54/setup_54_in148_fp32.onnx", None)]
    assert ort.sessions[-1].path.endswith("setup_54/setup_54_in148_fp32.onnx")
    assert model.config.slice_wise is False


# --- the browser's exported YAML -------------------------------------------------------------

# The model entries the browser's Export Config writes (web/src/backend/
# routes/config.ts exportYaml, as web/test/unit/backend-routes.test.ts pins
# it), under the data_path a YAML needs and the charge_group a cluster does.
EXPORTED = """data_path: {data}
models:
- type: onnx
  name: mito_aff_setup54_16nm
  repo: cellmap/onnx-web-models
  folder: setup_54
  file: setup_54_mito_affinities_16nm_in132_fp16.onnx
- type: onnx
  name: mito_aff_setup53-2d_16nm-ft1
  path: mito_aff_setup53-2d_16nm-ft1.onnx  # the file Download finetuned model saves: put it beside this YAML
charge_group: cellmap
"""


def test_the_browsers_exported_yaml_is_read_its_relative_path_from_the_yamls_folder(tmp_path, finetuned, monkeypatch):
    from cellmap_flow.cli import yaml_cli
    from cellmap_flow.config.yaml import load_config
    from cellmap_flow.dashboard.state import get_session

    folder = tmp_path / "exported"
    folder.mkdir()
    shutil.copy(finetuned, folder / finetuned.name)
    config_path = folder / "config.yaml"
    config_path.write_text(EXPORTED.format(data=ROOT / "tests" / "script_test" / "dummy.zarr" / "raw"))
    monkeypatch.chdir(tmp_path)  # elsewhere: the path is the YAML's folder's, not the working directory's
    hub_model, finetuned_model = registry.build_models(load_config(str(config_path))["models"], base_dir=str(folder))
    assert (hub_model.folder, hub_model.file, hub_model.name) == (
        "setup_54", "setup_54_mito_affinities_16nm_in132_fp16.onnx", "mito_aff_setup54_16nm")
    assert finetuned_model.path == str(folder / finetuned.name)
    assert finetuned_model.launch_entry == {"type": "onnx", "path": str(folder / finetuned.name),
                                            "name": "mito_aff_setup53-2d_16nm-ft1"}
    result = CliRunner().invoke(yaml_cli.main, [str(config_path), "--validate-only"])
    assert result.exit_code == 0, result.output
    assert [m.path or m.folder for m in get_session().models_config] == ["setup_54", str(folder / finetuned.name)]
    # Without the YAML's folder (a model entry from elsewhere), as before: the working directory's.
    assert registry.build_model({"type": "onnx", "path": "x.onnx"}, "x").path == str(tmp_path / "x.onnx")


# --- the real onnxruntime ------------------------------------------------------------------------

def test_a_forward_through_onnxruntime_is_the_browsers_slice_by_slice(finetuned):
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is in pixi's onnx environment, not this one")
    session = ort.InferenceSession(str(finetuned), providers=["CPUExecutionProvider"])
    # onnxruntime reads the manifest where the browser writes it.
    assert session.get_modelmeta().custom_metadata_map[MANIFEST_KEY] == json.dumps(FINETUNED_MANIFEST)
    model = OnnxModelConfig(path=str(finetuned), slices_per_chunk=3)
    block = torch.rand(1, 1, 3, 8, 8)
    out = model.config.model(block).numpy()
    expected = [session.run(["output"], {"input": block[:, :, z:z + 1].numpy()})[0] for z in range(3)]
    assert out.shape == (1, 2, 3, 6, 6)
    np.testing.assert_array_equal(out, np.concatenate(expected, axis=2))