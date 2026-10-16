"""The onnx model type: the metadata it reads from a file without onnx or
onnxruntime, its geometry from a manifest the file carries, from the entry
alone and from a Hugging Face folder's manifest (the Hub a fake), the
variant it serves, what it feeds the session, its model entry, YAML
entries as the browser's Export Config writes them, and a forward through
the real onnxruntime on the browser tests' head.onnx when it is installed
(pixi's onnx environment has it; the test environments do not, so those
tests skip there). Geometry runs against a stand-in onnxruntime, so it is
checked everywhere."""

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
from cellmap_flow.models.configs import onnx as onnx_type
from cellmap_flow.models.configs.onnx import MANIFEST_KEY, choose_variant, parse_manifest, read_manifest, read_metadata
from cellmap_flow.models.models_config import OnnxModelConfig
from tests.utils.onnx_helpers import FINETUNED_MANIFEST, HEAD, with_metadata

ROOT = Path(__file__).resolve().parents[2]


# --- the file's metadata --------------------------------------------------------------

@pytest.fixture
def finetuned(tmp_path):
    """head.onnx with the finetuned manifest written in, saved as the browser's download names it."""
    path = tmp_path / "mito_aff_setup53-2d_16nm-ft1.onnx"
    path.write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: json.dumps(FINETUNED_MANIFEST), "x": "y"}))
    return path


def test_the_metadata_is_read_without_onnx_or_onnxruntime(finetuned, tmp_path):
    assert read_metadata(HEAD) == {}
    assert read_metadata(finetuned) == {MANIFEST_KEY: json.dumps(FINETUNED_MANIFEST), "x": "y"}
    manifest = read_manifest(finetuned)
    assert (manifest["channels"], manifest["input_voxel_size"], manifest["io"]) == (
        ["mito_aff_z", "mito_aff_y"], [16, 16, 16], {"input": "input", "output": "output"})
    assert read_manifest(HEAD) is None
    # A failed download saved under the model's name is not a model.
    page = tmp_path / "page.onnx"
    page.write_text("<!doctype html><html>rate limited</html>")
    with pytest.raises(ValueError, match="not an ONNX model"):
        read_metadata(page)
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: "{not json"}))
    with pytest.raises(ValueError, match="is not JSON"):
        read_manifest(broken)


def test_a_manifest_is_widened_as_the_browser_widens_it():
    """The exports' short forms: a cube's side for a block, io as a sentence."""
    manifest = parse_manifest({
        "model": "setup_54", "channels": ["a"], "input_voxel_size_nm": [16, 16, 16],
        "output_voxel_size_nm": [16, 16, 16], "output": "raw logits",
        "io": "input 'raw' float32 [1,1,z,y,x]; output 'logits' float32 [1,C,z',y',x']",
        "variants": [{"variant": "fp32", "input_size": 148, "output_size": 60, "file": "a.onnx"},
                     {"variant": "bf16", "input_size": 148, "output_size": 60, "file": "b.onnx"}],
    })
    assert manifest["io"] == {"input": "raw", "output": "logits"}
    assert manifest["variants"] == [{"variant": "fp32", "input_size": [148] * 3, "output_size": [60] * 3,
                                     "file": "a.onnx"}]
    assert manifest["input_norm"] == []
    for raw, message in [
        ({"channels": "a"}, "channels is not a list"),
        ({"channels": ["a"], "input_voxel_size_nm": [16, 16]}, "three positive numbers"),
        ({"channels": ["a"], "input_voxel_size_nm": [16] * 3, "output_voxel_size_nm": [16] * 3,
          "variants": [{"variant": "bf16"}]}, "no variant of a known precision"),
    ]:
        with pytest.raises(ValueError, match=message):
            parse_manifest(raw)


# --- a stand-in onnxruntime ----------------------------------------------------------------

class FakeSession:
    """An InferenceSession that has the graph's input and output, and runs zeros:
    the output's fixed sizes, and where it has none, the input's less ``crop``."""

    def __init__(self, inputs, outputs, providers, crop, channels):
        self._inputs, self._outputs, self.providers = inputs, outputs, providers
        self.crop, self.channels = crop, channels
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        self.feeds.append(feed)
        (block,) = feed.values()
        dims = [d if isinstance(d, int) else None for d in self._outputs[0].shape]
        spatial = [d or n - c for d, n, c in zip(dims[-3:], block.shape[2:], self.crop)]
        shape = [1, dims[1] or self.channels, *spatial] if len(dims) == 5 else [1, *spatial]
        return [np.zeros(shape, dtype=np.float32)]


def arg(name, shape, type_="tensor(float)"):
    return SimpleNamespace(name=name, shape=shape, type=type_)


@pytest.fixture
def ort(monkeypatch):
    """A stand-in onnxruntime: a session built has the graph ``ort.graph`` (inputs,
    outputs) and, where its output's sizes are not fixed, writes the input less
    ``ort.crop`` voxels a side in ``ort.channels`` channels. By default the fixture's."""
    fake = types.ModuleType("onnxruntime")
    fake.__version__ = "0.0-fake"
    fake.graph = ([arg("input", [1, 1, 1, 8, 8])], [arg("output", [1, 2, 1, 6, 6])])
    fake.crop, fake.channels = (0, 2, 2), 2
    fake.sessions = []
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    fake.disable_telemetry_events = lambda: None

    def session(path, providers=None):
        built = FakeSession(*fake.graph, providers, fake.crop, fake.channels)
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


def test_the_entry_wins_over_the_manifest(finetuned, ort):
    """A 16 nm model run at 8 nm writes 8 nm voxels, as the manifest's ratio says."""
    model = OnnxModelConfig(path=str(finetuned), input_voxel_size=8, channels="z, y")
    assert _geometry(model)[:5] == ((8, 8, 8), (8, 8, 8), (64, 64, 64), (64, 48, 48), ["z", "y"])
    # A slice in is a slice out, at the input's z spacing.
    assert _geometry(OnnxModelConfig(path=str(finetuned), output_voxel_size=[32, 16, 16]))[1] == (16, 16, 16)
    with pytest.raises(ValueError, match="outputs 2 channel"):
        OnnxModelConfig(path=str(finetuned), channels=["only"]).config


def test_a_file_without_a_manifest_takes_its_voxel_size_from_the_entry_and_its_blocks_from_the_graph(ort):
    model = registry.build_model({"type": "onnx", "path": str(HEAD), "resolution": 8}, "plain")
    assert _geometry(model) == ((8, 8, 8), (8, 8, 8), (64, 64, 64), (64, 48, 48),
                                ["channel_0", "channel_1"], [8, 6, 6, 2])
    assert model.declared_input_norm() is None
    with pytest.raises(ValueError, match="does not say its input_voxel_size"):
        OnnxModelConfig(path=str(HEAD)).config


def test_a_graph_of_symbolic_sizes_takes_them_from_the_entry(ort):
    """A 3D model exported with dynamic axes: its blocks and channels are the entry's."""
    ort.graph = ([arg("raw", ["batch", 1, "z", "y", "x"])], [arg("out", ["batch", "c", "z", "y", "x"])])
    with pytest.raises(ValueError, match="does not say its input_size"):
        OnnxModelConfig(path=str(HEAD), input_voxel_size=8).config
    with pytest.raises(ValueError, match="no fixed number of channels"):
        OnnxModelConfig(path=str(HEAD), input_voxel_size=8, input_size=20, output_size=18).config
    ort.crop, ort.channels = (2, 2, 2), 1
    entry = {"type": "onnx", "path": str(HEAD), "input_voxel_size": [8, 4, 4], "input_size": 20, "output_size": 18,
             "classes": ["mito"]}
    model = registry.build_model(entry, "dyn")
    assert _geometry(model) == ((8, 4, 4), (8, 4, 4), (160, 80, 80), (144, 72, 72), ["mito"], [18, 18, 18, 1])
    assert model.config.slice_wise is False
    model.config.model(torch.zeros(1, 1, 20, 20, 20))
    assert ort.sessions[-1].feeds[-1]["raw"].shape == (1, 1, 20, 20, 20)


@pytest.mark.parametrize("graph, message", [
    (([arg("input", [1, 3, 8, 8])], [arg("output", [1, 2, 6, 6])]), r"feeds an ONNX model \[1, 1, z, y, x\]"),
    (([arg("input", [1, 1, 1, 8, 8])], [arg("output", [1, 2, 1, 4, 4])]), r"has an output of \[1, 4, 4\]"),
])
def test_a_graph_that_is_not_what_the_browser_feeds_is_refused(finetuned, ort, graph, message):
    ort.graph = graph
    with pytest.raises(ValueError, match=message):
        OnnxModelConfig(path=str(finetuned)).config


def test_an_fp16_graph_is_fed_float16_and_serves_float32(finetuned, ort):
    ort.graph = ([arg("input", [1, 1, 1, 8, 8], "tensor(float16)")], [arg("output", [1, 2, 1, 6, 6])])
    model = OnnxModelConfig(path=str(finetuned))
    out = model.config.model(torch.zeros(1, 1, 8, 8, 8, dtype=torch.float16))
    assert ort.sessions[-1].feeds[-1]["input"].dtype == np.float16 and out.dtype == torch.float32


def test_an_off_centre_output_is_refused(tmp_path, ort):
    """An output an odd number of voxels smaller would sit half a voxel off its input's grid."""
    manifest = dict(FINETUNED_MANIFEST, variants=[dict(FINETUNED_MANIFEST["variants"][0], output_size=[1, 5, 5])])
    path = tmp_path / "odd.onnx"
    path.write_bytes(with_metadata(HEAD.read_bytes(), {MANIFEST_KEY: json.dumps(manifest)}))
    ort.graph = ([arg("input", [1, 1, 1, 8, 8])], [arg("output", [1, 2, 1, None, None])])
    with pytest.raises(ValueError, match="not centred on whole voxels"):
        OnnxModelConfig(path=str(path)).config


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
    assert _geometry(model) == ((16, 16, 16), (16, 16, 16), (2368,) * 3, (960,) * 3,
                                ["mito_aff_z", "mito_aff_y", "mito_aff_x"], [60, 60, 60, 3])
    assert hub == [("cellmap/onnx-web-models", "setup_54/manifest.json", None),
                   ("cellmap/onnx-web-models", "setup_54/setup_54_in148_fp32.onnx", None)]
    assert ort.sessions[-1].path.endswith("setup_54/setup_54_in148_fp32.onnx")
    assert model.config.slice_wise is False


@pytest.mark.parametrize("entry, file", [
    ({"file": "setup_54_in116_int8w.onnx"}, "setup_54_in116_int8w.onnx"),
    ({"variant": "fp16", "input_size": 132}, "setup_54_in132_fp16.onnx"),
    ({"variant": "fp16"}, "setup_54_in148_fp16.onnx"),
])
def test_a_hub_folders_variant_is_chosen_by_file_or_precision_and_block(hub, ort, entry, file):
    ort.graph = ([arg("input", [1, 1, None, None, None])], [arg("output", [1, 3, None, None, None])])
    ort.crop, ort.channels = (88, 88, 88), 3
    model = OnnxModelConfig(folder="setup_54", repo="me/exports", revision="v2", **entry)
    model.config
    assert hub[-1] == ("me/exports", f"setup_54/{file}", "v2")


def test_a_variant_the_manifest_lacks_is_refused_with_the_ones_it_has():
    manifest = parse_manifest(SETUP_54)
    with pytest.raises(ValueError, match=r"no fp16 with an input of 100 variant; it has setup_54_in148_fp32.onnx"):
        choose_variant(manifest, variant="fp16", input_size=(100,))
    with pytest.raises(ValueError, match="no variant in other.onnx"):
        choose_variant(manifest, file="other.onnx")


def test_the_manifest_of_a_hub_folder_is_read_once_and_a_failure_retried_after_a_minute(hub, monkeypatch):
    """The dashboard's model advice asks on every poll."""
    model = OnnxModelConfig(folder="setup_54")
    assert model.declared_input_norm() == SETUP_54["input_norm"]
    model.declared_input_norm()
    assert len(hub) == 1
    asked = []

    def unreachable(repo, filename, revision=None):
        asked.append(filename)
        raise OSError("the Hub is unreachable")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", unreachable)
    failing = OnnxModelConfig(folder="gone")
    assert failing.declared_input_norm() is None
    now = onnx_type.time.monotonic()
    monkeypatch.setattr(onnx_type.time, "monotonic", lambda: now + 30)
    assert failing.declared_input_norm() is None and asked == ["gone/manifest.json"]
    monkeypatch.setattr(onnx_type.time, "monotonic", lambda: now + 61)
    assert failing.declared_input_norm() is None and len(asked) == 2


# --- the model entry --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({}, "a path .* or a folder"),
    ({"path": "/a.onnx", "folder": "setup_54"}, "and not both"),
    ({"path": "/a.onnx", "file": "b.onnx"}, "a path is the file itself"),
    ({"folder": "setup_54", "variant": "bf16"}, "variant must be one of"),
    ({"folder": "setup_54", "slices_per_chunk": 0}, "at least 1"),
])
def test_an_entry_that_names_no_one_model_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OnnxModelConfig(**kwargs)


def test_the_entry_says_only_what_it_was_given(finetuned):
    model = OnnxModelConfig(path=str(finetuned), input_voxel_size="16,16,16", input_size=8, channels="a, b",
                            slices_per_chunk=4, name="ft")
    assert model.to_dict() == {"type": "onnx", "path": str(finetuned), "input_voxel_size": [16, 16, 16],
                               "input_size": 8, "channels": ["a", "b"], "slices_per_chunk": 4, "name": "ft"}
    assert OnnxModelConfig(folder="setup_54", file="x.onnx").to_dict() == {
        "type": "onnx", "folder": "setup_54", "file": "x.onnx", "name": "setup_54"}
    assert OnnxModelConfig.finetunable is False and OnnxModelConfig(folder="setup_54").finetune_modes() == ()


# What the browser's Export Config writes (web/src/backend/routes/config.ts
# exportYaml, as web/test/unit/backend-routes.test.ts pins it), with the
# cluster's settings added.
EXPORTED = """# CellMap-Flow exported configuration
# Reload with: cellmap_flow yaml <this file>
#
data_path: {data}
resample: true
json_data:
  input_norm:
  - name: MinMaxNormalizer
    min_value: 0
    max_value: 255
  - name: LambdaNormalizer
    expression: 2*x-1
  postprocess:
  - name: SigmoidPostprocessor
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
queue: gpu_h100
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
    assert result.exit_code == 0 and "Configuration is valid" in result.output, result.output
    assert [m.path or m.folder for m in get_session().models_config] == ["setup_54", str(folder / finetuned.name)]
    # Without the YAML's folder (a model entry from elsewhere), as before: the working directory's.
    assert registry.build_model({"type": "onnx", "path": "x.onnx"}, "x").path == str(tmp_path / "x.onnx")


def test_its_server_runs_in_the_onnx_environment():
    import tomllib

    assert OnnxModelConfig.default_env == "onnx"
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text())
    assert pixi["environments"]["onnx"] == {"features": ["onnx"]}
    assert "onnxruntime-gpu" in pixi["feature"]["onnx"]["target"]["linux-64"]["pypi-dependencies"]


def test_building_an_entry_imports_no_onnxruntime(finetuned):
    """The CLIs and the dashboard build entries, and their environment has none."""
    code = (
        "import sys; from cellmap_flow.models.models_config import OnnxModelConfig; "
        f"m = OnnxModelConfig(path={str(finetuned)!r}); m.to_dict(); m.declared_input_norm(); "
        "assert 'onnxruntime' not in sys.modules and 'onnx' not in sys.modules, sorted(sys.modules)"
    )
    import subprocess

    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


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


def test_the_runner_warms_it_up_and_sees_its_logits_unbounded(finetuned):
    """The probe the dashboard's postprocessing advice reads: an export's logits
    are unbounded, so it suggests the sigmoid (or the affinity chain)."""
    pytest.importorskip("onnxruntime", reason="onnxruntime is in pixi's onnx environment, not this one")
    from cellmap_flow.inference.runner import ModelRunner

    runner = ModelRunner(OnnxModelConfig(path=str(finetuned)))
    assert runner.output_class == "unbounded"
