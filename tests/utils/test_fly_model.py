"""The fly model type: what it reads beside a checkpoint, what it takes as
given, its sigmoid for each kind of checkpoint, and its entry. Against a
stand-in ``fly_organelles.model.StandardUnet``: fly_organelles is not in the
test environments (it has an environment of its own, fly)."""

import json
import os
import subprocess
import sys
import types

import pytest
import torch

from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.models import registry
from cellmap_flow.models.configs.fly import _ends_in_sigmoid, run_metadata, unet_arguments
from cellmap_flow.models.models_config import FlyModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StandardUnet(torch.nn.Module):
    """fly_organelles' StandardUnet as far as this type sees it: its arguments
    and the names of its weights. Each level's convolutions run one after the
    other, unpooled, so an input loses (kernel - 1) voxels per convolution."""

    def __init__(self, out_channels, num_fmaps=16, fmap_inc_factor=6, downsample_factors=None,
                 kernel_size_down=None, kernel_size_up=None):
        super().__init__()
        self.unet_backbone = torch.nn.Module()
        self.unet_backbone.l_conv = torch.nn.ModuleList()
        channels = 1
        for level, kernels in enumerate(kernel_size_down):
            layers = []
            for kernel in kernels:
                layers += [torch.nn.Conv3d(channels, num_fmaps * fmap_inc_factor**level, kernel), torch.nn.ReLU()]
                channels = num_fmaps * fmap_inc_factor**level
            level_module = torch.nn.Module()
            level_module.conv_pass = torch.nn.Sequential(*layers)
            self.unet_backbone.l_conv.append(level_module)
        self.final_conv = torch.nn.Conv3d(channels, out_channels, 1)

    def forward(self, x):
        for level in self.unet_backbone.l_conv:
            x = level.conv_pass(x)
        return self.final_conv(x)


# Two levels: 12 voxels in lose 2 + 2 + 2, so 6 come out.
ARGUMENTS = {"out_channels": 2, "num_fmaps": 2, "fmap_inc_factor": 3, "downsample_factors": [(2, 2, 2)],
             "kernel_size_down": [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3)]]}


@pytest.fixture
def fly_organelles(monkeypatch):
    package, model = types.ModuleType("fly_organelles"), types.ModuleType("fly_organelles.model")
    model.StandardUnet = StandardUnet
    package.model = model
    monkeypatch.setitem(sys.modules, "fly_organelles", package)
    monkeypatch.setitem(sys.modules, "fly_organelles.model", model)


def _training_checkpoint(run_dir, name="model_checkpoint_1000"):
    """A training checkpoint as fly_organelles saves one: weights and optimizer state."""
    torch.manual_seed(0)
    network = StandardUnet(**ARGUMENTS)
    path = run_dir / name
    torch.save({"model_state_dict": network.state_dict(), "optimizer_state_dict": {}}, path)
    return str(path), network


def _snapshot(run_dir, input_size=178, output_size=56, channels=1, voxel_size=16, iteration=1001):
    """A training snapshot: zarr v2 raw and output arrays, (batch, channel, z, y, x)."""
    for array, size, count in (("raw", input_size, 1), ("output", output_size, channels)):
        folder = run_dir / "snapshots" / f"{iteration:08d}.zarr" / array
        folder.mkdir(parents=True)
        (folder / ".zarray").write_text(json.dumps({"shape": [14, count, size, size, size], "zarr_format": 2}))
        (folder / ".zattrs").write_text(json.dumps({"offset": [0, 0, 0], "voxel_size": [voxel_size] * 3}))


TRAIN_PY = """
import sys
sys.exit("a training script, which must not run")
voxel_size = (16, 16, 16)
labels = ["mito", "er"]
model = StandardUnet(len(labels))
"""


def test_a_training_runs_files_fill_what_is_not_given(tmp_path):
    """The salivary runs' layout: train.py names the labels, the snapshots
    the tile and voxel sizes, and nothing else does."""
    _snapshot(tmp_path, channels=2)
    (tmp_path / "train.py").write_text(TRAIN_PY)
    model = FlyModelConfig(checkpoint_path=str(tmp_path / "model_checkpoint_20000"), name="mito")
    assert (model.channels, model.input_voxel_size, model.output_voxel_size, model.input_size, model.output_size) == (
        ["mito", "er"], (16, 16, 16), (16, 16, 16), (178, 178, 178), (56, 56, 56))
    # Written out, so an exported YAML says what is served.
    assert model.to_dict() == {
        "type": "fly", "checkpoint_path": str(tmp_path / "model_checkpoint_20000"), "channels": ["mito", "er"],
        "input_voxel_size": [16, 16, 16], "output_voxel_size": [16, 16, 16], "name": "mito",
        "input_size": [178, 178, 178], "output_size": [56, 56, 56],
    }


def test_given_arguments_win_and_an_input_size_does_not_take_the_snapshots_output_size(tmp_path):
    _snapshot(tmp_path, channels=2)
    (tmp_path / "train.py").write_text(TRAIN_PY)
    model = FlyModelConfig(checkpoint_path=str(tmp_path / "model_checkpoint_1000"), channels="nuc,ves",
                           input_voxel_size=8, input_size=338)
    # One voxel size given is both: the snapshots' are the network at 16 nm.
    assert (model.channels, model.input_voxel_size, model.output_voxel_size, model.input_size, model.output_size) == (
        ["nuc", "ves"], (8, 8, 8), (8, 8, 8), (338, 338, 338), None)


@pytest.mark.parametrize("filename, content, found", [
    pytest.param("metadata.json", json.dumps({
        "channels_names": ["mito", "er"], "input_voxel_size": [8, 8, 8], "output_voxel_size": [8, 8, 8],
        "input_shape": [1, 1, 178, 178, 178], "output_shape": [56, 56, 56]}),
        {"channels": (["mito", "er"], "metadata.json"), "input_voxel_size": ([8, 8, 8], "metadata.json"),
         "output_voxel_size": ([8, 8, 8], "metadata.json"), "sizes": (([178] * 3, [56] * 3), "metadata.json")},
        id="a-cellmap-export"),
    pytest.param("config.yaml", "run: {labels: [mito, er], voxel_size: 8}\n"
                                "checkpoint: {input_shape: [194, 194, 194], output_shape: [72, 72, 72]}\n",
        {"channels": (["mito", "er"], "config.yaml"), "input_voxel_size": (8, "config.yaml"),
         "output_voxel_size": (8, "config.yaml"), "sizes": (([194] * 3, [72] * 3), "config.yaml")},
        id="fly-organelles-run-config"),
    pytest.param("config.yaml", "run: {labels: [mito], voxel_size: 8, lsd: true}\n",
        {"input_voxel_size": (8, "config.yaml"), "output_voxel_size": (8, "config.yaml")},
        id="an-lsd-runs-labels-are-not-its-channels"),
    pytest.param("config.yaml", "run: [unclosed\n", {}, id="an-unreadable-file-is-skipped"),
])
def test_what_each_file_beside_a_checkpoint_says(tmp_path, filename, content, found):
    (tmp_path / filename).write_text(content)
    assert run_metadata(str(tmp_path / "model.pt")) == found


def test_labels_that_do_not_name_each_channel_are_not_used(tmp_path):
    """An affinity run trains one label into many channels."""
    _snapshot(tmp_path, channels=19)
    (tmp_path / "train.py").write_text('labels = ["mito"]\n')
    with pytest.raises(ValueError, match="needs channels: give it, or put beside the checkpoint"):
        FlyModelConfig(checkpoint_path=str(tmp_path / "model_checkpoint_1000"))
    assert FlyModelConfig(checkpoint_path=str(tmp_path / "model_checkpoint_1000"), channels=["a"] * 19).channels


def test_a_computed_value_in_the_training_script_is_unknown(tmp_path):
    (tmp_path / "train.py").write_text('voxel_size = (8, 8, 8)\nlabels = ["mito"]\nvoxel_size = tuple(v * 2 for v in voxel_size)\n')
    assert run_metadata(str(tmp_path / "model_checkpoint_1000")) == {"channels": (["mito"], "train.py")}


def test_one_voxel_size_stands_for_the_other_and_fractions_are_kept(tmp_path):
    """_as_int_tuple truncated 5.24 nm to 5, putting every chunk on the wrong grid."""
    model = FlyModelConfig(checkpoint_path=str(tmp_path / "c.ts"), channels=["mito"], output_voxel_size="5.24,4,4",
                           input_size=(12, 12, 12), output_size=(6, 6, 6))
    model._model = torch.nn.Conv3d(1, 1, 7)
    config = model.config
    assert config.input_voxel_size == config.output_voxel_size == (5.24, 4, 4)
    assert config.read_shape == pytest.approx((62.88, 48, 48)) and config.write_shape == pytest.approx((31.44, 24, 24))


def test_a_cellmap_export_folder_is_the_cellmap_types(tmp_path):
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "metadata.json").write_text("{}")
    with pytest.raises(ConfigError, match=f"serve it with type: cellmap, folder_path: {tmp_path / 'export'}"):
        registry.build_model({"type": "fly", "checkpoint": str(tmp_path / "export")}, "m")
    with pytest.raises(ValueError, match="is a folder: give the checkpoint file in it"):
        FlyModelConfig(checkpoint_path=str(tmp_path), channels=["mito"], input_voxel_size=8)


def test_a_training_checkpoint_is_built_from_its_weights_shapes(tmp_path, fly_organelles):
    path, network = _training_checkpoint(tmp_path)
    assert unet_arguments(network.state_dict()) == ARGUMENTS
    model = FlyModelConfig(checkpoint_path=path, channels=["mito", "er"], input_voxel_size=8, input_size=12)
    x = torch.rand(1, 1, 12, 12, 12)
    with torch.no_grad():
        assert torch.equal(model.config.model(x), torch.sigmoid(network(x)))
    assert tuple(model.config.write_shape) == (48, 48, 48)

    with pytest.raises(ValueError, match="outputs 2 channels, but 1 are named"):
        FlyModelConfig(checkpoint_path=path, channels=["mito"], input_voxel_size=8, input_size=12).config


def test_with_no_size_the_training_tile_goes_in_and_the_network_says_what_comes_out(tmp_path, fly_organelles):
    path, _ = _training_checkpoint(tmp_path)
    model = FlyModelConfig(checkpoint_path=path, channels=["mito", "er"], input_voxel_size=8)
    model.validate_model_shapes = False
    assert (tuple(model.config.read_shape), tuple(model.config.write_shape)) == ((178 * 8,) * 3, (172 * 8,) * 3)
    # The computed size stays out of the entry: a rebuilt model computes it again.
    assert "output_size" not in model.to_dict()


def _bare():
    torch.manual_seed(0)
    return torch.nn.Conv3d(1, 1, 3)


@pytest.mark.parametrize("filename, network, sigmoid, adds", [
    pytest.param("c.ts", lambda: torch.jit.script(_bare()), True, True, id="torchscript-without-one"),
    pytest.param("model.ts", lambda: torch.jit.script(torch.nn.Sequential(_bare(), torch.nn.Sigmoid())), True, False,
                 id="a-cellmap-exports-torchscript-gets-no-second"),
    pytest.param("model.pt", lambda: torch.nn.Sequential(_bare(), torch.nn.Sigmoid()), True, False,
                 id="eager-with-one"),
    pytest.param("model.pt", _bare, True, True, id="eager-without-one"),
    pytest.param("c.ts", lambda: torch.jit.script(_bare()), False, False, id="false-adds-none"),
])
def test_the_output_goes_through_one_sigmoid(tmp_path, monkeypatch, filename, network, sigmoid, adds):
    """A .ts always got one, so a cellmap export's, which ends in its own,
    got two; a model.pt never did."""
    monkeypatch.setenv("CELLMAP_FLOW_ALLOW_PICKLE", "1")
    network = network()
    path = str(tmp_path / filename)
    torch.jit.save(network, path) if filename.endswith(".ts") else torch.save(network, path)
    model = FlyModelConfig(checkpoint_path=path, channels=["mito"], input_voxel_size=8, input_size=8, sigmoid=sigmoid)
    x = torch.randn(1, 1, 8, 8, 8) * 10
    with torch.no_grad():
        expected = network(x)
        assert torch.allclose(model.config.model(x), torch.sigmoid(expected) if adds else expected)
    assert _ends_in_sigmoid(model.config.model) == sigmoid
    assert tuple(model.config.write_shape) == (48, 48, 48)


@pytest.mark.parametrize("sigmoid", [True, False])
def test_a_training_checkpoint_is_a_sequential_either_way(tmp_path, fly_organelles, sigmoid):
    """A LoRA adapter trained on it is keyed by its module names."""
    path, _ = _training_checkpoint(tmp_path)
    model = FlyModelConfig(checkpoint_path=path, channels=["a", "b"], input_voxel_size=8, input_size=12,
                           output_size=6, sigmoid=sigmoid).config.model
    assert isinstance(model, torch.nn.Sequential) and _ends_in_sigmoid(model) == sigmoid
    assert next(iter(model.state_dict())) == "0.unet_backbone.l_conv.0.conv_pass.0.weight"


def test_an_entry_round_trips_through_to_dict_and_the_launch_entry(tmp_path):
    _snapshot(tmp_path)
    entry = {"type": "fly", "checkpoint_path": str(tmp_path / "model_checkpoint_1000"), "channels": ["mito"],
             "input_voxel_size": [16, 16, 16], "output_voxel_size": [8, 8, 8], "name": "fly",
             "input_size": [100, 100, 100], "output_size": [20, 20, 20], "scale": "s1", "sigmoid": False}
    model = registry.build_model(entry, "fly")
    assert model.to_dict() == entry and list(model.to_dict()) == list(entry)
    assert registry.build_model(model.launch_entry, "fly").to_dict() == entry
    # From the files beside it, and rebuilt the same from the entry it writes.
    found = registry.build_model({"type": "fly", "checkpoint": entry["checkpoint_path"], "classes": ["mito"]}, "f")
    assert registry.build_model(found.launch_entry, "f").to_dict() == found.to_dict()
    assert found.to_dict()["input_size"] == [178, 178, 178]


def test_the_trainer_takes_a_fly_model_as_its_entry(tmp_path):
    from cellmap_flow.finetune.job_manager.submit import model_entry, resolve_model_type
    from cellmap_flow.finetune.model_loading import model_config_from_entry

    _snapshot(tmp_path)
    (tmp_path / "train.py").write_text('labels = ["mito"]\n')
    model = FlyModelConfig(checkpoint_path=str(tmp_path / "c.ts"), name="mito", sigmoid=False)
    assert resolve_model_type(model) == "fly"
    entry = model_entry(model, "fly", None)
    assert entry == model.to_dict() and entry["sigmoid"] is False
    assert model_config_from_entry(entry).to_dict() == entry


def test_nothing_but_building_the_model_imports_torch_or_fly_organelles(tmp_path):
    """The CLIs, --help and the model form import every type, and read
    nothing but the files beside a checkpoint to build its config."""
    _snapshot(tmp_path)
    (tmp_path / "train.py").write_text('labels = ["mito"]\n')
    script = f"""
import sys
from cellmap_flow.models import registry
from cellmap_flow.cli import main
from click.testing import CliRunner
model = registry.build_model({{"type": "fly", "checkpoint": {str(tmp_path / "model_checkpoint_1")!r}}}, "m")
model.to_dict(), model.launch_entry, model.command, model.default_env
assert "FlyModelConfig" in registry.describe_types()
assert CliRunner().invoke(main.cli, ["infer", "fly", "--help"]).exit_code == 0
loaded = [m for m in ("torch", "fly_organelles") if m in sys.modules]
assert not loaded, loaded
"""
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": ROOT}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=ROOT,
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.skipif(not os.path.isdir("/groups/cellmap/cellmap/zouinkhim/salevary/train/v2/distance/mito_16_all"),
                    reason="the example's run is on Janelia's file system")
def test_the_example_yaml_is_a_fly_model():
    """It names the checkpoint only; the rest is in the run's train.py and snapshots."""
    import yaml

    with open(os.path.join(ROOT, "example", "sal_1_mito.yaml")) as f:
        (model,) = registry.build_models(yaml.safe_load(f)["models"])
    assert isinstance(model, FlyModelConfig) and model.default_env == "fly"
    assert (model.channels, model.input_voxel_size, model.input_size, model.output_size) == (
        ["mito"], (16, 16, 16), (178, 178, 178), (56, 56, 56))
