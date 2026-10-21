"""The fly model type: what it reads beside a checkpoint, a training
checkpoint built from its weights, its sigmoid for each kind of checkpoint,
and the trainer's entry for it. Against a stand-in
``fly_organelles.model.StandardUnet``: fly_organelles is not in the test
environments (it has an environment of its own, fly)."""

import json
import sys
import types

import pytest
import torch

from cellmap_flow.models.configs.fly import _ends_in_sigmoid, unet_arguments
from cellmap_flow.models.models_config import FlyModelConfig


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