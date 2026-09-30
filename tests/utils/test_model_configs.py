"""Model configs: the server rebuilds each from its command, and each type
takes its geometry from its model. The commands and to_dict() themselves
are pinned in tests/cli/test_cli_surface.py."""

import copy
import shlex
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from funlib.geometry import Coordinate

from cellmap_flow.models.models_config import (
    BioModelConfig,
    DaCapoModelConfig,
    FinetuneModelConfig,
    FlyModelConfig,
    HuggingFaceModelConfig,
    ModelConfig,
    ScriptModelConfig,
)


@pytest.mark.parametrize(
    "config",
    [
        lambda: ScriptModelConfig(script_path="/groups/my models/mito model.py", name="mito", scale="s1"),
        lambda: DaCapoModelConfig(run_name="run 1", iteration=5000, name="dc", scale="s0"),
        lambda: FlyModelConfig(checkpoint_path="/ckpt/model_checkpoint_1000", channels=["mito", "er"],
                               input_voxel_size=(16, 16, 16), output_voxel_size=(16, 16, 16), name="fly",
                               input_size=(100, 100, 100), output_size=(20, 20, 20)),
        lambda: BioModelConfig(model_name="affable-shark", voxel_size=(8, 8, 8), edge_length_to_process=64, name="bio"),
        lambda: FinetuneModelConfig(lora_adapter_path="/runs/my run/lora_adapter",
                                    base_model={"type": "script", "script_path": "/a b/c.py"}, name="ft"),
        lambda: HuggingFaceModelConfig(repo="cellmap/mito-v1", revision="abc123", name="m v1"),
    ],
    ids=["script", "dacapo", "fly", "bio", "finetune", "huggingface"],
)
def test_the_server_rebuilds_the_same_config_from_its_command(config, monkeypatch):
    """What `cellmap_flow_server <command> -d <data>` does before it serves."""
    from cellmap_flow.cli.server_cli import cli
    from cellmap_flow.models import registry

    def no_download(self):
        raise AssertionError("building a launch command must not fetch metadata")

    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", no_download)
    config = config()
    argv = shlex.split(config.command)
    params = cli.commands[argv[0]].make_context(argv[0], argv[1:] + ["-d", "/data/raw.zarr"]).params
    server_options = {"data_path", "debug", "port", "certfile", "keyfile"}
    kwargs = {k: v for k, v in params.items() if k not in server_options and v is not None}
    rebuilt = type(config)(**registry.coerce_cli_args(type(config), kwargs))
    if isinstance(config, HuggingFaceModelConfig):  # its to_dict() adds the downloaded metadata
        assert (rebuilt.repo, rebuilt.revision, rebuilt.name) == (config.repo, config.revision, config.name)
    else:
        assert rebuilt.to_dict() == config.to_dict()


@pytest.fixture
def fake_frameworks(monkeypatch):
    """Just enough of dacapo and bioimageio to import them."""
    for name in ("dacapo", "dacapo.experiments", "dacapo.store", "dacapo.store.create_store", "bioimageio"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["dacapo.experiments"].Run = object
    store = sys.modules["dacapo.store.create_store"]
    store.create_config_store = store.create_weights_store = lambda: None
    core = types.ModuleType("bioimageio.core")
    core.load_description = lambda name: object()
    monkeypatch.setitem(sys.modules, "bioimageio.core", core)


def _dacapo_run(out_channels):
    class AffinitiesTask:  # named like dacapo's; nine offsets
        predictor = SimpleNamespace(neighborhood=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0), (0, 3, 0),
                                                  (0, 0, 3), (9, 0, 0), (0, 9, 0), (0, 0, 9)])

    class Model(torch.nn.Module):
        eval_input_shape = Coordinate(20, 20, 20)

        def compute_output_shape(self, shape):
            return out_channels, Coordinate(12, 12, 12)

        def scale(self, voxel_size):
            return Coordinate(voxel_size) / 2

        def forward(self, x):
            return torch.zeros(1, out_channels, 12, 12, 12)

    raw = SimpleNamespace(voxel_size=Coordinate(8, 8, 8))
    return SimpleNamespace(model=Model(), datasplit=SimpleNamespace(train=[SimpleNamespace(raw=raw)]),
                           task=AffinitiesTask())


def test_each_type_takes_its_geometry_from_its_model(fake_frameworks, monkeypatch):
    fly = FlyModelConfig(checkpoint_path="unused", channels=["mito"], input_voxel_size=(8, 8, 8),
                         output_voxel_size=(4, 4, 4), input_size=(20, 20, 20), output_size=(12, 12, 12))
    fly._model = torch.nn.Module()
    fly._model.forward = lambda x: torch.zeros(1, 1, 12, 12, 12)
    assert tuple(fly.config.write_shape) == (48, 48, 48), "in output voxels"

    for out_channels, channels in [(9, "aff_3_0_0"), (3, ["x", "y", "z"])]:
        dacapo = DaCapoModelConfig(run_name="r", iteration=0)
        monkeypatch.setattr(dacapo, "_load_dacapo_run", lambda: _dacapo_run(out_channels))
        config = dacapo.config
        assert (tuple(config.output_voxel_size), tuple(config.write_shape), config.output_channels) == (
            (4, 4, 4), (48, 48, 48), out_channels)
        assert (config.channels[3] if out_channels == 9 else config.channels) == channels  # old names when they fit

    bio = BioModelConfig(model_name="m", voxel_size="8,8,8")
    axes = ["b", "c", "z", "y", "x"]
    monkeypatch.setattr(bio, "load_input_information", lambda model: ("in", axes, [16] * 3, (slice(None),) * 5, False))
    monkeypatch.setattr(bio, "load_output_information", lambda model: (["out"], [axes], [16, 16, 16, 1], [16] * 3, 1))
    assert np.dtype(bio.output_dtype) == np.uint8 and tuple(bio.config.input_voxel_size) == (8, 8, 8)


def test_a_plugin_config_without_its_own_to_dict_exports_its_constructor_arguments():
    class WeightsModelConfig(ModelConfig):
        def __init__(self, weights: str, sizes: tuple = (1, 1, 1), name=None, scale=None):
            super().__init__()
            self.weights, self.name, self.scale = weights, name, scale

    config = WeightsModelConfig("/w w.pt", sizes=(8, 8, 8), name="w")
    expected = {"type": "weights", "weights": "/w w.pt", "sizes": [8, 8, 8], "name": "w"}
    assert config.to_dict() == copy.deepcopy(config).to_dict() == expected
    assert shlex.split(config.command) == ["weights", "--weights", "/w w.pt", "--sizes", "8,8,8", "--name", "w"]
    assert "_init_params" not in str(config)


def test_a_config_can_skip_the_dummy_forward():
    class ForwardRaises(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("the dummy forward pass ran")

    class NoForwardModelConfig(ModelConfig):
        def _get_config(self):
            return SimpleNamespace(model=ForwardRaises(), read_shape=(8, 8, 8), write_shape=(8, 8, 8),
                                   input_voxel_size=(1, 1, 1), output_voxel_size=(1, 1, 1), output_channels=1,
                                   block_shape=np.array((8, 8, 8, 1)))

    config = NoForwardModelConfig()
    config.validate_model_shapes = False
    assert config.config.read_shape == (8, 8, 8)
