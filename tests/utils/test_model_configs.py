"""Model configs: launch commands, geometry and declared outputs."""

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
    ScriptModelConfig,
)

SERVER_OPTIONS = {"data_path", "debug", "port", "certfile", "keyfile"}


def _relaunch(config):
    """Parse config.command with the real server CLI and rebuild the config.

    Mirrors what ``cellmap_flow_server <command> -d <data>`` does before it
    starts serving.
    """
    from cellmap_flow.cli.server_cli import cli
    from cellmap_flow.utils.cli_utils import process_constructor_args

    argv = shlex.split(config.command)
    command = cli.commands[argv[0]]
    ctx = command.make_context(argv[0], argv[1:] + ["-d", "/data/raw.zarr"])
    kwargs = {
        k: v for k, v in ctx.params.items() if k not in SERVER_OPTIONS and v is not None
    }
    cls = type(config)
    return cls(**process_constructor_args(cls, kwargs))


@pytest.mark.parametrize(
    "config",
    [
        ScriptModelConfig(
            script_path="/groups/my models/mito model.py", name="mito", scale="s1"
        ),
        DaCapoModelConfig(run_name="run 1", iteration=5000, name="dc", scale="s0"),
        FlyModelConfig(
            checkpoint_path="/ckpt/model_checkpoint_1000",
            channels=["mito", "er"],
            input_voxel_size=(16, 16, 16),
            output_voxel_size=(16, 16, 16),
            name="fly",
            input_size=(100, 100, 100),
            output_size=(20, 20, 20),
        ),
        BioModelConfig(
            model_name="affable-shark",
            voxel_size=(8, 8, 8),
            edge_length_to_process=64,
            name="bio",
        ),
        FinetuneModelConfig(
            lora_adapter_path="/runs/my run/lora_adapter",
            base_model={"type": "script", "script_path": "/a b/c.py"},
            name="ft",
        ),
    ],
    ids=lambda c: type(c).__name__,
)
def test_command_rebuilds_the_same_config(config):
    assert _relaunch(config).to_dict() == config.to_dict()


def test_huggingface_command_keeps_the_name_and_needs_no_download(monkeypatch):
    def no_download(self):
        raise AssertionError("building a launch command must not fetch metadata")

    monkeypatch.setattr(HuggingFaceModelConfig, "_load_metadata", no_download)
    config = HuggingFaceModelConfig(repo="cellmap/mito-v1", revision="abc123", name="m v1")
    rebuilt = _relaunch(config)
    assert (rebuilt.repo, rebuilt.revision, rebuilt.name) == ("cellmap/mito-v1", "abc123", "m v1")


def test_fly_write_shape_uses_the_output_voxel_size():
    fly = FlyModelConfig(
        checkpoint_path="unused",
        channels=["mito"],
        input_voxel_size=(8, 8, 8),
        output_voxel_size=(4, 4, 4),
        input_size=(20, 20, 20),
        output_size=(12, 12, 12),
    )

    class Out(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(1, 1, 12, 12, 12)

    fly._model = Out()
    assert tuple(fly.config.write_shape) == (48, 48, 48)


@pytest.fixture
def fake_dacapo(monkeypatch):
    for name in ("dacapo", "dacapo.experiments", "dacapo.store", "dacapo.store.create_store"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["dacapo.experiments"].Run = object
    store = sys.modules["dacapo.store.create_store"]
    store.create_config_store = store.create_weights_store = lambda: None


class AffinitiesTask:
    """Named like dacapo's; nine offsets."""

    predictor = SimpleNamespace(
        neighborhood=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 0, 0), (0, 3, 0),
                      (0, 0, 3), (9, 0, 0), (0, 9, 0), (0, 0, 9)]
    )


def _dacapo_run(out_channels):
    class Model(torch.nn.Module):
        eval_input_shape = Coordinate(20, 20, 20)

        def compute_output_shape(self, shape):
            return out_channels, Coordinate(12, 12, 12)

        def scale(self, voxel_size):
            return Coordinate(voxel_size) / 2

        def forward(self, x):
            return torch.zeros(1, out_channels, 12, 12, 12)

    raw = SimpleNamespace(voxel_size=Coordinate(8, 8, 8))
    return SimpleNamespace(
        model=Model(),
        datasplit=SimpleNamespace(train=[SimpleNamespace(raw=raw)]),
        task=AffinitiesTask(),
    )


def test_dacapo_geometry_and_channels_follow_the_model(fake_dacapo, monkeypatch):
    dacapo = DaCapoModelConfig(run_name="r", iteration=0)
    monkeypatch.setattr(dacapo, "_load_dacapo_run", lambda: _dacapo_run(9))
    config = dacapo.config

    assert tuple(config.output_voxel_size) == (4, 4, 4)
    assert tuple(config.write_shape) == (48, 48, 48)
    assert config.output_channels == 9
    assert config.channels[3] == "aff_3_0_0"


def test_dacapo_keeps_the_old_names_when_they_fit(fake_dacapo, monkeypatch):
    dacapo = DaCapoModelConfig(run_name="r", iteration=0)
    monkeypatch.setattr(dacapo, "_load_dacapo_run", lambda: _dacapo_run(3))
    assert dacapo.config.channels == ["x", "y", "z"]


def test_bioimage_models_declare_their_uint8_output(monkeypatch):
    fake = types.ModuleType("bioimageio.core")
    fake.load_description = lambda name: object()
    monkeypatch.setitem(sys.modules, "bioimageio", types.ModuleType("bioimageio"))
    monkeypatch.setitem(sys.modules, "bioimageio.core", fake)

    bio = BioModelConfig(model_name="m", voxel_size="8,8,8")
    axes = ["b", "c", "z", "y", "x"]
    monkeypatch.setattr(
        bio,
        "load_input_information",
        lambda model: ("in", axes, [16, 16, 16], (slice(None),) * 5, False),
    )
    monkeypatch.setattr(
        bio,
        "load_output_information",
        lambda model: (["out"], [axes], [16, 16, 16, 1], [16, 16, 16], 1),
    )
    assert np.dtype(bio.output_dtype) == np.uint8
    assert tuple(bio.config.input_voxel_size) == (8, 8, 8)


def test_a_plugin_config_gets_a_command_under_its_registered_name():
    from cellmap_flow.models.models_config import ModelConfig

    class MyThingModelConfig(ModelConfig):
        def __init__(self, weights: str, name=None):
            super().__init__()
            self.weights, self.name = weights, name

        def to_dict(self):
            return {"type": "mything", "weights": self.weights, "name": self.name}

    assert shlex.split(MyThingModelConfig("/w w.pt").command) == [
        "mything",
        "--weights",
        "/w w.pt",
    ]


def test_a_plugin_config_without_to_dict_exports_its_constructor_arguments():
    import copy

    from cellmap_flow.models.models_config import ModelConfig

    class WeightsModelConfig(ModelConfig):
        def __init__(self, weights: str, sizes: tuple = (1, 1, 1), name=None, scale=None):
            super().__init__()
            self.weights, self.name, self.scale = weights, name, scale

    config = WeightsModelConfig("/w w.pt", sizes=(8, 8, 8), name="w")
    expected = {"type": "weights", "weights": "/w w.pt", "sizes": [8, 8, 8], "name": "w"}
    assert config.to_dict() == expected
    assert shlex.split(config.command) == [
        "weights", "--weights", "/w w.pt", "--sizes", "8,8,8", "--name", "w",
    ]
    assert copy.deepcopy(config).to_dict() == expected
    assert "_init_params" not in str(config)


def test_a_config_can_skip_the_dummy_forward_pass():
    from cellmap_flow.models.models_config import ModelConfig

    class ForwardRaises(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("the dummy forward pass ran")

    class NoForwardModelConfig(ModelConfig):
        def _get_config(self):
            return SimpleNamespace(
                model=ForwardRaises(),
                read_shape=(8, 8, 8),
                write_shape=(8, 8, 8),
                input_voxel_size=(1, 1, 1),
                output_voxel_size=(1, 1, 1),
                output_channels=1,
                block_shape=np.array((8, 8, 8, 1)),
            )

    config = NoForwardModelConfig()
    config.validate_model_shapes = False
    assert config.config.read_shape == (8, 8, 8)
