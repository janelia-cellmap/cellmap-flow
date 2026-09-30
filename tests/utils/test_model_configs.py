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


def _fly_whose_model_outputs(size, output_voxel_size=(8, 8, 8), name=None):
    """A Fly config declaring 20 voxels in and 12 out, whose model returns ``size`` a side."""
    fly = FlyModelConfig(checkpoint_path="unused", channels=["mito"], input_voxel_size=(8, 8, 8),
                         output_voxel_size=output_voxel_size, input_size=(20, 20, 20), output_size=(12, 12, 12),
                         name=name)
    fly._model = torch.nn.Module()
    fly._model.forward = lambda x: torch.zeros(1, 1, size, size, size)
    return fly


def test_a_fly_models_write_shape_is_in_output_voxels():
    fly = _fly_whose_model_outputs(12, output_voxel_size=(4, 4, 4))
    assert tuple(fly.config.write_shape) == (48, 48, 48)


def test_a_shape_mismatch_names_the_models_type():
    """It said "Script config shape validation failed" for every type."""
    with pytest.raises(ValueError, match="^FlyModelConfig shape validation failed for mito:"):
        _fly_whose_model_outputs(10, name="mito").config


@pytest.mark.parametrize("given, missing", [
    pytest.param({"input_size": (100, 100, 100)}, "output_size", id="input-size-only"),
    pytest.param({"output_size": (20, 20, 20)}, "input_size", id="output-size-only"),
])
def test_a_fly_model_given_one_size_asks_for_the_other(given, missing):
    """Both used to be replaced by the 178/56 default."""
    with pytest.raises(ValueError, match=f"no {missing}"):
        FlyModelConfig(checkpoint_path="unused", channels=["mito"], input_voxel_size=(8, 8, 8),
                       output_voxel_size=(8, 8, 8), **given)


@pytest.mark.parametrize("out_channels, channels", [
    pytest.param(9, "aff_3_0_0", id="nine-affinities-named-by-offset"),
    pytest.param(3, ["x", "y", "z"], id="three-keep-the-old-names"),
])
def test_a_dacapo_models_geometry_and_channels_follow_the_model(fake_frameworks, monkeypatch, out_channels, channels):
    dacapo = DaCapoModelConfig(run_name="r", iteration=0)
    monkeypatch.setattr(dacapo, "_load_dacapo_run", lambda: _dacapo_run(out_channels))
    config = dacapo.config
    assert (tuple(config.output_voxel_size), tuple(config.write_shape), config.output_channels) == (
        (4, 4, 4), (48, 48, 48), out_channels)
    assert (config.channels[3] if out_channels == 9 else config.channels) == channels


def test_a_bioimage_model_declares_its_uint8_output(fake_frameworks, monkeypatch):
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


def _declared(**attrs):
    """A Config declaring 8 voxels in and out at 1 nm, one channel, and ``attrs``."""
    return SimpleNamespace(read_shape=(8, 8, 8), write_shape=(8, 8, 8), input_voxel_size=(1, 1, 1),
                           output_voxel_size=(1, 1, 1), output_channels=1, block_shape=np.array((8, 8, 8, 1)),
                           **attrs)


def test_a_config_can_skip_the_dummy_forward():
    class ForwardRaises(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("the dummy forward pass ran")

    class NoForwardModelConfig(ModelConfig):
        def _get_config(self):
            return _declared(model=ForwardRaises())

    config = NoForwardModelConfig()
    config.validate_model_shapes = False
    assert config.config.read_shape == (8, 8, 8)


def test_a_config_without_a_name_or_an_output_dtype_serves_float32():
    """The fallback's warning read self.name, which ModelConfig does not set."""
    class UnnamedModelConfig(ModelConfig):
        def _get_config(self):
            return _declared(predict=lambda *args: None)

    assert UnnamedModelConfig().geometry.output_dtype == np.float32
