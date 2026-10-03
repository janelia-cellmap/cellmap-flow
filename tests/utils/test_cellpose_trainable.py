"""Finetuning the cellpose model type, against a stand-in ``cellpose`` whose
network has Cellpose-SAM's interface at a toy size: the module the trainer
trains, the patch it reads, LoRA's layers, and serving the result through
Cellpose's eval, live and as a finetuned model. The real network is checked
in the cellpose4 environment only (it is not in the test environments)."""

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from funlib.geometry import Coordinate, Roi  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from cellmap_flow.models.configs.base import Config  # noqa: E402
from cellmap_flow.models.models_config import CellposeModelConfig  # noqa: E402

TILE, PS = 256, 8


class TinyCPSAM(nn.Module):
    """Cellpose-SAM's network in miniature: a patch embedding whose weight the
    forward reads directly, fixed position embeddings (so one tile size
    only), a Linear block, a readout and the fixed W2 unpatching; it returns
    (output, style) and loads in bfloat16, as Cellpose's does."""

    def __init__(self, width=4):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.patch_embed = nn.Module()
        self.encoder.patch_embed.proj = nn.Conv2d(3, width, PS, stride=PS)
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, TILE // PS, TILE // PS, width))
        self.encoder.mlp = nn.Linear(width, width)
        self.out = nn.Conv2d(width, 3 * PS * PS, 1)
        self.W2 = nn.Parameter(torch.eye(3 * PS * PS).reshape(3 * PS * PS, 3, PS, PS), requires_grad=False)
        self._dtype = torch.float32
        self.dtype = torch.bfloat16

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self.to(value)
        self._dtype = value

    def forward(self, x):
        proj = self.encoder.patch_embed.proj
        x = F.conv2d(x, proj.weight.data[:, : x.shape[1]], bias=proj.bias.data, stride=PS)
        x = self.encoder.mlp(x.permute(0, 2, 3, 1) + self.encoder.pos_embed)
        x = F.conv_transpose2d(self.out(x.permute(0, 3, 1, 2)), self.W2, stride=PS)
        return x, torch.zeros(x.shape[0], 256)


class FakeCellposeModel:
    """cellpose.models.CellposeModel: its network (the same weights each time,
    as pretrained weights are), backbone, and an eval that records its
    arguments and gives a probability logit of 1 everywhere, and flows of 0."""

    built = []

    def __init__(self, gpu=False, pretrained_model="cpsam_v2"):
        torch.manual_seed(0)
        self.backbone = "dino_vitl" if pretrained_model.startswith("cpdino") else "sam_vitl"
        self.net = TinyCPSAM()
        self.calls = []
        FakeCellposeModel.built.append(self)

    def eval(self, x, channel_axis=None, **kwargs):
        self.calls.append((x.shape, kwargs))
        z, y, xs, _ = x.shape
        flows = [None, np.zeros((2, z, y, xs), np.float32).squeeze(), np.ones((z, y, xs), np.float32).squeeze()]
        return np.zeros((z, y, xs), np.uint16).squeeze(), flows, None


@pytest.fixture
def fake_cellpose(monkeypatch):
    package = types.ModuleType("cellpose")
    models = types.ModuleType("cellpose.models")
    models.MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]
    models.get_user_models = lambda: []
    models.CellposeModel = FakeCellposeModel
    package.models = models
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    FakeCellposeModel.built = []
    return models


def _cellpose(**kw):
    kw.setdefault("output", "probability")  # one channel, which these serving checks read
    return CellposeModelConfig(voxel_size=8, slices_per_chunk=2, slice_size=TILE, context=16, **kw)


def test_the_trainable_module_is_cellposes_network_on_each_slice(fake_cellpose):
    model = _cellpose()
    module = model.trainable_model()
    net = model.config.model.net
    assert any(part is net for part in module.modules())  # what is trained is what serves
    assert net.dtype == torch.float32 and next(net.parameters()).dtype == torch.float32
    x = torch.rand(2, 1, 3, TILE, TILE)
    out = module(x)
    assert out.shape == (2, 3, 3, TILE, TILE)
    # Slice 1 of sample 1 is the network on that slice, normalized as Cellpose does.
    flat = x[1, 0, 1].flatten()
    lo, hi = torch.quantile(flat, 0.01), torch.quantile(flat, 0.99)
    alone = net(((x[1, :, 1] - lo) / (hi - lo + 1e-6))[None])[0][0]
    assert torch.allclose(out[1, :, 1], alone, atol=1e-5)


def test_each_slice_is_normalized_on_its_own(fake_cellpose):
    module = _cellpose().trainable_model()
    x = torch.rand(1, 1, 2, TILE, TILE)
    scaled = x.clone()
    scaled[:, :, 1] = scaled[:, :, 1] * 40 + 7
    assert torch.allclose(module(x), module(scaled), atol=1e-4)


def test_another_tile_size_is_refused_with_a_message(fake_cellpose):
    module = _cellpose().trainable_model()
    with pytest.raises(RuntimeError, match="256 x 256 tiles"):
        module(torch.rand(1, 1, 1, 288, 288))


@pytest.mark.parametrize("pretrained, tile", [("cpsam", 256), ("cpdino", 384)])
def test_the_trainer_reads_one_slice_of_one_tile(fake_cellpose, pretrained, tile):
    assert _cellpose(pretrained_model=pretrained).training_patch_voxels() == ((1, tile, tile), (1, tile, tile))


def test_its_finetune_modes_are_known_without_building_it(fake_cellpose):
    from cellmap_flow.finetune.job_manager.submit import trainable_model_types

    assert _cellpose().finetune_modes() == ("lora", "full") and FakeCellposeModel.built == []
    assert "cellpose" in trainable_model_types()


def test_lora_leaves_the_patch_embedding_alone(fake_cellpose):
    """Cellpose's forward reads that layer's weight directly: an adapter there would never train."""
    pytest.importorskip("peft")
    from cellmap_flow.finetune.adaptation import LoraStrategy

    model = LoraStrategy(r=2, alpha=4, dropout=0.0).prepare(_cellpose().trainable_model())
    adapted = {name.split(".lora_")[0] for name, _ in model.named_parameters() if ".lora_" in name}
    assert adapted and not any("patch_embed" in name for name in adapted)
    assert any(name.endswith("encoder.mlp") for name in adapted) and any(name.endswith(".out") for name in adapted)


def _idi(shape):
    class ArrayIDI:
        def to_ndarray_ts(self, roi):
            self.roi = roi
            return np.zeros(shape, np.float32)

    return ArrayIDI()


def test_the_live_server_serves_the_trained_module_through_cellposes_eval(fake_cellpose):
    model = _cellpose()
    module = model.trainable_model()
    config = model.config
    model.serve_trained(config, module)
    assert config.trained_module is module and config.model is FakeCellposeModel.built[0]
    out = config.process_chunk(_idi((2, TILE + 32, TILE + 32)), Roi((0, 128, 128), (16, TILE * 8, TILE * 8)))
    assert out.shape == (1, 2, TILE, TILE) and np.allclose(out, 1 / (1 + np.exp(-1)))


def test_a_new_config_gets_what_segmenting_needs(fake_cellpose):
    """A finetuned model's config, at the voxel size it was trained at (twice the base's here)."""
    model = _cellpose(output="masks")
    module = model.trainable_model()
    config = Config()
    config.read_shape, config.write_shape = Coordinate(32, 288 * 16, 288 * 16), Coordinate(32, 256 * 16, 256 * 16)
    model.serve_trained(config, module)
    assert config.eval_kwargs == model.config.eval_kwargs and config.output_dtype == np.uint64
    idi = _idi((2, 288, 288))
    out = config.process_chunk(idi, Roi((0, 256, 256), (32, 256 * 16, 256 * 16)))
    assert out.shape == (1, 2, 256, 256) and out.dtype == np.uint64
    assert idi.roi == Roi((0, 0, 0), (32, 288 * 16, 288 * 16))  # its own context, in its own nm


def test_a_module_without_this_models_network_is_refused(fake_cellpose):
    model = _cellpose()
    other = _cellpose().trainable_model()
    with pytest.raises(ValueError, match="does not hold this model's Cellpose network"):
        model.serve_trained(model.config, other)


def test_a_full_finetune_is_served_from_a_fresh_base(fake_cellpose, tmp_path):
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig

    trained = _cellpose()
    module = trained.trainable_model()
    with torch.no_grad():
        trained.config.model.net.out.bias.add_(1.0)
    torch.save(module.state_dict(), tmp_path / "weights.pt")
    finetuned = FinetuneModelConfig(weights_path=str(tmp_path / "weights.pt"), base_model=trained.to_dict())
    config = finetuned.config
    served = config.model
    assert served is FakeCellposeModel.built[-1] and served is not trained.config.model
    assert torch.equal(served.net.out.bias, trained.config.model.net.out.bias)
    assert config.process_chunk(_idi((2, 288, 288)), Roi((0, 128, 128), (16, TILE * 8, TILE * 8))).shape == (1, 2, 256, 256)


@pytest.mark.parametrize("output, channels, dtype", [("flows", ["flow_y", "flow_x", "cell"], np.float32),
                                                     ("masks", ["cell"], np.uint64)])
def test_a_finetuned_model_serves_its_bases_output(fake_cellpose, tmp_path, output, channels, dtype):
    """Training is on the flows whatever the base serves; what is served
    after is what the base served, channels and all."""
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig

    trained = _cellpose(output=output)
    torch.save(trained.trainable_model().state_dict(), tmp_path / "weights.pt")
    config = FinetuneModelConfig(weights_path=str(tmp_path / "weights.pt"), base_model=trained.to_dict()).config
    assert (config.channels, config.output_channels, config.output_dtype) == (channels, len(channels), dtype)
    assert list(config.block_shape) == [2, TILE, TILE, len(channels)]
    out = config.process_chunk(_idi((2, 288, 288)), Roi((0, 128, 128), (16, TILE * 8, TILE * 8)))
    assert out.shape == (len(channels), 2, TILE, TILE) and out.dtype == dtype


def test_a_lora_adapter_loaded_on_a_fresh_base_is_in_the_network_eval_runs(fake_cellpose, tmp_path):
    pytest.importorskip("peft")
    from cellmap_flow.finetune.adaptation import LoraStrategy
    from cellmap_flow.finetune.lora_wrapper import save_lora_adapter
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig

    trained = _cellpose()
    module = LoraStrategy(r=2, alpha=4, dropout=0.0).prepare(trained.trainable_model())
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "lora_B" in name:
                param.normal_()
    save_lora_adapter(module, str(tmp_path / "adapter"))
    finetuned = FinetuneModelConfig(lora_adapter_path=str(tmp_path / "adapter"), base_model=trained.to_dict())
    served = finetuned.config.model
    assert served is not trained.config.model
    x = torch.rand(1, 1, TILE, TILE)
    fresh = FakeCellposeModel().net.float()
    with torch.no_grad():
        trained_out, served_out = trained.config.model.net(x)[0], served.net(x)[0]
        assert torch.allclose(served_out, trained_out, atol=1e-5)
        assert not torch.allclose(served_out, fresh(x)[0], atol=1e-3)


def test_a_finetuned_model_whose_slices_are_one_tile_passes_its_shape_check(fake_cellpose, tmp_path):
    """The shape check forwards config.model at the read shape; had it been the
    3-channel tile network, a one-tile read would have run it and failed."""
    from cellmap_flow.models.configs.finetune import FinetuneModelConfig

    trained = CellposeModelConfig(voxel_size=8, slices_per_chunk=1, slice_size=TILE - 32, context=16)
    torch.save(trained.trainable_model().state_dict(), tmp_path / "weights.pt")
    finetuned = FinetuneModelConfig(weights_path=str(tmp_path / "weights.pt"), base_model=trained.to_dict())
    assert finetuned.config.model is FakeCellposeModel.built[-1]


def test_serving_puts_the_networks_training_mode_back():
    """Cellpose's eval leaves its network in eval mode: in the trainer's live
    server, training went on without stochastic depth until the next epoch."""
    from types import SimpleNamespace

    from cellmap_flow.models.configs.cellpose import _serving

    net = torch.nn.Linear(1, 1)
    with _serving(SimpleNamespace(net=net)) as override:
        net.eval()  # as Cellpose's _forward does
        assert override({"batch_size": 72}) in ({}, {"batch_size": 36})  # halved on a GPU
    assert net.training
    net.eval()
    with _serving(SimpleNamespace(net=net)):
        pass
    assert not net.training
