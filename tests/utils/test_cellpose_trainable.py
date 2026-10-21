"""Finetuning the cellpose model type, against a stand-in ``cellpose`` whose
network has Cellpose-SAM's interface at a toy size: the module the trainer
trains, and serving the result through Cellpose's eval as a finetuned model,
from its weights or a LoRA adapter. The real network is checked in the
cellpose4 environment only (it is not in the test environments)."""

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from funlib.geometry import Roi  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

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
    as pretrained weights are), backbone, and an eval that gives a
    probability logit of 1 everywhere, and flows of 0."""

    built = []

    def __init__(self, gpu=False, pretrained_model="cpsam_v2"):
        torch.manual_seed(0)
        self.backbone = "dino_vitl" if pretrained_model.startswith("cpdino") else "sam_vitl"
        self.net = TinyCPSAM()
        FakeCellposeModel.built.append(self)

    def eval(self, x, channel_axis=None, **kwargs):
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
    # A Cellpose 4: when an earlier test imported the environment's real
    # Cellpose 3, its cellpose.version stayed loaded, and the version check
    # refused the stand-in.
    version = types.ModuleType("cellpose.version")
    version.version = "4.0.0"
    package.version = version
    monkeypatch.setitem(sys.modules, "cellpose", package)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    monkeypatch.setitem(sys.modules, "cellpose.version", version)
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


def _idi(shape):
    class ArrayIDI:
        def to_ndarray_ts(self, roi):
            self.roi = roi
            return np.zeros(shape, np.float32)

    return ArrayIDI()


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
