"""A finetuned model as the trainer loads it to train on (model_loading) and as
it is served (FinetuneModelConfig, and the YAML the trainer writes for it)."""

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from cellmap_flow.finetune.finetuned_model_templates import generate_finetuned_model_yaml
from cellmap_flow.finetune.model_loading import (
    decode_model_entry,
    load_trainable_model,
    model_config_from_entry,
    root_base_model_dict,
)
from cellmap_flow.models.models_config import CellMapModelConfig, FinetuneModelConfig, ScriptModelConfig
from cellmap_flow.utils.web_utils import encode_to_str


@pytest.mark.parametrize("kind", ["full", pytest.param("lora", marks=pytest.mark.finetune)])
def test_a_finetune_entry_trains_on_from_what_it_exported(tiny_script, tmp_path, kind):
    """The job manager passes a finetuned model to the trainer as its entry
    (base64, or JSON by hand), which the trainer did not take: continuing from
    the last run was out of the dashboard's reach."""
    script = tiny_script()
    base = {"type": "script", "script_path": str(script)}
    model = ScriptModelConfig(script_path=str(script)).config.model
    if kind == "full":
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        torch.save(model.state_dict(), tmp_path / "model_state_dict.pt")
        # to_dict() surfaces base fields for display; they must not break loading.
        text = encode_to_str({"type": "finetune", "name": "ft", "base_model": base, "lora_adapter_path": None,
                              "weights_path": str(tmp_path / "model_state_dict.pt"), "base_type": "script",
                              "channels": ["mito"]})
    else:
        from cellmap_flow.finetune.lora_wrapper import save_lora_adapter, wrap_model_with_lora

        model = wrap_model_with_lora(model, lora_r=2, lora_alpha=4, lora_dropout=0.0)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.fill_(0.3)
        save_lora_adapter(model, str(tmp_path / "adapter"))
        text = json.dumps({"type": "finetune", "name": "ft", "lora_adapter_path": str(tmp_path / "adapter"),
                           "base_model": base})

    config = model_config_from_entry(decode_model_entry(text))
    assert isinstance(config, FinetuneModelConfig) and root_base_model_dict(config) == base
    x = torch.rand(1, 1, 4, 4, 4)
    with torch.no_grad():
        assert torch.allclose(load_trainable_model(config).eval()(x), model.eval()(x), atol=1e-6)


def test_a_finetune_is_served_from_exactly_one_export():
    """A full finetune (--lora-r 0) from weights_path, a LoRA one from
    lora_adapter_path; the server's command carries the one it has."""
    base = {"type": "huggingface", "repo": "cellmap/mito-aff-unet-setup-16"}
    for bad in ({"base_model": base}, {"base_model": base, "lora_adapter_path": "/a", "weights_path": "/w.pt"},
                {"weights_path": "/w.pt"}):
        with pytest.raises(ValueError):
            FinetuneModelConfig(**bad)
    full = FinetuneModelConfig(weights_path="/w.pt", base_model=base, name="ft")
    assert "--weights-path /w.pt" in full.command and "--lora-adapter-path" not in full.command
    assert full.to_dict()["weights_path"] == "/w.pt"
    lora = FinetuneModelConfig(lora_adapter_path="/a", base_model=base, name="lora")
    assert "--lora-adapter-path /a" in lora.command and "--weights-path" not in lora.command


@pytest.fixture
def exported_cellmap_model(monkeypatch):
    """cellmap_models as it opens an exported folder: TorchScript to serve, and
    train() rebuilding it as torch.export's UnflattenedModule."""

    class UnflattenedModule(torch.nn.Module):  # the name load_trainable_model looks for
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(1, 1, 1)

        def forward(self, x):
            return self.conv(x)

    class CellmapModel:
        def __init__(self, folder_path):
            shape, voxel = [4, 4, 4], [8, 8, 8]
            self.metadata = SimpleNamespace(
                model_name="m", model_type="unet", framework="torch", spatial_dims=3, in_channels=1,
                out_channels=1, iteration=0, channels_names=["mito"], input_voxel_size=voxel,
                output_voxel_size=voxel, input_shape=shape, output_shape=shape,
                inference_input_shape=shape, inference_output_shape=shape,
            )
            self.ts_model = torch.jit.trace(torch.nn.Conv3d(1, 1, 1), torch.zeros(1, 1, *shape))

        def train(self):
            return UnflattenedModule()

    leaf = types.ModuleType("cellmap_models.model_export.cellmap_model")
    leaf.CellmapModel = CellmapModel
    for name in ("cellmap_models", "cellmap_models.model_export"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)


@pytest.mark.filterwarnings("ignore:`torch.jit.trace:DeprecationWarning")
def test_a_finetune_is_served_on_the_module_tree_it_was_trained_on(exported_cellmap_model, tmp_path):
    """The exported weights are keyed by the trainer's tree, BatchLoopWrapper's
    "model." prefix included; serving them needs exactly that tree."""
    base = {"type": "cellmap", "folder_path": "/models/m"}
    trained = load_trainable_model(CellMapModelConfig(folder_path=base["folder_path"]))
    with torch.no_grad():
        for p in trained.parameters():
            p.fill_(0.5)
    torch.save(trained.state_dict(), tmp_path / "model_state_dict.pt")

    served = FinetuneModelConfig(weights_path=str(tmp_path / "model_state_dict.pt"), base_model=base).config.model
    assert list(served.state_dict()) == list(trained.state_dict()) == ["model.conv.weight", "model.conv.bias"]
    assert all(torch.equal(served.state_dict()[k], v) for k, v in trained.state_dict().items())


@pytest.mark.parametrize("export", [
    dict(lora_adapter_path="/a", weights_path="/w.pt", data_path="/data.zarr"),  # one export, not both
    dict(lora_adapter_path="/a", data_path="/path/to/data.zarr"),  # placeholders the template knew
    dict(lora_adapter_path="/a", data_path="/path/to/your/data.zarr"),  # ... and one it did not
])
def test_the_serving_yaml_refuses_what_it_could_not_serve(tmp_path, export):
    with pytest.raises(ValueError):
        generate_finetuned_model_yaml(base_model_dict={"type": "fly"}, model_name="m",
                                      output_path=tmp_path / "m.yaml", **export)
    assert not (tmp_path / "m.yaml").exists()
