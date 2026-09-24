"""Full finetuning (--lora-r 0): no adapter, whole model trainable.

Background
----------
On mito-aff-unet-setup-16 (2026-09-23) LoRA r=64 was slower per step than
training every parameter (0.90 vs 0.50 s), used more memory (50 vs 31 GB)
and reached a higher training loss at every checkpoint: the adapter's
savings are in parameters, and this model's cost is activations. So the
trainer had to be able to run without PEFT at all, and the result had to
be servable. These tests pin down:
  - a plain (non-PEFT) model trains, keeps only the best full checkpoint
    (no 3 GB periodic files) and exports full_finetune/model_state_dict.pt
    that reproduces the trained model bit-for-bit,
  - FinetuneModelConfig accepts weights_path XOR lora_adapter_path and
    emits the matching server flag,
  - the generated serving YAML carries weights_path for a full finetune.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner


def _trainer(tmp_path, num_epochs=6):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv3d(4, 1, 1))
    dl = DataLoader(
        TensorDataset(torch.rand(4, 1, 6, 6, 6), (torch.rand(4, 1, 6, 6, 6) > 0.5).float()),
        batch_size=2,
    )
    return model, LoRAFinetuner(
        model, dl, output_dir=str(tmp_path / "run"), num_epochs=num_epochs,
        device="cpu", use_mixed_precision=False, mask_unannotated=False,
        loss_type="bce", tensorboard=False,
    )


def test_full_finetune_keeps_only_best_checkpoint_and_exports_full_weights(tmp_path):
    model, trainer = _trainer(tmp_path)
    assert not trainer._is_peft()
    trainer.train()
    run = tmp_path / "run"
    assert (run / "best_checkpoint.pth").exists()
    # 6 epochs would have produced checkpoint_epoch_5.pth and a final one.
    assert not list(run.glob("checkpoint_epoch_*.pth"))
    ckpt = torch.load(run / "best_checkpoint.pth", map_location="cpu", weights_only=False)
    assert ckpt["full_model"] is True and ckpt["lora_only"] is False
    assert "optimizer_state_dict" not in ckpt

    out = trainer.save_adapter()
    weights = run / "full_finetune" / "model_state_dict.pt"
    assert out == str(weights) and weights.exists()

    fresh = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv3d(4, 1, 1))
    fresh.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    x = torch.rand(1, 1, 6, 6, 6)
    with torch.no_grad():
        assert torch.equal(fresh(x), model.eval()(x))


def test_full_finetune_load_checkpoint_tolerates_missing_optimizer_state(tmp_path):
    model, trainer = _trainer(tmp_path, num_epochs=1)
    trainer.train()
    _, trainer2 = _trainer(tmp_path / "b", num_epochs=1)
    trainer2.load_checkpoint(str(tmp_path / "run" / "best_checkpoint.pth"))
    assert trainer2.current_epoch == 0


def test_finetune_model_config_requires_exactly_one_source():
    from cellmap_flow.models.models_config import FinetuneModelConfig
    base = {"type": "huggingface", "repo": "cellmap/mito-aff-unet-setup-16"}
    with pytest.raises(ValueError):
        FinetuneModelConfig(base_model=base)
    with pytest.raises(ValueError):
        FinetuneModelConfig(lora_adapter_path="/a", weights_path="/w.pt", base_model=base)
    with pytest.raises(ValueError):
        FinetuneModelConfig(weights_path="/w.pt")
    cfg = FinetuneModelConfig(weights_path="/w.pt", base_model=base, name="ft")
    assert "--weights-path /w.pt" in cfg.command and "--lora-adapter-path" not in cfg.command
    assert cfg.to_dict()["weights_path"] == "/w.pt"
    cfg = FinetuneModelConfig(lora_adapter_path="/a", base_model=base, name="lora")
    assert "--lora-adapter-path /a" in cfg.command and "--weights-path" not in cfg.command


def test_generated_yaml_carries_weights_path_for_full_finetune(tmp_path):
    import yaml
    from cellmap_flow.finetune.finetuned_model_templates import generate_finetuned_model_yaml
    base = {"type": "huggingface", "repo": "cellmap/mito-aff-unet-setup-16"}
    out = generate_finetuned_model_yaml(
        weights_path="/w.pt", base_model_dict=base, model_name="ft",
        output_path=tmp_path / "ft.yaml", data_path="/data.zarr",
    )
    entry = yaml.safe_load(open(out))["models"][0]
    assert entry["weights_path"] == "/w.pt" and "lora_adapter_path" not in entry
    with pytest.raises(ValueError):
        generate_finetuned_model_yaml(
            lora_adapter_path="/a", weights_path="/w.pt", base_model_dict=base,
            model_name="x", output_path=tmp_path / "x.yaml", data_path="/data.zarr",
        )


def test_export_kwargs_prefer_what_is_on_disk(tmp_path):
    from cellmap_flow.finetune.finetune_job_manager import finetune_export_kwargs
    # nothing written yet: fall back to the job's rank
    assert "lora_adapter_path" in finetune_export_kwargs(tmp_path, {"lora_r": 64})
    assert "weights_path" in finetune_export_kwargs(tmp_path, {"lora_r": 0})
    # a full-finetune export on disk wins regardless of params
    (tmp_path / "full_finetune").mkdir()
    (tmp_path / "full_finetune" / "model_state_dict.pt").write_bytes(b"x")
    out = finetune_export_kwargs(tmp_path, {"lora_r": 64})
    assert out == {"weights_path": str(tmp_path / "full_finetune" / "model_state_dict.pt")}
