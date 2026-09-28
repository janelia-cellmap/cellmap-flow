"""Every training iteration keeps its own export; the old names follow the latest.

Each iteration's YAML pointed at the run's one lora_adapter/ (or
full_finetune/), which the next restart overwrote, so older
*_finetuned_<ts>.yaml files -- and models registered from them -- silently
served the newest weights. Exports now go to iterations/<n>_<ts>/, and
lora_adapter / full_finetune become links to the latest one, so whatever
reads those names (finetune_export_kwargs, complete_job, older tools) still
finds an export.
"""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.finetune_cli import _generate_model_files, _point_latest_export
from cellmap_flow.finetune.finetune_job_manager import finetune_export_kwargs
from cellmap_flow.finetune.lora_trainer import LoRAFinetuner


def _trainer(run):
    loader = DataLoader(TensorDataset(torch.rand(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4)), batch_size=1)
    return LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1), loader, output_dir=str(run), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", tensorboard=False,
    )


def _export(trainer, run, name):
    export_dir = run / "iterations" / name
    trainer.save_adapter(export_dir=str(export_dir))
    _point_latest_export(run, export_dir, is_lora=False)
    return export_dir


def test_an_earlier_iterations_export_survives_the_next(tmp_path):
    run = tmp_path / "run"
    trainer = _trainer(run)
    with torch.no_grad():
        trainer.model.weight.fill_(1.0)
    first = _export(trainer, run, "001_20260101_000000")
    with torch.no_grad():
        trainer.model.weight.fill_(2.0)
    second = _export(trainer, run, "002_20260101_000100")

    first_weights = torch.load(first / "full_finetune" / "model_state_dict.pt")
    assert torch.all(first_weights["weight"] == 1.0)
    latest = run / "full_finetune" / "model_state_dict.pt"
    assert torch.all(torch.load(latest)["weight"] == 2.0)
    assert os.path.realpath(run / "full_finetune") == os.path.realpath(second / "full_finetune")
    # What the job manager reads still resolves, to the latest.
    assert finetune_export_kwargs(run) == {"weights_path": str(latest)}


def test_the_yaml_names_its_own_iteration(tmp_path):
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    (corrections / "_virtual_sources.json").write_text(json.dumps({"raw_dataset_path": "/data/raw.zarr"}))
    run = tmp_path / "run"
    export_dir = run / "iterations" / "001_20260101_000000"
    args = argparse.Namespace(output_dir=str(run), models_dir=None, lora_r=8,
                              corrections=str(corrections), serve_data_path=None, auto_serve=False)
    model_config = SimpleNamespace(name="m", to_dict=lambda: {"type": "fly"})

    _, path = _generate_model_files(args, model_config, "20260101_000000", is_lora=True, export_dir=export_dir)

    entry = yaml.safe_load(open(path))["models"][0]
    assert entry["lora_adapter_path"] == str(export_dir / "lora_adapter")


def test_a_run_directory_from_before_keeps_its_export(tmp_path):
    """A real full_finetune/ from an older run is moved aside, not deleted."""
    run = tmp_path / "run"
    (run / "full_finetune").mkdir(parents=True)
    (run / "full_finetune" / "model_state_dict.pt").write_bytes(b"old")
    trainer = _trainer(run)

    _export(trainer, run, "001_20260101_000000")

    assert (run / "full_finetune").is_symlink()
    kept = list((run / "iterations").glob("*/full_finetune/model_state_dict.pt"))
    assert any(p.read_bytes() == b"old" for p in kept)
