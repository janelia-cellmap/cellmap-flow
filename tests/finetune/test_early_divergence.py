"""A run that diverges before anything is served exits instead of waiting.

With --auto-serve the trainer waits for a restart after a diverged
iteration. The restart is sent to the job's inference server, which only
starts after an iteration completes -- so a first iteration that diverged
waited for a restart that could never arrive, holding the GPU until walltime.
And the unrecoverable-OOM path returned "diverged" without printing the
TRAINING_DIVERGED marker the job manager watches for.
"""

import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune import finetune_cli
from cellmap_flow.finetune.lora_trainer import LoRAFinetuner

SCRIPT = """
from funlib.geometry import Coordinate
import torch.nn as nn

input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
model = nn.Conv3d(1, 1, 1)
"""


def _nan_loader():
    raw = torch.full((2, 1, 4, 4, 4), float("nan"))
    ann = torch.ones(2, 1, 4, 4, 4)
    return DataLoader(TensorDataset(raw, ann), batch_size=2)


def test_a_diverged_first_iteration_exits_with_an_error(tmp_path, monkeypatch, capsys):
    script = tmp_path / "model.py"
    script.write_text(SCRIPT)
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    waited = []
    monkeypatch.setattr(finetune_cli, "create_dataloader", lambda *a, **k: _nan_loader())
    monkeypatch.setattr(
        finetune_cli, "_wait_for_restart_signal", lambda **k: waited.append(k)
    )
    monkeypatch.setattr(sys, "argv", [
        "finetune_cli", "--model-type", "script", "--model-script", str(script),
        "--corrections", str(corrections), "--output-dir", str(tmp_path / "run"),
        "--lora-r", "0", "--num-epochs", "1", "--loss-type", "bce",
        "--no-mixed-precision", "--no-tensorboard", "--num-workers", "0",
        "--auto-serve", "--serve-data-path", str(tmp_path),
    ])

    assert finetune_cli.main() == 1
    assert waited == [], "nothing can deliver a restart before the server starts"
    assert "TRAINING_DIVERGED" in capsys.readouterr().out


def test_an_unrecoverable_oom_prints_the_diverged_marker(tmp_path, monkeypatch, capsys):
    loader = DataLoader(TensorDataset(torch.rand(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4)), batch_size=1)
    trainer = LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1), loader, output_dir=str(tmp_path), num_epochs=1,
        device="cpu", use_mixed_precision=False, loss_type="bce", tensorboard=False,
    )

    def oom():
        raise torch.cuda.OutOfMemoryError("out of memory")

    monkeypatch.setattr(trainer, "_train_epoch", oom)
    stats = trainer.train()

    assert stats["diverged"]
    assert "TRAINING_DIVERGED" in capsys.readouterr().out


@pytest.fixture(autouse=True)
def _no_cuda_cache(monkeypatch):
    # The OOM path empties the CUDA cache; there is no CUDA device here.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
