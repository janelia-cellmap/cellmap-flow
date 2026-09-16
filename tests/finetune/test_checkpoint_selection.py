"""Checkpoint selection must rank epochs by the supervised term.

The training objective is ``supervised + lambda * distillation``, and the
distillation term is minimized by not changing the model at all: LoRA starts
at B=0, so the student is identical to the teacher and distillation is
exactly 0 on the first epoch. Ranking epochs by the combined loss therefore
handed epoch 1 a score no later epoch could beat, and since save_adapter()
exports best_checkpoint.pth, every run shipped a model one optimizer step
from where it started.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytest

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner


def _make_trainer(tmp_path, num_epochs, distillation_lambda=1.0):
    model = nn.Conv3d(1, 1, 1)
    dataloader = DataLoader(
        TensorDataset(torch.zeros(2, 1, 4, 4, 4), torch.zeros(2, 1, 4, 4, 4)),
        batch_size=1,
    )
    return LoRAFinetuner(
        model,
        dataloader,
        output_dir=str(tmp_path / "run"),
        num_epochs=num_epochs,
        device="cpu",
        use_mixed_precision=False,
        distillation_lambda=distillation_lambda,
    )


def _replay(trainer, per_epoch, monkeypatch):
    """Drive train() through a recorded (total, supervised) loss sequence."""
    seen = {"i": 0}

    def fake_epoch():
        total, supervised = per_epoch[seen["i"]]
        seen["i"] += 1
        trainer.last_supervised_loss = supervised
        return total

    monkeypatch.setattr(trainer, "_train_epoch", fake_epoch)
    trainer.train()


def _best_epoch(trainer):
    ckpt = torch.load(trainer.output_dir / "best_checkpoint.pth", map_location="cpu")
    return ckpt["epoch"] + 1


def test_epoch_one_does_not_win_on_its_free_zero_distillation(tmp_path, monkeypatch):
    """The real numbers from a run that shipped a one-step adapter.

    Supervised barely moves and total loss rises after epoch 1 purely because
    distillation switches on. The best supervised epoch is 2, not 1.
    """
    recorded = [
        (0.244829, 0.244829),  # distill == 0 here, and only here
        (0.339838, 0.244673),  # <- lowest supervised
        (0.316148, 0.244753),
        (0.393911, 0.244810),
        (0.302903, 0.244831),
        (0.292545, 0.244833),
        (0.317944, 0.244829),
        (0.258822, 0.244829),
    ]
    trainer = _make_trainer(tmp_path, len(recorded))
    _replay(trainer, recorded, monkeypatch)

    assert _best_epoch(trainer) == 2
    assert trainer.best_loss == pytest.approx(0.244673)


def test_steady_learning_is_not_discarded(tmp_path, monkeypatch):
    """Supervised falls every epoch while distillation grows faster.

    Combined loss rises monotonically, so the old rule kept epoch 1 and threw
    away all the learning. The last epoch is the right pick.
    """
    recorded = [(0.50, 0.50)] + [
        (0.50 - 0.02 * i + 0.10 * i, 0.50 - 0.02 * i) for i in range(1, 6)
    ]
    assert [t for t, _ in recorded] == sorted(t for t, _ in recorded)  # total only rises

    trainer = _make_trainer(tmp_path, len(recorded))
    _replay(trainer, recorded, monkeypatch)

    assert _best_epoch(trainer) == len(recorded)
    assert trainer.best_loss == pytest.approx(0.40)


def test_without_distillation_the_two_criteria_agree(tmp_path, monkeypatch):
    """lambda=0 means total == supervised, so behaviour is unchanged."""
    recorded = [(0.9, 0.9), (0.4, 0.4), (0.6, 0.6)]
    trainer = _make_trainer(tmp_path, len(recorded), distillation_lambda=0.0)
    _replay(trainer, recorded, monkeypatch)

    assert _best_epoch(trainer) == 2
    assert trainer.best_loss == pytest.approx(0.4)
