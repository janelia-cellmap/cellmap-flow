"""--resume continues after the saved epoch; a stop before any epoch is not a crash.

load_checkpoint() restored current_epoch but train() looped from 0, so a
resumed run did every epoch again. A stop signal before the first epoch
finished left epoch_loss None, and the closing summary's f"{None:.6f}"
raised TypeError.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner


def _trainer(tmp_path, num_epochs):
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 4, 4, 4), torch.ones(2, 1, 4, 4, 4)), batch_size=2)
    return LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1), loader, output_dir=str(tmp_path), num_epochs=num_epochs,
        device="cpu", use_mixed_precision=False, loss_type="bce", tensorboard=False,
    )


def _count_epochs(trainer, monkeypatch):
    ran = []

    def fake_epoch():
        ran.append(trainer.current_epoch)
        trainer.last_supervised_loss = 0.5
        return 0.5

    monkeypatch.setattr(trainer, "_train_epoch", fake_epoch)
    return ran


def test_resume_carries_on_after_the_checkpoints_epoch(tmp_path, monkeypatch):
    first = _trainer(tmp_path / "a", num_epochs=3)
    first.current_epoch = 2
    first.save_checkpoint(is_best=True)

    resumed = _trainer(tmp_path / "b", num_epochs=5)
    resumed.load_checkpoint(str(tmp_path / "a" / "best_checkpoint.pth"))
    ran = _count_epochs(resumed, monkeypatch)
    resumed.train()

    assert ran == [3, 4], "epochs 1-3 were done before the checkpoint"


def test_a_stop_before_the_first_epoch_ends_cleanly(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, num_epochs=3)
    ran = _count_epochs(trainer, monkeypatch)
    real_exists = type(tmp_path).exists
    # The stale-signal cleanup at the start of train() must not see it; the
    # epoch loop's check must.
    seen = {"n": 0}

    def exists(path):
        if path.name == "stop_signal.json":
            seen["n"] += 1
            return seen["n"] > 1
        return real_exists(path)

    monkeypatch.setattr(type(tmp_path), "exists", exists)
    stats = trainer.train()

    assert ran == []
    assert stats["final_loss"] is None
