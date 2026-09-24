"""TensorBoard event files from LoRAFinetuner.

Background
----------
Until 2026-09-23 the only record of a run was training_log.txt, and the
per-epoch lines in it had no timestamps, so questions like "is the loader
keeping up" or "did bf16 actually speed things up" needed LSF logs and
regexes. The trainer now writes TensorBoard scalars per optimizer step and
per epoch, a config card, and mid-slice patch images, next to the log.

These tests pin down that a short real training loop on CPU:
  - writes an event file under <output_dir>/tensorboard,
  - contains the step, epoch, timing and image tags the docs promise,
  - writes nothing when tensorboard=False.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner


def _make_trainer(tmp_path, tensorboard):
    model = nn.Conv3d(1, 1, 1)
    dataloader = DataLoader(
        TensorDataset(torch.rand(4, 1, 4, 4, 4), (torch.rand(4, 1, 4, 4, 4) > 0.5).float()),
        batch_size=2,
    )
    return LoRAFinetuner(
        model,
        dataloader,
        output_dir=str(tmp_path / "run"),
        num_epochs=1,
        device="cpu",
        use_mixed_precision=False,
        mask_unannotated=False,
        loss_type="bce",
        tensorboard=tensorboard,
    )


def _tags(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(logdir))
    acc.Reload()
    return acc.Tags()


def test_event_file_has_step_epoch_timing_and_image_tags(tmp_path):
    trainer = _make_trainer(tmp_path, tensorboard=True)
    trainer.train()
    trainer.tb.flush()

    logdir = tmp_path / "run" / "tensorboard"
    assert any(f.name.startswith("events.out.tfevents") for f in logdir.iterdir())

    tags = _tags(logdir)
    scalars = set(tags["scalars"])
    for tag in ("train/loss", "train/supervised", "train/lr", "time/step_s",
                "time/data_wait_s", "epoch/loss", "epoch/supervised",
                "epoch/best_supervised", "time/epoch_data_wait_s",
                "time/epoch_compute_s"):
        assert tag in scalars, f"missing scalar {tag}; have {sorted(scalars)}"
    assert "patch/raw|target|prediction|mask" in set(tags["images"])
    assert "config/text_summary" in tags["tensors"] or "config" in str(tags)


def test_two_optimizer_steps_give_two_step_points(tmp_path):
    trainer = _make_trainer(tmp_path, tensorboard=True)
    trainer.train()
    # 4 samples / batch 2 = 2 optimizer steps in the single epoch.
    assert trainer._tb_step == 2
    assert trainer._tb_epoch == 1


def test_disabled_writes_nothing(tmp_path):
    trainer = _make_trainer(tmp_path, tensorboard=False)
    trainer.train()
    assert trainer.tb is None
    assert not (tmp_path / "run" / "tensorboard").exists()
