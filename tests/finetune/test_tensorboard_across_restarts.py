"""TensorBoard curves continue across restart iterations, and writers are closed.

The CLI makes a new trainer for every iteration, all writing to the same
tensorboard/ directory, and each started its steps at 0: every iteration's
curves were drawn on top of the others. The SummaryWriter was never closed.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner

pytest.importorskip("tensorboard")


def _trainer(run, **kw):
    loader = DataLoader(TensorDataset(torch.rand(4, 1, 4, 4, 4), torch.ones(4, 1, 4, 4, 4)), batch_size=2)
    return LoRAFinetuner(
        torch.nn.Conv3d(1, 1, 1), loader, output_dir=str(run), num_epochs=1, device="cpu",
        use_mixed_precision=False, loss_type="bce", tensorboard=True, **kw,
    )


def test_a_restarted_iteration_continues_the_curves(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    first = _trainer(tmp_path)
    first.train()
    first.close()
    assert first.tb is None

    second = _trainer(tmp_path, tb_start_step=first._tb_step, tb_start_epoch=first._tb_epoch)
    second.train()
    second.close()

    acc = EventAccumulator(str(tmp_path / "tensorboard"))
    acc.Reload()
    steps = [e.step for e in acc.Scalars("train/loss")]
    epochs = [e.step for e in acc.Scalars("epoch/loss")]
    assert steps == [1, 2, 3, 4]
    assert epochs == [1, 2]
