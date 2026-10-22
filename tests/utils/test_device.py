"""The device torch work runs on (utils.device)."""

import torch


def test_the_trainer_and_the_servers_take_apples_gpu(monkeypatch):
    """On a Mac: its GPU, not the CPU every model ran and trained on there."""
    from cellmap_flow.finetune.trainer import loop
    from cellmap_flow.models.configs import base

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert base._get_device() == torch.device("mps")
    assert loop.best_device() == torch.device("mps")
