"""The device torch work runs on (utils.device)."""

import pytest
import torch

from cellmap_flow.utils.device import best_device


@pytest.mark.parametrize("cuda, mps, expected", [
    (True, True, "cuda"),
    # A Mac: its GPU, not the CPU every model ran and trained on there.
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_cuda_then_apples_gpu_then_the_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert best_device() == torch.device(expected)


def test_the_trainer_and_the_servers_take_it(monkeypatch):
    from cellmap_flow.finetune import lora_trainer
    from cellmap_flow.models.configs import base

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert base._get_device() == torch.device("mps")
    assert lora_trainer.best_device() == torch.device("mps")
