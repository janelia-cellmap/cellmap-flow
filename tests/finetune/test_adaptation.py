"""LoRA or full: the model decides, and each strategy resets, restarts and
teaches in its own way."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from cellmap_flow.finetune.adaptation import FullStrategy, LoraStrategy, strategy_for


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv3d(1, 4, 3), nn.ReLU(), nn.Conv3d(4, 2, 1))


@pytest.mark.parametrize("lora, lora_r, expected", [
    (False, None, ("full", 0)),
    (False, 8, ("full", 0)),  # a restart cannot make a full finetune LoRA
    pytest.param(True, None, ("lora", 2), marks=pytest.mark.finetune),  # the adapter's own rank
    pytest.param(True, 0, ("lora", 2), marks=pytest.mark.finetune),  # nor LoRA a full one
    pytest.param(True, 8, ("lora", 8), marks=pytest.mark.finetune),  # the rank a restart makes
])
def test_the_model_decides(lora, lora_r, expected):
    model = LoraStrategy(2, 4, 0.0).prepare(_net()) if lora else FullStrategy().prepare(_net())
    strategy = strategy_for(model, lora_r)
    assert (strategy.kind, strategy.r) == expected
    assert strategy.export_name == {"lora": "lora_adapter", "full": "full_finetune"}[strategy.kind]


@pytest.mark.parametrize("kind", [pytest.param("lora", marks=pytest.mark.finetune), "full"])
def test_reset_and_restart_go_back_to_where_training_started(kind):
    """After a NaN the trainer resets in place; a restart starts the next iteration
    where the job started, a LoRA one with the rank it asks for. A full finetune
    had nothing to reset to: it kept NaN weights, served them and restarted from
    them. A LoRA restart unloads the old adapter; merged, it would stay in the base."""
    x = torch.rand(1, 1, 6, 6, 6)
    strategy = LoraStrategy(2, 4, 0.0) if kind == "lora" else FullStrategy()
    model = strategy.prepare(_net())
    initial, before = strategy.initial_state(model), model(x).detach()

    def trained():
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(1.0)
        assert not torch.allclose(model(x), before)
        return model

    assert strategy.reset(trained(), initial) is model
    assert torch.allclose(model(x), before, atol=1e-6)
    restarted = strategy_for(trained(), 4, alpha=8).restart(model, initial)
    assert torch.allclose(restarted(x), before, atol=1e-6)
    assert strategy_for(restarted).r == (4 if kind == "lora" else 0)
    assert kind == "lora" or restarted is model, "the model the server shares stays the same"


@pytest.mark.finetune
def test_the_lora_teacher_is_the_base_as_it_is_served():
    """The model with its adapter off, in eval mode (in train mode the base's
    dropout made the teacher's targets noisy), and back as it was afterwards."""
    model = LoraStrategy(2, 4, 0.0).prepare(
        nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.Dropout(0.5), nn.Conv3d(4, 1, 1))
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.fill_(1.0)  # so that the adapter changes something
    strategy = strategy_for(model)
    strategy.train_mode(model)
    x = torch.rand(1, 1, 6, 6, 6)
    teacher = strategy.teacher(model)
    with torch.no_grad(), teacher as base:
        first = base(x)
    with torch.no_grad(), teacher as base:
        assert torch.equal(base(x), first)
    assert model.training
    with torch.no_grad():
        assert not torch.allclose(model.eval()(x), first), "the adapter was off"


def test_the_new_modules_import_nothing_heavy(tmp_path):
    """The session layer serves the CLI and the trainer without a dashboard, a
    viewer or torch; adaptation and losses may use torch, never peft or the
    dashboard. Importing the CLI module leaves the importer's logging alone: it
    called logging.basicConfig(force=True) at import, and the dashboard imports it."""
    heavy = ["cellmap_flow.globals", "flask", "neuroglancer", "huggingface_hub", "peft"]
    session = [f"cellmap_flow.finetune.session.{m}" for m in ("manifest", "volume", "store", "minio", "sync", "instance")]
    code = (
        "import logging, sys\n"
        + "".join(f"import {module}\n" for module in session)
        + f"loaded = [m for m in {heavy + ['torch']!r} if m in sys.modules]; assert not loaded, loaded\n"
        "import cellmap_flow.finetune.adaptation, cellmap_flow.finetune.losses\n"
        f"loaded = [m for m in {heavy!r} if m in sys.modules]; assert not loaded, loaded\n"
        "import cellmap_flow.globals\n"
        "mine = logging.StreamHandler(); logging.root.addHandler(mine)\n"
        "import cellmap_flow.finetune.finetune_cli\n"
        "assert mine in logging.root.handlers, 'importing finetune_cli reconfigured logging'\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
        env={**os.environ, "HOME": str(tmp_path), "PYTHONPATH": root},
    )
    assert result.returncode == 0, result.stderr[-2000:]
