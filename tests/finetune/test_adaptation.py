"""LoRA or full: the model decides, and each strategy resets and restarts in its own way."""

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


def _lora(r=2):
    return LoraStrategy(r, 2 * r, 0.0).prepare(_net())


def _lora_B(model):
    return [p for n, p in model.named_parameters() if "lora_B" in n]


@pytest.mark.parametrize("lora, lora_r, expected", [
    (False, None, ("full", 0)),
    (False, 8, ("full", 0)),  # a restart cannot make a full finetune LoRA
    pytest.param(True, None, ("lora", 2), marks=pytest.mark.finetune),  # the adapter's own rank
    pytest.param(True, 0, ("lora", 2), marks=pytest.mark.finetune),  # nor LoRA a full one
    pytest.param(True, 8, ("lora", 8), marks=pytest.mark.finetune),  # the rank a restart makes
])
def test_the_model_decides(lora, lora_r, expected):
    model = _lora() if lora else FullStrategy().prepare(_net())
    strategy = strategy_for(model, lora_r)
    assert (strategy.kind, strategy.r) == expected
    assert strategy.export_name == {"lora": "lora_adapter", "full": "full_finetune"}[strategy.kind]


@pytest.mark.finetune
def test_lora_resets_in_place_and_restarts_with_a_new_rank():
    strategy = LoraStrategy(2, 4, 0.0)
    model = strategy.prepare(_net())
    base = {k: v.clone() for k, v in model.state_dict().items() if "lora_" not in k}
    with torch.no_grad():
        for p in _lora_B(model):
            p.fill_(1.0)

    assert strategy.reset(model, None) is model  # the trainer's retry after a NaN
    assert all(not p.any() for p in _lora_B(model))

    with torch.no_grad():
        for p in _lora_B(model):
            p.fill_(1.0)
    restarted = strategy_for(model, 4, alpha=8).restart(model, None)
    assert restarted.peft_config["default"].r == 4
    assert all(not p.any() for p in _lora_B(restarted))
    # Unloaded, not merged: the old adapter left nothing in the base.
    after = {k: v for k, v in restarted.state_dict().items() if "lora_" not in k}
    assert after.keys() == base.keys() and all(torch.equal(v, base[k]) for k, v in after.items())


def test_the_new_modules_import_nothing_heavy(tmp_path):
    """adaptation and losses may use torch (finetune/*), but never peft or the dashboard.

    And importing the CLI module leaves the importer's logging alone: it
    used to call logging.basicConfig(force=True) at import, and the
    dashboard imports it. (Importing cellmap_flow.globals configures logging
    too; the dashboard has done that long before.)
    """
    heavy = ["cellmap_flow.globals", "flask", "neuroglancer", "huggingface_hub", "peft"]
    code = (
        "import logging, sys\n"
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
