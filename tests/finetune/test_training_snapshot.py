"""What a finetune serves is what it trained.

Two tiny nets -- a Conv3d net with a 1x1x1 two-channel head, and one with
BatchNorm that ends in a sigmoid -- are finetuned with a LoRA adapter and in
full, on the CPU with one thread, fixed seeds and no loader workers, with
distillation (on unlabeled voxels and on good-region anchors). The model served
from the export computes exactly what was trained, and the one export_merged
folds computes the same to float precision. Through the CLI, two iterations
with a restart between them: the second trains again and is what is served.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper
from cellmap_flow.finetune.target_transforms import (
    BinaryTargetTransform,
    BroadcastBinaryTargetTransform,
)


@pytest.fixture(autouse=True)
def _one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _net(kind):
    torch.manual_seed(0)
    if kind == "head":
        net = nn.Sequential(nn.Conv3d(1, 8, 3), nn.ReLU(), nn.Conv3d(8, 2, 1))
    else:
        net = nn.Sequential(
            nn.Conv3d(1, 4, 3), nn.BatchNorm3d(4), nn.ReLU(), nn.Conv3d(4, 1, 1), nn.Sigmoid()
        )
    return BatchLoopWrapper(net)


def _data(n, anchors=False):
    g = torch.Generator().manual_seed(1)
    raw = torch.rand(n, 1, 8, 8, 8, generator=g)
    ann = torch.randint(0, 3, (n, 1, 6, 6, 6), generator=g).float()
    ann[:, :, :2] = 0  # an unannotated slab, for distillation on unlabeled voxels
    tensors = [raw, ann]
    if anchors:
        tensors.append((torch.rand(n, 1, 6, 6, 6, generator=g) > 0.5).float())
    return TensorDataset(*tensors)


# name: (net, LoRA?, trainer settings, samples, anchors). The head net emits
# logits, the BatchNorm net probabilities, which switches the losses' sigmoid
# off and BCE-with-logits to BCE.
CASES = {
    "head-lora": ("head", True, dict(
        loss_type="bce", target_transform=BroadcastBinaryTargetTransform(2),
        label_smoothing=0.1, balance_classes=True, distillation_lambda=0.5,
    ), 4, False),
    "bn-lora": ("bn", True, dict(
        loss_type="margin", target_transform=BinaryTargetTransform(), balance_classes=True,
        distillation_lambda=1.0,
    ), 4, True),
    "bn-full": ("bn", False, dict(
        loss_type="bce", target_transform=BinaryTargetTransform(), learning_rate=1e-2,
    ), 4, False),
}


def _train(case, tmp_path):
    kind, lora, settings, n, anchors = CASES[case]
    model = _net(kind)
    if lora:
        from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora

        model = wrap_model_with_lora(model, lora_r=4, lora_alpha=8, lora_dropout=0.1)
    loader = DataLoader(_data(n, anchors), batch_size=2)
    trainer = LoRAFinetuner(
        model, loader, output_dir=str(tmp_path / "run"), num_epochs=3, device="cpu",
        use_mixed_precision=False, tensorboard=False, **settings,
    )
    trainer.train()
    exported = Path(trainer.save_adapter(export_dir=str(tmp_path / "export")))
    return trainer, exported


def _served(case, exported):
    kind, lora = CASES[case][:2]
    base = _net(kind).eval()
    if lora:
        from cellmap_flow.finetune.lora_wrapper import load_lora_adapter

        return load_lora_adapter(base, str(exported), is_trainable=False).eval()
    base.load_state_dict(torch.load(exported, weights_only=True))
    return base


def _merged(case, exported):
    from cellmap_flow.finetune.export_merged import apply_finetune

    kind, lora = CASES[case][:2]
    eager = _net(kind).model.eval()
    if lora:
        return apply_finetune(eager, lora_adapter_path=str(exported))
    return apply_finetune(eager, weights_path=str(exported))


@pytest.mark.parametrize("case", [
    pytest.param(case, marks=[pytest.mark.finetune] if CASES[case][1] else []) for case in CASES
])
def test_the_export_serves_and_merges_what_was_trained(case, tmp_path):
    trainer, exported = _train(case, tmp_path)
    x = torch.rand(1, 1, 8, 8, 8, generator=torch.Generator().manual_seed(2))

    # The export serves what was trained (save_adapter left the best epoch loaded).
    trainer.model.eval()
    with torch.no_grad():
        trained = trainer.model(x)
        served = _served(case, exported)(x)
        merged = _merged(case, exported)(x)
    assert torch.equal(served, trained)
    assert (merged - served).abs().max().item() <= 1e-6 * max(served.abs().max().item(), 1.0)


# --- through the CLI -----------------------------------------------------------


def _exported_weights(export, lora):
    if lora:
        from safetensors.torch import load_file

        return load_file(str(export / "adapter_model.safetensors"))
    return torch.load(export / "model_state_dict.pt", weights_only=True)


@pytest.mark.parametrize("lora_r", [
    pytest.param(4, marks=pytest.mark.finetune, id="lora"),
    pytest.param(0, id="full"),
])
def test_a_restart_trains_a_second_iteration_and_serves_it(run_cli, tmp_path, lora_r):
    """Two iterations of a served job: the restart arrives through the signal
    file the job manager writes, and the job ends on the next signal, which is
    not a restart request."""

    def through_the_signal_file(signal_file, controller):
        signal_file.write_text(json.dumps({"params": {"learning_rate": 2e-4}}))

    cli = run_cli("--lora-r", str(lora_r), "--num-epochs", "2", "--auto-serve", "--serve-data-path", str(tmp_path),
                  restarts=[through_the_signal_file])
    assert cli.code == 1  # the second signal, not a restart request, ends it
    assert (len(cli.loaded), len(cli.served), cli.waited) == (2, 1, 2)

    yamls = sorted((cli.session / "models").glob("tiny_finetuned_*.yaml"))
    entry = yaml.safe_load(yamls[-1].read_text())["models"][0]
    assert entry["type"] == "finetune" and entry["base_model"]["type"] == "script"
    name, source = ("lora_adapter", "lora_adapter_path") if lora_r else ("full_finetune", "weights_path")
    first, second = sorted((cli.run / "iterations").iterdir())
    assert second in Path(entry[source]).parents, "the last YAML serves the second iteration's export"
    weights = [_exported_weights(iteration / name, lora_r) for iteration in (first, second)]
    assert any(not torch.equal(weights[0][key], weights[1][key]) for key in weights[0]), "the second trained again"
