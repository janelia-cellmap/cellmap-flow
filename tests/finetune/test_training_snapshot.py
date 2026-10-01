"""What a finetune computes and writes, pinned bit for bit.

Two tiny nets -- a Conv3d net with a 1x1x1 two-channel head, and one with
BatchNorm that ends in a sigmoid -- are finetuned with a LoRA adapter and in
full, on the CPU with one thread, fixed seeds and no loader workers. The
cases between them take every loss the trainer has, on logits and on
probabilities (BCE, Dice + BCE, margin, MSE; class balancing, label
smoothing; distillation on unlabeled voxels, on all of them and on
good-region anchors), and gradient accumulation with a leftover step.

Pinned:
- the loss of every epoch, the best epoch and a hash of the final weights;
- the keys of the exported adapter or weights and of best_checkpoint.pth
  (the model is served and merged by those names);
- that the model served from the export, and the one export_merged folds,
  compute what was trained;
- through the CLI, two iterations with a restart between them: the stdout
  markers, the iterations/NNN_<ts>/ layout with its lora_adapter or
  full_finetune link, and the hash of each iteration's weights.

Float results depend on the torch version and the CPU's vector unit, so the
numbers are recorded per platform. On a platform with no record the losses
are compared to 1e-4 and the hashes are not checked.
"""

import hashlib
import json
import os
import re
import sys
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

PLATFORM = f"{torch.__version__.split('+')[0]}/{torch.backends.cpu.get_cpu_capability()}"


@pytest.fixture(autouse=True)
def _one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _sha(tensors):
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].detach().cpu().contiguous()
        h.update(f"{key}:{t.dtype}:{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


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
    "head-full": ("head", False, dict(
        loss_type="combined", target_transform=BroadcastBinaryTargetTransform(2),
        distillation_lambda=0.5, distillation_all_voxels=True, gradient_accumulation_steps=2,
    ), 6, False),
    "head-mse": ("head", False, dict(
        loss_type="mse", target_transform=BroadcastBinaryTargetTransform(2), learning_rate=1e-2,
    ), 4, False),
    "bn-lora": ("bn", True, dict(
        loss_type="margin", target_transform=BinaryTargetTransform(), balance_classes=True,
        distillation_lambda=1.0,
    ), 4, True),
    "bn-full": ("bn", False, dict(
        loss_type="bce", target_transform=BinaryTargetTransform(), learning_rate=1e-2,
    ), 4, False),
    "bn-combined": ("bn", False, dict(
        loss_type="combined", target_transform=BinaryTargetTransform(),
    ), 4, False),
}


def _lora_keys(*layers):
    return {
        "adapter": [f"base_model.model.model.{i}.lora_{ab}.weight" for i in layers for ab in "AB"],
        "target_modules": [f"model.{i}" for i in layers],
        "checkpoint": [f"base_model.model.model.{i}.lora_{ab}.default.weight" for i in layers for ab in "AB"],
    }


# Platform-independent: the names the exports are read by (net, LoRA?).
KEYS = {
    ("head", True): _lora_keys(0, 2),
    ("head", False): {"weights": ["model.0.bias", "model.0.weight", "model.2.bias", "model.2.weight"]},
    ("bn", True): _lora_keys(0, 3),
    ("bn", False): {"weights": [
        "model.0.bias", "model.0.weight", "model.1.bias", "model.1.num_batches_tracked",
        "model.1.running_mean", "model.1.running_var", "model.1.weight", "model.3.bias", "model.3.weight",
    ]},
}
LORA_CHECKPOINT = [
    "best_loss", "epoch", "global_step", "lora_only", "model_state_dict",
    "optimizer_state_dict", "scaler_state_dict", "training_stats",
]
FULL_CHECKPOINT = [
    "best_loss", "epoch", "full_model", "global_step", "lora_only", "model_state_dict",
    "training_stats",
]

# platform: case: numbers, recorded at 3af8578 (the finetune and py311 envs);
# the LoRA and BatchNorm cases re-recorded when the startup probes moved to eval mode.
NUMBERS = {
    "2.11.0/AVX512": {
        "head-lora": dict(losses=[0.6987209916114807, 0.698713093996048, 0.6987030804157257], best_epoch=3,
            final="b4b73e9c69c69ba5", served="763d74da39819ca9", merged="7463fa1fd5b1bd37"),
        "head-full": dict(losses=[0.5870955586433411, 0.5870503385861715, 0.5870087544123331], best_epoch=3,
            final="3b179152167fc898", served="a7e8a07a96a7d192", merged="c10ba1aafb918bde"),
        "head-mse": dict(losses=[0.255184605717659, 0.25077535957098007, 0.24827992916107178], best_epoch=3,
            final="6dea9ec7b5241659", served="f049423276a5ffee", merged="3470018947a22780"),
        "bn-lora": dict(losses=[0.04049694538116455, 0.040494199842214584, 0.04049184173345566], best_epoch=3,
            final="d2fbab97a3adffd7", served="3bca8c6c1ebbbf08", merged="09610b81b13e9af7"),
        "bn-full": dict(losses=[0.6908666491508484, 0.6729371845722198, 0.6613186895847321], best_epoch=3,
            final="7e24ea0fc66ed2f1", served="adfee9d4c0dfe3aa", merged="af112bc0f53a2732"),
        "bn-combined": dict(losses=[0.5986000299453735, 0.5984358191490173, 0.5982885360717773], best_epoch=3,
            final="9a8a5256df4e1228", served="bbd0738a34b201d1", merged="3f1e2ba12b3e35d5"),
        "cli-lora": dict(weights=["5f01c87384bc375a", "9f353fadafb00a4d"]),
        "cli-full": dict(weights=["7be962342aa2a858", "7972cf973bddacaa"]),
    },
    "2.5.1/AVX512": {
        "head-full": dict(losses=[0.5870955586433411, 0.5870503385861715, 0.5870087544123331], best_epoch=3,
            final="3b179152167fc898", served="a7e8a07a96a7d192", merged="c10ba1aafb918bde"),
        "head-mse": dict(losses=[0.255184605717659, 0.25077535957098007, 0.24827992916107178], best_epoch=3,
            final="6dea9ec7b5241659", served="f049423276a5ffee", merged="3470018947a22780"),
        "bn-full": dict(losses=[0.6908666491508484, 0.6729371845722198, 0.6613186895847321], best_epoch=3,
            final="cfb16c39814850ef", served="1306fa048cffbf5a", merged="7f75e17107964203"),
        "bn-combined": dict(losses=[0.5986000299453735, 0.5984358191490173, 0.598288506269455], best_epoch=3,
            final="254682e13f29af18", served="b0528b3794cc2ee6", merged="10acd5b5b2e9179e"),
        "cli-full": dict(weights=["7be962342aa2a858", "7972cf973bddacaa"]),
    },
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
    final = _sha(trainer.model.state_dict())
    exported = Path(trainer.save_adapter(export_dir=str(tmp_path / "export")))
    best = torch.load(tmp_path / "run" / "best_checkpoint.pth", weights_only=False)
    return trainer, exported, best, final


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


def _check_numbers(case, got):
    """Exact on a platform with a record; elsewhere the losses only, to 1e-4."""
    recorded = NUMBERS.get(PLATFORM, {}).get(case)
    if recorded is not None:
        assert got == recorded
        return
    for numbers in NUMBERS.values():
        if case in numbers and "losses" in got:
            assert got["losses"] == pytest.approx(numbers[case]["losses"], rel=1e-4)
            assert got["best_epoch"] == numbers[case]["best_epoch"]
            return


def _case_params():
    return [
        pytest.param(case, marks=[pytest.mark.finetune] if CASES[case][1] else [])
        for case in CASES
    ]


@pytest.mark.parametrize("case", _case_params())
def test_training_computes_and_exports_what_it_always_did(case, tmp_path):
    kind, lora = CASES[case][:2]
    trainer, exported, best, final = _train(case, tmp_path)
    x = torch.rand(1, 1, 8, 8, 8, generator=torch.Generator().manual_seed(2))

    keys = KEYS[kind, lora]
    if lora:
        from safetensors.torch import load_file

        assert sorted(os.listdir(exported)) == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
        assert sorted(load_file(str(exported / "adapter_model.safetensors"))) == keys["adapter"]
        config = json.loads((exported / "adapter_config.json").read_text())
        assert (config["r"], config["lora_alpha"], sorted(config["target_modules"])) == (4, 8, keys["target_modules"])
        assert sorted(best) == LORA_CHECKPOINT
        assert sorted(best["model_state_dict"]) == keys["checkpoint"]
    else:
        assert exported.relative_to(tmp_path) == Path("export/full_finetune/model_state_dict.pt")
        assert sorted(torch.load(exported, weights_only=True)) == keys["weights"]
        assert sorted(best) == FULL_CHECKPOINT
        assert sorted(best["model_state_dict"]) == keys["weights"]

    # The export serves what was trained (save_adapter left the best epoch loaded).
    trainer.model.eval()
    with torch.no_grad():
        trained = trainer.model(x)
        served = _served(case, exported)(x)
        merged_model = _merged(case, exported)
        merged = merged_model(x)
    assert torch.equal(served, trained)
    assert (merged - served).abs().max().item() <= 1e-6 * max(served.abs().max().item(), 1.0)

    _check_numbers(case, {
        "losses": [s["loss"] for s in trainer.training_stats],
        "best_epoch": best["epoch"] + 1,
        "final": final,
        "served": _sha({"out": served}),
        "merged": _sha(merged_model.state_dict()),
    })


# --- through the CLI -----------------------------------------------------------

SCRIPT = """
from funlib.geometry import Coordinate
import torch
import torch.nn as nn
from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper

input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(8, 8, 8) * input_voxel_size
write_shape = Coordinate(6, 6, 6) * output_voxel_size
output_channels = 2
torch.manual_seed(0)
model = BatchLoopWrapper(nn.Sequential(nn.Conv3d(1, 8, 3), nn.ReLU(), nn.Conv3d(8, 2, 1)))
"""

MARKERS = re.compile(
    r"^(TRAINING_ITERATION_COMPLETE:|FINETUNED_MODEL_YAML:|RESTART_FAILED:|INFERENCE_SERVER_FAILED:"
    r"|TRAINING_DIVERGED|RESTARTING_TRAINING|WAITING_FOR_RESTART|RESTART_STATUS:)"
)

CLI_MARKERS = [
    "FINETUNED_MODEL_YAML: <session>/models/tiny_finetuned_<ts>.yaml",
    "TRAINING_ITERATION_COMPLETE: tiny_finetuned_<ts>",
    "WAITING_FOR_RESTART",
    "RESTARTING_TRAINING",
    "RESTART_STATUS: Loading corrections...",
    "RESTART_STATUS: Preparing trainer...",
    "RESTART_STATUS: Starting training...",
    "FINETUNED_MODEL_YAML: <session>/models/tiny_finetuned_<ts>.yaml",
    "TRAINING_ITERATION_COMPLETE: tiny_finetuned_<ts>",
    "WAITING_FOR_RESTART",
]


def _cli_layout(name):
    layout = {
        f"iterations/001_<ts>/{name}", f"iterations/002_<ts>/{name}",
        f"{name} -> iterations/002_<ts>/{name}", "best_checkpoint.pth",
    }
    if name == "lora_adapter":
        layout.add("checkpoint_epoch_2.pth")  # a full finetune keeps only its best
    return layout


def _normalize(text, tmp_path):
    return re.sub(r"\d{8}_\d{6}", "<ts>", text.replace(str(tmp_path / "session"), "<session>"))


@pytest.mark.parametrize("lora_r", [
    pytest.param(4, marks=pytest.mark.finetune, id="lora"),
    pytest.param(0, id="full"),
])
def test_the_cli_writes_and_announces_what_it_always_did(lora_r, tmp_path, monkeypatch, capsys):
    from cellmap_flow.finetune import finetune_cli, session_loop

    script = tmp_path / "model.py"
    script.write_text(SCRIPT)
    session = tmp_path / "session"
    corrections = session / "corrections"
    corrections.mkdir(parents=True)
    (corrections / "_virtual_sources.json").write_text(
        json.dumps({"kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr"})
    )
    run = session / "runs" / "run"

    loaded, served, waited = [], [], []

    def dataloader(*a, **k):
        loaded.append(k)
        return DataLoader(_data(4), batch_size=2)

    signals = iter([json.dumps({"params": {"learning_rate": 2e-4}}), "not json"])
    real_wait = session_loop._wait_for_restart_signal

    def wait(**kwargs):
        waited.append(kwargs["signal_file"])
        kwargs["signal_file"].write_text(next(signals))
        return real_wait(**kwargs)

    monkeypatch.setattr(session_loop, "create_dataloader", dataloader)
    monkeypatch.setattr(session_loop, "_start_inference_server_background", lambda *a, **k: served.append(a) or (None, 0))
    monkeypatch.setattr(session_loop, "_wait_for_restart_signal", wait)
    monkeypatch.setattr(sys, "argv", [
        "finetune_cli", "--model-type", "script", "--model-script", str(script),
        "--model-name", "tiny", "--corrections", str(corrections), "--output-dir", str(run),
        "--lora-r", str(lora_r), "--num-epochs", "2", "--loss-type", "bce",
        "--output-type", "binary_broadcast", "--distillation-lambda", "0.5",
        "--no-mixed-precision", "--no-tensorboard", "--num-workers", "0",
        "--auto-serve", "--serve-data-path", str(tmp_path),
    ])

    assert finetune_cli.main() == 1  # the second, malformed, restart signal ends it
    assert (len(loaded), len(served), len(waited)) == (2, 1, 2)

    out = capsys.readouterr().out
    markers = [_normalize(line, tmp_path) for line in out.splitlines() if MARKERS.match(line)]
    assert markers == CLI_MARKERS

    name = "lora_adapter" if lora_r else "full_finetune"
    layout = set()
    for path in run.rglob("*"):
        rel = _normalize(str(path.relative_to(run)), tmp_path)
        if path.is_symlink():
            layout.add(f"{rel} -> {_normalize(os.readlink(path), tmp_path)}")
        elif path.is_file() and path.parent.name not in ("lora_adapter", "full_finetune"):
            layout.add(rel)
        elif path.name == name:
            layout.add(rel)
    assert layout == _cli_layout(name)

    yamls = sorted((session / "models").glob("tiny_finetuned_*.yaml"))
    entry = yaml.safe_load(yamls[-1].read_text())["models"][0]
    source = "lora_adapter_path" if lora_r else "weights_path"
    assert _normalize(entry[source], tmp_path).startswith("<session>/runs/run/iterations/002_<ts>/")
    assert entry["type"] == "finetune" and entry["base_model"]["type"] == "script"

    weights = []
    for it in sorted((run / "iterations").iterdir()):
        if lora_r:
            from safetensors.torch import load_file

            weights.append(_sha(load_file(str(it / name / "adapter_model.safetensors"))))
        else:
            weights.append(_sha(torch.load(it / name / "model_state_dict.pt", weights_only=True)))
    _check_numbers(f"cli-{'lora' if lora_r else 'full'}", {"weights": weights})
