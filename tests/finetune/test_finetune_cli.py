"""python -m cellmap_flow.finetune.finetune_cli, through main(): what a run
exports and announces, what it refuses, and what a restart may change.

The markers of a run with one restart, its iterations/NNN_<ts>/ layout and the
weights of each iteration are pinned by test_training_snapshot.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.adaptation import cpu_state_copy
from cellmap_flow.finetune.finetune_job_manager import finetune_export_kwargs
from cellmap_flow.models.models_config import ScriptModelConfig


def _yamls(cli):
    return [line.split(":", 1)[1].strip() for line in cli.markers if line.startswith("FINETUNED_MODEL_YAML:")]


@pytest.mark.parametrize("flags, manifest, attrs, yaml_dir", [
    # The YAMLs went to output_dir/../../.., which for this run is /: a
    # permission error, after training had succeeded, reported as a failure.
    ([], True, None, "run/models"),
    (["--models-dir", "<tmp>/m"], True, None, "m"),
    ([], {"kind": "volume_zarr_v1"}, {"dataset_path": "/data/raw.zarr"}, "run/models"),  # the volume's own attrs
    # A finetuned model trains on; its full weights are served on its root base's module tree.
    (["--model-type", "finetune", "--model-entry", "<entry>"], True, None, "run/models"),
    # Nothing names the raw data: no YAML (never a placeholder path), and the run still succeeds.
    ([], {"kind": "volume_zarr_v1"}, None, None),
], ids=["headless", "models dir", "data path from the volume", "a finetuned model", "no raw data"])
def test_a_run_exports_and_writes_the_yaml_that_serves_it(run_cli, tiny_script, tmp_path, flags, manifest,
                                                          attrs, yaml_dir):
    run = tmp_path / "run"
    (run / "full_finetune").mkdir(parents=True)
    (run / "full_finetune" / "model_state_dict.pt").write_bytes(b"old")  # from before per-iteration exports
    if attrs:
        (tmp_path / "session" / "corrections" / "vol.zarr").mkdir(parents=True)
        (tmp_path / "session" / "corrections" / "vol.zarr" / ".zattrs").write_text(json.dumps(attrs))
    base = {"type": "script", "script_path": str(tiny_script())}
    torch.save(ScriptModelConfig(script_path=base["script_path"]).config.model.state_dict(), tmp_path / "ft.pt")
    entry = json.dumps({"type": "finetune", "base_model": base, "weights_path": str(tmp_path / "ft.pt")})
    cli = run_cli(*[f.replace("<tmp>", str(tmp_path)).replace("<entry>", entry) for f in flags],
                  manifest=manifest, run_dir=run)

    assert cli.code == 0
    assert cli.markers[-1].startswith("TRAINING_ITERATION_COMPLETE: tiny_finetuned_")
    if yaml_dir is None:
        assert _yamls(cli) == []
    else:
        (path,) = _yamls(cli)
        assert Path(path).parent == tmp_path / yaml_dir
        served = yaml.safe_load(open(path))
        assert served["data_path"] == "/data/raw.zarr"  # what it trained on
        assert {k: served["models"][0]["base_model"][k] for k in base} == base  # the script, even under a finetune
    # The old export is moved aside, not deleted, and the old name follows the new one.
    assert (run / "full_finetune").is_symlink() and (run / "full_finetune" / "model_state_dict.pt").exists()
    assert b"old" in [p.read_bytes() for p in run.glob("iterations/*/full_finetune/model_state_dict.pt")]
    assert not (run / "tensorboard").exists()


def _no_port(*args):
    raise OSError("address in use")


NAN_PATCHES = DataLoader(TensorDataset(torch.full((2, 1, 4, 4, 4), float("nan")), torch.full((2, 1, 4, 4, 4), 2.0)))


@pytest.mark.parametrize("options, marker", [
    (dict(server=_no_port), "INFERENCE_SERVER_FAILED: address in use"),
    (dict(loaders=[NAN_PATCHES]), "TRAINING_DIVERGED"),
], ids=["the server cannot start", "the first iteration diverges"])
def test_a_served_run_that_cannot_serve_fails(run_cli, tmp_path, options, marker):
    """A server that would not start ended the job with 0, so it showed as
    COMPLETED; a first iteration that diverged waited, until walltime, for a
    restart that only the server it never started could have delivered."""
    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path), **options)
    assert (cli.code, cli.markers[-1], cli.waited) == (1, marker, 0)


@pytest.mark.parametrize("flags, trains", [
    # BCE compares shapes: a one-channel target against three channels died on the first batch.
    (["--output-type", "binary"], False),
    (["--output-type", "binary", "--loss-type", "dice"], True),  # dice broadcasts the target
    # --select-channel slices the prediction; these targets had every channel.
    (["--output-type", "distance", "--select-channel", "1"], True),
    (["--output-type", "binary_broadcast", "--select-channel", "0"], True),
    (["--select-channel", "3"], False),
    (["--output-type", "affinities", "--offsets", "[[1, 0, 0]]", "--select-channel", "0"], False),
])
def test_a_target_the_model_cannot_train_on_is_refused_before_training(run_cli, flags, trains):
    cli = run_cli(*flags, channels=3)
    assert cli.code == (0 if trains else 1)
    assert ("Starting epoch" in cli.out) == trains and len(cli.loaded) == 1


@pytest.mark.parametrize("rank, restart, export", [
    # The model decides what is exported: a restart asking for LoRA of a full
    # finetune had its next YAML point at a lora_adapter/ that was never written.
    pytest.param(0, {"lora_r": 8}, "full_finetune", id="full"),
    # alpha follows the rank, or raising it from 2 to 4 halved every update.
    pytest.param(2, {"lora_r": 4}, "lora_adapter", marks=pytest.mark.finetune, id="lora"),
])
def test_a_restart_applies_the_settings_it_may_change(run_cli, tmp_path, rank, restart, export):
    """A restart request may change training settings, typed as argparse would,
    and nothing else. One that cannot be set up (an emptied volume) is
    reported and waits for the next, the model served as trained meanwhile."""
    run = tmp_path / "session" / "runs" / "run"
    run.mkdir(parents=True)
    params = {"learning_rate": 1e-4, "augment": False, "num_epochs": 1, "loss_type": "bce", "corrections": "c",
              "balance_classes": False, "offsets": None}
    (run / "metadata.json").write_text(json.dumps({"params": params}))
    states = {}

    def emptied_volume(record):
        states["during the failed setup"] = cpu_state_copy(record.served[0])
        return ValueError("no populated chunks")

    first = {**restart, "augment": True, "learning_rate": "5e-4", "num_epochs": "many", "loss_type": "hinge",
             "corrections": "/elsewhere", "balance_classes": "true", "offsets": [[1, 0, 0]]}
    cli = run_cli(
        "--lora-r", str(rank), "--no-augment", "--auto-serve", "--serve-data-path", str(tmp_path),
        server=lambda args, config, model: states.setdefault("served", cpu_state_copy(model)),
        loaders=[None, emptied_volume], restarts=[{"params": first}, {"params": {"learning_rate": 1e-3}}],
    )

    assert cli.code == 1  # the last restart signal is malformed
    assert "RESTART_FAILED: no populated chunks" in cli.markers
    assert all(torch.equal(v, states["served"][k]) for k, v in states["during the failed setup"].items())
    assert [load["augment"] for load in cli.loaded] == [False, True, True]
    recorded = json.loads((run / "metadata.json").read_text())["params"]
    assert recorded == {**params, "learning_rate": 1e-3, "augment": True, "balance_classes": True,
                        "offsets": "[[1, 0, 0]]"}  # --offsets is JSON

    entry = yaml.safe_load(open(_yamls(cli)[-1]))["models"][0]
    exported = Path(entry.pop("lora_adapter_path" if rank else "weights_path"))
    assert "lora_adapter_path" not in entry and "weights_path" not in entry
    iteration = exported.parent if rank else exported.parent.parent
    assert iteration.parent == run / "iterations" and iteration.name.startswith("003_")
    if rank:
        config = json.loads((exported / "adapter_config.json").read_text())
        assert (config["r"], config["lora_alpha"]) == (4, 8)
    # What the job manager reads follows the link to this latest export.
    (latest,) = finetune_export_kwargs(run).values()
    assert os.path.realpath(latest) == os.path.realpath(exported)


@pytest.mark.skipif(importlib.util.find_spec("tensorboard") is None, reason="tensorboard is not installed")
def test_tensorboard_curves_run_on_across_a_restart(run_cli, tmp_path):
    """Each iteration's trainer started its steps at 0, on top of the last one's curves."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path), restarts=[{"params": {}}],
                  tensorboard=True)
    events = EventAccumulator(str(cli.run / "tensorboard"))
    events.Reload()
    assert [e.step for e in events.Scalars("train/loss")] == [1, 2]
    assert [e.step for e in events.Scalars("epoch/loss")] == [1, 2]
    tags = events.Tags()
    assert {"train/supervised", "train/lr", "time/step_s", "time/data_wait_s", "epoch/supervised",
            "epoch/best_supervised", "time/epoch_data_wait_s", "time/epoch_compute_s"} <= set(tags["scalars"])
    assert "patch/raw|target|prediction|mask" in tags["images"] and "config/text_summary" in tags["tensors"]


def test_the_cli_takes_every_model_type_the_job_manager_submits():
    """argparse knew fly, dacapo, huggingface and script, so a job for an exported
    (cellmap) or finetuned (finetune) model died on the GPU node with exit 2."""
    from cellmap_flow.finetune.finetune_cli import build_arg_parser
    from cellmap_flow.finetune.finetune_job_manager import TRAINABLE_MODEL_TYPES

    for model_type in TRAINABLE_MODEL_TYPES:
        args = ["--corrections", "/c", "--output-dir", "/o", "--model-type", model_type]
        assert build_arg_parser().parse_args(args).model_type == model_type
