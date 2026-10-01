"""python -m cellmap_flow.finetune.finetune_cli, through main(): what a run
exports and announces, what it refuses, and what a restart may change.

The markers of a run with one restart, its iterations/NNN_<ts>/ layout and the
weights of each iteration are pinned by test_training_snapshot.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.adaptation import cpu_state_copy
from cellmap_flow.finetune.job_manager.persistence import finetune_export_kwargs
from cellmap_flow.models.models_config import ScriptModelConfig


def _yamls(cli):
    """The serving YAMLs the run announced, in order."""
    return [line.split(":", 1)[1].strip() for line in cli.markers if line.startswith("FINETUNED_MODEL_YAML:")]


def _served_entry(cli):
    """The model entry of the last YAML the run announced."""
    return yaml.safe_load(open(_yamls(cli)[-1]))["models"][0]


@pytest.mark.parametrize("flags, models_dir", [
    # They went to output_dir/../../.., which for this run is /: a permission
    # error, after training had succeeded, reported as a failure.
    pytest.param([], "run/models", id="a headless run keeps them in its own dir"),
    pytest.param(["--models-dir", "<tmp>/m"], "m", id="--models-dir wins"),
])
def test_where_the_serving_yaml_goes(run_cli, tmp_path, flags, models_dir):
    """A dashboard run's YAMLs go into its session's models/ (test_training_snapshot)."""
    cli = run_cli(*[f.replace("<tmp>", str(tmp_path)) for f in flags], run_dir=tmp_path / "run")
    assert cli.code == 0
    assert [Path(path).parent for path in _yamls(cli)] == [tmp_path / models_dir]


@pytest.mark.parametrize("manifest, attrs", [
    pytest.param(True, None, id="the manifest's raw data"),
    pytest.param({"kind": "volume_zarr_v1"}, {"dataset_path": "/data/raw.zarr"}, id="else the volume's own attrs"),
])
def test_the_yaml_serves_the_data_the_run_trained_on(run_cli, tmp_path, manifest, attrs):
    if attrs:
        (tmp_path / "session" / "corrections" / "vol.zarr").mkdir(parents=True)
        (tmp_path / "session" / "corrections" / "vol.zarr" / ".zattrs").write_text(json.dumps(attrs))
    cli = run_cli(manifest=manifest)
    assert yaml.safe_load(open(_yamls(cli)[-1]))["data_path"] == "/data/raw.zarr"


def test_a_run_whose_data_nothing_names_still_succeeds_without_a_yaml(run_cli):
    """The YAML fell back to a made-up "/path/to/data.zarr"; now there is none,
    and the weights, saved by then, are not a failed run."""
    cli = run_cli(manifest={"kind": "volume_zarr_v1"})
    assert cli.code == 0 and _yamls(cli) == []
    assert cli.markers[-1].startswith("TRAINING_ITERATION_COMPLETE: tiny_finetuned_")
    assert (cli.run / "full_finetune" / "model_state_dict.pt").exists()


def test_a_finetuned_model_trains_on_and_is_served_on_its_root_base(run_cli, tiny_script, tmp_path):
    """The job manager passes a finetuned model as --model-entry. Full weights
    replace every parameter, so they are served on the root base's module tree;
    on the finetune entry's they would meet a tree whose names do not match."""
    base = {"type": "script", "script_path": str(tiny_script())}
    torch.save(ScriptModelConfig(script_path=base["script_path"]).config.model.state_dict(), tmp_path / "ft.pt")
    entry = json.dumps({"type": "finetune", "base_model": base, "weights_path": str(tmp_path / "ft.pt")})
    cli = run_cli("--model-type", "finetune", "--model-entry", entry)
    assert cli.code == 0
    served_base = _served_entry(cli)["base_model"]
    assert {key: served_base[key] for key in base} == base


def test_an_export_from_before_per_iteration_exports_is_kept(run_cli, tmp_path):
    """A real full_finetune/ from an older run is moved into iterations/, not
    deleted, and the old name becomes a link to the newest export."""
    run = tmp_path / "run"
    (run / "full_finetune").mkdir(parents=True)
    (run / "full_finetune" / "model_state_dict.pt").write_bytes(b"old")
    cli = run_cli(run_dir=run)
    assert cli.code == 0
    assert (run / "full_finetune").is_symlink() and (run / "full_finetune" / "model_state_dict.pt").exists()
    assert b"old" in [p.read_bytes() for p in run.glob("iterations/*/full_finetune/model_state_dict.pt")]


def _no_port(*args):
    raise OSError("address in use")


NAN_PATCHES = DataLoader(TensorDataset(torch.full((2, 1, 4, 4, 4), float("nan")), torch.full((2, 1, 4, 4, 4), 2.0)))


@pytest.mark.parametrize("options, marker", [
    pytest.param(dict(server=_no_port), "INFERENCE_SERVER_FAILED: address in use", id="the server cannot start"),
    pytest.param(dict(loaders=[NAN_PATCHES]), "TRAINING_DIVERGED", id="the first iteration diverges"),
])
def test_a_served_run_that_cannot_serve_fails(run_cli, tmp_path, options, marker):
    """A server that would not start ended the job with 0, so it showed as
    COMPLETED; a first iteration that diverged waited, until walltime, for a
    restart that only the server it never started could have delivered."""
    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path), **options)
    assert (cli.code, cli.markers[-1], cli.waited) == (1, marker, 0)


@pytest.mark.parametrize("flags, trains", [
    # BCE compares shapes: a one-channel target against three channels died on the first batch.
    pytest.param(["--output-type", "binary"], False, id="binary with bce: refused"),
    pytest.param(["--output-type", "binary", "--loss-type", "dice"], True, id="binary with dice: broadcast"),
    # --select-channel slices the prediction to one channel; these targets had every channel.
    pytest.param(["--output-type", "distance", "--select-channel", "1"], True, id="distance, one channel"),
    pytest.param(["--output-type", "binary_broadcast", "--select-channel", "0"], True,
                 id="broadcast binary, one channel"),
    pytest.param(["--select-channel", "3"], False, id="a channel the model does not have: refused"),
    pytest.param(["--output-type", "affinities", "--offsets", "[[1, 0, 0]]", "--select-channel", "0"], False,
                 id="one channel of affinities: refused"),
])
def test_a_target_a_three_channel_model_cannot_train_on_is_refused_before_training(run_cli, flags, trains):
    cli = run_cli(*flags, channels=3)
    assert cli.code == (0 if trains else 1)
    assert ("Starting epoch" in cli.out) == trains and len(cli.loaded) == 1


def _metadata(run, **params):
    """A metadata.json with ``params``, as the job manager writes one."""
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({"params": params}))
    return run / "metadata.json"


MIN_MAX = {"min_value": 0.0, "max_value": 255.0, "invert": False}


@pytest.mark.parametrize("input_norm", [
    pytest.param([{"name": "MinMaxNormalizer", **MIN_MAX}, {"name": "LambdaNormalizer", "expression": "x*2-1"}],
                 id="the dashboard's step list"),
    pytest.param({"MinMaxNormalizer": MIN_MAX, "LambdaNormalizer": {"expression": "x*2-1"}},
                 id="build_corrections' older dict"),
])
def test_a_run_records_the_normalization_it_trained_on(run_cli, tmp_path, caplog, input_norm):
    """metadata.json, next to the weights, says which normalization the
    training data went through, as the manifest gave it. With the dashboard's
    step list the job's log said the snapshot had failed, after writing it."""
    metadata = _metadata(tmp_path / "session" / "runs" / "run", learning_rate=1e-4)
    cli = run_cli(manifest={"kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr", "input_norm": input_norm})
    assert cli.code == 0
    assert json.loads(metadata.read_text())["params"] == {"learning_rate": 1e-4, "input_norm": input_norm}
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name.endswith("session_loop")]
    assert [r.getMessage() for r in warnings] == []


def test_a_restart_changes_only_the_training_settings(run_cli, tmp_path):
    """Values are typed as argparse would type them; one that cannot be used is
    ignored, and so is anything that is not a training setting, such as the
    corrections path. The dashboard's "augment" sets the trainer's --no-augment
    (the toggle was dropped), and offsets sent as a list become --offsets' JSON."""
    params = dict(learning_rate=1e-4, augment=False, num_epochs=1, loss_type="bce", corrections="c",
                  balance_classes=False, offsets=None)
    metadata = _metadata(tmp_path / "session" / "runs" / "run", **params)
    restart = dict(learning_rate="5e-4", augment=True, num_epochs="many", loss_type="hinge",
                   corrections="/elsewhere", balance_classes="true", offsets=[[1, 0, 0]])
    cli = run_cli("--no-augment", "--auto-serve", "--serve-data-path", str(tmp_path), restarts=[{"params": restart}])

    assert [load["augment"] for load in cli.loaded] == [False, True]
    assert json.loads(metadata.read_text())["params"] == {
        **params, "learning_rate": 5e-4, "augment": True, "balance_classes": True, "offsets": "[[1, 0, 0]]",
    }


@pytest.mark.parametrize("restart", [
    pytest.param({"num_epochs": 0}, id="no epochs"),
    pytest.param({"batch_size": 0}, id="an empty batch"),
    pytest.param({"gradient_accumulation_steps": 0}, id="no accumulation steps"),
    pytest.param({"learning_rate": -1}, id="a negative learning rate"),
    pytest.param({"offsets": [[1, 0]]}, id="an offset of two values"),
])
def test_a_restart_setting_that_cannot_train_is_ignored(run_cli, tmp_path, restart):
    """These type-checked, and failed only once training ran, on a job that
    was serving: 0 epochs exported the previous iteration's best checkpoint,
    0 accumulation steps was reported as divergence, a negative learning rate
    as a failed restart, and two-value offsets ended the job."""
    launched = dict(num_epochs=1, batch_size=8, gradient_accumulation_steps=1, learning_rate=1e-4, offsets=None)
    metadata = _metadata(tmp_path / "session" / "runs" / "run", **launched)
    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path), restarts=[{"params": restart}])
    assert json.loads(metadata.read_text())["params"] == launched
    assert sum(line.startswith("TRAINING_ITERATION_COMPLETE:") for line in cli.markers) == 2


def test_a_restart_that_cannot_be_set_up_waits_for_the_next(run_cli, tmp_path):
    """Its data and target were built outside the CLI's try, so a bad restart
    (an emptied volume, say) ended the job and took the served model with it;
    and the model was reset before the restart was known to work, so the base
    went on being served. Now it is reported, the model stays as trained, and
    the next restart runs."""
    states = {}

    def emptied_volume(record):
        states["meanwhile"] = cpu_state_copy(record.served[0])
        return ValueError("no populated chunks")

    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path), loaders=[None, emptied_volume],
                  server=lambda args, config, model: states.setdefault("served", cpu_state_copy(model)),
                  restarts=[{"params": {}}, {"params": {}}])
    assert "RESTART_FAILED: no populated chunks" in cli.markers
    assert all(torch.equal(value, states["served"][key]) for key, value in states["meanwhile"].items())
    assert len([line for line in cli.markers if line.startswith("TRAINING_ITERATION_COMPLETE:")]) == 2


@pytest.mark.finetune
def test_a_restart_whose_trainer_cannot_be_built_leaves_the_reset_to_the_next(run_cli, tmp_path, monkeypatch, caplog):
    """The model is reset before its trainer is built, and the reset was marked
    done before that: the job said the previous model was still served while
    it served the starting weights, and the next restart, finding nothing to
    reset, kept the failed one's adapter and ignored its own rank."""
    from cellmap_flow.finetune import session_loop

    built = []

    def trainer(*args, **kwargs):
        built.append(kwargs)
        if len(built) == 2:
            raise RuntimeError("CUDA out of memory")
        return real_trainer(*args, **kwargs)

    real_trainer = session_loop.LoRAFinetuner
    monkeypatch.setattr(session_loop, "LoRAFinetuner", trainer)
    cli = run_cli("--lora-r", "2", "--auto-serve", "--serve-data-path", str(tmp_path),
                  restarts=[{"params": {"lora_r": 4}}, {"params": {"lora_r": 8}}])
    assert "RESTART_FAILED: CUDA out of memory" in cli.markers
    config = json.loads((Path(_served_entry(cli)["lora_adapter_path"]) / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"]) == (8, 16)
    assert "Serving the starting weights until a restart" in caplog.text


class _UnreadableChunks(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise OSError("a chunk could not be read")


def test_a_restart_whose_training_fails_waits_for_the_next(run_cli, tmp_path):
    """Only a restart's set-up was guarded: an error once training ran, such as
    a chunk that could not be read, ended a job that was serving."""
    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path),
                  loaders=[None, DataLoader(_UnreadableChunks(), batch_size=2)],
                  restarts=[{"params": {}}, {"params": {}}])
    assert "RESTART_FAILED: a chunk could not be read" in cli.markers
    assert sum(line.startswith("TRAINING_ITERATION_COMPLETE:") for line in cli.markers) == 2


def test_a_restart_cannot_turn_a_full_finetune_into_lora(run_cli, tmp_path):
    """The model decides what is exported. A restart asking a full finetune for
    LoRA had its next YAML point at a lora_adapter/ that was never written."""
    cli = run_cli("--auto-serve", "--serve-data-path", str(tmp_path), restarts=[{"params": {"lora_r": 8}}])
    entry = _served_entry(cli)
    assert "lora_adapter_path" not in entry
    exported = Path(entry["weights_path"])
    assert exported.parent.name == "full_finetune" and exported.parent.parent.name.startswith("002_")
    # And what the job manager reads, through the run's own full_finetune link, is this export.
    (latest,) = finetune_export_kwargs(cli.run).values()
    assert os.path.realpath(latest) == os.path.realpath(exported)


@pytest.mark.finetune
@pytest.mark.parametrize("restart, adapter", [
    pytest.param({"lora_r": 4}, (4, 8), id="alpha follows the rank"),
    pytest.param({"lora_r": 4, "lora_alpha": 4}, (4, 4), id="an alpha given with it is kept"),
    # Rank 0 asks for a full finetune, which a LoRA job cannot become: it
    # kept its rank but took the alpha of 2 x 0, a scaling of 0, and trained
    # nothing.
    pytest.param({"lora_r": 0}, (2, 4), id="rank 0 keeps the adapter as it was"),
])
def test_a_lora_restarts_alpha_keeps_the_adapters_scaling(run_cli, tmp_path, restart, adapter):
    """peft scales an adapter by lora_alpha / r, and a restart carried only the
    rank: going from r=8 to r=64 took the scaling from 2 to 0.25, and every
    update to an eighth of its size."""
    cli = run_cli("--lora-r", "2", "--auto-serve", "--serve-data-path", str(tmp_path),
                  restarts=[{"params": restart}])
    path = Path(_served_entry(cli)["lora_adapter_path"])
    assert path.parent.name.startswith("002_")
    config = json.loads((path / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"]) == adapter


@pytest.mark.parametrize("rank, restart, recorded", [
    # It kept its adapter (above), and metadata.json said lora_r 0, lora_alpha 0.
    pytest.param(2, {"lora_r": 0}, {"lora_r": 2, "lora_alpha": 4}, id="a LoRA job asked for rank 0",
                 marks=pytest.mark.finetune),
    pytest.param(0, {"lora_r": 8}, {"lora_r": 0, "lora_alpha": 0}, id="a full finetune asked for rank 8"),
])
def test_a_restart_that_cannot_change_the_rank_records_the_rank_kept(run_cli, tmp_path, rank, restart, recorded):
    """A dashboard started later reads a job's settings from its metadata.json."""
    metadata = _metadata(tmp_path / "session" / "runs" / "run", lora_r=rank, lora_alpha=2 * rank)
    run_cli("--lora-r", str(rank), "--auto-serve", "--serve-data-path", str(tmp_path), restarts=[{"params": restart}])
    assert json.loads(metadata.read_text())["params"] == recorded


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


def test_without_tensorboard_nothing_is_written(run_cli):
    cli = run_cli()  # --no-tensorboard
    assert cli.code == 0 and not (cli.run / "tensorboard").exists()


def test_the_cli_takes_every_model_type_the_job_manager_submits():
    """argparse knew fly, dacapo, huggingface and script, so a job for an exported
    (cellmap) or finetuned (finetune) model died on the GPU node with exit 2."""
    from cellmap_flow.finetune.cli import build_arg_parser
    from cellmap_flow.finetune.job_manager.submit import TRAINABLE_MODEL_TYPES

    for model_type in TRAINABLE_MODEL_TYPES:
        args = ["--corrections", "/c", "--output-dir", "/o", "--model-type", model_type]
        assert build_arg_parser().parse_args(args).model_type == model_type
