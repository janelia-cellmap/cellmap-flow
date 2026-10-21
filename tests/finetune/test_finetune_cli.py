"""python -m cellmap_flow.finetune.finetune_cli, through main(): what a run
serves, what it refuses, what a restart may change, and how a run that
cannot serve or restart fails. A run with one restart between two
iterations is test_training_snapshot's.
"""

import json

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune.adaptation import cpu_state_copy


def _yamls(cli):
    """The serving YAMLs the run announced, in order."""
    return [line.split(":", 1)[1].strip() for line in cli.markers if line.startswith("FINETUNED_MODEL_YAML:")]


def _served_entry(cli):
    """The model entry of the last YAML the run announced."""
    return yaml.safe_load(open(_yamls(cli)[-1]))["models"][0]


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
    # At the level it trained on: the server picks it by the model's voxel
    # size, as the trainer did. The YAML said "scale: s0", whatever that was.
    assert "scale" not in _served_entry(cli)


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
    pytest.param(["--select-channel", "3"], False, id="a channel the model does not have: refused"),
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


def test_the_cli_takes_every_model_type_the_job_manager_submits():
    """argparse knew fly, dacapo, huggingface and script, so a job for an exported
    (cellmap) or finetuned (finetune) model died on the GPU node with exit 2."""
    from cellmap_flow.finetune.cli import build_arg_parser
    from cellmap_flow.finetune.job_manager.submit import trainable_model_types

    assert {"cellmap", "dacapo", "finetune", "fly", "huggingface", "script"} <= set(trainable_model_types())
    for model_type in trainable_model_types():
        args = ["--corrections", "/c", "--output-dir", "/o", "--model-type", model_type]
        assert build_arg_parser().parse_args(args).model_type == model_type
