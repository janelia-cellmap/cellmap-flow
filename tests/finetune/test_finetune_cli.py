"""python -m cellmap_flow.finetune.finetune_cli, through main(): what a run
serves, what it refuses, what a restart may change, when Stop Early ends its
training, and how a run that cannot serve or restart fails. A run with one
restart between two iterations is test_training_snapshot's.
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


# A 2D model's script: one slice of 4 x 4 at 16 nm in and out.
SCRIPT_2D = """
from funlib.geometry import Coordinate
import torch.nn as nn

input_voxel_size = output_voxel_size = Coordinate(16, 16, 16)
read_shape = write_shape = Coordinate(1, 4, 4) * input_voxel_size
output_channels = 1
model = nn.Conv3d(1, 1, 1)
"""


@pytest.mark.parametrize("two_d, patch", [
    pytest.param(True, [178] * 3, id="a 3D model's volume for a 2D model: refused"),
    pytest.param(False, [4] * 3, id="the model's own grid: trained"),
])
def test_a_volume_on_another_models_grid_is_refused_before_a_patch_is_read(run_cli, tmp_path, caplog, two_d, patch):
    """setup_54_2d's job (1 x 556 x 556 at 16 nm) read the 178^3 patches at
    8 nm of a volume made for a 3D model, and torch.export refused them
    ("Expected input at *args[0].shape[2] to be equal to 1, but got 178"),
    after the FP16 probe had blamed the precision."""
    flags = []
    if two_d:
        (tmp_path / "model_2d.py").write_text(SCRIPT_2D)
        flags = ["--model-script", str(tmp_path / "model_2d.py")]
    grid = {"input_size_voxels": patch, "output_size_voxels": [4] * 3 if patch == [4] * 3 else [56] * 3,
            "input_voxel_size_nm": [8] * 3, "output_voxel_size_nm": [8] * 3}
    cli = run_cli(*flags, manifest={"kind": "volume_zarr_v1", "raw_dataset_path": "/data/raw.zarr", **grid})
    assert (cli.code, len(cli.loaded)) == ((1, 0) if two_d else (0, 1))
    if two_d:
        # Both grids named.
        assert "178×178×178" in caplog.text and "1×4×4" in caplog.text


def test_a_restart_trains_the_same_channel_and_holds_the_others_again(run_cli, tmp_path, monkeypatch):
    """A restart that does not name the channel keeps the one the job trains."""
    from cellmap_flow.finetune import session_loop

    trainers = []

    class Recorded(session_loop.Finetuner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)

    monkeypatch.setattr(session_loop, "Finetuner", Recorded)
    run_cli("--select-channel", "1", "--auto-serve", "--serve-data-path", str(tmp_path), channels=3,
            restarts=[{"params": {"learning_rate": 1e-3}}])
    assert [(t.select_channel, t.distillation.holds_others) for t in trainers] == [(1, True)] * 2


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


def _epochs(cli):
    """How many epochs each iteration of the run started."""
    return [part.count("Starting epoch") for part in cli.out.split("RESTARTING_TRAINING")]


def test_a_stop_asked_for_while_the_job_sets_up_is_kept(run_cli):
    """No server is up in the first iteration, so the stop comes as
    stop_signal.json. Training deleted it as stale when it began, and a stop
    asked for while the model loaded was lost: the run trained every epoch."""
    def stop_meanwhile(record):
        record.run.mkdir(parents=True, exist_ok=True)  # as the job manager made it, at submit
        (record.run / "stop_signal.json").write_text("{}")
        return run_cli.patches()

    cli = run_cli("--num-epochs", "3", loaders=[stop_meanwhile])
    assert cli.code == 0 and _epochs(cli) == [0]
    assert not (cli.run / "stop_signal.json").exists(), "the stop is used up"


class _AskedWhileFetched(TensorDataset):
    """``tensors``' patches; fetching one calls ``ask``."""

    def __init__(self, ask, *tensors):
        super().__init__(*tensors)
        self.ask = ask

    def __getitem__(self, index):
        self.ask()
        return super().__getitem__(index)


def test_a_stop_over_the_control_endpoint_ends_training_after_the_epoch_it_came_in(run_cli, tmp_path):
    """In the iterations after the first, the server's control endpoint hands
    the stop over (JobControl is what it calls). One that came before the
    restart, as the previous training ended on its own, was for that one."""
    controls = []

    def restart(signal_file, control):
        controls.append(control)
        control.request_stop({})
        control.request_restart({"params": {}})

    def stop_in_epoch_1(record):
        patches = run_cli.patches().dataset.tensors
        return DataLoader(_AskedWhileFetched(lambda: controls[0].request_stop({}), *patches), batch_size=2)

    cli = run_cli("--num-epochs", "3", "--auto-serve", "--serve-data-path", str(tmp_path),
                  restarts=[restart], loaders=[None, stop_in_epoch_1])
    assert _epochs(cli) == [3, 1]


def test_the_cli_takes_every_model_type_the_job_manager_submits():
    """argparse knew fly, dacapo, huggingface and script, so a job for an exported
    (cellmap) or finetuned (finetune) model died on the GPU node with exit 2."""
    from cellmap_flow.finetune.cli import build_arg_parser
    from cellmap_flow.finetune.job_manager.submit import trainable_model_types

    assert {"cellmap", "dacapo", "finetune", "fly", "huggingface", "script"} <= set(trainable_model_types())
    for model_type in trainable_model_types():
        args = ["--corrections", "/c", "--output-dir", "/o", "--model-type", model_type]
        assert build_arg_parser().parse_args(args).model_type == model_type
