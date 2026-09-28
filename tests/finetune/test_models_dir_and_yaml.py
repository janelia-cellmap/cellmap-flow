"""Where the serving YAML goes, what data it names, and that it cannot fail training.

The models directory was output_dir.parent.parent.parent: for a dashboard
run (<session>/runs/<name>) that is <base>/models, outside the session and
shared by all of them, and for a headless --output-dir /nrs/x/run it is
/models -- a permission error after training had succeeded, which landed in
"Training failed", exit 1, and no completion marker. The data path fell back
to a "/path/to/data.zarr" placeholder the template's check did not know.
"""

import argparse
import json
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune import finetune_cli
from cellmap_flow.finetune.finetune_cli import _generate_model_files, _models_dir
from cellmap_flow.finetune.finetuned_model_templates import generate_finetuned_model_yaml


def _args(output_dir, **kw):
    base = dict(output_dir=str(output_dir), models_dir=None, lora_r=8,
                corrections="/nonexistent", serve_data_path=None, auto_serve=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_a_dashboard_run_writes_into_its_sessions_models_dir(tmp_path):
    run = tmp_path / "base" / "20260101_000000" / "runs" / "mito_20260101_000100"
    assert _models_dir(_args(run)) == tmp_path / "base" / "20260101_000000" / "models"


def test_a_headless_run_keeps_its_yamls_in_its_own_dir(tmp_path):
    assert _models_dir(_args(tmp_path / "run")) == tmp_path / "run" / "models"


def test_models_dir_flag_wins(tmp_path):
    assert _models_dir(_args(tmp_path / "run", models_dir=str(tmp_path / "m"))) == tmp_path / "m"


def _corrections(tmp_path, manifest_raw=None, zattrs=None):
    corrections = tmp_path / "corrections"
    (corrections / "a_crop.zarr").mkdir(parents=True)
    (corrections / "a_crop.zarr" / ".zattrs").write_text(json.dumps(zattrs or {}))
    if manifest_raw:
        (corrections / "_virtual_sources.json").write_text(
            json.dumps({"kind": "volume_zarr_v1", "raw_dataset_path": manifest_raw})
        )
    return corrections


def test_the_yaml_serves_the_data_the_manifest_trained_on(tmp_path):
    corrections = _corrections(tmp_path, manifest_raw="/data/raw.zarr")
    args = _args(tmp_path / "run", corrections=str(corrections))
    model_config = SimpleNamespace(name="m", to_dict=lambda: {"type": "fly"})
    _, path = _generate_model_files(args, model_config, "20260101_000000", is_lora=True)
    assert yaml.safe_load(open(path))["data_path"] == "/data/raw.zarr"


def test_no_data_path_is_an_error_not_a_placeholder(tmp_path):
    args = _args(tmp_path / "run", corrections=str(_corrections(tmp_path)))
    model_config = SimpleNamespace(name="m", to_dict=lambda: {"type": "fly"})
    with pytest.raises(ValueError, match="placeholder"):
        _generate_model_files(args, model_config, "20260101_000000", is_lora=True)


@pytest.mark.parametrize("placeholder", ["/path/to/data.zarr", "/path/to/your/data.zarr"])
def test_the_template_refuses_every_placeholder(tmp_path, placeholder):
    with pytest.raises(ValueError):
        generate_finetuned_model_yaml(
            lora_adapter_path="/a", base_model_dict={"type": "fly"}, model_name="m",
            output_path=tmp_path / "m.yaml", data_path=placeholder,
        )


SCRIPT = """
from funlib.geometry import Coordinate
import torch.nn as nn

input_voxel_size = Coordinate(8, 8, 8)
output_voxel_size = Coordinate(8, 8, 8)
read_shape = Coordinate(4, 4, 4) * input_voxel_size
write_shape = Coordinate(4, 4, 4) * output_voxel_size
output_channels = 1
model = nn.Conv3d(1, 1, 1)
"""


def _run_main(tmp_path, monkeypatch, corrections, output_dir):
    script = tmp_path / "model.py"
    script.write_text(SCRIPT)
    loader = DataLoader(
        TensorDataset(torch.rand(2, 1, 4, 4, 4), torch.full((2, 1, 4, 4, 4), 2.0)), batch_size=2
    )
    monkeypatch.setattr(finetune_cli, "create_dataloader", lambda *a, **k: loader)
    monkeypatch.setattr(sys, "argv", [
        "finetune_cli", "--model-type", "script", "--model-script", str(script),
        "--model-name", "tiny", "--corrections", str(corrections),
        "--output-dir", str(output_dir), "--lora-r", "0", "--num-epochs", "1",
        "--loss-type", "bce", "--no-mixed-precision", "--no-tensorboard", "--num-workers", "0",
    ])
    return finetune_cli.main()


def test_a_yaml_that_cannot_be_written_does_not_fail_the_run(tmp_path, monkeypatch, capsys):
    corrections = _corrections(tmp_path)  # nothing names the raw data
    assert _run_main(tmp_path, monkeypatch, corrections, tmp_path / "run") == 0
    out = capsys.readouterr().out
    assert "TRAINING_ITERATION_COMPLETE: tiny_finetuned_" in out
    assert "FINETUNED_MODEL_YAML:" not in out
    assert (tmp_path / "run" / "full_finetune" / "model_state_dict.pt").exists()


def test_the_yaml_path_is_announced(tmp_path, monkeypatch, capsys):
    corrections = _corrections(tmp_path, manifest_raw="/data/raw.zarr")
    assert _run_main(tmp_path, monkeypatch, corrections, tmp_path / "run") == 0
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("FINETUNED_MODEL_YAML:"))
    path = line.split(":", 1)[1].strip()
    assert path.startswith(str(tmp_path / "run" / "models"))
    assert yaml.safe_load(open(path))["models"][0]["name"].startswith("tiny_finetuned_")


def test_the_job_manager_points_the_trainer_at_the_sessions_models_dir(tmp_path):
    from pathlib import Path
    from unittest.mock import patch

    from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager

    class _Script:
        cli_name = "script"
        name = "m"
        script_path = "/s.py"

    corrections = _corrections(tmp_path, manifest_raw="/data/raw.zarr")

    class _Thread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    with patch("cellmap_flow.finetune.finetune_job_manager.is_bsub_available", return_value=False), \
         patch("cellmap_flow.finetune.finetune_job_manager.run_locally",
               return_value=SimpleNamespace(process=SimpleNamespace(pid=1))), \
         patch("cellmap_flow.finetune.finetune_job_manager.threading.Thread", _Thread):
        job = FinetuneJobManager().submit_finetuning_job(
            model_config=_Script(), corrections_path=corrections, output_base=tmp_path / "session",
        )
    command = json.loads((Path(job.output_dir) / "metadata.json").read_text())["command"].split()
    assert command[command.index("--models-dir") + 1] == str(tmp_path / "session" / "models")
