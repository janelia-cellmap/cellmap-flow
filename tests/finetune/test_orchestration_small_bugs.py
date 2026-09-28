"""Smaller orchestration bugs, each of which lost something quietly.

- Resume took whichever zarr os.listdir() gave first, crop zarrs included.
- The monitor parsed partial lines, so a loss or marker split across two reads
  was lost.
- A server that would not start ended the job with status 0: COMPLETED.
- Submit counted *.zarr directories instead of checking for the manifest the
  trainer needs, so a crop-only session queued a GPU job that then failed.
- An emptied number field reached the trainer as "--num-epochs None".
"""

import json
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJob, FinetuneJobManager, JobStatus
from cellmap_flow.utils.bsub_utils import JobStatus as LSFJobStatus


def _zarr(path, **attrs):
    path.mkdir(parents=True)
    (path / ".zattrs").write_text(json.dumps(attrs))


def test_resume_picks_the_volume_the_session_trains_on(tmp_path):
    from cellmap_flow.dashboard.routes.finetune.annotation_sessions import _annotation_volume_dirs

    corrections = tmp_path / "corrections"
    _zarr(corrections / "aaa_crop.zarr", dataset_path="/raw")          # a crop, listed first
    _zarr(corrections / "vol-old.zarr", type="annotation_volume")
    time.sleep(0.01)
    _zarr(corrections / "vol-new.zarr", type="annotation_volume")
    _zarr(corrections / "vol-x_chunk_1.zarr", type="annotation_volume")  # legacy extract
    (corrections / "_virtual_sources.json").write_text(
        json.dumps({"volume_zarr_path": str(corrections / "vol-old.zarr")})
    )

    assert _annotation_volume_dirs(str(corrections)) == ["vol-old.zarr", "vol-new.zarr"]
    (corrections / "_virtual_sources.json").unlink()
    assert _annotation_volume_dirs(str(corrections))[0] == "vol-new.zarr"


def test_a_line_split_across_two_reads_is_still_parsed(tmp_path, monkeypatch):
    out = tmp_path / "runs" / "r"
    out.mkdir(parents=True)
    log = out / "training_log.txt"
    log.write_text("Starting epoch 3 of 10...\nEpoch 3/10 - Lo")
    rest = iter(["ss: 0.25 - Supervised: 0.25\n"])

    def sleep(seconds):
        try:
            with open(log, "a") as f:
                f.write(next(rest))
        except StopIteration:
            pass

    replies = iter([LSFJobStatus.RUNNING, LSFJobStatus.RUNNING, LSFJobStatus.FAILED])
    job = FinetuneJob(
        job_id="j", lsf_job=SimpleNamespace(job_id="1", get_status=lambda: next(replies)),
        model_name="m", output_dir=out, params={}, status=JobStatus.RUNNING,
        created_at=datetime.now(), log_file=log,
    )
    monkeypatch.setattr(fjm.time, "sleep", sleep)
    FinetuneJobManager().monitor_job(job)
    assert job.latest_loss == pytest.approx(0.25)
    assert job.current_epoch == 3


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


def test_a_server_that_cannot_start_fails_the_job(tmp_path, monkeypatch, capsys):
    from cellmap_flow.finetune import finetune_cli

    script = tmp_path / "model.py"
    script.write_text(SCRIPT)
    loader = DataLoader(TensorDataset(torch.rand(2, 1, 4, 4, 4), torch.full((2, 1, 4, 4, 4), 2.0)), batch_size=2)
    monkeypatch.setattr(finetune_cli, "create_dataloader", lambda *a, **k: loader)

    def no_server(*a, **k):
        raise OSError("address in use")

    monkeypatch.setattr(finetune_cli, "_start_inference_server_background", no_server)
    monkeypatch.setattr(sys, "argv", [
        "finetune_cli", "--model-type", "script", "--model-script", str(script),
        "--corrections", str(tmp_path), "--output-dir", str(tmp_path / "run"), "--lora-r", "0",
        "--num-epochs", "1", "--loss-type", "bce", "--no-mixed-precision", "--no-tensorboard",
        "--num-workers", "0", "--auto-serve", "--serve-data-path", str(tmp_path),
    ])
    assert finetune_cli.main() == 1
    assert "INFERENCE_SERVER_FAILED: address in use" in capsys.readouterr().out


def test_submit_refuses_a_session_without_a_manifest(tmp_path):
    corrections = tmp_path / "corrections"
    _zarr(corrections / "crop.zarr", dataset_path="/raw")

    class _Script:
        cli_name = "script"
        name = "m"
        script_path = "/s.py"

    with patch.object(fjm, "is_bsub_available", return_value=False), \
         patch.object(fjm, "run_locally") as local:
        with pytest.raises(ValueError, match="_virtual_sources.json"):
            FinetuneJobManager().submit_finetuning_job(
                model_config=_Script(), corrections_path=corrections, output_base=tmp_path,
            )
    local.assert_not_called()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("value, expected", [("", 10), (None, 10), ("25", 25), (40, 40)])
def test_an_empty_number_field_gets_its_default(value, expected):
    from cellmap_flow.dashboard.routes.finetune.training import _number

    assert _number({"num_epochs": value}, "num_epochs", 10, int) == expected


@pytest.mark.parametrize("value", ["ten", "2.5"])
def test_a_number_field_that_is_not_one_is_refused(value):
    from cellmap_flow.dashboard.routes.finetune.training import _number

    with pytest.raises(ValueError, match="num_epochs"):
        _number({"num_epochs": value}, "num_epochs", 10, int)
