"""Submitting a job does not build the model in the dashboard.

_get_model_metadata fell back to model_config.config for the voxel sizes and
channel names, and for a script, Hugging Face or DaCapo model that builds it
in the dashboard process: weights download, torch.export, a CUDA context.
It now asks resolve_model_geometry, which asks the model's running server
(or its cache) first.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from cellmap_flow.finetune import finetune_job_manager as fjm
from cellmap_flow.finetune.finetune_job_manager import FinetuneJobManager


class _Thread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


class _HubModel:
    """No geometry attributes of its own; building it is not allowed."""

    cli_name = "huggingface"
    name = "hub_model"
    repo = "org/hub_model"
    revision = None

    @property
    def config(self):
        raise AssertionError("submit built the model")


def test_submit_reads_geometry_without_building_the_model(tmp_path, monkeypatch):
    from cellmap_flow.utils import model_geometry

    asked = []

    def fake_resolve(name, model_config):
        asked.append(name)
        return SimpleNamespace(
            input_voxel_size=[8, 8, 8], output_voxel_size=[4, 4, 4], channels=["nuc"],
            read_shape=[64] * 3, write_shape=[32] * 3, output_channels=1,
        )

    monkeypatch.setattr(model_geometry, "resolve_model_geometry", fake_resolve)
    corrections = tmp_path / "corrections"
    (corrections / "vol.zarr").mkdir(parents=True)
    (corrections / "vol.zarr" / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))

    with patch.object(fjm, "is_bsub_available", return_value=False), \
         patch.object(fjm, "run_locally", return_value=SimpleNamespace(process=SimpleNamespace(pid=1))), \
         patch.object(fjm.threading, "Thread", _Thread):
        job = FinetuneJobManager().submit_finetuning_job(
            model_config=_HubModel(), corrections_path=corrections, output_base=tmp_path,
        )

    command = json.loads((job.output_dir / "metadata.json").read_text())["command"]
    assert "--input-voxel-size 8 8 8" in command
    assert "--output-voxel-size 4 4 4" in command
    assert "--channels nuc" in command
    assert asked == ["hub_model"], "looked up once, not once per field"
