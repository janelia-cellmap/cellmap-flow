"""Tests for finetuning job manager helpers and metadata."""

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cellmap_flow.finetune.finetune_job_manager import (
    FinetuneJob,
    FinetuneJobManager,
    JobStatus,
)


class DummyScriptModelConfig:
    cli_name = "script"

    def __init__(self):
        self.name = "dummy_script_model"
        self.script_path = "/tmp/dummy_model.py"
        self.channels = ["mito"]
        self.input_voxel_size = [8, 8, 8]
        self.output_voxel_size = [8, 8, 8]


class DummyThread:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False

    def start(self):
        self.started = True


class FinetuneJobManagerTests(unittest.TestCase):
    def test_submit_job_uses_console_script_and_preserves_scheduler_metadata(self):
        manager = FinetuneJobManager()
        model_config = DummyScriptModelConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            corrections_dir = Path(tmpdir) / "corrections"
            correction = corrections_dir / "crop_1.zarr"
            correction.mkdir(parents=True)
            (correction / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))

            fake_job = SimpleNamespace(process=SimpleNamespace(pid=1234))

            with patch(
                "cellmap_flow.finetune.finetune_job_manager.is_bsub_available",
                return_value=False,
            ), patch(
                "cellmap_flow.finetune.finetune_job_manager.run_locally",
                return_value=fake_job,
            ), patch(
                "cellmap_flow.finetune.finetune_job_manager.threading.Thread",
                DummyThread,
            ):
                job = manager.submit_finetuning_job(
                    model_config=model_config,
                    corrections_path=corrections_dir,
                    output_base=Path(tmpdir),
                    queue="gpu_a100",
                    charge_group="my_lab",
                )

            metadata = json.loads((job.output_dir / "metadata.json").read_text())
            command = metadata["command"]

            self.assertIn(sys.executable, command)
            self.assertIn("-m cellmap_flow.finetune.finetune_cli", command)
            self.assertNotIn(
                "stdbuf -oL python -m cellmap_flow.finetune.finetune_cli",
                command,
            )
            self.assertIn("--model-type script", command)
            self.assertIn("--model-script /tmp/dummy_model.py", command)
            self.assertEqual(metadata["queue"], "gpu_a100")
            self.assertEqual(metadata["charge_group"], "my_lab")

    def test_complete_job_uses_metadata_scheduler_settings_in_yaml(self):
        manager = FinetuneJobManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            output_dir = session_dir / "finetuning" / "runs" / "run_1"
            output_dir.mkdir(parents=True)

            adapter_dir = output_dir / "lora_adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_model.bin").write_bytes(b"adapter")
            (adapter_dir / "adapter_config.json").write_text("{}")

            corrections_dir = session_dir / "corrections"
            correction = corrections_dir / "crop_1.zarr"
            correction.mkdir(parents=True)
            (correction / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))

            metadata = {
                "corrections_path": str(corrections_dir),
                "model_type": "script",
                "model_script": "/tmp/dummy_model.py",
                "queue": "gpu_l40s",
                "charge_group": "cellmap-special",
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata))

            job = FinetuneJob(
                job_id="job-1",
                lsf_job=None,
                model_name="dummy_script_model",
                output_dir=output_dir,
                params={},
                status=JobStatus.COMPLETED,
                created_at=datetime(2025, 1, 2, 3, 4, 5),
                log_file=output_dir / "training_log.txt",
            )
            job.current_epoch = 3
            job.latest_loss = 0.25

            manager.complete_job(job)

            self.assertIsNotNone(job.model_yaml_path)
            yaml_text = Path(job.model_yaml_path).read_text()
            self.assertIn("queue: gpu_l40s", yaml_text)
            self.assertIn("charge_group: cellmap-special", yaml_text)
            self.assertIn("type: finetune", yaml_text)


if __name__ == "__main__":
    unittest.main()
