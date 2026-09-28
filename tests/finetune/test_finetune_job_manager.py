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
            (corrections_dir / "_virtual_sources.json").write_text(json.dumps({"kind": "volume_zarr_v1"}))

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

    def test_submit_passes_the_scheduler_settings_for_the_serving_yaml(self):
        """The trainer writes the serving YAML, so it needs the job's queue."""
        manager = FinetuneJobManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            corrections_dir = Path(tmpdir) / "corrections"
            correction = corrections_dir / "crop_1.zarr"
            correction.mkdir(parents=True)
            (correction / ".zattrs").write_text(json.dumps({"dataset_path": "/data/raw.zarr"}))
            (corrections_dir / "_virtual_sources.json").write_text(json.dumps({"kind": "volume_zarr_v1"}))

            with patch(
                "cellmap_flow.finetune.finetune_job_manager.is_bsub_available",
                return_value=False,
            ), patch(
                "cellmap_flow.finetune.finetune_job_manager.run_locally",
                return_value=SimpleNamespace(process=SimpleNamespace(pid=1234)),
            ), patch(
                "cellmap_flow.finetune.finetune_job_manager.threading.Thread",
                DummyThread,
            ):
                job = manager.submit_finetuning_job(
                    model_config=DummyScriptModelConfig(),
                    corrections_path=corrections_dir,
                    output_base=Path(tmpdir),
                    queue="gpu_l40s",
                    charge_group="cellmap-special",
                )

            command = json.loads((job.output_dir / "metadata.json").read_text())["command"]
            self.assertIn("--queue gpu_l40s", command)
            self.assertIn("--charge-group cellmap-special", command)

    def test_complete_job_takes_the_name_and_yaml_the_trainer_reported(self):
        """It used to make up its own name (the job's creation time instead of
        the iteration's), never found that YAML, and generated a second one."""
        manager = FinetuneJobManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir)
            output_dir = session_dir / "runs" / "run_1"
            adapter_dir = output_dir / "iterations" / "002_20250102_050000" / "lora_adapter"
            adapter_dir.mkdir(parents=True)
            (adapter_dir / "adapter_model.bin").write_bytes(b"adapter")
            (adapter_dir / "adapter_config.json").write_text("{}")
            (output_dir / "lora_adapter").symlink_to(adapter_dir.relative_to(output_dir))
            yaml_1 = session_dir / "models" / "m_finetuned_20250102_040000.yaml"
            yaml_2 = session_dir / "models" / "m_finetuned_20250102_050000.yaml"
            (output_dir / "training_log.txt").write_text(
                f"FINETUNED_MODEL_YAML: {yaml_1}\n"
                "TRAINING_ITERATION_COMPLETE: m_finetuned_20250102_040000\n"
                "RESTARTING_TRAINING\n"
                f"FINETUNED_MODEL_YAML: {yaml_2}\n"
                "TRAINING_ITERATION_COMPLETE: m_finetuned_20250102_050000\n"
            )
            (output_dir / "metadata.json").write_text(json.dumps({"params": {}}))

            job = FinetuneJob(
                job_id="job-1",
                lsf_job=None,
                model_name="m",
                output_dir=output_dir,
                params={},
                status=JobStatus.COMPLETED,
                created_at=datetime(2025, 1, 2, 3, 4, 5),
                log_file=output_dir / "training_log.txt",
            )

            manager.complete_job(job)

            self.assertEqual(job.finetuned_model_name, "m_finetuned_20250102_050000")
            self.assertEqual(job.model_yaml_path, yaml_2)
            metadata = json.loads((output_dir / "metadata.json").read_text())
            self.assertEqual(metadata["finetuned_model_name"], "m_finetuned_20250102_050000")
            self.assertEqual(metadata["model_yaml_path"], str(yaml_2))
            self.assertFalse((session_dir / "models").exists(), "no YAML of its own")

    def test_an_iteration_without_a_yaml_is_not_given_the_previous_one(self):
        from cellmap_flow.finetune.finetune_job_manager import trainer_outputs_from_log

        log = (
            "FINETUNED_MODEL_YAML: /s/models/a.yaml\n"
            "TRAINING_ITERATION_COMPLETE: a\n"
            "TRAINING_ITERATION_COMPLETE: b\n"
        )
        self.assertEqual(trainer_outputs_from_log(log), ("b", None))
        self.assertEqual(trainer_outputs_from_log("nothing yet"), (None, None))


if __name__ == "__main__":
    unittest.main()
