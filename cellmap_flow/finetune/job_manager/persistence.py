"""A run's directory, as the job manager writes it and reads it back.

A job runs in ``<session>/runs/<model>_<YYYYmmdd_HHMMSS>/``. Beside what the
trainer writes there (its log, iterations and exports; finetune.run_outputs),
the manager keeps its record of the job in ``metadata.json``:
- written at submit (``submission_metadata``, ``write_metadata``), then
  given the job's ``lsf_job_id`` and ``status``;
- updated as the job's status changes, with its ``inference_server_url``,
  and when it ends, with its ``finetuned_model_name`` and
  ``model_yaml_path`` (``update_metadata``);
- completed when it succeeds (``record_completion``).
The trainer writes a restart's settings into its ``params``.

A dashboard started later, perhaps newer than the one that submitted the
jobs, finds a session's jobs again from these records (``rehydrate``), so
they are a format. It reads ``job_id``, ``lsf_job_id``, ``status``,
``model_name``, ``params``, ``created_at`` and ``corrections_path``, and
records a job LSF says has ended as final, with a ``status_detail`` when
LSF no longer knows the job.

``finetune_export_kwargs`` and ``check_export`` say which export a finished
run left: a LoRA run's ``lora_adapter/``, a full finetune's
``full_finetune/model_state_dict.pt``.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES, FinetuneJob, JobStatus
from cellmap_flow.finetune.job_manager.submit import MODEL_ENTRY_TYPES
from cellmap_flow.finetune.job_manager.tailer import finished_iterations
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.lsf import LSFJob
from cellmap_flow.jobs.spec import JobStatus as LSFJobStatus

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"


def submission_metadata(
    *,
    model_config,
    model_type: str,
    checkpoint_path: Optional[Path],
    corrections_path: Path,
    num_corrections: int,
    output_dir: Path,
    lora_r: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_type: str,
    label_smoothing: float,
    distillation_lambda: Optional[float],
    distillation_scope: str,
    margin: float,
    balance_classes: bool,
    augment: bool,
    channels: List[str],
    input_voxel_size: List[int],
    output_voxel_size: List[int],
    output_type: str,
    queue: str,
    charge_group: str,
    command: str,
) -> dict:
    """Build metadata persisted for a submitted finetuning job."""
    return {
        "job_id": str(uuid.uuid4()),
        "model_name": model_config.name,
        "model_type": model_type,
        "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "model_script": str(model_config.script_path) if hasattr(model_config, "script_path") else None,
        "repo": model_config.repo if model_type == "huggingface" else None,
        "revision": getattr(model_config, "revision", None) if model_type == "huggingface" else None,
        "model_entry": model_config.to_dict() if model_type in MODEL_ENTRY_TYPES else None,
        "corrections_path": str(corrections_path),
        "num_corrections": num_corrections,
        "output_dir": str(output_dir),
        "params": {
            "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_type": loss_type,
            "label_smoothing": label_smoothing,
            "distillation_lambda": distillation_lambda,
            "distillation_scope": distillation_scope,
            "margin": margin,
            "balance_classes": balance_classes,
            "augment": augment,
            "channels": channels,
            "input_voxel_size": input_voxel_size,
            "output_voxel_size": output_voxel_size,
            "output_type": output_type,
            "queue": queue,
            "charge_group": charge_group,
        },
        "queue": queue,
        "charge_group": charge_group,
        "created_at": datetime.now().isoformat(),
        "command": command,
    }


def write_metadata(output_dir, metadata: dict) -> None:
    """Write a new run's metadata.json."""
    metadata_file = Path(output_dir) / METADATA_FILE
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")


def update_metadata(output_dir, **fields) -> None:
    """Merge ``fields`` into the run's metadata.json, replacing it atomically."""
    path = Path(output_dir) / METADATA_FILE
    try:
        metadata = json.loads(path.read_text()) if path.exists() else {}
        metadata.update(fields)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(metadata, indent=2))
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"Could not update {path}: {e}")


def record_completion(finetune_job: FinetuneJob) -> None:
    """Record in metadata.json that the job completed, with its model and last epoch."""
    metadata_file = finetune_job.output_dir / METADATA_FILE
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["completed_at"] = datetime.now().isoformat()
        metadata["status"] = "COMPLETED"
        metadata["finetuned_model_name"] = finetune_job.finetuned_model_name
        yaml_path = finetune_job.model_yaml_path
        metadata["model_yaml_path"] = str(yaml_path) if yaml_path else None
        metadata["final_epoch"] = finetune_job.current_epoch
        metadata["final_loss"] = finetune_job.latest_loss

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Updated metadata file: {metadata_file}")


def rehydrate(session_path, known) -> List[FinetuneJob]:
    """The jobs of a session still alive on the cluster, rebuilt from their records.

    A run is rebuilt when its recorded status is not final, its job id is
    not in ``known`` (the jobs already followed), and bjobs still reports
    it as pending or running; its status is then LSF's, and what only its
    log says (its server, its model, its epoch) is left for the monitor to
    read. Local runs (a PID, not an LSF job) are not reattached.

    All the session's candidates are asked about in one bjobs call, and
    whatever bjobs says has ended -- including a job it no longer knows
    at all -- is recorded as final, so it is not asked about again. This
    runs on every load of the finetune tab.
    """
    candidates = []
    for metadata_file in sorted(Path(session_path).glob(f"runs/*/{METADATA_FILE}")):
        try:
            metadata = json.loads(metadata_file.read_text())
        except (OSError, ValueError):
            continue
        job_id = metadata.get("job_id")
        lsf_job_id = metadata.get("lsf_job_id")
        if (
            not job_id
            or job_id in known
            or not lsf_job_id
            or str(lsf_job_id).startswith("PID:")
            or metadata.get("status") in {s.value for s in TERMINAL_STATUSES}
        ):
            continue
        candidates.append((metadata_file, metadata, job_id, str(lsf_job_id)))
    if not candidates:
        return []

    reported = jobs_lsf.statuses([lsf_job_id for *_, lsf_job_id in candidates])
    jobs = []
    for metadata_file, metadata, job_id, lsf_job_id in candidates:
        if lsf_job_id not in reported:
            continue  # bjobs cannot say; try again next time
        observed = reported[lsf_job_id]
        output_dir = metadata_file.parent
        if observed is None:
            # LSF has forgotten it: it ended long enough ago to be purged,
            # while no dashboard was watching. Asked about again, it never
            # answers. The trainer never prints "done" -- after an
            # iteration it waits for restarts until it is stopped or runs
            # out of walltime -- so a run whose log shows a finished
            # iteration delivered a model, and that is what it is recorded
            # as having done.
            iterations, last = finished_iterations(output_dir / "training_log.txt")
            if iterations:
                status = JobStatus.COMPLETED
                detail = (
                    f"LSF no longer knows job {lsf_job_id}; it had finished "
                    f"{iterations} iteration(s), the last {last}"
                )
            else:
                status = JobStatus.FAILED
                detail = (
                    f"LSF no longer knows job {lsf_job_id}; it ended while no "
                    "dashboard was watching, and how is not known"
                )
            update_metadata(output_dir, status=status.value, status_detail=detail)
            continue
        if observed == LSFJobStatus.COMPLETED:
            # Finished while no dashboard was watching; say so, so it is
            # not asked about again. complete_job does not run for it.
            update_metadata(output_dir, status=JobStatus.COMPLETED.value)
            continue
        if observed == LSFJobStatus.FAILED:
            update_metadata(output_dir, status=JobStatus.FAILED.value)
            continue
        if observed not in (LSFJobStatus.RUNNING, LSFJobStatus.PENDING):
            continue
        lsf_job = LSFJob(job_id=lsf_job_id, model_name=metadata.get("model_name"))
        params = metadata.get("params") or {}
        try:
            created_at = datetime.fromisoformat(metadata["created_at"])
        except (KeyError, TypeError, ValueError):
            created_at = datetime.now()
        jobs.append(FinetuneJob(
            job_id=job_id,
            lsf_job=lsf_job,
            model_name=metadata.get("model_name") or "",
            output_dir=output_dir,
            params=params,
            status=JobStatus.RUNNING if observed == LSFJobStatus.RUNNING else JobStatus.PENDING,
            created_at=created_at,
            log_file=output_dir / "training_log.txt",
            total_epochs=int(params.get("num_epochs") or 10),
            corrections_path=(
                Path(metadata["corrections_path"]) if metadata.get("corrections_path") else None
            ),
        ))
    return jobs


def finetune_export_kwargs(output_dir, params=None) -> dict:
    """Which artifact a finished run produced, as FinetuneModelConfig kwargs.

    A LoRA run exports lora_adapter/; a full finetune (--lora-r 0) exports
    full_finetune/model_state_dict.pt. Decided by what is on disk first --
    the job's own record of lora_r is the fallback for a run that has not
    written its export yet -- so the dashboard never points a viewer at an
    adapter directory that a rank-0 run never made.
    """
    output_dir = Path(output_dir)
    weights = output_dir / "full_finetune" / "model_state_dict.pt"
    adapter = output_dir / "lora_adapter"
    if weights.exists():
        return {"weights_path": str(weights)}
    if adapter.exists():
        return {"lora_adapter_path": str(adapter)}
    if params and int(params.get("lora_r", 8) or 0) <= 0:
        return {"weights_path": str(weights)}
    return {"lora_adapter_path": str(adapter)}


def check_export(output_dir, params) -> None:
    """Raise RuntimeError unless the run's export (finetune_export_kwargs) is all on disk."""
    export = finetune_export_kwargs(output_dir, params)
    if "weights_path" in export:
        # Full finetune (--lora-r 0): a single state dict, no adapter dir.
        weights_file = Path(export["weights_path"])
        if not weights_file.exists():
            raise RuntimeError(
                f"Training completed but full-finetune weights not found: {weights_file}"
            )
        logger.info(f"Verified full-finetune weights exist: {weights_file}")
        return

    adapter_path = Path(export["lora_adapter_path"])

    # Check for adapter model (supports both .bin and .safetensors formats)
    adapter_model_bin = adapter_path / "adapter_model.bin"
    adapter_model_safetensors = adapter_path / "adapter_model.safetensors"

    if not (adapter_model_bin.exists() or adapter_model_safetensors.exists()):
        raise RuntimeError(
            f"Training completed but adapter model not found. "
            f"Checked: {adapter_model_bin} and {adapter_model_safetensors}"
        )

    adapter_config_file = adapter_path / "adapter_config.json"
    if not adapter_config_file.exists():
        raise RuntimeError(
            f"Training completed but adapter config not found: {adapter_config_file}"
        )

    logger.info(f"Verified LoRA adapter files exist in {adapter_path}")
