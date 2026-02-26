"""
Job manager for orchestrating finetuning jobs on LSF cluster.

This module provides:
- FinetuneJob: Track metadata and status of a single finetuning job
- FinetuneJobManager: Orchestrate job lifecycle from submission to completion
"""

import json
import logging
import re
import threading
import time
import uuid
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

from cellmap_flow.utils.bsub_utils import (
    submit_bsub_job,
    run_locally,
    is_bsub_available,
    LSFJob,
    JobStatus as LSFJobStatus
)

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a finetuning job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class FinetuneJob:
    """Track a finetuning job with metadata, status, and training progress.

    Manages lifecycle from submission through completion, including inference
    server state and restart chain linkage.
    """
    job_id: str
    lsf_job: Optional[LSFJob]
    model_name: str
    output_dir: Path
    params: Dict[str, Any]
    status: JobStatus
    created_at: datetime
    log_file: Path
    finetuned_model_name: Optional[str] = None
    model_script_path: Optional[Path] = None
    model_yaml_path: Optional[Path] = None
    current_epoch: int = 0
    total_epochs: int = 10
    latest_loss: Optional[float] = None
    inference_server_url: Optional[str] = None
    inference_server_ready: bool = False
    previous_job_id: Optional[str] = None
    next_job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Get LSF job ID or local PID
        lsf_job_id = None
        if self.lsf_job:
            if hasattr(self.lsf_job, 'job_id'):
                lsf_job_id = self.lsf_job.job_id
            elif hasattr(self.lsf_job, 'process'):
                lsf_job_id = f"PID:{self.lsf_job.process.pid}"

        return {
            "job_id": self.job_id,
            "lsf_job_id": lsf_job_id,
            "model_name": self.model_name,
            "output_dir": str(self.output_dir),
            "params": self.params,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "log_file": str(self.log_file),
            "finetuned_model_name": self.finetuned_model_name,
            "model_script_path": str(self.model_script_path) if self.model_script_path else None,
            "model_yaml_path": str(self.model_yaml_path) if self.model_yaml_path else None,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "latest_loss": self.latest_loss,
            "inference_server_url": self.inference_server_url,
            "inference_server_ready": self.inference_server_ready,
            "previous_job_id": self.previous_job_id,
            "next_job_id": self.next_job_id,
        }


class FinetuneJobManager:
    """
    Orchestrate finetuning jobs from submission to completion.

    Manages the full lifecycle:
    1. Validation and job submission to LSF
    2. Background monitoring of training progress
    3. Post-training model registration
    4. Job cancellation and cleanup
    """

    def __init__(self):
        """Initialize the job manager."""
        self.jobs: Dict[str, FinetuneJob] = {}
        self.logger = logging.getLogger(__name__)
        self._monitor_threads: Dict[str, threading.Thread] = {}

    def _get_model_metadata(self, model_config, attr_name: str, default=None):
        """
        Get metadata from model config, checking both direct attributes and loaded config.

        Args:
            model_config: The model configuration object
            attr_name: Name of the attribute to retrieve
            default: Default value if attribute not found

        Returns:
            The attribute value if found, otherwise the default value
        """
        # First try direct attribute access
        if hasattr(model_config, attr_name):
            value = getattr(model_config, attr_name, None)
            if value is not None:
                return value

        # Then try loading config and checking there
        try:
            config = model_config.config
            if hasattr(config, attr_name):
                value = getattr(config, attr_name, None)
                if value is not None:
                    return value
        except Exception as e:
            self.logger.debug(f"Could not load config to check for {attr_name}: {e}")

        return default

    def _extract_data_path_from_corrections(self, corrections_path: Path) -> str:
        """Extract dataset path from corrections metadata."""
        # Look for first .zarr directory
        zarr_dirs = list(corrections_path.glob("*.zarr"))
        if not zarr_dirs:
            raise ValueError("No .zarr directories found in corrections")

        # Read .zattrs
        zattrs_file = zarr_dirs[0] / ".zattrs"
        if not zattrs_file.exists():
            raise ValueError("No .zattrs metadata found in corrections")

        with open(zattrs_file) as f:
            metadata = json.load(f)

        if "dataset_path" not in metadata:
            raise ValueError("No 'dataset_path' found in corrections metadata")

        return metadata["dataset_path"]

    def submit_finetuning_job(
        self,
        model_config,
        corrections_path: Path,
        lora_r: int = 8,
        num_epochs: int = 10,
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        output_base: Optional[Path] = None,
        queue: str = "gpu_h100",
        charge_group: str = "cellmap",
        checkpoint_path_override: Optional[Path] = None,
        auto_serve: bool = True,
        mask_unannotated: bool = False,
        loss_type: str = "combined",
        label_smoothing: float = 0.0,
        distillation_lambda: float = 0.0,
        distillation_scope: str = "unlabeled",
        margin: float = 0.3,
        balance_classes: bool = False,
        output_type: str = "binary",
        select_channel: Optional[int] = None,
        offsets: Optional[str] = None,
    ) -> FinetuneJob:
        """
        Submit finetuning job to LSF cluster.

        Args:
            model_config: Model configuration object (FlyModelConfig, etc.)
            corrections_path: Path to corrections.zarr directory
            lora_r: LoRA rank (default: 8)
            num_epochs: Number of training epochs (default: 10)
            batch_size: Training batch size (default: 2)
            learning_rate: Learning rate (default: 1e-4)
            output_base: Base directory for outputs (default: output/finetuning)
            queue: LSF queue name (default: gpu_h100)
            charge_group: LSF charge group (default: cellmap)
            checkpoint_path_override: Optional path to override checkpoint detection (default: None)
            auto_serve: Automatically start inference server after training (default: True)

        Returns:
            FinetuneJob object tracking the submitted job

        Raises:
            ValueError: If validation fails
            RuntimeError: If job submission fails
        """
        # === Validation ===

        # 1. Check model config
        if not model_config:
            raise ValueError("Model config is required")

        # 2. Get checkpoint path if available (optional)
        # For script models: we'll pass the script path instead
        # For fly/dacapo models: we need the checkpoint path
        checkpoint_path = None

        # Check for checkpoint override first
        if checkpoint_path_override:
            checkpoint_path = Path(checkpoint_path_override)
            self.logger.info(f"Using checkpoint path override: {checkpoint_path}")
        # For FlyModelConfig, get checkpoint_path attribute
        elif hasattr(model_config, 'checkpoint_path') and model_config.checkpoint_path:
            checkpoint_path = Path(model_config.checkpoint_path)
            self.logger.info(f"Found checkpoint_path: {checkpoint_path}")

        # Validate checkpoint exists if specified
        if checkpoint_path and not checkpoint_path.exists():
            raise ValueError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                f"Please verify the path exists and is accessible."
            )

        # 3. Check corrections path exists
        if not corrections_path.exists():
            raise ValueError(f"Corrections path does not exist: {corrections_path}")

        # 4. Count corrections (warn if few)
        correction_dirs = list(corrections_path.glob("*/"))
        num_corrections = len([d for d in correction_dirs if (d / ".zattrs").exists()])

        if num_corrections == 0:
            raise ValueError(f"No corrections found in {corrections_path}")

        if num_corrections < 5:
            self.logger.warning(
                f"Only {num_corrections} corrections found. "
                "Recommend at least 5-10 for meaningful finetuning."
            )

        self.logger.info(f"Found {num_corrections} corrections for training")

        # === Setup output directory ===

        if output_base is None:
            output_base = Path("output/finetuning")
        else:
            output_base = Path(output_base)

        # Create timestamped run directory inside finetuning subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = model_config.name.replace("/", "_").replace(" ", "_")
        run_dir_name = f"{model_basename}_{timestamp}"
        output_dir = output_base / "finetuning" / "runs" / run_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / "training_log.txt"

        self.logger.info(f"Output directory: {output_dir}")

        # === Build training command ===

        # Get model metadata - try multiple sources
        model_type = self._get_model_metadata(model_config, "model_type", "fly")
        if model_type == "fly" and "dacapo" in model_config.name.lower():
            model_type = "dacapo"

        # Get channels - try multiple attribute names
        channels = None
        for attr_name in ["channels", "classes", "class_names"]:
            channels = self._get_model_metadata(model_config, attr_name, None)
            if channels:
                break
        if channels is None:
            channels = ["mito"]  # Default fallback
        if isinstance(channels, str):
            channels = [channels]

        # Get voxel sizes
        input_voxel_size = self._get_model_metadata(model_config, "input_voxel_size", [16, 16, 16])
        output_voxel_size = self._get_model_metadata(model_config, "output_voxel_size", [16, 16, 16])

        # Convert to list if needed (in case they're Coordinate objects)
        if not isinstance(input_voxel_size, list):
            input_voxel_size = list(input_voxel_size)
        if not isinstance(output_voxel_size, list):
            output_voxel_size = list(output_voxel_size)

        # Extract data path for inference server if auto-serve is enabled
        serve_data_path = None
        if auto_serve:
            try:
                serve_data_path = self._extract_data_path_from_corrections(corrections_path)
                self.logger.info(f"Extracted dataset path for inference: {serve_data_path}")
            except Exception as e:
                self.logger.warning(f"Could not extract dataset path from corrections: {e}")
                self.logger.warning("Auto-serve will be disabled")
                auto_serve = False

        # Build CLI command
        cli_command = f"python -m cellmap_flow.finetune.finetune_cli "
        cli_command += f"--model-type {model_type} "

        # Add checkpoint or script path depending on what's available
        if checkpoint_path:
            cli_command += f"--model-checkpoint {checkpoint_path} "
        elif hasattr(model_config, 'script_path'):
            cli_command += f"--model-script {model_config.script_path} "

        cli_command += (
            f"--corrections {corrections_path} "
            f"--output-dir {output_dir} "
            f"--model-name {model_config.name} "
            f"--channels {' '.join(channels)} "
            f"--input-voxel-size {' '.join(map(str, input_voxel_size))} "
            f"--output-voxel-size {' '.join(map(str, output_voxel_size))} "
            f"--lora-r {lora_r} "
            f"--lora-alpha {lora_r * 2} "
            f"--num-epochs {num_epochs} "
            f"--batch-size {batch_size} "
            f"--learning-rate {learning_rate} "
            f"--loss-type {loss_type} "
        )

        # Add label smoothing if specified
        if label_smoothing > 0:
            cli_command += f"--label-smoothing {label_smoothing} "

        # Add distillation lambda if specified
        if distillation_lambda > 0:
            cli_command += f"--distillation-lambda {distillation_lambda} "
            if distillation_scope == "all":
                cli_command += "--distillation-all-voxels "

        # Add margin if using margin loss
        if loss_type == "margin":
            cli_command += f"--margin {margin} "

        # Add auto-serve flags if enabled
        if auto_serve and serve_data_path:
            cli_command += f"--auto-serve --serve-data-path {serve_data_path} "

        # Add mask_unannotated flag for sparse annotations
        if mask_unannotated:
            cli_command += "--mask-unannotated "

        # Add class balancing flag
        if balance_classes:
            cli_command += "--balance-classes "

        # Add output type and related args
        if output_type != "binary":
            cli_command += f"--output-type {output_type} "
        if select_channel is not None:
            cli_command += f"--select-channel {select_channel} "
        if offsets is not None:
            cli_command += f"--offsets '{offsets}' "

        cli_command = f"stdbuf -oL {cli_command} 2>&1 | tee {log_file}"

        self.logger.info(f"Training command: {cli_command}")

        # === Save job metadata ===

        metadata = {
            "job_id": str(uuid.uuid4()),
            "model_name": model_config.name,
            "model_type": model_type,
            "model_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "model_script": str(model_config.script_path) if hasattr(model_config, 'script_path') else None,
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
                "channels": channels,
                "input_voxel_size": input_voxel_size,
                "output_voxel_size": output_voxel_size,
            },
            "queue": queue,
            "charge_group": charge_group,
            "created_at": datetime.now().isoformat(),
            "command": cli_command,
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Saved metadata to {metadata_file}")

        # === Submit job (LSF or local) ===

        job_name = f"finetune_{model_basename}_{timestamp}"

        # Check if bsub is available
        if is_bsub_available():
            self.logger.info("Submitting to LSF cluster via bsub")
            try:
                lsf_job = submit_bsub_job(
                    command=cli_command,
                    queue=queue,
                    charge_group=charge_group,
                    job_name=job_name,
                    num_gpus=1,
                    num_cpus=4
                )
                self.logger.info(f"Submitted LSF job {lsf_job.job_id} for finetuning")
            except Exception as e:
                self.logger.error(f"Failed to submit job to LSF: {e}")
                raise RuntimeError(f"Job submission to LSF failed: {e}")
        else:
            # Fallback to local execution
            self.logger.info("bsub not available - running finetuning locally")
            try:
                lsf_job = run_locally(
                    command=cli_command,
                    name=job_name
                )
                self.logger.info(f"Started local finetuning job (PID: {lsf_job.process.pid})")
            except Exception as e:
                self.logger.error(f"Failed to start local job: {e}")
                raise RuntimeError(f"Local job execution failed: {e}")

        # === Create FinetuneJob tracking object ===

        job_id = metadata["job_id"]

        finetune_job = FinetuneJob(
            job_id=job_id,
            lsf_job=lsf_job,
            model_name=model_config.name,
            output_dir=output_dir,
            params=metadata["params"],
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            log_file=log_file,
            total_epochs=num_epochs
        )

        self.jobs[job_id] = finetune_job

        # === Start monitoring thread ===

        monitor_thread = threading.Thread(
            target=self.monitor_job,
            args=(finetune_job,),
            daemon=True
        )
        monitor_thread.start()
        self._monitor_threads[job_id] = monitor_thread

        self.logger.info(f"Started monitoring thread for job {job_id}")

        return finetune_job

    def monitor_job(self, finetune_job: FinetuneJob):
        """
        Background thread for job monitoring.

        Polls LSF status and tails log file to track training progress.
        Triggers completion when job finishes.

        Args:
            finetune_job: The FinetuneJob to monitor
        """
        job_id = finetune_job.job_id
        self.logger.info(f"Monitoring job {job_id}...")

        last_log_position = 0
        check_interval = 3  # seconds

        try:
            while True:
                # === Check LSF job status ===

                if finetune_job.lsf_job:
                    lsf_status = finetune_job.lsf_job.get_status()

                    # Map LSF status to FinetuneJob status
                    if lsf_status == LSFJobStatus.RUNNING:
                        if finetune_job.status == JobStatus.PENDING:
                            self.logger.info(f"Job {job_id} started running")
                        finetune_job.status = JobStatus.RUNNING
                    elif lsf_status == LSFJobStatus.PENDING:
                        finetune_job.status = JobStatus.PENDING
                    elif lsf_status == LSFJobStatus.COMPLETED:
                        self.logger.info(f"Job {job_id} completed according to LSF")
                        finetune_job.status = JobStatus.COMPLETED
                        break
                    elif lsf_status == LSFJobStatus.FAILED:
                        self.logger.error(f"Job {job_id} failed according to LSF")
                        finetune_job.status = JobStatus.FAILED
                        break
                    elif lsf_status == LSFJobStatus.KILLED:
                        self.logger.warning(f"Job {job_id} was killed")
                        finetune_job.status = JobStatus.CANCELLED
                        break

                # === Tail log file for progress updates ===

                if finetune_job.log_file.exists():
                    try:
                        # Check if file was truncated (e.g., during restart archival)
                        file_size = finetune_job.log_file.stat().st_size
                        if file_size < last_log_position:
                            self.logger.info(f"Log file truncated (size {file_size} < position {last_log_position}), resetting")
                            last_log_position = 0

                        with open(finetune_job.log_file, "r") as f:
                            # Seek to last read position
                            f.seek(last_log_position)
                            new_content = f.read()
                            last_log_position = f.tell()

                            if new_content:
                                # Parse for epoch and loss information
                                self._parse_training_progress(finetune_job, new_content)
                                # Parse for inference server ready marker
                                self._parse_inference_server_ready(finetune_job, new_content)

                        # Always check for restart/iteration markers (reads full log).
                        # This must run every cycle, not just when there's new content,
                        # because the marker may have been at the end of the previous
                        # chunk and we need to detect it even if no new output follows.
                        self._parse_training_restart(finetune_job, new_content if new_content else "")
                    except Exception as e:
                        self.logger.debug(f"Error reading log file: {e}")

                # Sleep before next check
                time.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Error monitoring job {job_id}: {e}")
            finetune_job.status = JobStatus.FAILED

        finally:
            # === Post-completion actions ===

            if finetune_job.status == JobStatus.COMPLETED:
                try:
                    self.complete_job(finetune_job)
                except Exception as e:
                    self.logger.error(f"Error in post-completion for job {job_id}: {e}")
                    finetune_job.status = JobStatus.FAILED

            self.logger.info(f"Stopped monitoring job {job_id}. Final status: {finetune_job.status.value}")

    def _parse_training_progress(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log content for training progress (epoch, loss).

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        # Look for patterns like "Epoch 5/10" and "Loss: 0.1234"

        # Match: Epoch X/Y
        epoch_pattern = r"Epoch\s+(\d+)/(\d+)"
        epoch_matches = re.findall(epoch_pattern, log_content, re.IGNORECASE)
        if epoch_matches:
            last_match = epoch_matches[-1]
            finetune_job.current_epoch = int(last_match[0])
            finetune_job.total_epochs = int(last_match[1])

        # Match: Loss: X.XXXX (various formats)
        loss_patterns = [
            r"Loss:\s+([\d.]+)",
            r"loss:\s+([\d.]+)",
            r"avg_loss:\s+([\d.]+)",
        ]

        for pattern in loss_patterns:
            loss_matches = re.findall(pattern, log_content, re.IGNORECASE)
            if loss_matches:
                try:
                    finetune_job.latest_loss = float(loss_matches[-1])
                    break
                except ValueError:
                    pass

    def _add_finetuned_neuroglancer_layer(self, finetune_job: FinetuneJob, model_name: str):
        """
        Add (or replace) the finetuned model's neuroglancer layer.

        Mirrors run_model() from cellmap_flow/models/run.py:
        1. Create/update Job object in g.jobs
        2. Add neuroglancer ImageLayer with pre/post processing args

        Args:
            finetune_job: Job with inference_server_url set
            model_name: Layer name (e.g. "mito_finetuned_20240101_120000")
        """
        from cellmap_flow.globals import g
        from cellmap_flow.utils.web_utils import get_norms_post_args, ARGS_KEY
        import neuroglancer

        server_url = finetune_job.inference_server_url

        # Create a Job object for the running server
        inference_job = LSFJob(
            job_id=finetune_job.lsf_job.job_id if finetune_job.lsf_job else "local",
            model_name=model_name
        )
        inference_job.host = server_url
        inference_job.status = LSFJobStatus.RUNNING

        # Remove any old finetuned jobs for this base model
        g.jobs = [
            j for j in g.jobs
            if not (hasattr(j, 'model_name') and j.model_name
                    and j.model_name.startswith(f"{finetune_job.model_name}_finetuned"))
        ]

        # Add to g.jobs
        g.jobs.append(inference_job)
        self.logger.info(f"Added finetuned job to g.jobs: {model_name}")

        # Get pre/post processing args (same hash as other models)
        st_data = get_norms_post_args(g.input_norms, g.postprocess)

        if g.viewer is None:
            self.logger.error("g.viewer is None - neuroglancer not initialized yet")
            return

        source_url = f"zarr://{server_url}/{model_name}{ARGS_KEY}{st_data}{ARGS_KEY}"
        self.logger.info(f"Adding neuroglancer layer: {model_name}")
        self.logger.info(f"  source: {source_url}")

        with g.viewer.txn() as s:
            # Remove old finetuned layer if it exists (exact name match)
            old_layer_name = finetune_job.finetuned_model_name
            if old_layer_name and old_layer_name in s.layers:
                self.logger.info(f"Removing old finetuned layer: {old_layer_name}")
                del s.layers[old_layer_name]

            # Also remove by current name in case of re-add
            if model_name in s.layers:
                del s.layers[model_name]

            # Add new layer - exact same format as run_model()
            s.layers[model_name] = neuroglancer.ImageLayer(
                source=source_url,
                shader=f"""#uicontrol invlerp normalized(range=[0, 255], window=[0, 255]);
                    #uicontrol vec3 color color(default="red");
                    void main(){{emitRGB(color * normalized());}}""",
            )

        # Update the stored name
        finetune_job.finetuned_model_name = model_name
        self.logger.info(f"Successfully added neuroglancer layer: {model_name}")

    def _parse_inference_server_ready(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log for CELLMAP_FLOW_SERVER_IP marker and add finetuned model
        to neuroglancer exactly like a normal inference model.

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        if finetune_job.inference_server_ready:
            return

        # Look for the standard server IP marker (same one start_hosts() uses)
        from cellmap_flow.utils.web_utils import IP_PATTERN
        ip_start = IP_PATTERN[0]
        ip_end = IP_PATTERN[1]

        pattern = re.escape(ip_start) + r"(.+?)" + re.escape(ip_end)
        matches = re.findall(pattern, log_content)
        if not matches:
            return

        server_url = matches[-1]
        finetune_job.inference_server_url = server_url
        finetune_job.inference_server_ready = True
        self.logger.info(f"Finetuned inference server detected at {server_url}")

        try:
            # Read the FULL log file to find TRAINING_ITERATION_COMPLETE marker.
            # This marker is printed BEFORE the server starts, so it's typically
            # in an earlier log chunk than the server IP marker.
            iter_pattern = r"TRAINING_ITERATION_COMPLETE:\s+(\S+)"
            full_log = finetune_job.log_file.read_text()
            iter_matches = re.findall(iter_pattern, full_log)
            if iter_matches:
                model_name = iter_matches[-1]
            else:
                model_name = f"{finetune_job.model_name}_finetuned"

            self._add_finetuned_neuroglancer_layer(finetune_job, model_name)

        except Exception as e:
            self.logger.error(f"Failed to add finetuned model to neuroglancer: {e}", exc_info=True)

    def _parse_training_restart(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log for RESTARTING_TRAINING and TRAINING_ITERATION_COMPLETE markers
        to handle iterative training restarts.

        On RESTARTING_TRAINING: reset training progress counters.
        On TRAINING_ITERATION_COMPLETE: update the neuroglancer layer name with new timestamp.

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        # Check for restart marker - reset progress
        if "RESTARTING_TRAINING" in log_content:
            self.logger.info(f"Training restart detected for job {finetune_job.job_id}")
            finetune_job.current_epoch = 0
            finetune_job.latest_loss = None
            finetune_job.status = JobStatus.RUNNING
            finetune_job.inference_server_ready = False

        # Check for iteration complete marker - update neuroglancer layer.
        # Read full log in case the marker was in a previous chunk.
        iter_pattern = r"TRAINING_ITERATION_COMPLETE:\s+(\S+)"
        try:
            full_log = finetune_job.log_file.read_text()
        except Exception:
            full_log = log_content
        iter_matches = re.findall(iter_pattern, full_log)
        if iter_matches:
            # For in-process restarts, the inference server usually stays on the same
            # URL and does not emit a fresh CELLMAP_FLOW_SERVER_IP marker. Mark the
            # server as ready once we see a completed training iteration if URL exists.
            if finetune_job.inference_server_url:
                finetune_job.inference_server_ready = True

            new_model_name = iter_matches[-1]
            if new_model_name != finetune_job.finetuned_model_name:
                self.logger.info(f"New training iteration complete: {new_model_name}")
                try:
                    self._add_finetuned_neuroglancer_layer(finetune_job, new_model_name)
                except Exception as e:
                    self.logger.error(f"Failed to update neuroglancer layer: {e}", exc_info=True)
                    # Still update the stored name so the frontend reflects the new model
                    # and we don't retry the failed neuroglancer update every cycle
                    finetune_job.finetuned_model_name = new_model_name

    def complete_job(self, finetune_job: FinetuneJob):
        """
        Post-training actions after job completes successfully.

        1. Verify adapter files exist
        2. Generate model script and YAML
        3. Register in g.models_config
        4. Update job status and metadata

        Args:
            finetune_job: The completed job

        Raises:
            RuntimeError: If adapter files missing or registration fails
        """
        job_id = finetune_job.job_id
        self.logger.info(f"Running post-completion for job {job_id}...")

        # === Verify adapter files exist ===

        adapter_path = finetune_job.output_dir / "lora_adapter"

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

        self.logger.info(f"Verified LoRA adapter files exist in {adapter_path}")

        # === Generate finetuned model name ===

        timestamp = finetune_job.created_at.strftime("%Y%m%d_%H%M%S")
        model_basename = finetune_job.model_name.replace("/", "_").replace(" ", "_")
        finetuned_model_name = f"{model_basename}_finetuned_{timestamp}"

        finetune_job.finetuned_model_name = finetuned_model_name

        self.logger.info(f"Generated finetuned model name: {finetuned_model_name}")

        # === Generate model script and YAML ===

        # Import here to avoid circular dependencies
        from cellmap_flow.finetune.finetuned_model_templates import (
            generate_finetuned_model_script,
            generate_finetuned_model_yaml
        )

        # Models output directory (at session level, not in finetuning subdirectory)
        # output_dir structure: session_path/finetuning/runs/model_timestamp/
        # So parent.parent.parent gets us to session_path
        models_dir = finetune_job.output_dir.parent.parent.parent / "models"

        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Models directory ready: {models_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create models directory {models_dir}: {e}")
            raise RuntimeError(f"Failed to create models directory: {e}")

        # Check if files already exist (generated by CLI with auto-serve)
        expected_script = models_dir / f"{finetuned_model_name}.py"
        expected_yaml = models_dir / f"{finetuned_model_name}.yaml"
        files_already_generated = expected_script.exists() and expected_yaml.exists()

        if files_already_generated:
            self.logger.info(f"Model files already generated by CLI, skipping generation")
            finetune_job.model_script_path = expected_script
            finetune_job.model_yaml_path = expected_yaml
            script_path = expected_script
            yaml_path = expected_yaml
            # Skip to registration
        else:
            self.logger.info(f"Generating model files...")

            # Get base model script path from metadata if available
            metadata_file = finetune_job.output_dir / "metadata.json"
            base_script_path = None
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        base_script_path = metadata.get("model_script", None)
                        self.logger.info(f"Found base model script in metadata: {base_script_path}")
                except Exception as e:
                    self.logger.warning(f"Could not read base script from metadata: {e}")

            try:
                # Generate .py script
                self.logger.info(f"Generating finetuned model script for {finetuned_model_name}...")
                self.logger.info(f"  Base script path: {base_script_path}")
                self.logger.info(f"  LoRA adapter path: {adapter_path}")
                self.logger.info(f"  Output path: {models_dir / f'{finetuned_model_name}.py'}")

                script_path = generate_finetuned_model_script(
                base_checkpoint=finetune_job.params.get("model_checkpoint", ""),
                lora_adapter_path=str(adapter_path),
                model_name=finetuned_model_name,
                channels=finetune_job.params.get("channels", ["mito"]),
                input_voxel_size=tuple(finetune_job.params.get("input_voxel_size", [16, 16, 16])),
                output_voxel_size=tuple(finetune_job.params.get("output_voxel_size", [16, 16, 16])),
                lora_r=finetune_job.params.get("lora_r", 8),
                lora_alpha=finetune_job.params.get("lora_alpha", 16),
                num_epochs=finetune_job.params.get("num_epochs", 10),
                learning_rate=finetune_job.params.get("learning_rate", 1e-4),
                output_path=models_dir / f"{finetuned_model_name}.py",
                base_script_path=base_script_path
                )

                finetune_job.model_script_path = script_path
                self.logger.info(f"Generated model script: {script_path}")

                # === Extract configuration from base model and corrections ===

                data_path = None
                json_data = None
                base_scale = "s0"  # Default scale (only safe default)

                # 1. Get dataset_path from corrections metadata (REQUIRED)
                corrections_dir = Path(metadata.get("corrections_path", ""))
                try:
                    data_path = self._extract_data_path_from_corrections(corrections_dir)
                    self.logger.info(f"Found dataset_path from corrections: {data_path}")
                except (ValueError, Exception) as e:
                    self.logger.error(f"Could not extract dataset_path: {e}")

                # 2. Get normalization and preprocessing from base model YAML
                if base_script_path:
                    self.logger.info("Extracting normalization from base model YAML...")
                    import yaml
                    base_yaml_path = Path(base_script_path).with_suffix('.yaml')
                    if base_yaml_path.exists():
                        try:
                            with open(base_yaml_path, 'r') as f:
                                base_config = yaml.safe_load(f)

                                # Get json_data (normalization and postprocessing)
                                if 'json_data' in base_config:
                                    json_data = base_config['json_data']
                                    self.logger.info(f"✓ Found json_data from base model YAML")
                                else:
                                    self.logger.warning(f"No json_data in base model YAML: {base_yaml_path}")

                                # Get data_path from base model (fallback if not in corrections)
                                if not data_path and 'data_path' in base_config:
                                    data_path = base_config['data_path']
                                    self.logger.info(f"✓ Using data_path from base model YAML: {data_path}")

                                # Get scale
                                if 'models' in base_config and len(base_config['models']) > 0:
                                    base_scale = base_config['models'][0].get('scale', 's0')
                                    self.logger.info(f"✓ Found scale from base model: {base_scale}")
                        except Exception as e:
                            self.logger.error(f"Failed to read base model YAML {base_yaml_path}: {e}")
                    else:
                        self.logger.warning(f"Base model YAML not found: {base_yaml_path}")

                # 3. Validate we have required data (NO PLACEHOLDERS!)
                if not data_path:
                    raise RuntimeError(
                        "Could not determine dataset_path for finetuned model. "
                        "Checked corrections metadata and base model YAML. "
                        "Cannot generate model config without actual dataset path."
                    )

                if not json_data:
                    self.logger.warning(
                        "No json_data (normalization/postprocessing) found. "
                        "Finetuned model may not work correctly without proper normalization. "
                        "Consider adding json_data to base model YAML."
                    )

                # Generate .yaml config
                yaml_path = generate_finetuned_model_yaml(
                    script_path=script_path,
                    model_name=finetuned_model_name,
                    resolution=finetune_job.params.get("input_voxel_size", [16, 16, 16])[0],
                    output_path=models_dir / f"{finetuned_model_name}.yaml",
                    data_path=data_path,
                    queue=finetune_job.params.get("queue", "gpu_h100"),
                    json_data=json_data,
                    scale=base_scale
                )

                finetune_job.model_yaml_path = yaml_path
                self.logger.info(f"Generated model YAML: {yaml_path}")

            except Exception as e:
                import traceback
                self.logger.error(f"Error generating model files: {e}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise RuntimeError(f"Failed to generate model files: {e}")

        # === Update metadata file with completion info ===

        metadata_file = finetune_job.output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["completed_at"] = datetime.now().isoformat()
            metadata["status"] = "COMPLETED"
            metadata["finetuned_model_name"] = finetuned_model_name
            metadata["model_script_path"] = str(script_path)
            metadata["model_yaml_path"] = str(yaml_path)
            metadata["final_epoch"] = finetune_job.current_epoch
            metadata["final_loss"] = finetune_job.latest_loss

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Updated metadata file: {metadata_file}")

        self.logger.info(f"Job {job_id} completed successfully!")

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        if job_id not in self.jobs:
            self.logger.error(f"Job {job_id} not found")
            return False

        finetune_job = self.jobs[job_id]

        if finetune_job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            self.logger.warning(f"Job {job_id} already finished with status {finetune_job.status}")
            return False

        self.logger.info(f"Cancelling job {job_id}...")

        if finetune_job.lsf_job:
            try:
                finetune_job.lsf_job.kill()
                finetune_job.status = JobStatus.CANCELLED
                self.logger.info(f"Successfully cancelled job {job_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error cancelling job {job_id}: {e}")
                return False
        else:
            self.logger.error(f"No LSF job associated with {job_id}")
            return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific job.

        Args:
            job_id: Job ID to query

        Returns:
            Dictionary with job status details, or None if not found
        """
        if job_id not in self.jobs:
            return None

        finetune_job = self.jobs[job_id]
        result = finetune_job.to_dict()
        result["loss"] = result.pop("latest_loss", None)
        result["progress_percent"] = (
            finetune_job.current_epoch / finetune_job.total_epochs * 100
        ) if finetune_job.total_epochs > 0 else 0
        return result

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        Get list of all jobs with their status.

        Returns:
            List of job status dictionaries
        """
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]

    def get_job_logs(self, job_id: str) -> Optional[str]:
        """
        Get full log content for a job.

        Args:
            job_id: Job ID

        Returns:
            Log file content as string, or None if not found
        """
        if job_id not in self.jobs:
            return None

        finetune_job = self.jobs[job_id]

        if not finetune_job.log_file.exists():
            return "Log file not yet created..."

        try:
            with open(finetune_job.log_file, "r") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading log file: {e}")
            return f"Error reading log file: {e}"

    def get_job(self, job_id: str) -> Optional[FinetuneJob]:
        """
        Get a FinetuneJob object by ID.

        Args:
            job_id: Job ID to retrieve

        Returns:
            FinetuneJob object, or None if not found
        """
        return self.jobs.get(job_id)

    def _archive_job_logs(self, job: FinetuneJob):
        """
        Archive logs before restart.

        Args:
            job: The job whose logs to archive
        """
        log_file = job.log_file
        metadata_file = job.output_dir / "metadata.json"

        # Find next archive number
        archive_num = 1
        while (job.output_dir / f"training_log_{archive_num}.txt").exists():
            archive_num += 1

        # Archive log (copy only - do NOT truncate, as tee still has an open file descriptor)
        if log_file.exists():
            import shutil
            archive_log = job.output_dir / f"training_log_{archive_num}.txt"
            shutil.copy(log_file, archive_log)
            self.logger.info(f"Archived log to {archive_log}")

        # Archive metadata
        if metadata_file.exists():
            import shutil
            archive_meta = job.output_dir / f"metadata_{archive_num}.json"
            shutil.copy(metadata_file, archive_meta)
            self.logger.info(f"Archived metadata to {archive_meta}")

    def restart_finetuning_job(
        self,
        job_id: str,
        updated_params: Optional[Dict[str, Any]] = None
    ) -> FinetuneJob:
        """
        Restart training on the same GPU via control endpoint.

        Primary path sends an HTTP restart request to the running
        inference server in the same process as the training loop.
        Falls back to file signal if control endpoint is unavailable.

        Args:
            job_id: ID of job to restart
            updated_params: Dict of updated training parameters

        Returns:
            Same FinetuneJob object (updated in-place)

        Raises:
            ValueError: If job not found or not in a restartable state
        """
        restart_t0 = time.perf_counter()

        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]

        # Only allow restart if the job is running (serving after training)
        if job.status not in [JobStatus.RUNNING, JobStatus.COMPLETED]:
            raise ValueError(
                f"Job {job_id} is in state {job.status.value} - "
                f"can only restart jobs that are RUNNING (serving) or COMPLETED"
            )

        if not job.inference_server_ready:
            raise ValueError(
                f"Job {job_id} inference server not ready - "
                f"training must complete and server must start before restarting"
            )

        # 1. Archive current logs
        self.logger.info(f"Archiving logs for job {job_id}...")
        archive_t0 = time.perf_counter()
        self._archive_job_logs(job)
        archive_elapsed = time.perf_counter() - archive_t0

        signal_data = {
            "restart": True,
            "timestamp": datetime.now().isoformat(),
            "params": updated_params or {}
        }

        # 2. Send restart request to running inference server (primary path)
        signal_write_mode = "http_control"
        write_t0 = time.perf_counter()
        http_error = None
        if job.inference_server_url:
            try:
                control_url = job.inference_server_url.rstrip("/") + "/__control__/restart"
                response = requests.post(control_url, json=signal_data, timeout=5)
                response.raise_for_status()
                data = response.json()
                if not data.get("success", False):
                    raise RuntimeError(data.get("error", "Unknown restart control failure"))
                self.logger.info(f"Sent restart request via HTTP control endpoint: {control_url}")
            except Exception as e:
                http_error = e
                self.logger.warning(f"HTTP restart control failed for job {job_id}: {e}")
        else:
            http_error = RuntimeError("No inference_server_url for HTTP restart control")

        # 3. Fallback to signal file if HTTP control endpoint is unavailable
        if http_error is not None:
            signal_write_mode = "file_signal_fallback"
            signal_file = job.output_dir / "restart_signal.json"
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            self.logger.info(f"Wrote fallback restart signal to {signal_file}")
        write_elapsed = time.perf_counter() - write_t0

        # 4. Reset training progress (keep inference server info)
        job.current_epoch = 0
        job.latest_loss = None
        job.status = JobStatus.RUNNING
        job.inference_server_ready = False

        # 5. Update stored params
        if updated_params:
            job.params.update(updated_params)

        total_elapsed = time.perf_counter() - restart_t0
        self.logger.info(
            f"Restart signal timings for job {job_id}: "
            f"archive={archive_elapsed:.2f}s write={write_elapsed:.2f}s "
            f"mode={signal_write_mode} total={total_elapsed:.2f}s"
        )
        self.logger.info(f"Job {job_id} restart request sent, waiting for CLI to pick it up")

        return job
