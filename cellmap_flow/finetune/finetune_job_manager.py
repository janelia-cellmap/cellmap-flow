"""
Job manager for orchestrating finetuning jobs on LSF cluster.

This module provides:
- FinetuneJob: Track metadata and status of a single finetuning job
- FinetuneJobManager: Orchestrate job lifecycle from submission to completion
"""

import json
import logging
import os
import string
import sys
import threading
import time
import uuid
import requests
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Any

from cellmap_flow.finetune import markers
from cellmap_flow.finetune.job_log import LogTailer
from cellmap_flow.jobs import lsf as jobs_lsf
from cellmap_flow.jobs.site import current_site
from cellmap_flow.jobs.spec import JobSpec
from cellmap_flow.jobs.spec import JobStatus as LSFJobStatus
from cellmap_flow.jobs.lsf import LSFJob
# Module globals, looked up when a job is submitted, so tests can replace
# them here.
from cellmap_flow.jobs.lsf import available as is_bsub_available
from cellmap_flow.jobs.local import run as run_locally
from cellmap_flow.utils.restart_token import (
    TOKEN_HEADER,
    read_restart_token,
    write_restart_token,
)

logger = logging.getLogger(__name__)


# The --model-type values finetune_cli accepts, and the ones among them that
# it takes as --model-entry (the model's to_dict()) because they have no
# dedicated flags.
TRAINABLE_MODEL_TYPES = frozenset({"fly", "dacapo", "huggingface", "script", "cellmap", "finetune"})
MODEL_ENTRY_TYPES = frozenset({"cellmap", "finetune"})


class JobStatus(Enum):
    """Status of a finetuning job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    # Alive and idle: an iteration finished (and is being served) or
    # diverged, and the trainer is waiting for a restart request.
    WAITING_FOR_RESTART = "WAITING_FOR_RESTART"


# The trainer's markers; see finetune/markers.py.
_STATUS_MARKER_RE = markers.STATUS_MARKER_RE


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


# Values in this command survive two rounds of shell quoting: the one bsub
# starts on the exec host, and LSF's own handling, which re-wraps the whole
# `bash -c` argument in single quotes. A single quote of ours therefore closes
# LSF's and the argument word-splits. That is not hypothetical:
#
#     --offsets '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]'
#
# reached the trainer as the bare word "[[1," with the rest scattered as stray
# arguments, and json.loads died with "Expecting value: line 1 column 5".
# Double quotes nest inside LSF's single quotes safely, so quote with those.
_SHELL_SAFE = frozenset(string.ascii_letters + string.digits + "@%+=:,./-_")


def _sh_quote(part: str) -> str:
    """Shell-quote without ever emitting a single quote."""
    part = str(part)
    if part and all(c in _SHELL_SAFE for c in part):
        return part
    escaped = (
        part.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def finetune_export_kwargs(output_dir, params=None) -> dict:
    """Which artifact a finished run produced, as FinetuneModelConfig kwargs.

    A LoRA run exports lora_adapter/; a full finetune (--lora-r 0) exports
    full_finetune/model_state_dict.pt. Decided by what is on disk first --
    the job's own record of lora_r is the fallback for a run that has not
    written its export yet -- so the dashboard never points a viewer at an
    adapter directory that a rank-0 run never made.
    """
    from pathlib import Path
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


_ITERATION_COMPLETE_RE = markers.ITERATION_COMPLETE_RE
_MODEL_YAML_RE = markers.MODEL_YAML_RE


def _yaml_model_entry(yaml_path) -> Optional[dict]:
    """The first models: entry of a serving YAML, or None if it cannot be read."""
    if not yaml_path:
        return None
    try:
        import yaml

        with open(yaml_path) as f:
            models = (yaml.safe_load(f) or {}).get("models") or []
    except Exception:
        return None
    entry = models[0] if models else None
    return entry if isinstance(entry, dict) and entry.get("base_model") else None


def _finished_iterations(log_file):
    """(how many iterations the log says finished, the last one's model name).

    Read a line at a time, since the log is everything the run printed. A log
    that is missing or cannot be read is no evidence: (0, None).
    """
    count, last = 0, None
    try:
        with open(log_file, errors="replace") as f:
            for line in f:
                for name in markers.ITERATION_COMPLETE_RE.findall(line):
                    count, last = count + 1, name
    except OSError:
        return 0, None
    return count, last


def trainer_outputs_from_log(log_text: str):
    """(model name, serving YAML path) of the last iteration the log reports.

    Either is None when the log has none. The YAML is only taken when it
    belongs to that iteration: the trainer prints it just before the
    iteration's completion marker, and skips it when it could not write one.
    """
    names = list(_ITERATION_COMPLETE_RE.finditer(log_text))
    if not names:
        return None, None
    last = names[-1]
    previous_end = names[-2].end() if len(names) > 1 else 0
    yamls = [
        m for m in _MODEL_YAML_RE.finditer(log_text, previous_end, last.start())
    ]
    return last.group(1), (yamls[-1].group(1) if yamls else None)


@dataclass
class FinetuneJob:
    """Track a finetuning job with metadata, status, and training progress.

    Manages lifecycle from submission through completion, including inference
    server state.
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
    model_yaml_path: Optional[Path] = None
    current_epoch: int = 0
    total_epochs: int = 10
    latest_loss: Optional[float] = None
    inference_server_url: Optional[str] = None
    inference_server_ready: bool = False
    # The corrections directory the job trains on. The restart route reads it
    # to refresh the manifest; it was never set, so a restart silently kept
    # the old patches_per_epoch, rehearsal fraction, input norm and
    # postprocessing even though the confirm dialog showed the new ones.
    corrections_path: Optional[Path] = None
    # Set by cancel_job before it kills the job, so the monitor reports the
    # exit that follows as CANCELLED rather than FAILED.
    cancel_requested: bool = False
    _processed_iteration_count: int = 0

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
            "model_yaml_path": str(self.model_yaml_path) if self.model_yaml_path else None,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "latest_loss": self.latest_loss,
            "inference_server_url": self.inference_server_url,
            "inference_server_ready": self.inference_server_ready,
            "corrections_path": str(self.corrections_path) if self.corrections_path else None,
        }


class FinetuneJobListener:
    """What the job manager tells its listeners (FinetuneJobManager.add_listener).

    Both are called on the job's monitor thread. A listener need not define
    both; one that raises is logged and does not stop the others.
    """

    def on_server_ready(self, job: FinetuneJob, url: str, model_name: str) -> None:
        """The job's inference server is up at ``url``, serving ``model_name``."""

    def on_iteration_complete(self, job: FinetuneJob, model_name: str) -> None:
        """The job finished a training iteration and named its model ``model_name``.

        ``job.finetuned_model_name`` is still the previous iteration's name
        (None before the first) while listeners run, so one that replaces a
        viewer layer can find the old one; the manager updates it after.
        """


class ViewerListener(FinetuneJobListener):
    """What the dashboard has always done: a viewer layer and a pipeline model.

    Each event adds (or replaces) the finetuned model's neuroglancer layer and
    registers its FinetuneModelConfig, through the manager's own methods.
    Either failing is logged, and does not stop the other.
    """

    def __init__(self, manager: "FinetuneJobManager"):
        self.manager = manager

    def _add_layer(self, job, model_name, failure):
        try:
            self.manager._add_finetuned_neuroglancer_layer(job, model_name)
        except Exception as e:
            self.manager.logger.error(f"{failure}: {e}", exc_info=True)

    def _register(self, job, model_name):
        try:
            self.manager._register_finetune_model_config(job, model_name)
        except Exception as e:
            self.manager.logger.error(f"Failed to register FinetuneModelConfig: {e}", exc_info=True)

    def on_server_ready(self, job, url, model_name):
        self._add_layer(job, model_name, "Failed to add finetuned model to neuroglancer")
        self._register(job, model_name)

    def on_iteration_complete(self, job, model_name):
        self._add_layer(job, model_name, "Failed to update neuroglancer layer")
        self._register(job, model_name)


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
        # Told when a job's server comes up and when an iteration finishes.
        self.viewer_listener = ViewerListener(self)
        self._listeners: List[Any] = [self.viewer_listener]

    def add_listener(self, listener) -> None:
        """Tell ``listener`` about job events; see FinetuneJobListener."""
        self._listeners.append(listener)

    def remove_listener(self, listener) -> None:
        """Stop telling ``listener``, which may be the default viewer_listener."""
        self._listeners = [other for other in self._listeners if other is not listener]

    def _notify(self, event: str, *args) -> None:
        for listener in list(self._listeners):
            handler = getattr(listener, event, None)
            if handler is None:
                continue
            try:
                handler(*args)
            except Exception as e:
                self.logger.error(f"Finetune listener {listener!r} failed in {event}: {e}", exc_info=True)

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

        # Then the model's geometry. This used to read model_config.config,
        # which for a script, Hugging Face or DaCapo model builds the model in
        # the dashboard process -- weights download, torch.export, a CUDA
        # context -- just to read two voxel sizes and the channel names.
        # resolve_model_geometry asks the model's running server, then its
        # cache, and builds the model only when neither can answer.
        config = self._model_geometry(model_config)
        if config is not None:
            value = getattr(config, attr_name, None)
            if value is not None:
                return value

        return default

    def _warn_made_up(self, model_config, made_up: dict) -> None:
        """Say, once per model, which of its settings the trainer is guessing.

        A model that says nothing of its channels or voxel sizes is trained as
        if it predicted mito at 16 nm, which is quietly wrong for most models.
        """
        name = getattr(model_config, "name", None)
        warned = self.__dict__.setdefault("_made_up_warned", set())
        if not made_up or name in warned:
            return
        warned.add(name)
        guesses = ", ".join(f"{key}={value}" for key, value in made_up.items())
        self.logger.warning(
            f"Model {name!r} does not say its {', '.join(made_up)}; training it "
            f"with {guesses}. Set them in the model's config if that is wrong."
        )

    def _model_geometry(self, model_config):
        """The model's geometry (see utils/model_geometry), looked up once per config."""
        cache = self.__dict__.setdefault("_geometry_cache", {})
        key = id(model_config)
        if key not in cache:
            from cellmap_flow.utils.model_geometry import resolve_model_geometry

            try:
                cache[key] = (model_config, resolve_model_geometry(
                    getattr(model_config, "name", None), model_config
                ))
            except Exception as e:
                self.logger.debug(f"Could not resolve the geometry of {model_config}: {e}")
                cache[key] = (model_config, None)
        return cache[key][1]

    def _extract_data_path_from_corrections(self, corrections_path: Path) -> str:
        """Extract dataset path from corrections metadata.

        The manifest's raw_dataset_path first -- it is what the trainer reads
        -- and only then the first correction zarr's attrs, which a crop zarr
        may not have.
        """
        from cellmap_flow.finetune.session.manifest import read_manifest

        try:
            raw = (read_manifest(str(corrections_path)) or {}).get("raw_dataset_path")
        except (OSError, ValueError):
            raw = None
        if raw:
            return raw

        # Look for first .zarr directory
        zarr_dirs = sorted(corrections_path.glob("*.zarr"))
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

    def _resolve_model_type(self, model_config) -> str:
        """Infer the finetuning CLI model type from the model config.

        Raises ValueError for a type the trainer cannot train, rather than
        submitting a GPU job whose argparse exits with code 2.
        """
        model_type = getattr(type(model_config), "cli_name", "fly")
        if model_type == "fly" and "dacapo" in model_config.name.lower():
            return "dacapo"
        if model_type not in TRAINABLE_MODEL_TYPES:
            raise ValueError(
                f"Models of type {model_type!r} cannot be finetuned; the trainer "
                f"supports {sorted(TRAINABLE_MODEL_TYPES)}."
            )
        return model_type

    def _normalize_metadata_list(self, value, default):
        """Return model metadata as a plain list for CLI serialization."""
        if value is None:
            return list(default)
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return list(value)

    def _build_finetune_command(
        self,
        *,
        model_config,
        model_type: str,
        checkpoint_path: Optional[Path],
        corrections_path: Path,
        output_dir: Path,
        log_file: Path,
        channels: List[str],
        input_voxel_size: List[int],
        output_voxel_size: List[int],
        lora_r: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        loss_type: str,
        label_smoothing: float,
        distillation_lambda: Optional[float],
        distillation_scope: str,
        margin: float,
        auto_serve: bool,
        serve_data_path: Optional[str],
        mask_unannotated: bool,
        balance_classes: bool,
        augment: bool,
        output_type: str,
        select_channel: Optional[int],
        offsets: Optional[str],
        models_dir: Optional[Path] = None,
        queue: Optional[str] = None,
        charge_group: Optional[str] = None,
    ) -> str:
        """Build the shell command used to launch finetuning."""
        command_parts = [
            sys.executable,
            "-m",
            "cellmap_flow.finetune.finetune_cli",
            "--model-type", model_type,
        ]

        if model_type in MODEL_ENTRY_TYPES:
            # No dedicated flags: hand the trainer the model's own entry.
            # encode_to_str() is URL-safe base64, so it needs no quoting.
            from cellmap_flow.utils.web_utils import encode_to_str

            command_parts += ["--model-entry", encode_to_str(model_config.to_dict())]
        elif model_type == "huggingface":
            command_parts += ["--repo", str(model_config.repo)]
            if getattr(model_config, "revision", None):
                command_parts += ["--revision", str(model_config.revision)]
        elif checkpoint_path:
            command_parts += ["--model-checkpoint", str(checkpoint_path)]
        elif hasattr(model_config, "script_path"):
            command_parts += ["--model-script", str(model_config.script_path)]

        command_parts += [
            "--corrections", str(corrections_path),
            "--output-dir", str(output_dir),
            "--model-name", str(model_config.name),
            "--channels", *map(str, channels),
            "--input-voxel-size", *map(str, input_voxel_size),
            "--output-voxel-size", *map(str, output_voxel_size),
            "--lora-r", str(lora_r),
            "--lora-alpha", str(lora_r * 2),
            "--num-epochs", str(num_epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--loss-type", str(loss_type),
        ]

        if label_smoothing > 0:
            command_parts += ["--label-smoothing", str(label_smoothing)]
        # Passed whenever it was chosen, 0 included: leaving the flag out means
        # "unset", which the trainer turns into 1.0 when good regions exist,
        # so omitting it for 0 made "0 (Disabled)" impossible to express.
        if distillation_lambda is not None:
            command_parts += ["--distillation-lambda", str(distillation_lambda)]
        if distillation_scope == "all" and (distillation_lambda is None or distillation_lambda > 0):
            command_parts.append("--distillation-all-voxels")
        if loss_type == "margin":
            command_parts += ["--margin", str(margin)]
        if auto_serve and serve_data_path:
            command_parts += ["--auto-serve", "--serve-data-path", str(serve_data_path)]
        if mask_unannotated:
            command_parts.append("--mask-unannotated")
        if balance_classes:
            command_parts.append("--balance-classes")
        # Opt-out flag: only passed when augmentation is disabled.
        if not augment:
            command_parts.append("--no-augment")
        if output_type != "binary":
            command_parts += ["--output-type", str(output_type)]
        if select_channel is not None:
            command_parts += ["--select-channel", str(select_channel)]
        if offsets is not None:
            command_parts += ["--offsets", str(offsets)]
        if models_dir is not None:
            command_parts += ["--models-dir", str(models_dir)]
        # Only written into the serving YAMLs, so a model served from one runs
        # where this job did rather than on hard-coded defaults.
        if queue:
            command_parts += ["--queue", str(queue)]
        if charge_group:
            command_parts += ["--charge-group", str(charge_group)]

        command = " ".join(_sh_quote(part) for part in command_parts)

        # Put this interpreter's own lib directory first on the loader path.
        #
        # We launch sys.executable directly rather than through the
        # environment's activation script, so nothing sets LD_LIBRARY_PATH for
        # us -- and LSF runs the exec host's login shell first, which can put
        # system paths ahead of ours. When that happens the system
        # libstdc++.so.6 is loaded before anything from this environment, and
        # the first extension module built against a newer toolchain fails
        # with "version `CXXABI_1.3.15' not found" even though the
        # environment ships a libstdc++ that has it. Seen with scipy pulled in
        # via cellpose on a GCC 13+ build.
        env_lib = os.path.join(sys.prefix, "lib")
        loader_path = (
            f"LD_LIBRARY_PATH={_sh_quote(env_lib)}"
            '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} '
        )
        # stdbuf on *both* sides. The trainer already flushes every line it
        # prints, but tee writes to the log file through stdio, which is
        # block-buffered when the destination is not a terminal -- so roughly
        # 8KB of output, five to ten epochs' worth, landed in the file at
        # once and the dashboard showed nothing in between.
        return (
            f"{loader_path}stdbuf -oL {command} 2>&1 "
            f"| stdbuf -oL tee {_sh_quote(log_file)}"
        )

    def _build_submission_metadata(
        self,
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

    def submit_finetuning_job(
        self,
        model_config,
        corrections_path: Path,
        lora_r: int = 8,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        output_base: Optional[Path] = None,
        queue: str = "gpu_h100",
        charge_group: str = "cellmap",
        checkpoint_path_override: Optional[Path] = None,
        auto_serve: bool = True,
        mask_unannotated: bool = False,
        loss_type: str = "combined",
        label_smoothing: float = 0.0,
        # None leaves it to the trainer: 1.0 with good regions, else 0.
        distillation_lambda: Optional[float] = None,
        distillation_scope: str = "unlabeled",
        margin: float = 0.3,
        balance_classes: bool = False,
        augment: bool = False,
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
            batch_size: Training batch size (default: 8)
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

        # Get model type from the config class's cli_name (e.g., "fly",
        # "dacapo", "huggingface"); refuses types the trainer cannot train.
        model_type = self._resolve_model_type(model_config)
        self._geometry_cache = {}  # looked up afresh for every submit

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

        # 4. Check there is something to train on. The trainer reads the
        # session's virtual-sources manifest and nothing else, so a session
        # without one failed on the GPU node with FileNotFoundError after
        # queueing. This used to count *.zarr directories instead: always
        # "Only 1 corrections" for a volume session, and any crop zarr passed.
        from cellmap_flow.finetune.session.manifest import VIRTUAL_MANIFEST_FILENAME, read_manifest

        if read_manifest(str(corrections_path)) is None:
            raise ValueError(
                f"No {VIRTUAL_MANIFEST_FILENAME} in {corrections_path}, so there is "
                "nothing to train on. Create an annotation volume, or import crops, first."
            )
        correction_dirs = list(corrections_path.glob("*/"))
        num_corrections = len([d for d in correction_dirs if (d / ".zattrs").exists()])

        # === Setup output directory ===

        if output_base is None:
            output_base = Path("output/finetuning")
        else:
            output_base = Path(output_base)

        # Create timestamped run directory under output_base/runs/ (output_base
        # already encodes the "finetuning" segment — see default above).
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = model_config.name.replace("/", "_").replace(" ", "_")
        run_dir_name = f"{model_basename}_{timestamp}"
        output_dir = output_base / "runs" / run_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Before submitting, so the job's server finds it and requires it.
        write_restart_token(output_dir)

        log_file = output_dir / "training_log.txt"
        # The session's models/, next to runs/, where every iteration's
        # serving YAML goes.
        models_dir = output_base / "models"

        self.logger.info(f"Output directory: {output_dir}")

        # === Build training command ===

        # Get channels - try multiple attribute names
        channels = None
        for attr_name in ["channels", "classes", "class_names"]:
            channels = self._get_model_metadata(model_config, attr_name, None)
            if channels:
                break
        made_up = {}
        if channels is None:
            channels = made_up["channels"] = ["mito"]  # Default fallback
        channels = self._normalize_metadata_list(channels, ["mito"])

        # Get voxel sizes
        input_voxel_size = self._get_model_metadata(model_config, "input_voxel_size", None)
        if input_voxel_size is None:
            input_voxel_size = made_up["input_voxel_size"] = [16, 16, 16]
        output_voxel_size = self._get_model_metadata(model_config, "output_voxel_size", None)
        if output_voxel_size is None:
            output_voxel_size = made_up["output_voxel_size"] = [16, 16, 16]
        self._warn_made_up(model_config, made_up)

        input_voxel_size = self._normalize_metadata_list(input_voxel_size, [16, 16, 16])
        output_voxel_size = self._normalize_metadata_list(output_voxel_size, [16, 16, 16])

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

        cli_command = self._build_finetune_command(
            model_config=model_config,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            corrections_path=corrections_path,
            output_dir=output_dir,
            log_file=log_file,
            channels=channels,
            input_voxel_size=input_voxel_size,
            output_voxel_size=output_voxel_size,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            auto_serve=auto_serve,
            serve_data_path=serve_data_path,
            mask_unannotated=mask_unannotated,
            balance_classes=balance_classes,
            augment=augment,
            output_type=output_type,
            select_channel=select_channel,
            offsets=offsets,
            models_dir=models_dir,
            queue=queue,
            charge_group=charge_group,
        )

        self.logger.info(f"Training command: {cli_command}")

        # === Save job metadata ===

        metadata = self._build_submission_metadata(
            model_config=model_config,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            corrections_path=corrections_path,
            num_corrections=num_corrections,
            output_dir=output_dir,
            lora_r=lora_r,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            distillation_lambda=distillation_lambda,
            distillation_scope=distillation_scope,
            margin=margin,
            balance_classes=balance_classes,
            augment=augment,
            channels=channels,
            input_voxel_size=input_voxel_size,
            output_voxel_size=output_voxel_size,
            output_type=output_type,
            queue=queue,
            charge_group=charge_group,
            command=cli_command,
        )

        metadata["models_dir"] = str(models_dir)

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
                # Training runs epochs, not chunks, so it is the likeliest
                # thing here to outlive the queue's 120-minute default.
                from cellmap_flow.globals import g as _g

                lsf_job = jobs_lsf.submit(JobSpec(
                    name=job_name,
                    # A shell line: it sets LD_LIBRARY_PATH and pipes through tee.
                    shell=cli_command,
                    queue=queue,
                    charge_group=charge_group,
                    gpus=1,
                    cpus=4,
                    walltime=getattr(_g, "walltime", None) or current_site().default_walltime,
                ))
                self.logger.info(f"Submitted LSF job {lsf_job.job_id} for finetuning")
            except Exception as e:
                self.logger.error(f"Failed to submit job to LSF: {e}")
                raise RuntimeError(f"Job submission to LSF failed: {e}")
        else:
            # Fallback to local execution
            self.logger.info("bsub not available - running finetuning locally")
            try:
                # cli_command is a shell command: it sets LD_LIBRARY_PATH as a
                # prefix assignment and pipes through tee. run_locally splits
                # with shlex and runs shell=False on purpose, so handing it
                # this string would exec "LD_LIBRARY_PATH=..." as a program.
                # Give it an argv list with an explicit shell instead -- the
                # list form skips run_locally's shlex.split entirely.
                lsf_job = run_locally(
                    command=["bash", "-c", cli_command],
                    name=job_name,
                    # The command tees its own output to training_log.txt;
                    # a second copy under ~/.cellmap_flow/server_logs is noise.
                    log_file=os.devnull,
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
            total_epochs=num_epochs,
            corrections_path=corrections_path,
        )

        self.jobs[job_id] = finetune_job
        # What a dashboard started later needs to find this job again: the
        # scheduler's id for it, and where it stands (see rehydrate_session).
        self._update_metadata(
            finetune_job, lsf_job_id=finetune_job.to_dict()["lsf_job_id"],
            status=finetune_job.status.value,
        )

        self._start_monitor(finetune_job)

        return finetune_job

    def _start_monitor(self, finetune_job: FinetuneJob):
        monitor_thread = threading.Thread(
            target=self.monitor_job,
            args=(finetune_job,),
            daemon=True
        )
        monitor_thread.start()
        self.logger.info(f"Started monitoring thread for job {finetune_job.job_id}")

    def _update_metadata(self, finetune_job: FinetuneJob, **fields):
        """Merge ``fields`` into the job's metadata.json, replacing it atomically."""
        path = Path(finetune_job.output_dir) / "metadata.json"
        try:
            metadata = json.loads(path.read_text()) if path.exists() else {}
            metadata.update(fields)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(metadata, indent=2))
            os.replace(tmp, path)
        except Exception as e:
            self.logger.warning(f"Could not update {path}: {e}")

    def rehydrate_session(self, session_path) -> int:
        """Pick up the jobs of a session that are still alive on the cluster.

        Jobs live only in this process's memory, so after a dashboard restart
        a running job disappeared: it could not be seen or cancelled, and
        with auto-serve it waited for a restart until walltime. Each job's
        metadata.json now records its LSF job id and status; a run whose
        recorded status is not final and that bjobs still reports as pending
        or running is monitored again, which also brings back its viewer
        layer once the log shows its server. Local runs (a PID, not an LSF
        job) are not reattached. Returns how many jobs were picked up.

        All the session's candidates are asked about in one bjobs call, and
        whatever bjobs says has ended -- including a job it no longer knows
        at all -- is recorded as final, so it is not asked about again. This
        runs on every load of the finetune tab.
        """
        candidates = []
        for metadata_file in sorted(Path(session_path).glob("runs/*/metadata.json")):
            try:
                metadata = json.loads(metadata_file.read_text())
            except (OSError, ValueError):
                continue
            job_id = metadata.get("job_id")
            lsf_job_id = metadata.get("lsf_job_id")
            if (
                not job_id
                or job_id in self.jobs
                or not lsf_job_id
                or str(lsf_job_id).startswith("PID:")
                or metadata.get("status") in {s.value for s in TERMINAL_STATUSES}
            ):
                continue
            candidates.append((metadata_file, metadata, job_id, str(lsf_job_id)))
        if not candidates:
            return 0

        reported = jobs_lsf.statuses([lsf_job_id for *_, lsf_job_id in candidates])
        count = 0
        for metadata_file, metadata, job_id, lsf_job_id in candidates:
            if lsf_job_id not in reported:
                continue  # bjobs cannot say; try again next time
            observed = reported[lsf_job_id]
            record = SimpleNamespace(output_dir=metadata_file.parent)
            if observed is None:
                # LSF has forgotten it: it ended long enough ago to be purged,
                # while no dashboard was watching. Asked about again, it never
                # answers. The trainer never prints "done" -- after an
                # iteration it waits for restarts until it is stopped or runs
                # out of walltime -- so a run whose log shows a finished
                # iteration delivered a model, and that is what it is recorded
                # as having done.
                iterations, last = _finished_iterations(metadata_file.parent / "training_log.txt")
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
                self._update_metadata(record, status=status.value, status_detail=detail)
                continue
            if observed == LSFJobStatus.COMPLETED:
                # Finished while no dashboard was watching; say so, so it is
                # not asked about again. complete_job does not run for it.
                self._update_metadata(record, status=JobStatus.COMPLETED.value)
                continue
            if observed == LSFJobStatus.FAILED:
                self._update_metadata(record, status=JobStatus.FAILED.value)
                continue
            if observed not in (LSFJobStatus.RUNNING, LSFJobStatus.PENDING):
                continue
            lsf_job = LSFJob(job_id=lsf_job_id, model_name=metadata.get("model_name"))
            params = metadata.get("params") or {}
            output_dir = metadata_file.parent
            try:
                created_at = datetime.fromisoformat(metadata["created_at"])
            except (KeyError, TypeError, ValueError):
                created_at = datetime.now()
            job = FinetuneJob(
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
            )
            self.jobs[job_id] = job
            self.logger.info(f"Reattached to job {job_id} (LSF {lsf_job_id}) from {output_dir}")
            self._start_monitor(job)
            count += 1
        return count

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

        log = LogTailer(finetune_job.log_file)
        check_interval = 3  # seconds
        persisted_status = finetune_job.status

        try:
            while True:
                # A cancel (or anything else that ended the job) is final. The
                # poll below used to overwrite it: after cancel_job set
                # CANCELLED, LSF reported the kill as EXIT (a local job as
                # return code -15) and the job ended up FAILED.
                if finetune_job.status in TERMINAL_STATUSES:
                    break

                # === Check LSF job status ===

                if finetune_job.lsf_job:
                    lsf_status = finetune_job.lsf_job.get_status()

                    # Map LSF status to FinetuneJob status
                    if finetune_job.cancel_requested and lsf_status in (
                        LSFJobStatus.COMPLETED, LSFJobStatus.FAILED, LSFJobStatus.KILLED
                    ):
                        finetune_job.status = JobStatus.CANCELLED
                        break
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
                        # Whole lines only; see LogTailer.
                        new_content = log.read()
                        if new_content:
                            # Parse for epoch and loss information
                            self._parse_training_progress(finetune_job, new_content)
                            # Parse for inference server ready marker
                            self._parse_inference_server_ready(finetune_job, new_content)

                        # Always check for restart/iteration markers (reads full log).
                        # This must run every cycle, not just when there's new content,
                        # because the marker may have been at the end of the previous
                        # chunk and we need to detect it even if no new output follows.
                        self._parse_training_restart(finetune_job, new_content)
                    except Exception as e:
                        self.logger.debug(f"Error reading log file: {e}")

                if finetune_job.status != persisted_status:
                    persisted_status = finetune_job.status
                    self._update_metadata(
                        finetune_job, status=persisted_status.value,
                        inference_server_url=finetune_job.inference_server_url,
                    )

                # Sleep before next check
                time.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Error monitoring job {job_id}: {e}")
            if finetune_job.status not in TERMINAL_STATUSES:
                finetune_job.status = JobStatus.FAILED

        finally:
            # === Post-completion actions ===

            if finetune_job.status == JobStatus.COMPLETED:
                try:
                    self.complete_job(finetune_job)
                except Exception as e:
                    self.logger.error(f"Error in post-completion for job {job_id}: {e}")
                    finetune_job.status = JobStatus.FAILED
            else:
                # A job that failed after training -- its server would not
                # start, say -- still produced a model; record what it was.
                self._read_trainer_outputs(finetune_job)

            self._update_metadata(
                finetune_job,
                status=finetune_job.status.value,
                finetuned_model_name=finetune_job.finetuned_model_name,
                model_yaml_path=str(finetune_job.model_yaml_path) if finetune_job.model_yaml_path else None,
            )
            self.logger.info(f"Stopped monitoring job {job_id}. Final status: {finetune_job.status.value}")

    def _parse_training_progress(self, finetune_job: FinetuneJob, log_content: str):
        """
        Parse log content for training progress (epoch, loss).

        Args:
            finetune_job: Job to update
            log_content: New log content to parse
        """
        # One loss per epoch, from the per-epoch summary line ("Epoch X/Y -
        # Loss: Z"), so the plot has one point per epoch and the loss is
        # always the one belonging to the epoch shown beside it. Per-batch
        # lines are deliberately not read: pairing them with an epoch is
        # fiddly, and what made the display look stuck was tee's buffering,
        # not the reporting interval.
        #
        # "Starting epoch N of M" is read too, so the epoch counter advances
        # as soon as an epoch begins rather than when it ends.
        for cur, total in markers.EPOCH_START_RE.findall(log_content):
            finetune_job.current_epoch = int(cur)
            finetune_job.total_epochs = int(total)

        summary_matches = markers.EPOCH_SUMMARY_RE.findall(log_content)
        if summary_matches:
            cur, total, loss = summary_matches[-1]
            finetune_job.current_epoch = max(
                finetune_job.current_epoch, int(cur)
            )
            finetune_job.total_epochs = int(total)
            try:
                finetune_job.latest_loss = float(loss)
            except ValueError:
                pass

    def _add_finetuned_neuroglancer_layer(self, finetune_job: FinetuneJob, model_name: str):
        """
        Add (or replace) the finetuned model's neuroglancer layer.

        Mirrors run_model() from cellmap_flow/dashboard/services/launch.py:
        1. Create/update Job object in g.jobs
        2. Add neuroglancer ImageLayer with pre/post processing args

        Args:
            finetune_job: Job with inference_server_url set
            model_name: Layer name (e.g. "mito_finetuned_20240101_120000")
        """
        from cellmap_flow.globals import g
        from cellmap_flow.utils.web_utils import get_norms_post_args
        import neuroglancer

        server_url = finetune_job.inference_server_url
        if not server_url:
            # The trainer prints its completion marker before it starts the
            # server, and with auto-serve off never starts one. A layer made
            # now had the source zarr://None/..., and came with a g.jobs entry
            # whose host was None -- permanently, without auto-serve.
            self.logger.info(
                f"No inference server for {model_name} yet; the layer is added once it is up."
            )
            return

        # Create a Job object for the running server
        # A local run is a LocalJob, which has a process and no job_id.
        inference_job = LSFJob(
            job_id=getattr(finetune_job.lsf_job, "job_id", None) or "local",
            model_name=model_name
        )
        # The address viewers use (see jobs.spec.public_server_url); the
        # dashboard's own requests and the restart control keep server_url.
        from cellmap_flow.jobs.spec import public_server_url

        inference_job.host = public_server_url(server_url)
        inference_job.status = LSFJobStatus.RUNNING

        # Replace any old finetuned jobs for this base model. One assignment
        # of a new list, rather than filter-then-append on the shared one:
        # this runs on the monitor thread while request threads use g.jobs.
        g.jobs = [
            j for j in list(g.jobs)
            if not (hasattr(j, 'model_name') and j.model_name
                    and j.model_name.startswith(f"{finetune_job.model_name}_finetuned"))
        ] + [inference_job]
        self.logger.info(f"Added finetuned job to g.jobs: {model_name}")

        # Get pre/post processing args (same hash as other models)
        st_data = get_norms_post_args(g.input_norms, g.postprocess)

        if g.viewer is None:
            self.logger.error("g.viewer is None - neuroglancer not initialized yet")
            return

        # Lie about the model's voxel size so the layer overlays the raw at
        # the closest available scale (e.g. trained at 16nm but raw is
        # multiscale 6/12/24 -> tell neuroglancer it's 12nm).
        from cellmap_flow.io.multiscale import closest_raw_scale
        from cellmap_flow.viewer.layers import prediction_source

        override_scales = None
        try:
            output_voxel_size = tuple(
                finetune_job.params.get("output_voxel_size") or ()
            )
            dataset_path = getattr(g, "dataset_path", None)
            if output_voxel_size and dataset_path:
                closest = closest_raw_scale(dataset_path, output_voxel_size)
                if closest is not None and tuple(closest) != tuple(output_voxel_size):
                    override_scales = closest
                    self.logger.info(
                        f"Finetuned model '{model_name}' output_voxel_size="
                        f"{output_voxel_size} overridden to closest raw scale "
                        f"{closest} for viewer overlay"
                    )
        except Exception as e:
            self.logger.warning(
                f"Could not compute override scales for finetuned '{model_name}': {e}"
            )

        source_spec = prediction_source(
            inference_job.host, model_name, st_data, override_scales
        )
        self.logger.info(f"Adding neuroglancer layer: {model_name}")
        self.logger.info(f"  source: {source_spec}")

        with g.viewer.txn() as s:
            # Remove old finetuned layer if it exists (exact name match)
            old_layer_name = finetune_job.finetuned_model_name
            if old_layer_name and old_layer_name in s.layers:
                self.logger.info(f"Removing old finetuned layer: {old_layer_name}")
                del s.layers[old_layer_name]

            # Also remove by current name in case of re-add
            if model_name in s.layers:
                del s.layers[model_name]

            s.layers[model_name] = neuroglancer.ImageLayer(
                source=source_spec,
                shader=self._finetuned_shader(server_url),
            )

        # Update the stored name
        finetune_job.finetuned_model_name = model_name
        self.logger.info(f"Successfully added neuroglancer layer: {model_name}")

    def _finetuned_shader(self, server_url):
        """The same display range an ordinary model layer gets.

        This used to be hardcoded to range=[0, 255]. A sigmoid output lives in
        [0, 1], so the finetuned layer rendered as near-black however good the
        predictions were, while the identical model added through the normal
        path looked fine -- an unfair comparison built into the viewer.

        Falls back to the old fixed range only if the server cannot be asked.
        """
        from cellmap_flow.globals import g
        from cellmap_flow.utils.output_probe import output_display_range
        from cellmap_flow.viewer.raw import prediction_shader
        from cellmap_flow.utils.server_info import fetch_model_info

        try:
            info = fetch_model_info(server_url)
            steps = [
                p.to_dict() for p in (g.postprocess or []) if hasattr(p, "to_dict")
            ]
            value_range = output_display_range(steps, info.get("output_class"))
        except Exception as e:
            self.logger.warning(
                f"Could not work out a display range for the finetuned layer "
                f"({e}); falling back to 0-255."
            )
            value_range = (0.0, 255.0)
        return prediction_shader("red", value_range)

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
        matches = markers.SERVER_URL_RE.findall(log_content)
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
            full_log = finetune_job.log_file.read_text()
            iter_matches = markers.ITERATION_COMPLETE_RE.findall(full_log)
            if iter_matches:
                model_name = iter_matches[-1]
            else:
                model_name = f"{finetune_job.model_name}_finetuned"
            self._read_trainer_outputs(finetune_job, set_name=False)
        except Exception as e:
            self.logger.error(f"Failed to add finetuned model to neuroglancer: {e}", exc_info=True)
            return

        self._notify("on_server_ready", finetune_job, server_url, model_name)

    def _register_finetune_model_config(
        self, finetune_job: FinetuneJob, finetuned_model_name: str
    ):
        """Register a FinetuneModelConfig in g.models_config so it appears
        in the pipeline builder with auto-populated parameters."""
        from cellmap_flow.globals import g
        from cellmap_flow.models.models_config import FinetuneModelConfig

        params = finetune_job.params

        # The trainer's own YAML for this iteration says exactly what it
        # exported and on which base; registering from it keeps the pipeline
        # builder's model identical to the one the YAML serves. Without one,
        # fall back to the run's latest export and the base model's entry.
        entry = _yaml_model_entry(finetune_job.model_yaml_path)
        if entry is not None:
            export = {
                k: entry[k] for k in ("lora_adapter_path", "weights_path") if entry.get(k)
            }
            base_model_dict = entry.get("base_model")
        else:
            export = finetune_export_kwargs(finetune_job.output_dir, params)
            base_model_dict = None

        # Find the base model's to_dict() from g.models_config
        if base_model_dict is None and hasattr(g, "models_config") and g.models_config:
            for mc in g.models_config:
                if getattr(mc, "name", None) == finetune_job.model_name:
                    base_model_dict = mc.to_dict()
                    break

        if base_model_dict is None:
            # Fallback: reconstruct from job params
            base_model_dict = {"type": "fly"}
            if params.get("model_checkpoint"):
                base_model_dict["checkpoint_path"] = params["model_checkpoint"]
            for key in ("channels", "input_voxel_size", "output_voxel_size"):
                if key in params:
                    base_model_dict[key] = params[key]

        ft_config = FinetuneModelConfig(
            base_model=base_model_dict,
            name=finetuned_model_name,
            scale=params.get("scale"),
            **export,
        )

        if not hasattr(g, "models_config"):
            g.models_config = []

        # Remove any previous finetuned versions of the same base model
        base_model_name = finetune_job.model_name
        g.models_config = [
            mc
            for mc in g.models_config
            if not (
                hasattr(mc, "name")
                and mc.name.startswith(f"{base_model_name}_finetuned")
            )
        ]

        g.models_config.append(ft_config)
        self.logger.info(
            f"Registered FinetuneModelConfig: {finetuned_model_name}"
        )

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
        # Status markers, in the order they were printed: a restart that
        # follows a divergence in the same chunk leaves the job running, and
        # the reverse leaves it waiting.
        for marker in _STATUS_MARKER_RE.findall(log_content):
            if finetune_job.status in TERMINAL_STATUSES:
                break
            if marker == "TRAINING_DIVERGED":
                # Training produced NaN/Inf loss. The trainer then waits for a
                # restart, or exits if nothing is served yet (LSF then says so).
                self.logger.warning(f"Training diverged for job {finetune_job.job_id}")
                finetune_job.status = JobStatus.WAITING_FOR_RESTART
                finetune_job.latest_loss = None
            elif marker == "WAITING_FOR_RESTART":
                finetune_job.status = JobStatus.WAITING_FOR_RESTART
            else:  # RESTARTING_TRAINING: reset progress
                self.logger.info(f"Training restart detected for job {finetune_job.job_id}")
                finetune_job.current_epoch = 0
                finetune_job.latest_loss = None
                finetune_job.status = JobStatus.RUNNING
                finetune_job.inference_server_ready = False

        # Check for iteration complete marker - update neuroglancer layer.
        # Read full log in case the marker was in a previous chunk.
        try:
            full_log = finetune_job.log_file.read_text()
        except Exception:
            full_log = log_content
        iter_matches = markers.ITERATION_COMPLETE_RE.findall(full_log)
        # Only process new iteration-complete markers (ignore ones already handled).
        # After a restart, _processed_iteration_count stays at the old count so
        # previously-seen markers don't re-trigger inference_server_ready or
        # neuroglancer layer updates.
        if len(iter_matches) > finetune_job._processed_iteration_count:
            finetune_job._processed_iteration_count = len(iter_matches)

            # For in-process restarts, the inference server usually stays on the same
            # URL and does not emit a fresh CELLMAP_FLOW_SERVER_IP marker. Mark the
            # server as ready once we see a completed training iteration if URL exists.
            if finetune_job.inference_server_url:
                finetune_job.inference_server_ready = True

            new_model_name = iter_matches[-1]
            self._read_trainer_outputs(finetune_job, set_name=False)
            if new_model_name != finetune_job.finetuned_model_name:
                self.logger.info(f"New training iteration complete: {new_model_name}")
                self._notify("on_iteration_complete", finetune_job, new_model_name)
                # Whatever the listeners managed -- without a server no layer
                # is added -- show the new name, and don't retry every poll.
                finetune_job.finetuned_model_name = new_model_name

    def _read_trainer_outputs(self, finetune_job: FinetuneJob, set_name: bool = True):
        """Take the latest iteration's model name and serving YAML from the log.

        The trainer prints "FINETUNED_MODEL_YAML: <path>" and then
        "TRAINING_ITERATION_COMPLETE: <name>" for every iteration it
        finishes. ``set_name=False`` leaves finetuned_model_name alone, for the
        monitor, which uses the old name to replace the old viewer layer.
        """
        try:
            log_text = finetune_job.log_file.read_text()
        except OSError:
            return
        name, yaml_path = trainer_outputs_from_log(log_text)
        if set_name and name:
            finetune_job.finetuned_model_name = name
        if yaml_path:
            finetune_job.model_yaml_path = Path(yaml_path)

    def complete_job(self, finetune_job: FinetuneJob):
        """
        Post-training actions after job completes successfully.

        1. Verify adapter files exist
        2. Take the model name and serving YAML the trainer reported
        3. Update job status and metadata

        Args:
            finetune_job: The completed job

        Raises:
            RuntimeError: If adapter files missing or registration fails
        """
        job_id = finetune_job.job_id
        self.logger.info(f"Running post-completion for job {job_id}...")

        # === Verify the training export exists ===

        export = finetune_export_kwargs(finetune_job.output_dir, finetune_job.params)
        if "weights_path" in export:
            # Full finetune (--lora-r 0): a single state dict, no adapter dir.
            weights_file = Path(export["weights_path"])
            if not weights_file.exists():
                raise RuntimeError(
                    f"Training completed but full-finetune weights not found: {weights_file}"
                )
            self.logger.info(f"Verified full-finetune weights exist: {weights_file}")
        else:
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

            self.logger.info(f"Verified LoRA adapter files exist in {adapter_path}")

        # === The name and YAML the trainer gave the result ===
        #
        # This used to make up its own: the job's creation time instead of
        # the iteration's, and a sanitized model name instead of the
        # trainer's. So the YAML it looked for never existed, it always
        # generated a second one (with the dashboard's current norms rather
        # than the training ones), and metadata.json named a model that
        # neither the viewer layer nor the registered config did. The trainer
        # prints both, and is the only thing that knows them.
        self._read_trainer_outputs(finetune_job)
        finetuned_model_name = finetune_job.finetuned_model_name
        yaml_path = finetune_job.model_yaml_path
        if yaml_path is None:
            self.logger.warning(
                f"Job {job_id}: the trainer reported no serving YAML (see its "
                f"log for why); the weights are in {finetune_job.output_dir}."
            )
        else:
            self.logger.info(f"Serving YAML for {finetuned_model_name}: {yaml_path}")

        # === Update metadata file with completion info ===

        metadata_file = finetune_job.output_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["completed_at"] = datetime.now().isoformat()
            metadata["status"] = "COMPLETED"
            metadata["finetuned_model_name"] = finetuned_model_name
            metadata["model_yaml_path"] = str(yaml_path) if yaml_path else None
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
                finetune_job.cancel_requested = True
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
        # A snapshot: monitor threads and rehydration add jobs concurrently.
        return [
            status for status in (self.get_job_status(job_id) for job_id in list(self.jobs))
            if status is not None
        ]

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

        # Only a trainer that is alive and waiting can take a restart. A
        # COMPLETED job has exited: nothing reads the request, and the monitor
        # that would have seen it through has stopped, so it used to sit at
        # RUNNING forever. WAITING_FOR_RESTART also covers a later iteration
        # that diverged: its server is up but not marked ready, and the
        # restart the user needed was refused. A RUNNING job whose server is
        # up is a trainer that predates the WAITING_FOR_RESTART marker.
        waiting = job.status == JobStatus.WAITING_FOR_RESTART
        serving = job.status == JobStatus.RUNNING and job.inference_server_ready
        if not (waiting or serving):
            raise ValueError(
                f"Job {job_id} is in state {job.status.value} - can only restart a "
                f"job that is waiting for a restart (its training iteration has "
                f"finished or diverged)"
            )

        signal_data = {
            "restart": True,
            "timestamp": datetime.now().isoformat(),
            "params": updated_params or {}
        }

        # 1. Send restart request to running inference server (primary path)
        signal_write_mode = "http_control"
        write_t0 = time.perf_counter()
        http_error = None
        if job.inference_server_url:
            try:
                control_url = job.inference_server_url.rstrip("/") + "/__control__/restart"
                restart_token = read_restart_token(job.output_dir)
                if restart_token is None:
                    raise RuntimeError(f"No restart token in {job.output_dir}")
                headers = {TOKEN_HEADER: restart_token}
                response = requests.post(control_url, json=signal_data, headers=headers, timeout=5)
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

        # 2. Fallback to signal file if HTTP control endpoint is unavailable
        if http_error is not None:
            signal_write_mode = "file_signal_fallback"
            signal_file = job.output_dir / "restart_signal.json"
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            self.logger.info(f"Wrote fallback restart signal to {signal_file}")
        write_elapsed = time.perf_counter() - write_t0

        # 3. Reset training progress (keep inference server info)
        job.current_epoch = 0
        job.latest_loss = None
        job.status = JobStatus.RUNNING
        job.inference_server_ready = False

        # 4. Update stored params
        if updated_params:
            job.params.update(updated_params)

        total_elapsed = time.perf_counter() - restart_t0
        self.logger.info(
            f"Restart signal timings for job {job_id}: "
            f"write={write_elapsed:.2f}s "
            f"mode={signal_write_mode} total={total_elapsed:.2f}s"
        )
        self.logger.info(f"Job {job_id} restart request sent, waiting for CLI to pick it up")

        return job
