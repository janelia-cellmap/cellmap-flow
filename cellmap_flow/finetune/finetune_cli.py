#!/usr/bin/env python
"""
Command-line interface for LoRA finetuning.

Usage:
    python -m cellmap_flow.finetune.finetune_cli \
        --model-checkpoint /path/to/checkpoint \
        --corrections corrections.zarr \
        --output-dir output/fly_organelles_v1.1

    # With custom settings
    python -m cellmap_flow.finetune.finetune_cli \
        --model-checkpoint /path/to/checkpoint \
        --corrections corrections.zarr \
        --output-dir output/fly_organelles_v1.1 \
        --lora-r 8 \
        --batch-size 8 \
        --num-epochs 20 \
        --learning-rate 2e-4
"""

import argparse
import gc
import json
import logging
import socket
import sys
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from cellmap_flow.models.models_config import FlyModelConfig, DaCapoModelConfig, HuggingFaceModelConfig, ModelConfig
from cellmap_flow.utils.ds import _is_remote_path
from cellmap_flow.utils.restart_token import read_or_create_restart_token
from cellmap_flow.finetune.finetuned_model_templates import FINETUNED_MODEL_YAML_MARKER
from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora
from cellmap_flow.finetune.model_loading import (
    decode_model_entry,
    load_trainable_model,
    model_config_from_entry,
    root_base_model_dict,
)
from cellmap_flow.finetune.virtual_dataset import create_dataloader
from cellmap_flow.finetune.lora_trainer import LoRAFinetuner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


class RestartController:
    """In-memory restart control shared between training loop and server endpoint."""

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._pending = None

    def request_restart(self, payload: Optional[dict]) -> bool:
        signal_data = {
            "restart": True,
            "timestamp": datetime.now().isoformat(),
            "params": {},
        }
        if isinstance(payload, dict):
            if "timestamp" in payload and payload["timestamp"]:
                signal_data["timestamp"] = payload["timestamp"]
            if isinstance(payload.get("params"), dict):
                signal_data["params"] = payload["params"]

        with self._lock:
            self._pending = signal_data
            self._event.set()
        return True

    def get_if_triggered(self) -> Optional[dict]:
        if not self._event.is_set():
            return None
        with self._lock:
            signal_data = self._pending
            self._pending = None
            self._event.clear()
        return signal_data


def _wait_for_port_ready(host: str, port: int, timeout_s: float = 30.0, interval_s: float = 0.1) -> bool:
    """Wait until a TCP port is accepting connections."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.5)):
                return True
        except OSError:
            time.sleep(interval_s)
    return False


def _start_inference_server_background(
    args, model_config: ModelConfig, trained_model, restart_controller: Optional[RestartController] = None
):
    """
    Start inference server in a background daemon thread.

    The server shares the same model object, so retraining updates weights
    automatically without needing to restart the server.

    Args:
        args: Command-line arguments
        model_config: Base model configuration
        trained_model: The trained LoRA model

    Returns:
        (thread, port) tuple
    """
    logger.info("=" * 60)
    logger.info("Starting inference server with finetuned model...")
    logger.info("=" * 60)

    startup_t0 = time.perf_counter()

    # Clear GPU cache from training
    cleanup_t0 = time.perf_counter()
    logger.info("Clearing GPU cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    cleanup_elapsed = time.perf_counter() - cleanup_t0

    # Validate serve data path
    if not args.serve_data_path:
        raise ValueError("--serve-data-path is required when --auto-serve is enabled")

    if not _is_remote_path(args.serve_data_path) and not Path(args.serve_data_path).exists():
        raise ValueError(f"Data path not found: {args.serve_data_path}")

    # Use the already-trained model
    logger.info("Using trained LoRA model for inference...")

    from cellmap_flow.models.models_config import _get_device
    device = _get_device()
    trained_model.eval()
    logger.info(f"Model set to eval mode on {device}")

    # Replace the model in the config with our finetuned version
    model_config.config.model = trained_model

    # Start server
    from cellmap_flow.server import CellMapFlowServer
    from cellmap_flow.utils.web_utils import get_free_port

    setup_t0 = time.perf_counter()
    logger.info(f"Creating server for dataset: {model_config.name}_finetuned")
    restart_callback = restart_controller.request_restart if restart_controller is not None else None
    restart_token = (
        read_or_create_restart_token(args.output_dir) if restart_callback is not None else None
    )
    server = CellMapFlowServer(
        args.serve_data_path,
        model_config,
        restart_callback=restart_callback,
        restart_token=restart_token,
    )

    # Get port
    port = args.serve_port if args.serve_port != 0 else get_free_port()

    # Start in daemon thread (server.run() prints CELLMAP_FLOW_SERVER_IP marker automatically)
    server_thread = threading.Thread(
        target=server.run,
        kwargs={'port': port, 'debug': False},
        daemon=True
    )
    server_thread.start()
    setup_elapsed = time.perf_counter() - setup_t0

    wait_t0 = time.perf_counter()
    server_ready = _wait_for_port_ready("127.0.0.1", port)
    wait_elapsed = time.perf_counter() - wait_t0

    host_url = f"http://{socket.gethostname()}:{port}"
    total_elapsed = time.perf_counter() - startup_t0
    logger.info("=" * 60)
    if server_ready:
        logger.info(f"Inference server port is ready on 127.0.0.1:{port}")
    else:
        logger.warning(f"Inference server did not become ready within timeout on 127.0.0.1:{port}")
    logger.info(f"Inference server running at {host_url}")
    logger.info(
        f"Startup timings (s): cleanup={cleanup_elapsed:.2f}, setup={setup_elapsed:.2f}, "
        f"wait_for_bind={wait_elapsed:.2f}, total={total_elapsed:.2f}"
    )
    logger.info("Server is running in background. Watching for restart signals...")
    logger.info("=" * 60)

    return server_thread, port


def _wait_for_restart_signal(
    signal_file: Optional[Path],
    check_interval: float = 1.0,
    restart_controller: Optional[RestartController] = None,
):
    """
    Watch for a restart signal file. Blocks until signal appears.

    Prefers in-memory restart events from the control endpoint, and
    falls back to a signal file for backward compatibility.

    Args:
        signal_file: Optional path to watch for legacy signal file
        check_interval: Seconds between checks

    Returns:
        Dict with restart parameters, or None if signal file is malformed
    """
    logger.info(f"Watching for restart signal (controller + file fallback: {signal_file})")

    while True:
        if restart_controller is not None:
            in_memory_signal = restart_controller.get_if_triggered()
            if in_memory_signal is not None:
                logger.info(f"Restart signal received via HTTP control endpoint: {in_memory_signal}")
                return in_memory_signal

        if signal_file and signal_file.exists():
            try:
                with open(signal_file) as f:
                    signal_data = json.load(f)
                signal_file.unlink()  # Remove signal file
                logger.info(f"Restart signal received: {signal_data}")
                return signal_data
            except Exception as e:
                logger.error(f"Error reading restart signal: {e}")
                # Remove malformed signal file
                try:
                    signal_file.unlink()
                except OSError:
                    pass
                return None
        time.sleep(check_interval)


# What a restart may change: training settings only. The model, the data and
# every path stay as launched, so a restart request cannot point the job at
# other files. Matches the dashboard's RESTART_PASSTHROUGH_KEYS, plus the
# distillation_all_voxels flag it derives from distillation_scope.
RESTARTABLE_ARGS = frozenset(
    {
        "lora_r", "lora_alpha", "num_epochs", "batch_size", "learning_rate",
        "loss_type", "label_smoothing", "distillation_lambda",
        "distillation_all_voxels", "margin", "balance_classes", "augment",
        "mask_unannotated", "gradient_accumulation_steps", "num_workers",
        "no_augment", "no_mixed_precision", "patch_shape", "output_type",
        "select_channel", "offsets",
    }
)


def _as_bool(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


def _as_offsets(value):
    # --offsets is a JSON string. A restart may carry the list itself, which
    # _build_target_transform's json.loads() would then reject.
    if isinstance(value, str):
        json.loads(value)
        return value
    return json.dumps(value)


def _as_shape(value):
    shape = [int(v) for v in value]
    if len(shape) != 3:
        raise ValueError(f"expected three values, got {value!r}")
    return shape


def _one_of(*choices):
    def convert(value):
        if value not in choices:
            raise ValueError(f"{value!r} is not one of {list(choices)}")
        return value
    return convert


# How each restartable setting is read: the argparse destination it sets and
# the conversion applied to the requested value. Most keys are the
# destination itself. The dashboard's "augment" is the inverse of the CLI's
# --no-augment, and used to be dropped by a hasattr(args, key) filter, so a
# restart could never switch augmentation on or off.
_RESTART_ARG_CONVERTERS = {
    "lora_r": ("lora_r", int),
    "lora_alpha": ("lora_alpha", int),
    "num_epochs": ("num_epochs", int),
    "batch_size": ("batch_size", int),
    "learning_rate": ("learning_rate", float),
    "loss_type": ("loss_type", _one_of("dice", "bce", "combined", "mse", "margin")),
    "label_smoothing": ("label_smoothing", float),
    "distillation_lambda": ("distillation_lambda", float),
    "distillation_all_voxels": ("distillation_all_voxels", _as_bool),
    "margin": ("margin", float),
    "balance_classes": ("balance_classes", _as_bool),
    "augment": ("no_augment", lambda value: not _as_bool(value)),
    "mask_unannotated": ("mask_unannotated", _as_bool),
    "gradient_accumulation_steps": ("gradient_accumulation_steps", int),
    "num_workers": ("num_workers", int),
    "no_augment": ("no_augment", _as_bool),
    "no_mixed_precision": ("no_mixed_precision", _as_bool),
    "patch_shape": ("patch_shape", _as_shape),
    "output_type": (
        "output_type", _one_of("binary", "binary_broadcast", "affinities", "distance")
    ),
    "select_channel": ("select_channel", int),
    "offsets": ("offsets", _as_offsets),
}
assert set(_RESTART_ARG_CONVERTERS) == RESTARTABLE_ARGS


def _apply_restart_params(args, signal_data: dict):
    """
    Update args with parameters from restart signal and persist to metadata.json.

    Args:
        args: argparse Namespace to update
        signal_data: Dict from restart signal file
    """
    params = signal_data.get("params", {}) or {}
    refused = sorted(set(params) - RESTARTABLE_ARGS)
    if refused:
        logger.warning(f"Ignoring restart parameters that restarts cannot change: {refused}")
    changed = False
    # What was applied, under the name the request used, for metadata.json.
    # Refused keys reach neither args nor metadata.json.
    recorded = {}
    for key, value in params.items():
        if key not in RESTARTABLE_ARGS or value is None:
            continue
        dest, convert = _RESTART_ARG_CONVERTERS[key]
        if not hasattr(args, dest):
            logger.warning(f"Restart parameter {key!r} has no matching training setting; ignoring it.")
            continue
        try:
            new_value = convert(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"Ignoring restart parameter {key}={value!r}: {e}")
            continue
        old_value = getattr(args, dest)
        setattr(args, dest, new_value)
        recorded[key] = _as_bool(value) if key == "augment" else new_value
        if old_value != new_value:
            logger.info(f"Updated {dest}: {old_value} -> {new_value}")
            changed = True

    # alpha is what sets LoRA's step size: peft scales the adapter by
    # lora_alpha / r. Submit derives alpha = 2 * r, but a restart only carries
    # lora_r -- so raising the rank from 8 to 64 while alpha stayed at 16 cut
    # the effective update to an eighth, and produced a loss curve that looks
    # reassuringly smooth because very little is happening per step.
    if "lora_r" in recorded and "lora_alpha" not in recorded:
        derived = int(recorded["lora_r"]) * 2
        if getattr(args, "lora_alpha", None) != derived:
            logger.info(
                f"Updated lora_alpha: {getattr(args, 'lora_alpha', None)} -> "
                f"{derived} (held at 2x rank so the adapter scaling does not "
                f"change when you change the rank)"
            )
            args.lora_alpha = derived
            recorded["lora_alpha"] = derived
            changed = True

    # Persist updated params to metadata.json
    if changed and hasattr(args, 'output_dir') and args.output_dir:
        metadata_file = Path(args.output_dir) / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                if "params" in metadata:
                    for key, value in recorded.items():
                        if key in metadata["params"]:
                            metadata["params"][key] = value
                metadata["last_restart_at"] = signal_data.get("timestamp")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info("Updated metadata.json with restart params")
            except Exception as e:
                logger.warning(f"Failed to update metadata.json: {e}")


def _is_peft_model(model) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return False
    return isinstance(model, PeftModel)


def _reset_for_restart(lora_model, args, initial_state=None):
    """Put the model back where training started, for the next iteration.

    LoRA: unload the adapter and wrap a fresh one around the base (which is
    how the rank can change on restart). Full finetune: load the starting
    weights back; before, the weights carried on from the previous iteration,
    NaNs included when it had diverged, so a full finetune never really
    restarted. The model object -- which the inference server shares -- is
    built once, so a restart cannot switch between the two kinds.

    Returns the model to train next.
    """
    if _is_peft_model(lora_model):
        logger.info("Resetting LoRA adapter weights for fresh restart...")
        if args.lora_r <= 0:
            logger.warning("Restart asked for lora_r=0 (full finetune) but this job trains a LoRA adapter; "
                           "submit a new job for that. Keeping the current adapter setup.")
            args.lora_r = max(1, int(lora_model.peft_config['default'].r))
        base = lora_model.unload()
        lora_model = wrap_model_with_lora(
            base,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_min_channels=args.lora_min_channels,
        )
    else:
        if args.lora_r > 0:
            # The mirror image of the case above. Left alone, args.lora_r > 0
            # made the next iteration's YAML point at a lora_adapter/ this job
            # never writes.
            logger.warning(f"Restart asked for LoRA rank {args.lora_r} but this job is a full finetune; "
                           "submit a new job for that. Keeping the full finetune.")
            args.lora_r = 0
        if initial_state is not None:
            logger.info("Resetting the full finetune to its starting weights for a fresh restart...")
            lora_model.load_state_dict(initial_state)

    lora_model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Restarting training from the starting weights...")
    return lora_model


def _finetuned_model_name(model_config, timestamp) -> str:
    return f"{model_config.name}_finetuned_{timestamp}"


def _models_dir(args) -> Path:
    """Where the served-model YAMLs go: --models-dir, else the session's models/.

    A dashboard run's output directory is <session>/runs/<name>, so the
    session is two levels up. This used to take three, which put the YAMLs in
    <base>/models, outside the session and shared by all of them, and turned
    a headless --output-dir /nrs/x/run into /models: a permission error after
    training had succeeded, reported as "Training failed". A run that is not
    under a runs/ directory keeps its YAMLs in its own models/.
    """
    if getattr(args, "models_dir", None):
        return Path(args.models_dir)
    output_dir = Path(args.output_dir)
    if output_dir.parent.name == "runs":
        return output_dir.parent.parent / "models"
    return output_dir / "models"


def _generate_model_files(args, model_config, timestamp, is_lora: Optional[bool] = None):
    """
    Generate YAML config file after training.

    Args:
        args: Command-line arguments
        model_config: Model configuration
        timestamp: Timestamp string for naming
        is_lora: Whether the trained model is a LoRA adapter (else a full
            finetune). Pass what the model is; ``args.lora_r`` is only the
            fallback, because a restart can change it without changing the
            model.

    Returns:
        (finetuned_model_name, yaml_path) tuple
    """
    if is_lora is None:
        is_lora = args.lora_r > 0
    from cellmap_flow.finetune.finetuned_model_templates import (
        generate_finetuned_model_yaml
    )

    finetuned_model_name = _finetuned_model_name(model_config, timestamp)

    output_dir_path = Path(args.output_dir)
    models_dir = _models_dir(args)
    models_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Generating model config for {finetuned_model_name} in {models_dir}...")

    # The raw data the model is served on: the manifest's, which is what it
    # was trained on; else the first correction zarr's attrs (which are also
    # a fallback source of normalization/postprocessing metadata below); else
    # the path the job serves. There is no placeholder: a YAML pointing at a
    # made-up path is worse than none, and generate_finetuned_model_yaml
    # refuses to write one.
    corrections_path = Path(args.corrections)
    data_path = None
    try:
        from cellmap_flow.finetune.virtual_dataset import read_manifest

        data_path = (read_manifest(str(corrections_path)) or {}).get("raw_dataset_path")
    except Exception as _e:
        logger.warning(f"Could not read the manifest in {corrections_path}: {_e}")
    zarr_dirs = sorted(corrections_path.glob("*.zarr"))
    zattrs_input_norm = None
    zattrs_postprocess = None
    if zarr_dirs:
        zattrs_file = zarr_dirs[0] / ".zattrs"
        if zattrs_file.exists():
            with open(zattrs_file) as f:
                metadata = json.load(f)
                data_path = data_path or metadata.get("dataset_path")
                zattrs_input_norm = metadata.get("input_norm")
                zattrs_postprocess = metadata.get("postprocess")

    if not data_path:
        logger.warning("Could not extract data_path from corrections, using serve_data_path")
        data_path = getattr(args, "serve_data_path", None)

    # Bake the training-time input_norm/postprocess into the generated yaml
    # so the served finetuned model gets queried with the same normalization
    # (and produces output through the same postprocessing, e.g.
    # SigmoidPostprocessor) the adapter was trained on. Without this,
    # training-vs-inference scale mismatch silently destroys finetuning
    # quality.
    #
    # Two correction workflows exist and store this differently:
    #  - the manifest-based workflow (_virtual_sources.json, written by
    #    yaml_crops.py) -- checked first, since it's kept fresh on restart.
    #  - the annotation-volume/MinIO workflow (stored directly on the
    #    correction zarr's own .zattrs, written by annotation_core.py /
    #    finetune_utils.create_annotation_volume_zarr) -- used as a fallback
    #    when no manifest exists.
    train_input_norm = None
    train_postprocess = None
    try:
        from cellmap_flow.finetune.virtual_dataset import read_manifest

        manifest = read_manifest(str(corrections_path)) or {}
        train_input_norm = manifest.get("input_norm")
        train_postprocess = manifest.get("postprocess")
    except Exception as _e:
        logger.warning(f"Could not load manifest from {corrections_path}: {_e}")

    train_input_norm = train_input_norm or zattrs_input_norm
    train_postprocess = train_postprocess or zattrs_postprocess

    if train_input_norm or train_postprocess:
        json_data = {
            "input_norm": train_input_norm or {},
            "postprocess": train_postprocess or {},
        }
    else:
        json_data = None
        logger.warning(
            "Could not find training input_norm/postprocess in either the "
            "corrections manifest or the correction zarr's own attrs. "
            "Generated finetuned yaml will lack normalization metadata."
        )

    yaml_path = generate_finetuned_model_yaml(
        lora_adapter_path=str(output_dir_path / "lora_adapter") if is_lora else None,
        weights_path=None if is_lora else str(output_dir_path / "full_finetune" / "model_state_dict.pt"),
        # A LoRA adapter was trained on top of the whole base, finetune
        # layers included, and is served on top of it. Full weights replace
        # every parameter, so they only need the base's module tree -- and on
        # a finetune base that tree would be a PeftModel their names no
        # longer match.
        base_model_dict=model_config.to_dict() if is_lora else root_base_model_dict(model_config),
        model_name=finetuned_model_name,
        output_path=models_dir / f"{finetuned_model_name}.yaml",
        data_path=data_path,
        json_data=json_data,
    )
    logger.info(f"Generated YAML: {yaml_path}")

    return finetuned_model_name, yaml_path


def _build_target_transform(args, model_config):
    """Build a TargetTransform based on CLI args."""
    from cellmap_flow.finetune.target_transforms import (
        BinaryTargetTransform,
        BroadcastBinaryTargetTransform,
        AffinityTargetTransform,
        DistanceTargetTransform,
    )

    output_type = args.output_type
    num_channels = model_config.config.output_channels

    if output_type == "binary":
        if num_channels > 1 and args.select_channel is None:
            logger.warning(
                f"Model has {num_channels} output channels but --output-type is 'binary' "
                f"and --select-channel is not set. Consider using --select-channel or "
                f"--output-type binary_broadcast."
            )
        return BinaryTargetTransform()

    elif output_type == "binary_broadcast":
        logger.info(f"Broadcasting binary target to {num_channels} channels")
        return BroadcastBinaryTargetTransform(num_channels)

    elif output_type == "affinities":
        offsets = None

        # Try CLI arg first
        if args.offsets:
            offsets = json.loads(args.offsets)

        # Try reading from model script
        if offsets is None and args.model_script:
            offsets = _read_offsets_from_script(args.model_script)

        if offsets is None:
            raise ValueError(
                "Affinity output type requires offsets. Provide --offsets as a JSON list "
                "(e.g. '[[1,0,0],[0,1,0],[0,0,1]]') or define an 'offsets' variable in "
                "the model script."
            )

        if len(offsets) > num_channels:
            raise ValueError(
                f"Number of offsets ({len(offsets)}) exceeds model output channels "
                f"({num_channels})."
            )

        if len(offsets) < num_channels:
            logger.info(
                f"Model has {num_channels} output channels but only {len(offsets)} affinity offsets. "
                f"Remaining {num_channels - len(offsets)} channels (e.g. LSDs) will be masked out."
            )

        logger.info(f"Using affinity target transform with {len(offsets)} offsets: {offsets}")
        return AffinityTargetTransform(offsets, num_channels=num_channels)

    elif output_type == "distance":
        if args.loss_type != "bce":
            raise ValueError(
                "--output-type distance produces soft targets in [0, 1]; only "
                "--loss-type bce (BCE with logits) is defined for them. Margin and "
                "dice assume hard labels, and mse is applied to raw logits."
            )
        if args.label_smoothing > 0:
            logger.warning(
                "Label smoothing is meaningless on a soft distance target; "
                f"ignoring --label-smoothing {args.label_smoothing}."
            )
            args.label_smoothing = 0.0
        if getattr(args, "mask_unannotated", False):
            logger.warning(
                "--output-type distance with --mask-unannotated (sparse/scribble "
                "annotations): a distance transform needs dense 3D labels, and "
                "voxels next to unannotated ones are left out of the loss, so "
                "very little of a scribble session will be supervised. Use "
                "--output-type binary --loss-type margin for scribbles."
            )
        logger.info(
            f"Using distance target transform (sigma={args.distance_sigma} voxels, "
            f"broadcast to {num_channels} channel(s))"
        )
        return DistanceTargetTransform(args.distance_sigma, num_channels=num_channels)

    else:
        raise ValueError(f"Unknown output type: {output_type}")


def _read_offsets_from_script(script_path):
    """Try to read an 'offsets' variable from a model script via AST parsing."""
    import ast

    try:
        with open(script_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "offsets":
                        return ast.literal_eval(node.value)
    except Exception as e:
        logger.debug(f"Could not read offsets from {script_path}: {e}")

    return None


def _model_config_from_args(args) -> ModelConfig:
    """The ModelConfig the command line describes."""
    if args.model_entry:
        # The model's own to_dict(), as the job manager passes it for the
        # types that have no dedicated flags (cellmap, finetune).
        entry = decode_model_entry(args.model_entry)
        logger.info(f"Using model entry of type {entry.get('type')!r}")
        return model_config_from_entry(entry, name=args.model_name)
    if args.model_script:
        from cellmap_flow.models.models_config import ScriptModelConfig
        logger.info(f"Using script-based model: {args.model_script}")
        return ScriptModelConfig(
            script_path=args.model_script,
            name=args.model_name or "script_model"
        )
    if args.model_type == "script":
        raise ValueError("For script models, --model-script is required")
    if args.model_type == "fly":
        if not args.model_checkpoint:
            raise ValueError(
                "For fly models, either --model-checkpoint or --model-script must be provided"
            )
        return FlyModelConfig(
            checkpoint_path=args.model_checkpoint,
            channels=args.channels,
            input_voxel_size=tuple(args.input_voxel_size),
            output_voxel_size=tuple(args.output_voxel_size),
            name=args.model_name,
        )
    if args.model_type == "dacapo":
        if not args.model_checkpoint:
            raise ValueError("For dacapo models, --model-checkpoint is required")
        checkpoint_path = Path(args.model_checkpoint)
        iteration = int(checkpoint_path.stem.split('_')[-1])
        run_name = checkpoint_path.parent.name
        return DaCapoModelConfig(
            run_name=run_name,
            iteration=iteration,
        )
    if args.model_type == "huggingface":
        if not args.repo:
            raise ValueError("For huggingface models, --repo is required")
        return HuggingFaceModelConfig(
            repo=args.repo,
            revision=args.revision,
            name=args.model_name,
        )
    if args.model_type == "cellmap":
        if not args.model_folder:
            raise ValueError("For cellmap models, --model-folder (or --model-entry) is required")
        from cellmap_flow.models.models_config import CellMapModelConfig
        return CellMapModelConfig(folder_path=args.model_folder, name=args.model_name)
    if args.model_type == "finetune":
        raise ValueError(
            "For finetune models, --model-entry is required: the model's entry "
            "(type: finetune, base_model, lora_adapter_path or weights_path) as JSON"
        )
    raise ValueError(f"Unknown model type: {args.model_type}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Finetune CellMap-Flow models with LoRA using user corrections"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="fly",
        choices=["fly", "dacapo", "huggingface", "script", "cellmap", "finetune"],
        help="Model type (fly, dacapo, huggingface, script, cellmap, or finetune). "
             "cellmap takes --model-folder; finetune (continue from a finetuned "
             "model) takes --model-entry."
    )
    parser.add_argument(
        "--model-entry",
        type=str,
        default=None,
        help="The model as its model entry (what ModelConfig.to_dict() gives, "
             "the same shape as a models: entry in a YAML), as JSON or "
             "encode_to_str()'d JSON. Takes precedence over the other model flags."
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        default=None,
        help="Folder of a cellmap model (for --model-type cellmap)"
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to model checkpoint (optional - can train from scratch)"
    )
    parser.add_argument(
        "--model-script",
        type=str,
        required=False,
        default=None,
        help="Path to model script (alternative to checkpoint)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=False,
        default=None,
        help="HuggingFace model repository (e.g., janelia-cellmap/mito_aff_unet_setup_16)"
    )
    parser.add_argument(
        "--revision",
        type=str,
        required=False,
        default=None,
        help="HuggingFace model revision (optional)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (for filtering corrections)"
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["mito"],
        help="Model output channels"
    )
    parser.add_argument(
        "--input-voxel-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Input voxel size (Z Y X)"
    )
    parser.add_argument(
        "--output-voxel-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Output voxel size (Z Y X)"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        # Low rank is itself the anti-forgetting mechanism here: this is
        # correcting a model that is mostly right, so the adapter wants just
        # enough capacity to fix the bad regions and not enough to rewrite
        # the good ones.
        help="LoRA rank (default: 8). 0 = full finetune: every parameter trainable, no adapter; "
             "exports full_finetune/model_state_dict.pt instead of lora_adapter/."
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha scaling (default: twice --lora-r)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
    )
    parser.add_argument(
        "--lora-min-channels",
        type=int,
        default=0,
        help="Skip LoRA on layers narrower than this on either side. "
             "Narrow full-resolution layers are where the adapter is expensive "
             "and nearly parameter-free: on mito-aff-unet-setup-16, 96 skips "
             "7 of 19 layers (~1%% of adapter params) for a 1.7x faster step. "
             "(default: 0, adapt every layer)"
    )

    # Data arguments
    parser.add_argument(
        "--corrections",
        type=str,
        required=True,
        help="Path to corrections.zarr directory"
    )
    parser.add_argument(
        "--patch-shape",
        type=int,
        nargs=3,
        default=None,
        help="Patch shape for training (Z Y X). Default: None (use full corrections)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (random flips, XY rotations, brightness "
             "and noise). Patch-center jitter is always applied and is not "
             "affected. Augmentation pays off when a run revisits the same "
             "patches many times; below a few hundred gradient steps it mostly "
             "adds variance, which is why the dashboard defaults it off."
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and adapter"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory for the generated serving YAMLs (default: <session>/models "
             "when --output-dir is <session>/runs/<name>, else <output-dir>/models)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="combined",
        choices=["dice", "bce", "combined", "mse", "margin"],
        help="Loss function (default: combined)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (e.g., 0.1 maps targets from 0/1 to 0.05/0.95). "
             "Helps preserve gradual distance-like outputs. (default: 0.0)"
    )
    parser.add_argument(
        "--distillation-lambda",
        type=float,
        default=0.0,
        help="Teacher distillation weight. Keeps model close to base on unlabeled voxels. "
             "0.0=disabled, try 0.5-1.0 for sparse scribbles. (default: 0.0)"
    )
    parser.add_argument(
        "--distillation-all-voxels",
        action="store_true",
        help="Apply distillation loss on all voxels instead of only unlabeled voxels. (default: unlabeled only)"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Margin threshold for margin loss. "
             "Foreground must exceed 1-margin, background must stay below margin. (default: 0.3)"
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance fg/bg loss contribution so each class is weighted equally, "
             "regardless of scribble voxel counts. Helps prevent foreground overprediction. (default: off)"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision (FP16) training"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4)"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not write TensorBoard event files to <output-dir>/tensorboard "
             "(default: write them; view with `tensorboard --logdir <training dir>`)"
    )

    # Resuming
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Auto-serve arguments
    parser.add_argument(
        "--auto-serve",
        action="store_true",
        help="Automatically start inference server after training completes"
    )
    parser.add_argument(
        "--serve-data-path",
        type=str,
        default=None,
        help="Dataset path for inference server (required if --auto-serve is used)"
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=0,
        help="Port for inference server (0 for auto-assignment)"
    )
    parser.add_argument(
        "--mask-unannotated",
        action="store_true",
        help="Enable masked loss for sparse annotations (0=ignore, 1=bg, 2+=fg)"
    )

    # Output type and target transform arguments
    parser.add_argument(
        "--output-type",
        type=str,
        default="binary",
        choices=["binary", "binary_broadcast", "affinities", "distance"],
        help="How to generate training targets from annotations. "
             "'binary': single-channel fg/bg (use with --select-channel for multi-channel models). "
             "'binary_broadcast': broadcast binary target to all output channels. "
             "'affinities': compute affinity targets from instance labels (requires offsets). "
             "'distance': soft signed-distance target (tanh(d/sigma)+1)/2 for models trained "
             "the fly_organelles way, e.g. the cellmap *_distance_* repos; requires --loss-type bce. "
             "(default: binary)"
    )
    parser.add_argument(
        "--distance-sigma",
        type=float,
        default=6.0,
        help="tanh scale in output voxels for --output-type distance. The cellmap "
             "distance models were trained with 6. (default: 6.0)"
    )
    parser.add_argument(
        "--select-channel",
        type=int,
        default=None,
        help="Select a single channel from multi-channel model output for binary training. "
             "Only used with --output-type binary. (default: None, use all channels)"
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default=None,
        help="JSON list of [dz,dy,dx] offsets for affinity target generation. "
             "Example: '[[1,0,0],[0,1,0],[0,0,1]]'. "
             "If not provided with --output-type affinities, will try to read 'offsets' "
             "from the model script."
    )

    return parser


def main():
    parser = build_arg_parser()

    args = parser.parse_args()

    # Keep the LoRA scaling factor (alpha/r) fixed at 2 regardless of rank,
    # which is what FinetuneJobManager already does for dashboard-submitted
    # jobs via lora_alpha = lora_r * 2. A fixed alpha default would silently
    # change the scaling whenever the rank default moved -- at r=64 an
    # alpha of 16 is a scaling of 0.25 rather than 2.
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2

    # Print configuration
    logger.info("=" * 60)
    logger.info("LoRA Finetuning Configuration")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Corrections: {args.corrections}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(
        f"LoRA rank: {args.lora_r} (alpha: {args.lora_alpha}, "
        f"min_channels: {args.lora_min_channels})"
    )
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")

    # === Load model (once) ===
    logger.info("Loading model...")

    model_config = _model_config_from_args(args)
    base_model = load_trainable_model(model_config)

    # === Wrap with LoRA (once - same object is reused across restarts) ===
    if args.lora_r <= 0:
        # Full finetune. Measured against LoRA r=64 on mito-aff-unet-setup-16
        # (2026-09-23): faster per step (0.50 vs 0.90 s), lower memory (31 vs
        # 50 GB at batch 8), and lower training loss at every checkpoint --
        # the adapter's savings are in parameters, which is not where this
        # model's cost is. The export is a full state dict under
        # full_finetune/, served via FinetuneModelConfig(weights_path=...).
        logger.info("lora_r=0: full finetuning -- every parameter trainable, no adapter. "
                    "Restarts start again from the starting weights.")
        # A finetuned model given as the base still carries its adapter; fold
        # it into the weights, which are what a full finetune trains.
        from cellmap_flow.finetune.lora_wrapper import _merge_existing_adapters

        base_model = _merge_existing_adapters(base_model)
        for p in base_model.parameters():
            p.requires_grad_(True)
        lora_model = base_model
        n_train = sum(p.numel() for p in lora_model.parameters())
        logger.info(f"trainable params: {n_train:,} || all params: {n_train:,} || trainable%: 100.0000")
    else:
        logger.info(f"Wrapping model with LoRA (r={args.lora_r})...")
        lora_model = wrap_model_with_lora(
            base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_min_channels=args.lora_min_channels,
        )

    # === Training loop (supports restart via signal file) ===
    server_started = False
    restart_controller = RestartController()
    iteration = 0
    # A full finetune's distillation teacher: a frozen copy of the starting
    # weights, made by the first trainer that needs one and reused after, so a
    # restart neither copies the model again nor distils toward weights an
    # earlier iteration already changed.
    teacher_model = None
    # The weights a full finetune starts from, on the CPU, to reset it to on
    # restart. LoRA resets by re-making its adapter and needs none.
    initial_state = None
    if not _is_peft_model(lora_model):
        from cellmap_flow.finetune.lora_trainer import cpu_state_copy

        initial_state = cpu_state_copy(lora_model)

    while True:
        iteration += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if iteration > 1:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Training Iteration {iteration}")
            logger.info("=" * 60)

        # Create dataloader (re-created each iteration to pick up new annotations)
        if iteration > 1:
            print("RESTART_STATUS: Loading corrections...", flush=True)
        logger.info(f"Loading corrections from {args.corrections}...")
        dataloader = create_dataloader(
            args.corrections,
            batch_size=args.batch_size,
            patch_shape=tuple(args.patch_shape) if args.patch_shape is not None else None,
            augment=not args.no_augment,
            num_workers=args.num_workers,
            shuffle=True,
            model_name=args.model_name,
        )
        logger.info(f"DataLoader created: {len(dataloader.dataset)} corrections")

        # Snapshot the active input_norm into metadata.json so any saved
        # checkpoint in this iteration is reproducible -- you can read
        # metadata.json next to the .pth and know exactly which
        # normalization was applied to the training data.
        try:
            from cellmap_flow.finetune.virtual_dataset import read_manifest

            manifest_norm = (read_manifest(args.corrections) or {}).get("input_norm")
            if manifest_norm is not None and args.output_dir:
                metadata_file = Path(args.output_dir) / "metadata.json"
                if metadata_file.exists():
                    import json as json_mod
                    with open(metadata_file) as f:
                        md = json_mod.load(f)
                    md.setdefault("params", {})["input_norm"] = manifest_norm
                    with open(metadata_file, "w") as f:
                        json_mod.dump(md, f, indent=2)
                    logger.info(
                        f"Snapshot input_norm into {metadata_file} "
                        f"(keys: {list(manifest_norm.keys())})"
                    )
        except Exception as _e:
            logger.warning(f"Could not snapshot input_norm into metadata.json: {_e}")

        # Build target transform (re-built each iteration to pick up restart params)
        select_channel = args.select_channel
        target_transform = _build_target_transform(args, model_config)
        logger.info(f"output_type={args.output_type}, select_channel={select_channel}")

        # Create trainer (re-created each iteration for fresh optimizer/scheduler)
        if iteration > 1:
            print("RESTART_STATUS: Preparing trainer...", flush=True)
        logger.info("Creating trainer...")
        trainer = LoRAFinetuner(
            lora_model,
            dataloader,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_mixed_precision=not args.no_mixed_precision,
            loss_type=args.loss_type,
            select_channel=select_channel,
            mask_unannotated=args.mask_unannotated,
            label_smoothing=args.label_smoothing,
            distillation_lambda=args.distillation_lambda,
            distillation_all_voxels=args.distillation_all_voxels,
            margin=args.margin,
            balance_classes=args.balance_classes,
            target_transform=target_transform,
            tensorboard=not args.no_tensorboard,
            teacher_model=teacher_model,
            initial_state=initial_state,
        )

        # Resume from checkpoint if specified (first iteration only)
        if args.resume and iteration == 1:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Train
        try:
            if iteration > 1:
                print("RESTART_STATUS: Starting training...", flush=True)
            stats = trainer.train()
            # None again if an OOM made the trainer drop distillation.
            teacher_model = trainer.teacher_model

            # If training diverged (NaN/Inf), skip saving and wait for restart
            if stats.get('diverged'):
                logger.warning("Training diverged — skipping model save.")
                if args.auto_serve and not server_started:
                    # Nothing can restart this job: the dashboard sends a
                    # restart to the job's inference server, which only starts
                    # after an iteration completes. Waiting here held the GPU
                    # until walltime. Exit so the job shows as failed and the
                    # GPU is freed; resubmit with other settings.
                    logger.error(
                        "The first training iteration diverged, so no inference "
                        "server is running to receive a restart. Exiting; "
                        "resubmit with other settings (e.g. a lower learning rate)."
                    )
                    return 1
                if args.auto_serve:
                    # Still wait for restart so the user can adjust params
                    signal_file = Path(args.output_dir) / "restart_signal.json"
                    restart_data = _wait_for_restart_signal(
                        signal_file=signal_file,
                        check_interval=1.0,
                        restart_controller=restart_controller,
                    )
                    if restart_data is None:
                        logger.error("Malformed restart signal, exiting")
                        return 1
                    _apply_restart_params(args, restart_data)

                    lora_model = _reset_for_restart(lora_model, args, initial_state)
                    print("RESTARTING_TRAINING", flush=True)
                    continue
                else:
                    return 1

            # What was exported is decided by the model, not by args.lora_r,
            # which a restart can change without changing the model.
            is_lora = _is_peft_model(lora_model)

            # Save final adapter (or, for a full finetune, the full weights)
            logger.info("\nSaving LoRA adapter..." if is_lora else "\nSaving full finetuned weights...")
            trainer.save_adapter()

            logger.info("\n" + "=" * 60)
            logger.info("Finetuning Complete!")
            logger.info(f"Best loss: {stats['best_loss']:.6f}")
            if is_lora:
                logger.info(f"Adapter saved to: {args.output_dir}/lora_adapter")
            else:
                logger.info(f"Weights saved to: {args.output_dir}/full_finetune/model_state_dict.pt")
            logger.info("=" * 60)

            # Generate model files. The weights are saved by now, so a YAML
            # that cannot be written is reported, not treated as a failed
            # training run.
            finetuned_model_name = _finetuned_model_name(model_config, timestamp)
            try:
                finetuned_model_name, yaml_path = _generate_model_files(
                    args, model_config, timestamp, is_lora=is_lora
                )
            except Exception as e:
                logger.error(f"Training succeeded but the serving YAML could not be written: {e}", exc_info=True)
            else:
                # The job manager takes the YAML from here rather than
                # guessing where it went. Printed before the completion
                # marker so both are in the log when that is seen.
                print(f"{FINETUNED_MODEL_YAML_MARKER} {yaml_path}", flush=True)

            # Print completion marker with timestamp (for job manager to detect)
            print(f"TRAINING_ITERATION_COMPLETE: {finetuned_model_name}", flush=True)

            # Auto-serve if requested
            if args.auto_serve:
                if not server_started:
                    # First time: start inference server in background thread
                    try:
                        _start_inference_server_background(
                            args, model_config, lora_model, restart_controller=restart_controller
                        )
                        server_started = True
                    except Exception as e:
                        logger.error(f"Failed to start inference server: {e}", exc_info=True)
                        print(f"INFERENCE_SERVER_FAILED: {e}", flush=True)
                        return 0
                else:
                    # Server already running - just set model back to eval mode
                    # The server shares the same model object, so it automatically
                    # serves with the updated weights
                    lora_model.eval()
                    logger.info("Model updated and set to eval mode. Server continuing with new weights.")

                # Watch for restart signal
                signal_file = Path(args.output_dir) / "restart_signal.json"
                restart_data = _wait_for_restart_signal(
                    signal_file=signal_file,
                    check_interval=1.0,
                    restart_controller=restart_controller,
                )

                if restart_data is None:
                    logger.error("Malformed restart signal, exiting")
                    return 1

                # Apply updated parameters
                _apply_restart_params(args, restart_data)

                # A true restart: training starts again from the model it
                # started from, not from the previous iteration's weights.
                lora_model = _reset_for_restart(lora_model, args, initial_state)
                print("RESTARTING_TRAINING", flush=True)
                continue  # Loop back to retrain

            # No auto-serve: just exit after training
            return 0

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            logger.info("Saving current state...")
            trainer.save_checkpoint(is_best=False)
            return 1

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
