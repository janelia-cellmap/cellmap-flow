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
        --lora-r 16 \
        --batch-size 4 \
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
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cellmap_flow.models.models_config import FlyModelConfig, DaCapoModelConfig, HuggingFaceModelConfig, ModelConfig
from cellmap_flow.utils.ds import _is_remote_path
from cellmap_flow.finetune.lora_wrapper import wrap_model_with_lora
from cellmap_flow.finetune.correction_dataset import create_dataloader
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
    server = CellMapFlowServer(args.serve_data_path, model_config, restart_callback=restart_callback)

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


def _apply_restart_params(args, signal_data: dict):
    """
    Update args with parameters from restart signal and persist to metadata.json.

    Args:
        args: argparse Namespace to update
        signal_data: Dict from restart signal file
    """
    params = signal_data.get("params", {})
    changed = False
    for key, value in params.items():
        if hasattr(args, key) and value is not None:
            old_value = getattr(args, key)
            setattr(args, key, value)
            if old_value != value:
                logger.info(f"Updated {key}: {old_value} -> {value}")
                changed = True

    # Persist updated params to metadata.json
    if changed and hasattr(args, 'output_dir') and args.output_dir:
        metadata_file = Path(args.output_dir) / "metadata.json"
        if metadata_file.exists():
            try:
                import json as json_mod
                with open(metadata_file, "r") as f:
                    metadata = json_mod.load(f)
                if "params" in metadata:
                    for key, value in params.items():
                        if key in metadata["params"]:
                            metadata["params"][key] = value
                metadata["last_restart_at"] = signal_data.get("timestamp")
                with open(metadata_file, "w") as f:
                    json_mod.dump(metadata, f, indent=2)
                logger.info(f"Updated metadata.json with restart params")
            except Exception as e:
                logger.warning(f"Failed to update metadata.json: {e}")


def _generate_model_files(args, model_config, timestamp):
    """
    Generate YAML config file after training.

    Args:
        args: Command-line arguments
        model_config: Model configuration
        timestamp: Timestamp string for naming

    Returns:
        (finetuned_model_name, yaml_path) tuple
    """
    from cellmap_flow.finetune.finetuned_model_templates import (
        generate_finetuned_model_yaml
    )

    model_basename = model_config.name
    finetuned_model_name = f"{model_basename}_finetuned_{timestamp}"

    # Create models directory in output
    output_dir_path = Path(args.output_dir)
    session_path = output_dir_path.parent.parent.parent
    models_dir = session_path / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Generating model config for {finetuned_model_name}...")

    # Extract data path from corrections
    corrections_path = Path(args.corrections)
    zarr_dirs = list(corrections_path.glob("*.zarr"))
    data_path = None
    if zarr_dirs:
        zattrs_file = zarr_dirs[0] / ".zattrs"
        if zattrs_file.exists():
            with open(zattrs_file) as f:
                metadata = json.load(f)
                data_path = metadata.get("dataset_path")

    if not data_path:
        logger.warning("Could not extract data_path from corrections, using serve_data_path")
        data_path = args.serve_data_path if args.auto_serve else "/path/to/data.zarr"

    # Bake the training-time input_norm into the generated yaml so the
    # served finetuned model gets queried with the same normalization the
    # adapter was trained on. Without this, training-vs-inference scale
    # mismatch silently destroys finetuning quality.
    json_data = None
    try:
        from cellmap_flow.finetune.virtual_dataset import read_manifest

        manifest = read_manifest(str(corrections_path)) or {}
        train_input_norm = manifest.get("input_norm")
        if train_input_norm:
            json_data = {"input_norm": train_input_norm, "postprocess": {}}
    except Exception as _e:
        logger.warning(
            f"Could not load training input_norm from manifest: {_e}. "
            "Generated finetuned yaml will lack normalization metadata."
        )

    yaml_path = generate_finetuned_model_yaml(
        lora_adapter_path=str(output_dir_path / "lora_adapter"),
        base_model_dict=model_config.to_dict(),
        model_name=finetuned_model_name,
        output_path=models_dir / f"{finetuned_model_name}.yaml",
        data_path=data_path,
        json_data=json_data,
    )
    logger.info(f"Generated YAML: {yaml_path}")

    return finetuned_model_name, yaml_path


class _SkootsHead(nn.Module):
    """Multi-head output layer matching the official SKOOTS architecture
    (`bism.models.spatial_embedding.SpatialEmbedding`): separate 3x3x3
    (padding=1) conv heads per output group -- [semantic(1), skeleton(1),
    vector(3)] or [semantic(1), skeleton(1), distance(1)] -- concatenated
    into one raw-logit tensor, instead of one shared 1x1x1 conv slicing a
    single 5-channel output into meaning-groups after the fact. Unlike the
    reference, activation (sigmoid/tanh) is deliberately NOT applied here:
    `SkootsLoss`/`SkeletonDistanceLoss` already apply it on the assumption
    the model returns raw logits, so baking it in here too would silently
    double-apply sigmoid/tanh on top of itself.
    """

    def __init__(self, in_channels: int, group_sizes: Tuple[int, ...], kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Conv3d(in_channels, size, kernel_size, padding=kernel_size // 2, bias=bias)
            for size in group_sizes
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            if bias:
                nn.init.zeros_(head.bias)

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


def _swap_final_conv_for_skoots(model, model_config, num_channels: int = 5):
    """Replace a pretrained model's output head with a fresh N-channel one.

    Warm-starting SKOOTS (or its skeleton_distance sibling) from an existing
    checkpoint (e.g. an affinity or distance model) means reusing its trained
    backbone but not its output layer, which was learned for a different,
    incompatible channel meaning (e.g. 3 affinity directions) -- there's
    nothing meaningful to transfer from that into a semantic/skeleton/(vector
    or distance) head, so it's reinitialized rather than resized/padded.

    Assumes the loaded model exposes its output layer as a top-level
    `final_conv` attribute (true for the torch.export-unflattened models
    produced by `CellmapModel.train()`, which is the only path this has been
    exercised against). Also patches `model_config.config.output_channels`
    to `num_channels` so downstream channel-count checks (e.g. in
    `_build_target_transform`) see the model's real, post-swap shape rather
    than the original checkpoint's.
    """
    if not hasattr(model, "final_conv"):
        raise ValueError(
            f"--output-type skoots/skeleton_distance/skeleton_semantic requires the base model to expose a "
            f"'final_conv' output layer to replace; {type(model).__name__} has none. "
            f"Only models loaded via CellmapModel.train() (huggingface/dacapo) are supported."
        )

    old_conv = model.final_conv
    weight = old_conv.weight
    out_c, in_c = weight.shape[0], weight.shape[1]
    has_bias = getattr(old_conv, "bias", None) is not None

    if num_channels == 5:
        group_sizes = (1, 1, 3)  # semantic, skeleton, vector(z,y,x)
    elif num_channels == 3:
        group_sizes = (1, 1, 1)  # semantic, skeleton, distance
    elif num_channels == 2:
        group_sizes = (1, 1)  # semantic, skeleton
    else:
        raise ValueError(f"_swap_final_conv_for_skoots: unsupported num_channels={num_channels}")

    # PyTorch's default Conv3d init assumes roughly unit-scale input
    # activations; this backbone's penultimate feature map is not unit-scale,
    # so the default random init lands each output channel at an arbitrary,
    # uncontrolled logit magnitude -- empirically anywhere from -27 to +28
    # across the old 5-channel swap. Any channel that lands deep in
    # sigmoid/tanh's saturated region starts training with ~zero gradient
    # (sigmoid(-21) is already indistinguishable from 0), so it can never
    # recover no matter how many epochs run. `_SkootsHead` zero-initializes
    # every head, putting every channel at exactly the neutral point
    # (sigmoid(0)=0.5, tanh(0)=0) regardless of the backbone's activation
    # scale, so no channel starts pre-saturated.
    #
    # kernel_size=3 (not the original 1x1x1) matches the official SKOOTS
    # head (`bism.models.spatial_embedding.SpatialEmbedding`), giving the
    # output heads a real receptive field instead of a bare per-voxel linear
    # projection of the backbone's last feature map.
    model.final_conv = _SkootsHead(in_c, group_sizes, kernel_size=3, bias=has_bias)
    logger.info(
        f"Swapped final_conv: {out_c}->{num_channels} output channels via "
        f"{len(group_sizes)} separate 3x3x3 heads {group_sizes} "
        f"(in_channels={in_c}, backbone weights kept, zero-initialized to "
        f"avoid pre-saturated sigmoid/tanh channels)"
    )

    model_config.config.output_channels = num_channels
    return model


def _build_target_transform(args, model_config):
    """Build a TargetTransform based on CLI args."""
    from cellmap_flow.finetune.target_transforms import (
        BinaryTargetTransform,
        BroadcastBinaryTargetTransform,
        AffinityTargetTransform,
        SkootsTargetTransform,
        SkeletonDistanceTargetTransform,
        SkeletonSemanticTargetTransform,
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

    elif output_type == "skoots":
        if num_channels != 5:
            raise ValueError(
                f"SKOOTS output type requires exactly 5 model output channels "
                f"(semantic, skeleton, vec_z, vec_y, vec_x); model has {num_channels}."
            )
        # SkootsDataset precomputes the (target, mask) pair per crop and packs
        # both into the batch's "target" tensor (see SkootsTargetTransform's
        # docstring) -- this transform only unpacks them, it does no
        # on-the-fly target computation.
        return SkootsTargetTransform()

    elif output_type == "skeleton_distance":
        if num_channels != 3:
            raise ValueError(
                f"skeleton_distance output type requires exactly 3 model output channels "
                f"(semantic, skeleton, distance); model has {num_channels}."
            )
        return SkeletonDistanceTargetTransform()

    elif output_type == "skeleton_semantic":
        if num_channels != 2:
            raise ValueError(
                f"skeleton_semantic output type requires exactly 2 model output channels "
                f"(semantic, skeleton); model has {num_channels}."
            )
        return SkeletonSemanticTargetTransform()

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


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Finetune CellMap-Flow models with LoRA using user corrections"
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="fly",
        choices=["fly", "dacapo", "huggingface", "script"],
        help="Model type (fly, dacapo, huggingface, or script)"
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
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling (default: 16)"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)"
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
        help="Disable data augmentation"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and adapter"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
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
        choices=["dice", "bce", "combined", "mse", "margin", "skoots", "skoots_simple_vector", "skeleton_distance", "skeleton_semantic"],
        help="Loss function (default: combined). 'skoots' or "
             "'skoots_simple_vector' (plain balanced MSE on the vector head "
             "instead of the annealed Gaussian/Tversky embedding loss) is "
             "required for --output-type skoots, 'skeleton_distance' is "
             "required for --output-type skeleton_distance, 'skeleton_semantic' "
             "is required for --output-type skeleton_semantic, and all are "
             "ignored otherwise."
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
        choices=["binary", "binary_broadcast", "affinities", "skoots", "skeleton_distance", "skeleton_semantic"],
        help="How to generate training targets from annotations. "
             "'binary': single-channel fg/bg (use with --select-channel for multi-channel models). "
             "'binary_broadcast': broadcast binary target to all output channels. "
             "'affinities': compute affinity targets from instance labels (requires offsets). "
             "'skoots': semantic+skeleton+vector instance targets (requires --skoots-crops-yaml "
             "and a 5-channel model; targets are precomputed per crop, not per batch). "
             "'skeleton_distance': semantic+skeleton+distance-from-skeleton instance targets -- "
             "same --skoots-crops-yaml infra as 'skoots' but a 3-channel model and a scalar "
             "distance-to-medial-axis head instead of a 3-channel vector field; appropriate only "
             "when instances don't actually touch (no directional disambiguation at boundaries). "
             "'skeleton_semantic': semantic+skeleton only, no instance-splitting channel at all -- "
             "same --skoots-crops-yaml infra but a 2-channel model; use when you only need a "
             "semantic mask + skeleton, or as a diagnostic to isolate those two heads from any "
             "vector/distance-loss effects. "
             "(default: binary)"
    )
    parser.add_argument(
        "--skoots-crops-yaml",
        type=str,
        default=None,
        help="Path to a CropsConfig YAML (see crop_loader.py) of dense instance-labeled "
             "crops to train a SKOOTS/skeleton_distance/skeleton_semantic head on. Required and "
             "only used with --output-type skoots/skeleton_distance/skeleton_semantic -- bypasses "
             "--corrections/create_dataloader entirely, since all three need dense full-instance "
             "labels, not sparse correction scribbles."
    )
    parser.add_argument(
        "--skoots-raw-dataset-path",
        type=str,
        default=None,
        help="Path to the raw EM zarr the --skoots-crops-yaml crops were annotated on "
             "(paired to each crop by physical nm coordinates, since it may be at a "
             "different resolution than the label crops). Required with --output-type skoots."
    )
    parser.add_argument(
        "--skoots-cache-dir",
        type=str,
        default=None,
        help="Directory to cache precomputed SKOOTS targets (semantic/skeleton/vectors), "
             "keyed by crop name. Defaults to a 'skoots_cache' dir next to --skoots-crops-yaml. "
             "Building targets is expensive (skeletonize + vector bake per instance), so "
             "this cache persists across training restarts."
    )
    parser.add_argument(
        "--skoots-patches-per-epoch",
        type=int,
        default=1000,
        help="Number of random patches sampled per epoch for --output-type skoots "
             "(default: 1000). Patch locations are freshly sampled every epoch; only "
             "target *computation* is precomputed/cached, not the patch sampling."
    )
    parser.add_argument(
        "--skoots-target-voxel-size",
        type=float,
        nargs=3,
        default=None,
        help="If set, decimate crop labels to this voxel size (nm, Z Y X) before "
             "building SKOOTS targets, so they match a base model trained at a "
             "coarser resolution than the label crops' own native voxel size. "
             "Must be an integer multiple of the crops' native voxel size. "
             "(default: None, use crops' native resolution)"
    )
    parser.add_argument(
        "--skoots-context-voxels",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="Extra halo (Z Y X, in --skoots-target-voxel-size/native voxels) added "
             "per side when reading the raw EM patch, beyond --patch-shape. Needed for "
             "valid-padding base models whose input tile is larger than their output "
             "tile (e.g. 61 61 61 for a 178-in/56-out model). (default: 0 0 0)"
    )
    parser.add_argument(
        "--skoots-skeleton-radius",
        type=int,
        default=2,
        help="Ball radius (voxels, at --skoots-target-voxel-size/native resolution) "
             "stamped around each skeleton voxel to build the skeleton-mask training "
             "target. Too large bridges nearby-but-separate instances' skeleton masks "
             "into one connected component, which then makes skeleton-seeded "
             "postprocessing wrongly merge them. (default: 2)"
    )
    parser.add_argument(
        "--skoots-min-branch-length",
        type=float,
        default=3.0,
        help="Prune terminal skeleton branches (voxels, at "
             "--skoots-target-voxel-size/native resolution) shorter than this from the "
             "raw skimage.skeletonize output -- removes small stair-step spurs from "
             "voxelization without touching real (non-spur) branch structure. Set to 0 "
             "to disable. (default: 3.0)"
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

    # Print configuration
    logger.info("=" * 60)
    logger.info("LoRA Finetuning Configuration")
    logger.info("=" * 60)
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Corrections: {args.corrections}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("")

    if args.output_type in ("skoots", "skeleton_distance", "skeleton_semantic"):
        if not args.skoots_crops_yaml or not args.skoots_raw_dataset_path:
            raise ValueError(
                f"--output-type {args.output_type} requires --skoots-crops-yaml and "
                "--skoots-raw-dataset-path."
            )
        # "skoots" output allows either loss_type variant (the default
        # annealed Gaussian/Tversky embedding loss, or the plain balanced-MSE
        # A/B alternative); "skeleton_distance"/"skeleton_semantic" each have
        # just the one.
        if args.output_type == "skoots":
            allowed_loss_types = {"skoots", "skoots_simple_vector"}
        elif args.output_type == "skeleton_distance":
            allowed_loss_types = {"skeleton_distance"}
        else:
            allowed_loss_types = {"skeleton_semantic"}
        if args.loss_type not in allowed_loss_types:
            raise ValueError(
                f"--output-type {args.output_type} requires --loss-type in {sorted(allowed_loss_types)} "
                f"(got {args.loss_type!r})."
            )
        # --corrections/create_dataloader is skipped entirely for these output types
        # (SkootsDataset reads directly from --skoots-crops-yaml), but --corrections
        # stays required at the argparse level to keep every other output type
        # unaffected -- pass any placeholder value here.
        logger.info(f"SKOOTS crops YAML: {args.skoots_crops_yaml}")
        logger.info(f"SKOOTS raw dataset: {args.skoots_raw_dataset_path}")

    # === Load model (once) ===
    logger.info("Loading model...")

    if args.model_script:
        from cellmap_flow.models.models_config import ScriptModelConfig
        logger.info(f"Using script-based model: {args.model_script}")
        model_config = ScriptModelConfig(
            script_path=args.model_script,
            name=args.model_name or "script_model"
        )
    elif args.model_type == "script":
        raise ValueError("For script models, --model-script is required")
    elif args.model_type == "fly":
        if not args.model_checkpoint:
            raise ValueError(
                "For fly models, either --model-checkpoint or --model-script must be provided"
            )
        model_config = FlyModelConfig(
            checkpoint_path=args.model_checkpoint,
            channels=args.channels,
            input_voxel_size=tuple(args.input_voxel_size),
            output_voxel_size=tuple(args.output_voxel_size),
            name=args.model_name,
        )
    elif args.model_type == "dacapo":
        if not args.model_checkpoint:
            raise ValueError("For dacapo models, --model-checkpoint is required")
        checkpoint_path = Path(args.model_checkpoint)
        iteration = int(checkpoint_path.stem.split('_')[-1])
        run_name = checkpoint_path.parent.name

        model_config = DaCapoModelConfig(
            run_name=run_name,
            iteration=iteration,
        )
    elif args.model_type == "huggingface":
        if not args.repo:
            raise ValueError("For huggingface models, --repo is required")
        model_config = HuggingFaceModelConfig(
            repo=args.repo,
            revision=args.revision,
            name=args.model_name,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    base_model = model_config.config.model
    logger.info(f"Model loaded: {type(base_model).__name__}")

    # TorchScript models (RecursiveScriptModule) can't be used with LoRA.
    # Use cellmap_model.train() to get a trainable nn.Module via torch.export
    # unflatten — no fly_organelles dependency needed.
    if isinstance(base_model, torch.jit.ScriptModule):
        logger.info("TorchScript model detected — loading trainable model via cellmap_model.train()...")
        cellmap_model = None
        if args.model_type == "huggingface":
            from cellmap_models.model_export.cellmap_model import get_huggingface_model
            cellmap_model = get_huggingface_model(args.repo, args.revision)
        elif hasattr(model_config, 'cellmap_model'):
            cellmap_model = model_config.cellmap_model

        if cellmap_model is not None:
            trainable = cellmap_model.train()
            if trainable is not None:
                if args.output_type in ("skoots", "skeleton_distance", "skeleton_semantic"):
                    num_channels = {"skoots": 5, "skeleton_distance": 3, "skeleton_semantic": 2}[args.output_type]
                    trainable = _swap_final_conv_for_skoots(trainable, model_config, num_channels)
                # UnflattenedModule (from torch.export) often has fixed batch=1.
                # Wrap it so the trainer can use any batch size.
                if type(trainable).__name__ == 'UnflattenedModule':
                    from cellmap_flow.finetune.lora_wrapper import BatchLoopWrapper
                    trainable = BatchLoopWrapper(trainable)
                    logger.info("Wrapped UnflattenedModule with BatchLoopWrapper for variable batch sizes")
                base_model = trainable
                logger.info(f"Trainable model loaded: {type(base_model).__name__}")
            else:
                logger.warning("cellmap_model.train() returned None — LoRA may fail")
        else:
            logger.warning("No CellmapModel available — LoRA may fail on TorchScript model")

    # === Wrap with LoRA (once - same object is reused across restarts) ===
    logger.info(f"Wrapping model with LoRA (r={args.lora_r})...")
    lora_model = wrap_model_with_lora(
        base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # A freshly reinitialized head needs full gradient descent, not a
        # low-rank delta on top of random weights -- LoRA-decomposing it
        # would badly bottleneck how much it can learn.
        modules_to_save=["final_conv"] if args.output_type in ("skoots", "skeleton_distance", "skeleton_semantic") else None,
    )

    # === Training loop (supports restart via signal file) ===
    server_started = False
    restart_controller = RestartController()
    iteration = 0

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
        if args.output_type in ("skoots", "skeleton_distance", "skeleton_semantic"):
            from cellmap_flow.finetune.skoots_dataset import SkootsDataset

            logger.info(f"Loading SKOOTS crops from {args.skoots_crops_yaml}...")
            skoots_dataset = SkootsDataset(
                crops_yaml_path=args.skoots_crops_yaml,
                raw_dataset_path=args.skoots_raw_dataset_path,
                output_size_voxels=tuple(args.patch_shape) if args.patch_shape is not None else (64, 64, 64),
                target_voxel_size_nm=tuple(args.skoots_target_voxel_size) if args.skoots_target_voxel_size is not None else None,
                context_voxels=tuple(args.skoots_context_voxels),
                skeleton_radius=args.skoots_skeleton_radius,
                min_branch_length=args.skoots_min_branch_length,
                cache_dir=args.skoots_cache_dir,
                patches_per_epoch=args.skoots_patches_per_epoch,
                target_mode=args.output_type,
            )
            # num_workers must stay 0: forked DataLoader workers crash immediately
            # ("aborting: fork() use detected") because ImageDataInterface's raw
            # reads go through tensorstore, which installs a pthread_atfork guard
            # that refuses to survive fork() once its internal thread pool exists
            # in the parent process -- and by this point (model/config loading)
            # it already does. Spawning instead of forking would dodge that, but
            # would also re-pickle this dataset's already-in-memory precomputed
            # per-crop target arrays (several GB) into every worker process, which
            # is worse than just reading raw patches from the main process.
            dataloader = DataLoader(
                skoots_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
            logger.info(f"DataLoader created: {len(skoots_dataset)} patches/epoch over {len(skoots_dataset.crops)} crops")
        else:
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
        if args.output_type not in ("skoots", "skeleton_distance", "skeleton_semantic"):
            # --corrections isn't a manifest path in skoots/skeleton_distance
            # mode, so there's nothing to snapshot here.
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

            # If training diverged (NaN/Inf), skip saving and wait for restart
            if stats.get('diverged'):
                logger.warning("Training diverged — skipping model save.")
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

                    # Reset LoRA weights for fresh restart
                    logger.info("Resetting LoRA adapter weights for fresh restart...")
                    from peft import PeftModel
                    if isinstance(lora_model, PeftModel):
                        base = lora_model.unload()
                        lora_model = wrap_model_with_lora(
                            base,
                            lora_r=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout,
                            modules_to_save=["final_conv"] if args.output_type in ("skoots", "skeleton_distance", "skeleton_semantic") else None,
                        )

                    lora_model.train()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.info("Restarting training with fresh LoRA weights...")
                    print("RESTARTING_TRAINING", flush=True)
                    continue
                else:
                    return 1

            # Save final adapter
            logger.info("\nSaving LoRA adapter...")
            trainer.save_adapter()

            logger.info("\n" + "=" * 60)
            logger.info("Finetuning Complete!")
            logger.info(f"Best loss: {stats['best_loss']:.6f}")
            logger.info(f"Adapter saved to: {args.output_dir}/lora_adapter")
            logger.info("=" * 60)

            # Generate model files
            finetuned_model_name, _ = _generate_model_files(
                args, model_config, timestamp
            )

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

                # Reset LoRA weights to initial state for a true restart.
                # Delete the current adapter and re-create it so training
                # starts from the frozen base model, not from the previous
                # finetuned weights.
                logger.info("Resetting LoRA adapter weights for fresh restart...")
                from peft import PeftModel
                if isinstance(lora_model, PeftModel):
                    base = lora_model.unload()
                    lora_model = wrap_model_with_lora(
                        base,
                        lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        modules_to_save=["final_conv"] if args.output_type in ("skoots", "skeleton_distance", "skeleton_semantic") else None,
                    )

                lora_model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Restarting training with fresh LoRA weights...")
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
