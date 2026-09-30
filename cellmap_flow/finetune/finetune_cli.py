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

from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.utils.ds import _is_remote_path
from cellmap_flow.utils.restart_token import read_or_create_restart_token
from cellmap_flow.finetune import markers, run_outputs
from cellmap_flow.finetune.adaptation import FullStrategy, LoraStrategy, strategy_for
from cellmap_flow.finetune.cli import (
    apply_restart_params,
    build_target_transform,
    model_config_from_args,
    parse_args,
)
from cellmap_flow.finetune.model_loading import load_trainable_model
from cellmap_flow.finetune.data import create_dataloader
from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.finetune.target_transforms import read_offsets_from_script

# The dashboard's import of it (routes/finetune/common.py), until W4-A moves it.
_read_offsets_from_script = read_offsets_from_script

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
    # The job manager's cue that this job is idle and can take a restart.
    markers.emit(markers.WAITING_FOR_RESTART)

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


def _reset_for_restart(lora_model, args, initial_state=None):
    """Put the model back where training started, for the next iteration.

    LoRA: unload the adapter and wrap a fresh one around the base (which is
    how the rank can change on restart). Full finetune: load the starting
    weights back (see the strategies' restart()). The model object -- which
    the inference server shares -- is built once, so a restart cannot switch
    between the two kinds: the model decides, and args.lora_r is made to
    agree with it.

    Returns the model to train next.
    """
    kept = strategy_for(lora_model)
    if kept.kind == "lora" and args.lora_r <= 0:
        logger.warning("Restart asked for lora_r=0 (full finetune) but this job trains a LoRA adapter; "
                       "submit a new job for that. Keeping the current adapter setup.")
        args.lora_r = max(1, int(kept.r))
    elif kept.kind == "full" and args.lora_r > 0:
        # The mirror image of the case above. Left alone, args.lora_r > 0
        # made the next iteration's YAML point at a lora_adapter/ this job
        # never writes.
        logger.warning(f"Restart asked for LoRA rank {args.lora_r} but this job is a full finetune; "
                       "submit a new job for that. Keeping the full finetune.")
        args.lora_r = 0

    strategy = strategy_for(
        lora_model,
        args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        min_channels=args.lora_min_channels,
    )
    lora_model = strategy.restart(lora_model, initial_state)
    lora_model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Restarting training from the starting weights...")
    return lora_model


def main():
    # Here, not at import: importing this module (the dashboard does, for
    # _read_offsets_from_script) reconfigured the importing process's logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )
    args = parse_args()

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

    model_config = model_config_from_args(args)
    base_model = load_trainable_model(model_config)

    # === Wrap with LoRA (once - same object is reused across restarts) ===
    # The request decides, here only: --lora-r 0 is a full finetune (see
    # FullStrategy). From here on the model does (strategy_for). A finetuned
    # model given as the base still carries its adapter; either way it is
    # folded into the weights first.
    if args.lora_r <= 0:
        strategy = FullStrategy()
    else:
        strategy = LoraStrategy(
            args.lora_r, args.lora_alpha, args.lora_dropout, min_channels=args.lora_min_channels
        )
    lora_model = strategy.prepare(base_model)

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
    initial_state = strategy.initial_state(lora_model)
    # Where the next iteration's TensorBoard curves start: (step, epoch).
    tb_position = (0, 0)
    # Set by a restart; the reset waits until the next iteration is set up.
    pending_reset = False

    while True:
        iteration += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if iteration > 1:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Training Iteration {iteration}")
            logger.info("=" * 60)

        # Set up this iteration: its data, its target and its trainer. A
        # restart's new settings or annotations are first used here, so an
        # error here -- an empty volume, offsets that do not fit -- used to
        # escape main() and end the whole job, taking the served model down
        # with it.
        try:
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
                from cellmap_flow.finetune.session.manifest import read_manifest

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
            target_transform = build_target_transform(args, model_config)
            logger.info(f"output_type={args.output_type}, select_channel={select_channel}")

            # Only now that the iteration can run: put the model back where
            # training started (see _reset_for_restart). Until here it is still
            # the previous iteration's, which the server keeps serving.
            if pending_reset:
                lora_model = _reset_for_restart(lora_model, args, initial_state)
                pending_reset = False

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
                tb_start_step=tb_position[0],
                tb_start_epoch=tb_position[1],
            )

            # Resume from checkpoint if specified (first iteration only)
            if args.resume and iteration == 1:
                logger.info(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)

        except Exception as e:
            logger.error(f"Could not set up training iteration {iteration}: {e}", exc_info=True)
            if not (args.auto_serve and server_started):
                return 1
            # The previous iteration's model is still loaded and served;
            # wait for a restart with settings that work.
            markers.emit(markers.RESTART_FAILED, e)
            restart_data = _wait_for_restart_signal(
                signal_file=Path(args.output_dir) / "restart_signal.json",
                check_interval=1.0,
                restart_controller=restart_controller,
            )
            if restart_data is None:
                logger.error("Malformed restart signal, exiting")
                return 1
            apply_restart_params(args, restart_data)
            markers.emit(markers.RESTARTING_TRAINING)
            continue

        # Train
        try:
            if iteration > 1:
                print("RESTART_STATUS: Starting training...", flush=True)
            stats = trainer.train()
            # None again if an OOM made the trainer drop distillation.
            teacher_model = trainer.teacher_model
            tb_position = (trainer._tb_step, trainer._tb_epoch)
            trainer.close()

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
                    apply_restart_params(args, restart_data)

                    pending_reset = True
                    markers.emit(markers.RESTARTING_TRAINING)
                    continue
                else:
                    return 1

            # What was exported is decided by the model, not by args.lora_r,
            # which a restart can change without changing the model.
            strategy = strategy_for(lora_model)
            is_lora = strategy.kind == "lora"

            # Save final adapter (or, for a full finetune, the full weights),
            # into this iteration's own directory, so the YAML written for it
            # keeps serving these weights after the next restart.
            export_dir = run_outputs.iteration_dir(args.output_dir, iteration, timestamp)
            logger.info("\nSaving LoRA adapter..." if is_lora else "\nSaving full finetuned weights...")
            exported = trainer.save_adapter(export_dir=str(export_dir))
            run_outputs.point_latest_export(Path(args.output_dir), export_dir, strategy.export_name)

            logger.info("\n" + "=" * 60)
            logger.info("Finetuning Complete!")
            logger.info(f"Best loss: {stats['best_loss']:.6f}")
            logger.info(
                f"{'Adapter' if is_lora else 'Weights'} saved to: {exported} "
                f"({Path(args.output_dir) / strategy.export_name} follows the latest iteration)"
            )
            logger.info("=" * 60)

            # Generate model files. The weights are saved by now, so a YAML
            # that cannot be written is reported, not treated as a failed
            # training run.
            finetuned_model_name = run_outputs.finetuned_model_name(model_config, timestamp)
            try:
                finetuned_model_name, yaml_path = run_outputs.write_serving_yaml(
                    args, model_config, timestamp, is_lora=is_lora, export_dir=export_dir
                )
            except Exception as e:
                logger.error(f"Training succeeded but the serving YAML could not be written: {e}", exc_info=True)
            else:
                # The job manager takes the YAML from here rather than
                # guessing where it went. Printed before the completion
                # marker so both are in the log when that is seen.
                markers.emit(markers.FINETUNED_MODEL_YAML, yaml_path)

            # Print completion marker with timestamp (for job manager to detect)
            markers.emit(markers.TRAINING_ITERATION_COMPLETE, finetuned_model_name)

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
                        markers.emit(markers.INFERENCE_SERVER_FAILED, e)
                        # The job was asked to train and serve, and cannot
                        # serve. This returned 0, so it showed as COMPLETED
                        # with nothing served and no sign why. The weights
                        # and YAML are saved all the same.
                        return 1
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
                apply_restart_params(args, restart_data)

                # A true restart: training starts again from the model it
                # started from, not from the previous iteration's weights --
                # once the next iteration is set up (see pending_reset).
                pending_reset = True
                markers.emit(markers.RESTARTING_TRAINING)
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
