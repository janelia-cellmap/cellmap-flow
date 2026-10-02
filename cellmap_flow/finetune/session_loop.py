"""The finetune job's loop: train, export, serve, wait for a restart, reset, again.

``TrainingSession.run`` is the job. Each iteration builds its data, target
and trainer, trains, exports into its own iterations/NNN_<ts>/ (see
run_outputs), and announces the result on stdout (see markers). A served
job (--auto-serve) then starts, or keeps, its inference server, which shares
the model object, and waits for a restart: from the server's control
endpoint (RestartController) or a restart_signal.json in the output
directory. A restart may change training settings only (cli.RESTARTABLE_ARGS);
training then starts again from the model the job started from.

The job manager follows the job through those stdout lines, and a job
outlives a dashboard upgrade, so their text and order only change
compatibly.
"""

import gc
import json
import logging
import socket
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Collection, Optional

import torch

from cellmap_flow.finetune import markers, run_outputs
from cellmap_flow.finetune.adaptation import strategy_for
from cellmap_flow.finetune.cli import apply_restart_params, build_target_transform, update_run_metadata
from cellmap_flow.finetune.data import create_dataloader
from cellmap_flow.finetune.lora_trainer import LoRAFinetuner
from cellmap_flow.io.paths import is_remote
from cellmap_flow.models.models_config import ModelConfig
from cellmap_flow.serving.restart_token import read_or_create_restart_token

logger = logging.getLogger(__name__)


class RestartController:
    """Hands a restart from the inference server's thread to the training loop.

    The server's ``/__control__/restart`` endpoint, once it has checked the
    job's token, calls ``request_restart`` with the request's body; the
    loop's wait (_wait_for_restart_signal) polls ``get_if_triggered``. A
    request that comes while the job is still training is kept until the
    wait; a second one replaces the first.
    """

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
        trained_model: The model just trained: LoRA-wrapped, or fully
            finetuned
        restart_controller: Where the server's restart endpoint hands
            restart requests, with the job's restart token required; None
            for a server that takes none

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

    if not is_remote(args.serve_data_path) and not Path(args.serve_data_path).exists():
        raise ValueError(f"Data path not found: {args.serve_data_path}")

    # Use the already-trained model
    logger.info("Using the trained model for inference...")

    from cellmap_flow.models.configs.base import _get_device
    device = _get_device()
    trained_model.eval()
    logger.info(f"Model set to eval mode on {device}")

    # Replace the model in the config with our finetuned version
    model_config.config.model = trained_model

    # Start server
    from cellmap_flow.server import CellMapFlowServer, get_free_port

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


def _restart_timestamp(signal_data: dict) -> Optional[str]:
    """The job manager's timestamp of a restart, which names it; None without one."""
    timestamp = signal_data.get("timestamp")
    return timestamp if isinstance(timestamp, str) and timestamp else None


def _wait_for_restart_signal(
    signal_file: Optional[Path],
    check_interval: float = 1.0,
    restart_controller: Optional[RestartController] = None,
    applied: Collection[str] = (),
):
    """Block until a restart is asked for, and return its signal.

    A restart comes from the inference server's control endpoint, through
    ``restart_controller``, or, when the job manager cannot reach that, as
    the ``signal_file`` it writes instead. Both carry the job manager's
    timestamp; a signal whose timestamp is in ``applied`` has been acted on
    already and is dropped. That happens when the HTTP request got through
    but its reply timed out: the job manager then writes the file as well.

    The job manager writes the file from another host, so it can be seen
    empty or cut short. Until it parses, it is left alone and the wait goes
    on.

    Returns:
        The signal, a dict with the restart's ``params``; or None for a file
        that parses but is not a restart signal.
    """
    logger.info(f"Watching for restart signal (controller + file fallback: {signal_file})")
    # The job manager's cue that this job is idle and can take a restart.
    markers.emit(markers.WAITING_FOR_RESTART)

    unreadable = None  # what the file last failed with, so it is logged once
    while True:
        if restart_controller is not None:
            in_memory_signal = restart_controller.get_if_triggered()
            if in_memory_signal is not None:
                if _restart_timestamp(in_memory_signal) in applied:
                    logger.info(f"Ignoring a restart that was applied already: {in_memory_signal}")
                    continue
                logger.info(f"Restart signal received via HTTP control endpoint: {in_memory_signal}")
                return in_memory_signal

        if signal_file and signal_file.exists():
            try:
                signal_data = json.loads(signal_file.read_text())
            except (OSError, ValueError) as e:
                if str(e) != unreadable:
                    logger.warning(f"Cannot read {signal_file} yet ({e}); waiting for it to be written.")
                    unreadable = str(e)
            else:
                unreadable = None
                signal_file.unlink(missing_ok=True)
                if not isinstance(signal_data, dict):
                    logger.error(f"Restart signal is not a restart request: {signal_data!r}")
                    return None
                if _restart_timestamp(signal_data) in applied:
                    logger.info(f"Ignoring a restart signal file for a restart applied already: {signal_data}")
                    continue
                logger.info(f"Restart signal received: {signal_data}")
                return signal_data
        time.sleep(check_interval)


class TrainingSession:
    """One finetune job, iteration after iteration, until it ends.

    ``model`` is the model to train, already prepared by ``strategy`` (the
    request decides the strategy, once, in finetune_cli.main; from then on
    the model does, through strategy_for). It is built once and reused
    across restarts: the inference server shares the object, so a restart
    retrains what is being served.
    """

    def __init__(self, args, model_config: ModelConfig, model, strategy):
        self.args = args
        self.model_config = model_config
        self.model = model
        # Given to the inference server when it starts, which hands it the
        # restarts it is sent.
        self.restart_controller = RestartController()
        self.server_started = False
        self.iteration = 0
        # A full finetune's distillation teacher: a frozen copy of the
        # starting weights, made by the first trainer that needs one and
        # reused after, so a restart neither copies the model again nor
        # distils toward weights an earlier iteration already changed.
        self.teacher_model = None
        # The weights a full finetune starts from, on the CPU, to reset it to
        # on restart. LoRA resets by re-making its adapter and needs none.
        self.initial_state = strategy.initial_state(model)
        # Where the next iteration's TensorBoard curves start: (step, epoch).
        self.tb_position = (0, 0)
        # The timestamps of the restarts applied, so that the same restart
        # arriving twice (over HTTP and as a file) is applied once.
        self.applied_restarts = set()
        # Set by a restart; the reset waits until the next iteration is set
        # up, and stays pending until its trainer is built.
        self.pending_reset = False
        # What the shared model holds, so that a restart that fails can say
        # what goes on being served.
        self.model_holds = "the starting weights"

    def run(self) -> int:
        """Train (and serve, and retrain on each restart) until the job ends; its exit code."""
        args = self.args
        while True:
            self.iteration += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.iteration > 1:
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"Training Iteration {self.iteration}")
                logger.info("=" * 60)

            # Set up this iteration: its data, its target and its trainer. A
            # restart's new settings or annotations are first used here, so an
            # error here -- an empty volume, offsets that do not fit -- must
            # not end the whole job and take the served model down with it.
            try:
                trainer = self._set_up_iteration()
            except Exception as e:
                logger.error(f"Could not set up training iteration {self.iteration}: {e}", exc_info=True)
                if not self._keep_serving_after(e):
                    return 1
                continue

            try:
                # Training, too, can fail on what a restart changed: data that
                # cannot be read, a target the model's output does not fit.
                try:
                    if self.iteration > 1:
                        markers.emit(markers.RESTART_STATUS, "Starting training...")
                    stats = self._train(trainer)
                    if not stats.get('diverged'):
                        self._export(trainer, stats, timestamp)
                except Exception as e:
                    logger.error(f"Training failed: {e}", exc_info=True)
                    self.model_holds = f"the partly trained weights of training iteration {self.iteration}"
                    # The next iteration starts from the starting weights again.
                    self.pending_reset = True
                    if not self._keep_serving_after(e):
                        return 1
                    continue

                if stats.get('diverged'):
                    # Skip saving, and wait for a restart with other settings.
                    logger.warning("Training diverged — skipping model save.")
                    self.model_holds = f"the weights of training iteration {self.iteration}, which diverged"
                    if not args.auto_serve:
                        return 1
                    if not self.server_started:
                        # No iteration has completed, so there is no model to
                        # serve while the job waits. A restart could still
                        # reach it through the signal file, but waiting holds
                        # the GPU, perhaps until walltime, for what a resubmit
                        # does as well. Exit so the job shows as failed and the
                        # GPU is freed.
                        logger.error(
                            "The first training iteration diverged, so no inference "
                            "server is running to receive a restart. Exiting; "
                            "resubmit with other settings (e.g. a lower learning rate)."
                        )
                        return 1
                else:
                    if not args.auto_serve:
                        return 0
                    if not self._serve():
                        return 1

                if not self._await_restart():
                    return 1
                # A true restart: training starts again from the model it
                # started from, not from the previous iteration's weights --
                # once the next iteration is set up.
                self.pending_reset = True

            except KeyboardInterrupt:
                logger.info("\nTraining interrupted by user")
                logger.info("Saving current state...")
                trainer.save_checkpoint(is_best=False)
                return 1

            except Exception as e:
                logger.error(f"The finetune job failed: {e}", exc_info=True)
                return 1

    def _train(self, trainer) -> dict:
        """Train this iteration; what the next one carries on with is kept even if training fails."""
        try:
            return trainer.train()
        finally:
            # None again if an OOM made the trainer drop distillation.
            self.teacher_model = trainer.teacher_model
            self.tb_position = (trainer._tb_step, trainer._tb_epoch)
            trainer.close()

    def _keep_serving_after(self, error) -> bool:
        """Report an iteration that failed and wait for the next restart; False if the job ends instead.

        Only a job that is serving outlives the failure: its model is still
        of use, and the dashboard can restart it with other settings.
        Before that -- the first iteration, or a run that does not serve --
        nothing could restart it (see the divergence case in run), so it
        fails.
        """
        if not (self.args.auto_serve and self.server_started):
            return False
        markers.emit(markers.RESTART_FAILED, error)
        # What the model holds depends on where the iteration failed: the
        # previous iteration's model if its data or target did, as the reset
        # waits for those (see _set_up_iteration); the starting weights if
        # its trainer did; partly trained weights if training did. The server
        # shares the model, and serves it in eval mode, whatever it holds.
        self.model.eval()
        logger.warning(f"Serving {self.model_holds} until a restart with settings that work.")
        return self._await_restart()

    def _set_up_iteration(self) -> LoRAFinetuner:
        """This iteration's data, target and trainer, with the model reset if a restart asked."""
        args = self.args
        # Re-created each iteration, to pick up new annotations.
        if self.iteration > 1:
            markers.emit(markers.RESTART_STATUS, "Loading corrections...")
        logger.info(f"Loading corrections from {args.corrections}...")
        dataloader = create_dataloader(
            args.corrections,
            batch_size=args.batch_size,
            augment=not args.no_augment,
            num_workers=args.num_workers,
        )
        logger.info(f"DataLoader created: {len(dataloader.dataset)} corrections")
        self._record_input_norm()

        # Re-built each iteration, to pick up restart params.
        target_transform = build_target_transform(args, self.model_config)
        logger.info(f"output_type={args.output_type}, select_channel={args.select_channel}")

        # Only now that its data and target are built: put the model back
        # where training started (see _reset_model). Until here it is still
        # the previous iteration's, which the server keeps serving. The
        # trainer has to be built on the reset model -- a LoRA reset makes
        # new adapter parameters for its optimizer -- so it comes after.
        if self.pending_reset:
            self._reset_model()

        # Re-created each iteration, for a fresh optimizer and scheduler.
        if self.iteration > 1:
            markers.emit(markers.RESTART_STATUS, "Preparing trainer...")
        logger.info("Creating trainer...")
        trainer = LoRAFinetuner(
            self.model,
            dataloader,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_mixed_precision=not args.no_mixed_precision,
            loss_type=args.loss_type,
            select_channel=args.select_channel,
            mask_unannotated=args.mask_unannotated,
            label_smoothing=args.label_smoothing,
            distillation_lambda=args.distillation_lambda,
            distillation_all_voxels=args.distillation_all_voxels,
            margin=args.margin,
            balance_classes=args.balance_classes,
            target_transform=target_transform,
            tensorboard=not args.no_tensorboard,
            teacher_model=self.teacher_model,
            initial_state=self.initial_state,
            tb_start_step=self.tb_position[0],
            tb_start_epoch=self.tb_position[1],
        )
        # Done only now: had the trainer failed, the model would have been
        # reset all the same, and the next restart must reset it again, with
        # its own rank.
        self.pending_reset = False

        # Resume from checkpoint if specified (first iteration only)
        if args.resume and self.iteration == 1:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        return trainer

    def _record_input_norm(self) -> None:
        """Snapshot the manifest's input_norm into metadata.json.

        So any checkpoint saved in this iteration is reproducible: the
        metadata.json next to the .pth says which normalization the training
        data went through. It is stored as the manifest has it: the
        dashboard's ordered ``[{name, **params}]`` steps, or the older
        ``{Name: params}`` dict (build_corrections' default).
        """
        args = self.args
        try:
            from cellmap_flow.finetune.session.manifest import read_manifest

            manifest_norm = (read_manifest(args.corrections) or {}).get("input_norm")
            if manifest_norm is not None and args.output_dir:
                def record(metadata):
                    metadata.setdefault("params", {})["input_norm"] = manifest_norm

                if update_run_metadata(args.output_dir, record):
                    logger.info(f"Snapshot input_norm into {args.output_dir}/metadata.json: {manifest_norm}")
        except Exception as e:
            logger.warning(f"Could not snapshot input_norm into metadata.json: {e}")

    def _export(self, trainer, stats, timestamp) -> None:
        """Export this iteration, write the YAML that serves it, and announce both."""
        args = self.args
        output_dir = Path(args.output_dir)
        # What was exported is decided by the model, not by args.lora_r,
        # which a restart can change without changing the model.
        strategy = strategy_for(self.model)
        is_lora = strategy.kind == "lora"

        # The adapter (or, for a full finetune, the full weights), into this
        # iteration's own directory, so the YAML written for it keeps serving
        # these weights after the next restart.
        export_dir = run_outputs.iteration_dir(output_dir, self.iteration, timestamp)
        logger.info("\nSaving LoRA adapter..." if is_lora else "\nSaving full finetuned weights...")
        exported = trainer.save_adapter(export_dir=str(export_dir))
        run_outputs.point_latest_export(output_dir, export_dir, strategy.export_name)
        self.model_holds = f"the weights exported to {export_dir}"

        logger.info("\n" + "=" * 60)
        logger.info("Finetuning Complete!")
        logger.info(f"Best loss: {stats['best_loss']:.6f}")
        logger.info(
            f"{'Adapter' if is_lora else 'Weights'} saved to: {exported} "
            f"({output_dir / strategy.export_name} follows the latest iteration)"
        )
        logger.info("=" * 60)

        # The weights are saved by now, so a YAML that cannot be written is
        # reported, not treated as a failed training run.
        finetuned_model_name = run_outputs.finetuned_model_name(self.model_config, timestamp)
        try:
            finetuned_model_name, yaml_path = run_outputs.write_serving_yaml(
                args, self.model_config, timestamp, is_lora=is_lora, export_dir=export_dir
            )
        except Exception as e:
            logger.error(f"Training succeeded but the serving YAML could not be written: {e}", exc_info=True)
        else:
            # The job manager takes the YAML from here rather than guessing
            # where it went. Printed before the completion marker so both are
            # in the log when that is seen.
            markers.emit(markers.FINETUNED_MODEL_YAML, yaml_path)

        # The job manager's cue that the iteration is done, and its model's name.
        markers.emit(markers.TRAINING_ITERATION_COMPLETE, finetuned_model_name)

    def _serve(self) -> bool:
        """Serve the model just trained; False if the server could not start.

        The first iteration starts the inference server in a background
        thread. After that it is running, and serves the updated weights
        already: it shares the model object.
        """
        if self.server_started:
            self.model.eval()
            logger.info("Model updated and set to eval mode. Server continuing with new weights.")
            return True
        try:
            _start_inference_server_background(
                self.args, self.model_config, self.model, restart_controller=self.restart_controller
            )
        except Exception as e:
            logger.error(f"Failed to start inference server: {e}", exc_info=True)
            markers.emit(markers.INFERENCE_SERVER_FAILED, e)
            # The job was asked to train and serve, and cannot serve: it
            # fails rather than show as COMPLETED with nothing served and no
            # sign why. The weights and YAML are saved all the same.
            return False
        self.server_started = True
        return True

    def _await_restart(self) -> bool:
        """Wait for the next restart and apply its settings; False for a signal file that is not a restart request."""
        restart_data = _wait_for_restart_signal(
            signal_file=Path(self.args.output_dir) / "restart_signal.json",
            check_interval=1.0,
            restart_controller=self.restart_controller,
            applied=self.applied_restarts,
        )
        if restart_data is None:
            logger.error("Malformed restart signal, exiting")
            return False
        if _restart_timestamp(restart_data):
            self.applied_restarts.add(_restart_timestamp(restart_data))
        apply_restart_params(self.args, restart_data)
        markers.emit(markers.RESTARTING_TRAINING)
        return True

    def _record_rank(self) -> None:
        """Put the rank and alpha the job keeps into its metadata.json.

        apply_restart_params recorded the ones the restart asked for, which
        a dashboard started later would take for the job's.
        """
        args = self.args
        if not args.output_dir:
            return

        def record(metadata):
            params = metadata.get("params", {})
            for key in ("lora_r", "lora_alpha"):
                if key in params:
                    params[key] = getattr(args, key)

        try:
            update_run_metadata(args.output_dir, record)
        except Exception as e:
            logger.warning(f"Could not record the kept rank in {args.output_dir}/metadata.json: {e}")

    def _reset_model(self) -> None:
        """Put the model back where training started, for the next iteration.

        LoRA: unload the adapter and wrap a fresh one around the base (which
        is how the rank can change on restart). Full finetune: load the
        starting weights back (see the strategies' restart()). The model
        object -- which the inference server shares -- is built once, so a
        restart cannot switch between the two kinds: the model decides, and
        args.lora_r is made to agree with it.
        """
        args = self.args
        kept = strategy_for(self.model)
        if kept.kind == "lora" and args.lora_r <= 0:
            logger.warning("Restart asked for lora_r=0 (full finetune) but this job trains a LoRA adapter; "
                           "submit a new job for that. Keeping the current adapter setup.")
            args.lora_r = max(1, int(kept.r))
            # Its alpha too: the restart derived 2 x 0 = 0 for it, and an
            # adapter scaled by alpha / r = 0 trains nothing.
            args.lora_alpha = kept.alpha
            self._record_rank()
        elif kept.kind == "full" and args.lora_r > 0:
            # The mirror image of the case above. Left alone, args.lora_r > 0
            # made the next iteration's YAML point at a lora_adapter/ this job
            # never writes. A full finetune has no adapter for an alpha to
            # scale: 0, as submit records it.
            logger.warning(f"Restart asked for LoRA rank {args.lora_r} but this job is a full finetune; "
                           "submit a new job for that. Keeping the full finetune.")
            args.lora_r = args.lora_alpha = 0
            self._record_rank()

        strategy = strategy_for(
            self.model,
            args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            min_channels=args.lora_min_channels,
        )
        self.model = strategy.restart(self.model, self.initial_state)
        self.model_holds = "the starting weights"
        self.model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Restarting training from the starting weights...")
