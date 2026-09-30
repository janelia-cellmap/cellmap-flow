"""Following a finetune job until it ends.

``monitor_job`` runs on the job's own thread, which FinetuneJobManager
starts when it submits the job or finds it again. Every few seconds it asks
the scheduler about the job (state.on_scheduler_status) and reads what the
trainer has added to its log (tailer.LogTailer):
- the epoch and loss;
- the inference server's address, once it is up, which the listeners hear
  as ``on_server_ready``;
- each finished iteration's model, their ``on_iteration_complete``;
- the status markers (state.on_status_marker).

It records each change of status in the run's metadata.json, and, when the
job ends, its final status, model and serving YAML. A job the scheduler says
completed must have left its export (``complete_job``): it is COMPLETED only
once that is found, and FAILED if it is not.
"""

import logging
import time
from pathlib import Path

from cellmap_flow.finetune import markers
from cellmap_flow.finetune.job_manager import persistence, state
from cellmap_flow.finetune.job_manager.listener import Listeners
from cellmap_flow.finetune.job_manager.state import TERMINAL_STATUSES, FinetuneJob, JobStatus
from cellmap_flow.finetune.job_manager.tailer import LogTailer, trainer_outputs_from_log

logger = logging.getLogger(__name__)


def monitor_job(finetune_job: FinetuneJob, listeners: Listeners):
    """
    Background thread for job monitoring.

    Polls LSF status and tails log file to track training progress.
    Triggers completion when job finishes.

    Args:
        finetune_job: The FinetuneJob to monitor
        listeners: Whom to tell of its server and its iterations
    """
    job_id = finetune_job.job_id
    logger.info(f"Monitoring job {job_id}...")

    log = LogTailer(finetune_job.log_file)
    check_interval = 3  # seconds
    persisted_status = finetune_job.status
    ended = None  # how the scheduler says the job ended

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
                ended = state.on_scheduler_status(finetune_job, finetune_job.lsf_job.get_status())
                if ended:
                    break

            # === Tail log file for progress updates ===

            if finetune_job.log_file.exists():
                try:
                    # Whole lines only; see LogTailer.
                    new_content = log.read()
                    if new_content:
                        # Parse for epoch and loss information
                        _parse_training_progress(finetune_job, new_content)
                        # Parse for inference server ready marker
                        _parse_inference_server_ready(finetune_job, new_content, listeners)

                    # Always check for restart/iteration markers (reads full log).
                    # This must run every cycle, not just when there's new content,
                    # because the marker may have been at the end of the previous
                    # chunk and we need to detect it even if no new output follows.
                    _parse_training_restart(finetune_job, new_content, listeners)
                except Exception as e:
                    logger.debug(f"Error reading log file: {e}")

            if finetune_job.status != persisted_status:
                persisted_status = finetune_job.status
                persistence.update_metadata(
                    finetune_job.output_dir, status=persisted_status.value,
                    inference_server_url=finetune_job.inference_server_url,
                )

            # Sleep before next check
            time.sleep(check_interval)

    except Exception as e:
        logger.error(f"Error monitoring job {job_id}: {e}")
        if finetune_job.status not in TERMINAL_STATUSES:
            finetune_job.status = JobStatus.FAILED

    finally:
        # === Post-completion actions ===

        if ended == JobStatus.COMPLETED:
            try:
                complete_job(finetune_job)
                outcome = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error in post-completion for job {job_id}: {e}")
                outcome = JobStatus.FAILED
            # Only now, so that no one sees COMPLETED turn into FAILED (see
            # state.on_scheduler_status). A cancel that came in meanwhile stands.
            if finetune_job.status not in TERMINAL_STATUSES:
                finetune_job.status = outcome
        else:
            # A job that failed after training -- its server would not
            # start, say -- still produced a model; record what it was.
            _read_trainer_outputs(finetune_job)

        persistence.update_metadata(
            finetune_job.output_dir,
            status=finetune_job.status.value,
            finetuned_model_name=finetune_job.finetuned_model_name,
            model_yaml_path=str(finetune_job.model_yaml_path) if finetune_job.model_yaml_path else None,
        )
        logger.info(f"Stopped monitoring job {job_id}. Final status: {finetune_job.status.value}")


def _parse_training_progress(finetune_job: FinetuneJob, log_content: str):
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


def _parse_inference_server_ready(finetune_job: FinetuneJob, log_content: str, listeners: Listeners):
    """
    Parse log for CELLMAP_FLOW_SERVER_IP marker and tell the listeners
    the job's inference server is up.

    Args:
        finetune_job: Job to update
        log_content: New log content to parse
        listeners: Whom to tell
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
    logger.info(f"Finetuned inference server detected at {server_url}")

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
        _read_trainer_outputs(finetune_job, set_name=False)
    except Exception as e:
        logger.error(f"Could not read the served model's name from {finetune_job.log_file}: {e}",
                     exc_info=True)
        return

    listeners.notify("on_server_ready", finetune_job, server_url, model_name)
    # Whatever the listeners managed (see FinetuneJobListener), and so
    # that the iteration it serves is not announced again.
    finetune_job.finetuned_model_name = model_name


def _parse_training_restart(finetune_job: FinetuneJob, log_content: str, listeners: Listeners):
    """
    Parse log for RESTARTING_TRAINING and TRAINING_ITERATION_COMPLETE markers
    to handle iterative training restarts.

    On RESTARTING_TRAINING: reset training progress counters.
    On TRAINING_ITERATION_COMPLETE: tell the listeners the iteration's model.

    Args:
        finetune_job: Job to update
        log_content: New log content to parse
        listeners: Whom to tell
    """
    # Status markers, in the order they were printed: a restart that
    # follows a divergence in the same chunk leaves the job running, and
    # the reverse leaves it waiting.
    for marker in markers.STATUS_MARKER_RE.findall(log_content):
        state.on_status_marker(finetune_job, marker)

    # Check for iteration complete marker - tell the listeners.
    # Read full log in case the marker was in a previous chunk.
    try:
        full_log = finetune_job.log_file.read_text()
    except Exception:
        full_log = log_content
    iter_matches = markers.ITERATION_COMPLETE_RE.findall(full_log)
    # Only process new iteration-complete markers (ignore ones already handled).
    # After a restart, _processed_iteration_count stays at the old count so
    # previously-seen markers don't re-trigger inference_server_ready or
    # the listeners.
    if len(iter_matches) > finetune_job._processed_iteration_count:
        finetune_job._processed_iteration_count = len(iter_matches)

        # For in-process restarts, the inference server usually stays on the same
        # URL and does not emit a fresh CELLMAP_FLOW_SERVER_IP marker. Mark the
        # server as ready once we see a completed training iteration if URL exists.
        if finetune_job.inference_server_url:
            finetune_job.inference_server_ready = True

        new_model_name = iter_matches[-1]
        _read_trainer_outputs(finetune_job, set_name=False)
        if new_model_name != finetune_job.finetuned_model_name:
            logger.info(f"New training iteration complete: {new_model_name}")
            listeners.notify("on_iteration_complete", finetune_job, new_model_name)
            # Whatever the listeners managed -- without a server no layer
            # is added -- show the new name, and don't retry every poll.
            finetune_job.finetuned_model_name = new_model_name


def _read_trainer_outputs(finetune_job: FinetuneJob, set_name: bool = True):
    """Take the latest iteration's model name and serving YAML from the log.

    The trainer prints "FINETUNED_MODEL_YAML: <path>" and then
    "TRAINING_ITERATION_COMPLETE: <name>" for every iteration it
    finishes. ``set_name=False`` leaves finetuned_model_name alone, for the
    monitor, whose listeners use the old name (see FinetuneJobListener).
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


def complete_job(finetune_job: FinetuneJob):
    """
    Post-training actions after job completes successfully.

    1. Verify adapter files exist
    2. Take the model name and serving YAML the trainer reported
    3. Record the completion in metadata.json; the monitor makes the job
       COMPLETED once this has returned

    Args:
        finetune_job: The completed job

    Raises:
        RuntimeError: If its export is missing (persistence.check_export);
            nothing else here raises
    """
    job_id = finetune_job.job_id
    logger.info(f"Running post-completion for job {job_id}...")

    # === Verify the training export exists ===

    persistence.check_export(finetune_job.output_dir, finetune_job.params)

    # === The name and YAML the trainer gave the result ===
    #
    # This used to make up its own: the job's creation time instead of
    # the iteration's, and a sanitized model name instead of the
    # trainer's. So the YAML it looked for never existed, it always
    # generated a second one (with the dashboard's current norms rather
    # than the training ones), and metadata.json named a model that
    # neither the viewer layer nor the registered config did. The trainer
    # prints both, and is the only thing that knows them.
    _read_trainer_outputs(finetune_job)
    finetuned_model_name = finetune_job.finetuned_model_name
    yaml_path = finetune_job.model_yaml_path
    if yaml_path is None:
        logger.warning(
            f"Job {job_id}: the trainer reported no serving YAML (see its "
            f"log for why); the weights are in {finetune_job.output_dir}."
        )
    else:
        logger.info(f"Serving YAML for {finetuned_model_name}: {yaml_path}")

    # === Update metadata file with completion info ===

    persistence.record_completion(finetune_job)

    logger.info(f"Job {job_id} completed successfully!")
