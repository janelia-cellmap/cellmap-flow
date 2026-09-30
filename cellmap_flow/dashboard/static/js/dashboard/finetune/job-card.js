// The Training Status card: which job, its status and progress, and its
// model's inference server; and the state of its Restart and Stop Early
// buttons. What the card shows comes from three places: a job just
// submitted, the job's status (polled, or its entry in the job list after a
// reload), and the lines of its log.
import { EPOCH_LOSS } from "./loss-plot.js";

// The status badge's color, by the job manager's JobStatus. SUBMITTED is the
// card's own, until the first status answer.
const STATUS_COLORS = {
  SUBMITTED: "warning",
  PENDING: "warning",
  RUNNING: "primary",
  COMPLETED: "success",
  FAILED: "danger",
  CANCELLED: "secondary",
  WAITING_FOR_RESTART: "info",
};

const TERMINAL = ["FAILED", "CANCELLED", "COMPLETED"];

// lossPlot: the Training Logs card's plot, which a new run empties and each
// epoch's loss goes into.
export function createJobCard(lossPlot) {
  const $ = (id) => document.getElementById(id);
  const progressText = $("jobProgress");
  const progressBar = $("trainingProgressBar");
  const serverStatus = $("inferenceServerStatus");
  const restartBtn = $("restartJobBtn");
  const stopEarlyBtn = $("stopEarlyBtn");

  // The epoch comes both from the log and from the status, which can
  // disagree for a moment. The bar keeps the latest epoch of the run it
  // shows, so a late report of an earlier one is ignored; once the model is
  // served the bar says so, until a restart begins a new run.
  let liveEpochState = null;  // { current, total } the bar shows
  let restartEpochResetPending = false;  // a restart began; its epochs start over
  let isServingReady = false;
  let stopRequested = false;

  function shouldAcceptEpochProgress(current, total) {
    if (isServingReady && !restartEpochResetPending) {
      return false;
    }
    if (!liveEpochState) {
      return true;
    }
    if (restartEpochResetPending) {
      return true;
    }
    if (total !== liveEpochState.total) {
      return true;
    }
    if (current < liveEpochState.current) {
      return false;
    }
    return true;
  }

  function renderEpochProgress(current, total) {
    const safeTotal = total > 0 ? total : 1;
    const progressPercent = Math.max(0, Math.min(100, (current / safeTotal) * 100));
    progressBar.style.width = progressPercent + "%";
    progressBar.className = "progress-bar progress-bar-striped progress-bar-animated";
    progressBar.textContent = `${current}/${safeTotal} epochs`;
    liveEpochState = { current, total: safeTotal };
    if (current > 0 || restartEpochResetPending) {
      restartEpochResetPending = false;
    }
  }

  function renderServingReadyProgress() {
    progressBar.style.width = "100%";
    progressBar.className = "progress-bar bg-success";
    progressBar.textContent = "Serving - Ready for inference";
    isServingReady = true;
    restartEpochResetPending = false;
  }

  function resetProgressBarToWaiting() {
    progressBar.style.width = "0%";
    progressBar.className = "progress-bar progress-bar-striped progress-bar-animated";
    progressBar.textContent = "0%";
  }

  function showJob(jobId, modelName) {
    $("trainingStatusCard").style.display = "block";
    $("jobId").textContent = jobId;
    $("jobModelName").textContent = modelName;
  }

  function showOutputType(outputType) {
    const badge = $("jobOutputType");
    badge.textContent = outputType;
    badge.className = "badge " + (outputType === "affinities" ? "bg-warning" : "bg-info");
  }

  function showStatus(status) {
    const badge = $("jobStatus");
    badge.textContent = status;
    badge.className = "badge bg-" + (STATUS_COLORS[status] || "secondary");
  }

  // The epoch a job's status reports, if the bar takes it; says whether it did.
  function showReportedEpoch(job) {
    if (!(job.current_epoch && job.total_epochs &&
          shouldAcceptEpochProgress(job.current_epoch, job.total_epochs))) {
      return false;
    }
    renderEpochProgress(job.current_epoch, job.total_epochs);
    progressText.textContent =
      `Epoch ${job.current_epoch}/${job.total_epochs}` +
      (job.loss ? ` - Loss: ${job.loss.toFixed(4)}` : "");
    return true;
  }

  function showServer(modelName) {
    serverStatus.style.display = "block";
    $("neuroglancerLayer").textContent = modelName;
  }

  // Stop Early, back to its label and disabled; the status poll enables it
  // while a training loop actually runs.
  function resetStopEarly() {
    stopRequested = false;
    stopEarlyBtn.textContent = "⏸ Stop Early";
    stopEarlyBtn.disabled = true;
  }

  // A new run is about to begin: nothing of the last one stays.
  function restarting() {
    restartEpochResetPending = true;
    isServingReady = false;
    liveEpochState = null;
    serverStatus.style.display = "none";
    progressText.textContent = "Restarting training...";
    resetProgressBarToWaiting();
    lossPlot.reset();
  }

  return {
    // The job's id as the card shows it, or "-" when it shows none.
    jobId: () => $("jobId").textContent,

    // A job just submitted, before its first status answer.
    submitted(jobId, modelName, outputType) {
      showJob(jobId, modelName);
      showOutputType(outputType);
      showStatus("SUBMITTED");
      liveEpochState = null;
      restartEpochResetPending = false;
      isServingReady = false;
      resetProgressBarToWaiting();
      lossPlot.reset();
      serverStatus.style.display = "none";
      restartBtn.style.display = "none";
    },

    // A job found again after a page reload, from its entry in the job
    // list. Its loss plot comes from its log instead, and what only the
    // poll shows (the waiting text, Stop Early) waits for the first poll.
    restored(job) {
      showJob(job.job_id, job.model_name);
      showOutputType((job.params && job.params.output_type) || "binary");
      showStatus(job.status);
      showReportedEpoch(job);
      if (job.status === "WAITING_FOR_RESTART") {
        restartBtn.style.display = "inline-block";
      }
      if (job.inference_server_ready) {
        renderServingReadyProgress();
        restartBtn.style.display = "inline-block";
        if (job.finetuned_model_name) showServer(job.finetuned_model_name);
      }
    },

    // A status answer (/api/finetune/job/<id>/status).
    polled(data) {
      if (data.params && data.params.output_type) {
        showOutputType(data.params.output_type);
      }
      showStatus(data.status);

      // Surface status transitions even if log stream is temporarily delayed
      if (data.status === "PENDING") {
        progressText.textContent = "Queued and waiting for worker";
        isServingReady = false;
      } else if (data.status === "RUNNING" && !(data.current_epoch && data.total_epochs) && !liveEpochState) {
        progressText.textContent = "Worker started; waiting for first epoch log...";
        isServingReady = false;
      }

      if (showReportedEpoch(data) && Number.isFinite(data.loss)) {
        lossPlot.add(data.current_epoch, data.loss);
      }

      // Restart: while the model is serving, or the trainer is waiting for
      // a restart (an iteration finished, or diverged).
      if (data.inference_server_ready || data.status === "WAITING_FOR_RESTART") {
        restartBtn.style.display = "inline-block";
      }

      // Stop Early only means something while a training loop is running,
      // so it is enabled then, and only then; unless a stop has been
      // requested, which keeps it disabled with its own label.
      if (!stopRequested) {
        stopEarlyBtn.disabled = !(data.status === "RUNNING" && !data.inference_server_ready);
      }

      // The model is served (the training iteration is complete): the loop
      // has exited, naturally or by Stop Early.
      if (data.inference_server_ready && data.finetuned_model_name) {
        showServer(data.finetuned_model_name);
        renderServingReadyProgress();
        resetStopEarly();
      }

      if (TERMINAL.includes(data.status)) {
        resetStopEarly();
      }
    },

    // A line of the job's log: restarts, their progress, and the epochs.
    logLine(line) {
      if (!line) return;

      if (line.includes("RESTARTING_TRAINING")) {
        restarting();
        return;
      }

      // Show restart sub-status updates (e.g. "Loading corrections...", "Preparing trainer...")
      const restartStatusMatch = line.match(/RESTART_STATUS:\s*(.+)/);
      if (restartStatusMatch) {
        progressText.textContent = restartStatusMatch[1].trim();
        return;
      }

      // Parse "Epoch X/Y" anywhere in the line.
      const epochMatch = line.match(/Epoch\s+(\d+)\/(\d+)/i);
      if (epochMatch) {
        const current = parseInt(epochMatch[1], 10);
        const total = parseInt(epochMatch[2], 10);
        if (!Number.isNaN(current) && !Number.isNaN(total) && total > 0 && shouldAcceptEpochProgress(current, total)) {
          renderEpochProgress(current, total);
          progressText.textContent = `Epoch ${current}/${total}`;
        }
      }

      // Only update progress text and plot from epoch-level summary lines
      // (e.g. "Epoch 18/20 - Loss: 0.011371"), not per-batch lines.
      const epochLossMatch = line.match(EPOCH_LOSS);
      if (epochLossMatch) {
        const epochVal = parseInt(epochLossMatch[1], 10);
        const totalVal = parseInt(epochLossMatch[2], 10);
        const lossVal = parseFloat(epochLossMatch[3]);
        if (!Number.isNaN(epochVal) && !Number.isNaN(lossVal)) {
          progressText.textContent = `Epoch ${epochVal}/${totalVal} - Loss: ${lossVal.toFixed(4)}`;
          lossPlot.add(epochVal, lossVal);
        }
      }
    },

    // A restart the job has accepted.
    restarted() {
      restarting();
      resetStopEarly();
      restartBtn.style.display = "none";
    },

    // Stop Early clicked: disabled, saying so, until the stop is refused or
    // the loop has exited.
    stopRequested() {
      stopRequested = true;
      stopEarlyBtn.disabled = true;
      stopEarlyBtn.textContent = "⏸ Stop requested...";
    },
    resetStopEarly,
  };
}
