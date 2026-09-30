// The Training panel's job: submitting it, following it (the status poll and
// the log stream), Restart, Stop Early and Cancel, and finding it again
// after a page reload.
import { poll } from "../../lib/poll.js";
import { createJobLog } from "./log-stream.js";
import { createLossPlot, EPOCH_LOSS } from "./loss-plot.js";
import { getAnswer, postAnswer } from "./requests.js";

function getStatusColor(status) {
  const colors = {
    "PENDING": "warning",
    "RUNNING": "primary",
    "COMPLETED": "success",
    "FAILED": "danger",
    "CANCELLED": "secondary",
    "WAITING_FOR_RESTART": "info"
  };
  return colors[status] || "secondary";
}

function showNotification(title, message) {
  // Browser notification
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification(title, { body: message });
  }

  // Also show in-page alert
  alert(`${title}\n\n${message}`);
}

// picker: the model picker; form: the training form.
export function initJobMonitor({ picker, form }) {
  let statusPoller = null;
  let liveEpochState = null;
  let restartEpochResetPending = false;
  let isServingReady = false;
  const lossPlot = createLossPlot(
    document.getElementById("lossPlotCanvas"), document.getElementById("lossPlotSummary"));
  const jobLog = createJobLog({ onLine: updateProgressFromLogLine });
  const appendLog = jobLog.append;

  // Start Finetuning button
  const startFinetuningBtn = document.getElementById("startFinetuningBtn");
  startFinetuningBtn.addEventListener("click", async function() {
    const selectedModel = picker.selected();
    if (!selectedModel) {
      alert("Please select a model first");
      return;
    }

    // Get corrections path from output path field
    const correctionsPath = document.getElementById("outputPath").value.trim();
    if (!correctionsPath) {
      alert("Please specify the output path where annotation crops are saved");
      return;
    }

    let patchesPerEpochOverride;
    let rehearsalFractionOverride;
    try {
      patchesPerEpochOverride = form.readPatchesPerEpochOverride();
      rehearsalFractionOverride = form.readRehearsalFractionOverride();
    } catch (error) {
      alert(error.message);
      return;
    }

    // Get training parameters from form
    const params = {
      model_name: selectedModel.name,
      corrections_path: correctionsPath,
      lora_r: parseInt(document.getElementById("loraRank").value),
      num_epochs: parseInt(document.getElementById("numEpochs").value),
      batch_size: parseInt(document.getElementById("batchSize").value),
      learning_rate: parseFloat(document.getElementById("learningRate").value),
      auto_serve: document.getElementById("autoServeCheck").checked,
      loss_type: document.getElementById("lossType").value,
      distillation_lambda: parseFloat(document.getElementById("distillationLambda").value),
      distillation_scope: document.getElementById("distillationScope").value,
      balance_classes: document.getElementById("balanceClasses").checked,
      augment: document.getElementById("augment").checked,
      label_smoothing: parseFloat(document.getElementById("labelSmoothing").value) || 0,
      queue: document.getElementById("gpuQueue").value
    };
    if (patchesPerEpochOverride !== undefined) {
      params.patches_per_epoch = patchesPerEpochOverride;
    }
    if (rehearsalFractionOverride !== undefined) {
      params.rehearsal_fraction = rehearsalFractionOverride;
    }

    // Forward margin value when using margin loss
    if (params.loss_type === "margin") {
      const m = parseFloat(document.getElementById("marginValue").value);
      if (Number.isFinite(m)) params.margin = m;
    }

    // Add optional checkpoint path override if provided
    const checkpointPath = document.getElementById("checkpointPath").value.trim();
    if (checkpointPath) {
      params.checkpoint_path = checkpointPath;
    }

    // Disable button and show loading. The submit endpoint may run a
    // pre-submit MinIO sync; if that takes a while, switch the spinner
    // label to make it obvious that the dashboard is doing real work,
    // not stuck.
    startFinetuningBtn.disabled = true;
    startFinetuningBtn.innerHTML =
      '<span class="spinner-border spinner-border-sm"></span> Submitting...';
    const slowLabelTimer = setTimeout(() => {
      startFinetuningBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm"></span> ' +
        'Syncing annotations from MinIO before submit (this can take several minutes)...';
    }, 3000);

    try {
      // Submit job
      const result = await postAnswer("/api/finetune/submit", params);

      if (result.success) {
        // Show training status card
        document.getElementById("trainingStatusCard").style.display = "block";
        document.getElementById("jobId").textContent = result.job_id;
        document.getElementById("jobModelName").textContent = params.model_name;
        const outputType = result.output_type || "binary";
        document.getElementById("jobOutputType").textContent = outputType;
        document.getElementById("jobOutputType").className = "badge " + (outputType === "affinities" ? "bg-warning" : "bg-info");
        document.getElementById("jobStatus").textContent = "SUBMITTED";
        document.getElementById("jobStatus").className = "badge bg-warning";

        // Clear logs and start streaming
        jobLog.clear();
        liveEpochState = null;
        restartEpochResetPending = false;
        isServingReady = false;
        resetProgressBarToWaiting();
        lossPlot.reset();
        document.getElementById('inferenceServerStatus').style.display = 'none';
        document.getElementById('restartJobBtn').style.display = 'none';
        appendLog("Finetuning job submitted successfully!");
        appendLog(`Job ID: ${result.job_id}`);
        appendLog(`LSF Job ID: ${result.lsf_job_id || 'N/A'}`);
        appendLog(`Output directory: ${result.output_dir}`);
        if (result.tensorboard_command) appendLog(`TensorBoard: ${result.tensorboard_command}`);
        if (result.note) {
          appendLog(`Note: ${result.note}`);
        }
        appendLog(`\nWaiting for training to start...\n`);

        // Start log streaming and status polling
        jobLog.follow(result.job_id);
        startStatusPolling(result.job_id);

        // Request notification permission if not already granted
        if ("Notification" in window && Notification.permission === "default") {
          Notification.requestPermission();
        }
      } else {
        alert(`Failed to submit job: ${result.error}`);
        appendLog(`✗ Error: ${result.error}\n`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
      appendLog(`✗ Error: ${error.message}\n`);
    } finally {
      // Re-enable button
      clearTimeout(slowLabelTimer);
      startFinetuningBtn.disabled = false;
      startFinetuningBtn.innerHTML = '🚀 Start Finetuning';
    }
  });

  function updateProgressFromLogLine(line) {
    if (!line) return;

    if (line.includes("RESTARTING_TRAINING")) {
      restartEpochResetPending = true;
      isServingReady = false;
      liveEpochState = null;
      document.getElementById('inferenceServerStatus').style.display = 'none';
      document.getElementById("jobProgress").textContent = "Restarting training...";
      resetProgressBarToWaiting();
      lossPlot.reset();
      return;
    }

    // Show restart sub-status updates (e.g. "Loading corrections...", "Preparing trainer...")
    const restartStatusMatch = line.match(/RESTART_STATUS:\s*(.+)/);
    if (restartStatusMatch) {
      document.getElementById("jobProgress").textContent = restartStatusMatch[1].trim();
      return;
    }

    // Parse "Epoch X/Y" anywhere in the line.
    const epochMatch = line.match(/Epoch\s+(\d+)\/(\d+)/i);
    if (epochMatch) {
      const current = parseInt(epochMatch[1], 10);
      const total = parseInt(epochMatch[2], 10);
      if (!Number.isNaN(current) && !Number.isNaN(total) && total > 0 && shouldAcceptEpochProgress(current, total)) {
        renderEpochProgress(current, total);
        document.getElementById("jobProgress").textContent = `Epoch ${current}/${total}`;
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
        document.getElementById("jobProgress").textContent =
          `Epoch ${epochVal}/${totalVal} - Loss: ${lossVal.toFixed(4)}`;
        lossPlot.add(epochVal, lossVal);
      }
    }
  }

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
    const progressBar = document.getElementById("trainingProgressBar");
    progressBar.style.width = progressPercent + "%";
    progressBar.className = "progress-bar progress-bar-striped progress-bar-animated";
    progressBar.textContent = `${current}/${safeTotal} epochs`;
    liveEpochState = { current, total: safeTotal };
    if (current > 0 || restartEpochResetPending) {
      restartEpochResetPending = false;
    }
  }

  function renderServingReadyProgress() {
    const progressBar = document.getElementById("trainingProgressBar");
    progressBar.style.width = "100%";
    progressBar.className = "progress-bar bg-success";
    progressBar.textContent = "Serving - Ready for inference";
    isServingReady = true;
    restartEpochResetPending = false;
  }

  function resetProgressBarToWaiting() {
    const progressBar = document.getElementById("trainingProgressBar");
    progressBar.style.width = "0%";
    progressBar.className = "progress-bar progress-bar-striped progress-bar-animated";
    progressBar.textContent = "0%";
  }

  // The job's status, every 3 seconds (the first after 3 s), until it fails
  // or is cancelled; this replaces the poller running before, if any. The
  // poll asks once at a time and stops at once when the job is over, so a
  // failed job notifies once, not once per tick that was waiting.
  function startStatusPolling(jobId) {
    if (statusPoller) statusPoller.stop();
    statusPoller = poll(async ({ stale }) => {
      try {
        const data = await getAnswer(`/api/finetune/job/${jobId}/status`);
        if (stale()) return;

        if (!data.success) {
          console.error("Error getting job status:", data.error);
          return;
        }

        // Update output type from job params
        if (data.params && data.params.output_type) {
          const ot = data.params.output_type;
          document.getElementById("jobOutputType").textContent = ot;
          document.getElementById("jobOutputType").className = "badge " + (ot === "affinities" ? "bg-warning" : "bg-info");
        }

        // Update status display
        const statusBadge = document.getElementById("jobStatus");
        statusBadge.textContent = data.status;

        // Color code status
        statusBadge.className = "badge bg-" + getStatusColor(data.status);

        // Surface status transitions even if log stream is temporarily delayed
        if (data.status === "PENDING") {
          document.getElementById("jobProgress").textContent = "Queued and waiting for worker";
          isServingReady = false;
        } else if (data.status === "RUNNING" && !(data.current_epoch && data.total_epochs) && !liveEpochState) {
          document.getElementById("jobProgress").textContent = "Worker started; waiting for first epoch log...";
          isServingReady = false;
        }

        // Update progress
        if (data.current_epoch && data.total_epochs && shouldAcceptEpochProgress(data.current_epoch, data.total_epochs)) {
          renderEpochProgress(data.current_epoch, data.total_epochs);

          document.getElementById("jobProgress").textContent =
            `Epoch ${data.current_epoch}/${data.total_epochs}` +
            (data.loss ? ` - Loss: ${data.loss.toFixed(4)}` : "");
          if (Number.isFinite(data.loss)) {
            lossPlot.add(data.current_epoch, data.loss);
          }
        }

        // Show restart button when the model is serving, or the trainer is
        // waiting for a restart (an iteration finished, or diverged)
        const restartBtn = document.getElementById("restartJobBtn");
        if (data.inference_server_ready || data.status === "WAITING_FOR_RESTART") {
          restartBtn.style.display = "inline-block";
        }

        // Stop Early is only meaningful while a training loop is actively
        // running. Enable on RUNNING + no inference server yet; disable
        // otherwise. Skip the toggle while the user has explicitly requested
        // a stop (the click handler set its own disabled+label).
        const stopEarlyBtnPoll = document.getElementById("stopEarlyBtn");
        if (stopEarlyBtnPoll.textContent.indexOf("requested") === -1) {
          const trainingActive = data.status === "RUNNING" && !data.inference_server_ready;
          stopEarlyBtnPoll.disabled = !trainingActive;
        }

        // Handle inference server becoming ready (training iteration complete)
        if (data.inference_server_ready && data.finetuned_model_name) {
          // Update inference server status display
          document.getElementById('inferenceServerStatus').style.display = 'block';
          document.getElementById('neuroglancerLayer').textContent = data.finetuned_model_name;

          // Update progress bar to show serving state
          renderServingReadyProgress();

          // Training loop has exited (either naturally or via stop-early) and
          // inference is up — clear any lingering "Stop requested..." state.
          resetStopEarlyButton();
        }

        // Handle terminal states
        if (data.status === "FAILED" || data.status === "CANCELLED" || data.status === "COMPLETED") {
          resetStopEarlyButton();
          // The stream normally ends itself with "done"; see jobLog.closeSoon.
          jobLog.closeSoon();
        }
        if (data.status === "FAILED" || data.status === "CANCELLED") {
          showNotification(
            "Training " + data.status,
            "Check the job log below for the traceback."
          );
          return false;
        }
      } catch (error) {
        console.error("Status polling error:", error);
      }
    }, { intervalMs: 3000, immediate: false });
  }

  // Cancel button
  document.getElementById("cancelJobBtn").addEventListener("click", async function() {
    const jobId = document.getElementById("jobId").textContent;
    if (!jobId || jobId === '-') {
      alert("No active job to cancel");
      return;
    }

    if (!confirm("Are you sure you want to cancel this training job?")) {
      return;
    }

    try {
      const result = await postAnswer(`/api/finetune/job/${jobId}/cancel`);

      if (result.success) {
        appendLog("\n=== TRAINING CANCELLED BY USER ===\n");
      } else {
        alert(`Failed to cancel: ${result.error}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  });

  // Stop Early — graceful, keeps the job alive for Restart.
  // Reset clears the "requested" label, then leaves the button DISABLED.
  // The status-polling loop re-enables it when training is actually running.
  function resetStopEarlyButton() {
    const btn = document.getElementById("stopEarlyBtn");
    btn.textContent = "⏸ Stop Early";
    btn.disabled = true;
  }

  const stopEarlyBtn = document.getElementById("stopEarlyBtn");
  stopEarlyBtn.addEventListener("click", async function() {
    const jobId = document.getElementById("jobId").textContent;
    if (!jobId || jobId === "-") {
      alert("No active job to stop.");
      return;
    }
    if (!confirm(
      "Stop training early after the current epoch?\n\n" +
      "The job stays alive and the inference server will come up on the " +
      "partially-trained adapter. You can then click Restart Training to " +
      "re-run with updated parameters."
    )) {
      return;
    }
    stopEarlyBtn.disabled = true;
    stopEarlyBtn.textContent = "⏸ Stop requested...";
    try {
      const result = await postAnswer(`/api/finetune/job/${jobId}/stop-early`);
      if (result.success) {
        appendLog("\n=== STOP EARLY REQUESTED — exiting after current epoch ===\n");
      } else {
        alert(`Failed to request stop: ${result.error}`);
        resetStopEarlyButton();
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
      resetStopEarlyButton();
    }
  });

  // Restart uses whatever's currently in the main training form as the source
  // of truth — no separate modal. User edits the form to change params, hits
  // Restart, confirms.
  const restartJobBtn = document.getElementById('restartJobBtn');
  restartJobBtn.addEventListener('click', async () => {
    const lossType = document.getElementById('lossType').value;
    let patchesPerEpochOverride;
    let rehearsalFractionOverride;
    try {
      patchesPerEpochOverride = form.readPatchesPerEpochOverride();
      rehearsalFractionOverride = form.readRehearsalFractionOverride();
    } catch (error) {
      alert(error.message);
      return;
    }
    const requestBody = {
      lora_r: parseInt(document.getElementById('loraRank').value),
      num_epochs: parseInt(document.getElementById('numEpochs').value),
      batch_size: parseInt(document.getElementById('batchSize').value),
      learning_rate: parseFloat(document.getElementById('learningRate').value),
      loss_type: lossType,
      distillation_lambda: parseFloat(document.getElementById('distillationLambda').value),
      distillation_scope: document.getElementById('distillationScope').value,
      balance_classes: document.getElementById('balanceClasses').checked,
      augment: document.getElementById('augment').checked,
      label_smoothing: parseFloat(document.getElementById('labelSmoothing').value) || 0,
    };
    if (patchesPerEpochOverride !== undefined) {
      requestBody.patches_per_epoch = patchesPerEpochOverride;
    }
    if (rehearsalFractionOverride !== undefined) {
      requestBody.rehearsal_fraction = rehearsalFractionOverride;
    }
    if (lossType === 'margin') {
      const m = parseFloat(document.getElementById('marginValue').value);
      if (Number.isFinite(m)) requestBody.margin = m;
    }

    // Build a short summary so user can sanity-check before confirming.
    const summary =
      `Restart training with current form values?\n\n` +
      `  loss:           ${requestBody.loss_type}` +
      (requestBody.margin !== undefined ? ` (margin=${requestBody.margin})` : '') + '\n' +
      `  learning_rate:  ${requestBody.learning_rate}\n` +
      `  lora_r:         ${requestBody.lora_r}\n` +
      `  epochs:         ${requestBody.num_epochs}\n` +
      `  batch_size:     ${requestBody.batch_size}\n` +
      `  patches/epoch:  ${requestBody.patches_per_epoch !== undefined ? requestBody.patches_per_epoch : 'manifest'}\n` +
      `  rehearsal:      ${requestBody.rehearsal_fraction !== undefined ? requestBody.rehearsal_fraction : 'manifest'}\n` +
      `  distillation:   ${requestBody.distillation_lambda} (${requestBody.distillation_scope})\n` +
      `  label_smoothing:${requestBody.label_smoothing}\n` +
      `  balance_classes:${requestBody.balance_classes}\n` +
      `  augment:${requestBody.augment}\n`;
    if (!confirm(summary)) return;

    const originalBtnText = restartJobBtn.textContent;
    restartJobBtn.disabled = true;
    restartJobBtn.textContent = 'Restarting...';

    const jobId = document.getElementById('jobId').textContent;
    try {
      const data = await postAnswer(`/api/finetune/job/${jobId}/restart`, requestBody);

      if (data.success) {
        jobLog.clear();
        appendLog('Restart request sent - training will restart on same GPU...\n');
        isServingReady = false;
        restartEpochResetPending = true;
        liveEpochState = null;
        document.getElementById('jobProgress').textContent = 'Restarting training...';
        document.getElementById('inferenceServerStatus').style.display = 'none';
        resetStopEarlyButton();
        lossPlot.reset();
        resetProgressBarToWaiting();

        restartJobBtn.style.display = 'none';
        // A card restored after a reload has no poller running, so start
        // one (this replaces the poller if one is already running).
        startStatusPolling(jobId);
        // Likewise the log: reopen it where it stopped if it is not
        // streaming, so the restarted run's lines show up.
        jobLog.resume(jobId);
      } else {
        alert('Failed to restart training: ' + data.error);
      }
    } catch (error) {
      console.error('Error restarting training:', error);
      alert('Failed to restart training');
    } finally {
      restartJobBtn.disabled = false;
      restartJobBtn.textContent = originalBtnText;
    }
  });

  // === RESTORE ACTIVE JOB ON PAGE LOAD ===

  async function restoreActiveJob() {
    try {
      const data = await getAnswer('/api/finetune/jobs');
      if (!data.success || !data.jobs || data.jobs.length === 0) return;

      // Find the most recent live job: queued, running, waiting for a
      // restart, or finished and still serving (it can be restarted). A
      // queued job needs its card too, or it cannot be cancelled after a
      // reload. The list comes back oldest first.
      const activeJob = data.jobs
        .slice()
        .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))
        .find(j =>
          j.status === 'PENDING' ||
          j.status === 'RUNNING' ||
          j.status === 'WAITING_FOR_RESTART' ||
          (j.status === 'COMPLETED' && j.inference_server_ready)
        );
      if (!activeJob) return;

      const jobId = activeJob.job_id;

      // Show the training status card
      document.getElementById('trainingStatusCard').style.display = 'block';
      document.getElementById('jobId').textContent = jobId;
      document.getElementById('jobModelName').textContent = activeJob.model_name;

      const outputType = (activeJob.params && activeJob.params.output_type) || 'binary';
      document.getElementById('jobOutputType').textContent = outputType;
      document.getElementById('jobOutputType').className = 'badge ' + (outputType === 'affinities' ? 'bg-warning' : 'bg-info');

      // Restore status badge
      const statusBadge = document.getElementById('jobStatus');
      statusBadge.textContent = activeJob.status;
      statusBadge.className = 'badge bg-' + getStatusColor(activeJob.status);

      // Restore progress
      if (activeJob.current_epoch && activeJob.total_epochs) {
        renderEpochProgress(activeJob.current_epoch, activeJob.total_epochs);
        document.getElementById('jobProgress').textContent =
          `Epoch ${activeJob.current_epoch}/${activeJob.total_epochs}` +
          (activeJob.loss ? ` - Loss: ${activeJob.loss.toFixed(4)}` : '');
      }

      if (activeJob.status === 'WAITING_FOR_RESTART') {
        document.getElementById('restartJobBtn').style.display = 'inline-block';
      }
      // Restore inference server ready state
      if (activeJob.inference_server_ready) {
        renderServingReadyProgress();
        document.getElementById('restartJobBtn').style.display = 'inline-block';
        if (activeJob.finetuned_model_name) {
          document.getElementById('inferenceServerStatus').style.display = 'block';
          document.getElementById('neuroglancerLayer').textContent = activeJob.finetuned_model_name;
        }
      }

      // Restore logs and loss plot from the log file. The live stream below
      // then starts where this ends (logData.offset), instead of sending the
      // whole log again on top of it.
      let restoredLogOffset = 0;
      try {
        const logData = await getAnswer(`/api/finetune/job/${jobId}/logs`);
        if (logData.success && logData.logs) {
          restoredLogOffset = logData.offset || 0;
          jobLog.show(logData.logs);
          lossPlot.addLog(logData.logs);
        }
      } catch (e) {
        console.warn('Could not restore logs:', e);
      }

      // Resume polling and streaming if job is still queued, running, or
      // waiting for a restart
      if (activeJob.status === 'RUNNING' || activeJob.status === 'PENDING' ||
          activeJob.status === 'WAITING_FOR_RESTART') {
        jobLog.follow(jobId, restoredLogOffset);
        startStatusPolling(jobId);
      } else {
        // Nothing more will be written, but a restart reopens the stream from
        // here (jobLog.resume).
        jobLog.resumeFrom(restoredLogOffset);
        // Finished but still serving: keep the status current, so Restart
        // (and a later training run) shows up here as it happens.
        startStatusPolling(jobId);
      }

    } catch (e) {
      console.warn('Could not restore active finetuning job:', e);
    }
  }

  restoreActiveJob();

  // Initialize empty loss plot on page load
  lossPlot.render();
}
