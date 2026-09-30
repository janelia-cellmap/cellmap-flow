// The Training panel's job: submitting it, following it (the status poll and
// the log stream), Restart, Stop Early and Cancel, and finding it again after
// a page reload. What the Training Status card shows is job-card.js's.
import { setBusy } from "../../lib/dom.js";
import { poll } from "../../lib/poll.js";
import { TERMINAL, createJobCard } from "./job-card.js";
import { createJobLog } from "./log-stream.js";
import { createLossPlot } from "./loss-plot.js";
import { getAnswer, getAnswerIfFound, postAnswer } from "./requests.js";

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
  const lossPlot = createLossPlot(
    document.getElementById("lossPlotCanvas"), document.getElementById("lossPlotSummary"));
  const card = createJobCard(lossPlot);
  const jobLog = createJobLog({ onLine: card.logLine });
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

    let training;
    try {
      training = form.trainingParams();
    } catch (error) {
      alert(error.message);
      return;
    }
    const params = {
      model_name: selectedModel.name,
      corrections_path: correctionsPath,
      ...training,
      auto_serve: document.getElementById("autoServeCheck").checked,
      queue: document.getElementById("gpuQueue").value,
    };
    // Add optional checkpoint path override if provided
    const checkpointPath = document.getElementById("checkpointPath").value.trim();
    if (checkpointPath) {
      params.checkpoint_path = checkpointPath;
    }

    // The submit endpoint may run a pre-submit MinIO sync; if that takes a
    // while, the spinner's label changes to make it obvious that the
    // dashboard is doing real work, not stuck.
    setBusy(startFinetuningBtn, true, "Submitting...");
    const slowLabelTimer = setTimeout(() => {
      setBusy(startFinetuningBtn, true,
        "Syncing annotations from MinIO before submit (this can take several minutes)...");
    }, 3000);

    try {
      const result = await postAnswer("/api/finetune/submit", params);

      if (result.success) {
        card.submitted(result.job_id, params.model_name, result.output_type || "binary");
        jobLog.clear();
        // The answer names the job's output directory, not its log file:
        // the job manager writes that there as training_log.txt, and the
        // status polls name it.
        jobLog.setLogFile(result.output_dir ? `${result.output_dir}/training_log.txt` : null);
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
      clearTimeout(slowLabelTimer);
      setBusy(startFinetuningBtn, false);
    }
  });

  // The job's status, every 3 seconds (the first after 3 s); this replaces
  // the poller running before, if any. It stops once the answer cannot
  // change any more:
  // - the job is over: COMPLETED, FAILED or CANCELLED. Its monitor in the
  //   job manager has stopped, and a restart only goes to a job waiting for
  //   one, so a finished job's status is final;
  // - the server does not know the job (a 404), e.g. the dashboard was
  //   restarted since this page was loaded.
  // An error, or no answer at all, is skipped, and the next tick asks again.
  // The poll asks once at a time and stops at once when the job is over, so
  // a failed job notifies once, not once per tick that was waiting. Unlike
  // the tab's other polls it goes on while the page is hidden: that is when
  // the browser notification of a failed job is useful.
  function startStatusPolling(jobId) {
    if (statusPoller) statusPoller.stop();
    statusPoller = poll(async ({ stale }) => {
      try {
        const data = await getAnswerIfFound(`/api/finetune/job/${jobId}/status`);
        if (stale()) return;

        if (data === null) {
          appendLog(`Status updates stopped: the dashboard does not know job ${jobId}. ` +
            "If it was restarted, reload the page to find the job again.");
          return false;
        }
        if (!data.success) {
          console.error("Error getting job status:", data.error);
          return;
        }
        card.polled(data);
        if (data.log_file) jobLog.setLogFile(data.log_file);

        if (TERMINAL.includes(data.status)) {
          // The stream normally ends itself with "done"; see jobLog.closeSoon.
          jobLog.closeSoon();
          if (data.status !== "COMPLETED") {
            showNotification(
              "Training " + data.status,
              "Check the job log below for the traceback."
            );
          }
          return false;
        }
      } catch (error) {
        console.error("Status polling error:", error);
      }
    }, { intervalMs: 3000, immediate: false });
  }

  // Cancel button
  document.getElementById("cancelJobBtn").addEventListener("click", async function() {
    const jobId = card.jobId();
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

  // Stop Early: graceful, the job stays alive for Restart.
  document.getElementById("stopEarlyBtn").addEventListener("click", async function() {
    const jobId = card.jobId();
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
    card.stopRequested();
    try {
      const result = await postAnswer(`/api/finetune/job/${jobId}/stop-early`);
      if (result.success) {
        appendLog("\n=== STOP EARLY REQUESTED — exiting after current epoch ===\n");
      } else {
        alert(`Failed to request stop: ${result.error}`);
        card.resetStopEarly();
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
      card.resetStopEarly();
    }
  });

  // Restart uses whatever's currently in the main training form as the source
  // of truth — no separate modal. User edits the form to change params, hits
  // Restart, confirms.
  const restartJobBtn = document.getElementById('restartJobBtn');
  restartJobBtn.addEventListener('click', async () => {
    let requestBody;
    try {
      requestBody = form.trainingParams();
    } catch (error) {
      alert(error.message);
      return;
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

    setBusy(restartJobBtn, true);
    restartJobBtn.textContent = 'Restarting...';

    const jobId = card.jobId();
    try {
      const data = await postAnswer(`/api/finetune/job/${jobId}/restart`, requestBody);

      if (data.success) {
        jobLog.clear();
        appendLog('Restart request sent - training will restart on same GPU...\n');
        card.restarted();
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
      setBusy(restartJobBtn, false);
    }
  });

  // After a page reload, the most recent live job's card, log and plot, and
  // its status poll and log stream again.
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
      card.restored(activeJob);
      jobLog.setLogFile(activeJob.log_file || null);

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
        // Finished, and its model was served: one poll brings the card up to
        // date, and the poll stops there, as the status is final.
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
