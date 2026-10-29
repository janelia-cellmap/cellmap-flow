// The Training panel's job: submitting it, following it (the status poll and
// the log stream), Restart, Stop Early and Cancel, and finding it again after
// a page reload. What the Training Status card shows is job-card.js's.
//
// In the browser mode (lib/page-data isBrowserMode) the job runs in the page
// (web/src/finetune) and answers the same routes: no corrections path is
// asked for, and the finetuned model a status names can be downloaded. In
// both modes it can be published to Hugging Face (publish.js).
//
// Shift+R restarts the card's job as Restart does, without its confirm
// (restartFromKey). It arrives three ways: pressed over the viewer in server
// mode, where routes/finetune/restart_key.py counts the press and the job's
// status answer carries the count (restart_requests), so it acts within one
// poll; pressed over the viewer in browser mode, where
// web/src/backend/routes/finetune.ts binds it in the page and sends
// RESTART_EVENT; and pressed on this page itself.
import { setBusy } from "../../lib/dom.js";
import { announce, listen, MODELS_CHANGED_EVENT, RESTART_EVENT } from "../../lib/events.js";
import { isBrowserMode } from "../../lib/page-data.js";
import { poll } from "../../lib/poll.js";
import { sessionVolumeChanged } from "./crops.js";
import { TERMINAL, createJobCard } from "./job-card.js";
import { createJobLog } from "./log-stream.js";
import { createLossPlot } from "./loss-plot.js";
import { initPublish } from "./publish.js";
import { getAnswer, getAnswerIfFound, postAnswer } from "./requests.js";
import { channelLabel, channelUnpicked, fillChannel } from "./training-form.js";

// Shift+R on this page, as Neuroglancer reads it over the viewer
// ("shift+keyr": the key, whatever the keyboard layout types with it):
// without another modifier, not held down, and not in a field, where it
// types an R.
function isRestartKey(event) {
  if (event.code !== "KeyR" || !event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) return false;
  if (event.repeat || event.isComposing || event.defaultPrevented) return false;
  const target = event.target;
  if (!(target instanceof Element)) return true;
  return !(target.closest("input, textarea, select") || target.isContentEditable);
}

// The confirm Restart asks, to check the form's values first, as rows of
// [label, the request's key or a function of it, shown if the key is not
// sent]; a function's null leaves its row out. Each mode's rows are what its
// run reads: the server's trainer the whole form, the browser's last layer
// and whole network the loss fields (loss, distillation, label smoothing,
// class balance: web/src/backend/routes/finetune.ts readLoss) beside their
// own, and every run the channel picked, for a model trained a channel at a
// time.
const LOSS_ROW = ["loss", (body) => body.loss_type + (body.margin !== undefined ? ` (margin=${body.margin})` : "")];
const RATE_ROW = ["learning_rate", (body) => `${body.learning_rate} (${body.learning_rate_schedule})`];
const DISTILLATION_ROW = ["distillation", (body) => `${body.distillation_lambda} (${body.distillation_scope})`];
const CHANNEL_ROW = ["channel", (body) => (body.select_channel === undefined ? null : channelText(body.select_channel))];
const RESTART_SUMMARY_ROWS = {
  server: [
    LOSS_ROW,
    RATE_ROW,
    ["lora_r", "lora_r"],
    ["epochs", "num_epochs"],
    ["batch_size", "batch_size"],
    ["patches/epoch", "patches_per_epoch", "manifest"],
    ["rehearsal", "rehearsal_fraction", "manifest"],
    DISTILLATION_ROW,
    ["label_smoothing", "label_smoothing"],
    ["balance_classes", "balance_classes"],
    ["augment", "augment"],
    CHANNEL_ROW,
  ],
  head: [
    LOSS_ROW,
    RATE_ROW,
    ["epochs", "num_epochs"],
    DISTILLATION_ROW,
    ["label_smoothing", "label_smoothing"],
    ["balance_classes", "balance_classes"],
    CHANNEL_ROW,
  ],
  full: [
    ["training", () => "whole network"],
    LOSS_ROW,
    RATE_ROW,
    ["epochs", "num_epochs"],
    ["batch_size", "batch_size"],
    ["tile_size", "tile_size", "default"],
    DISTILLATION_ROW,
    ["label_smoothing", "label_smoothing"],
    ["balance_classes", "balance_classes"],
    ["augment", "augment"],
    CHANNEL_ROW,
  ],
};

// The channel `index` as the training form's Channel names it.
function channelText(index) {
  const option = [...(document.getElementById("trainChannel")?.options || [])].find((o) => o.value === String(index));
  return option ? option.text : `channel ${index}`;
}

export function restartSummary(requestBody, browser) {
  const mode = !browser ? "server" : requestBody.finetune_mode === "full" ? "full" : "head";
  const rows = RESTART_SUMMARY_ROWS[mode].flatMap(([label, key, fallback]) => {
    const value = typeof key === "function" ? key(requestBody) : requestBody[key];
    return value === null ? [] : [`  ${(label + ":").padEnd(16)}${value !== undefined ? value : fallback}\n`];
  });
  return "Restart training with current form values?\n\n" + rows.join("");
}

function showNotification(title, message) {
  // Browser notification
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification(title, { body: message });
  }

  // Also show in-page alert
  alert(`${title}\n\n${message}`);
}

// picker: the model picker; form: the training form; log: the Annotation
// Crops panel's log, where Shift+R says what it did.
export function initJobMonitor({ picker, form, log }) {
  let statusPoller = null;
  const browser = isBrowserMode();
  const publish = initPublish();
  const lossPlot = createLossPlot(
    document.getElementById("lossPlotCanvas"), document.getElementById("lossPlotSummary"));
  const card = createJobCard(lossPlot);
  const jobLog = createJobLog({ onLine: card.logLine });
  const appendLog = jobLog.append;
  // A restart request on its way, from the button or the key: the button is
  // disabled meanwhile, and Shift+R is ignored, saying so.
  let restartInFlight = false;
  // The Shift+R presses over the viewer that server mode counted, by job,
  // as the last answer gave them (keyPressed).
  const keyPresses = new Map();

  // What Shift+R has to say: in the Annotation Crops panel's log, where the
  // patch tools (Shift+G among them) report, and in the Training Logs. The
  // key is pressed over the viewer, with either of the tab's panels showing.
  function say(text) {
    log.add(text);
    log.showEnd();
    appendLog(text);
  }

  // Whether an answer's count of Shift+R presses for the job (server mode:
  // restart_requests) says it was pressed since the last answer. The first
  // count seen for a job is where it starts, so a press made before this
  // page followed the job (before a reload, or for another tab) is not acted
  // on now. The browser mode's answers carry no count: its key sends
  // RESTART_EVENT instead.
  function keyPressed(jobId, count) {
    if (typeof count !== "number") return false;
    const seen = keyPresses.get(jobId);
    keyPresses.set(jobId, count);
    return seen !== undefined && count > seen;
  }

  // The Training Status card's Publish to Hugging Face, for the model a
  // job's status says is served (inference_server_ready with
  // finetuned_model_name), and in browser mode its Download finetuned model
  // link (the template's browser-only anchor), pointed at the same model;
  // both hidden on any other status, and for a job just submitted or
  // restarted, whose model is not made yet. The page fetches the link on a
  // click (web/src/rebase.ts), like the Export Config link, and the service
  // worker relays the GET to the in-page backend. The Flask page has no such
  // anchor: a server's model is exported by its publish.
  function showFinetunedModel(job) {
    const name = job && job.inference_server_ready && job.finetuned_model_name;
    publish.showFor(name || null);
    if (!browser) return;
    const link = document.getElementById("downloadFinetunedBtn");
    if (name) {
      link.href = `/api/finetune/model/${encodeURIComponent(name)}/download`;
      link.style.display = "inline-block";
    } else {
      link.removeAttribute("href");
      link.style.display = "none";
    }
  }

  // The rest of the page is told when the job's model may have come, gone
  // or been replaced: when its status, its server's readiness or its
  // model's name changes from what was last seen.
  let servedSeen = null;
  function noteServed(job) {
    const seen = job ? [job.status, Boolean(job.inference_server_ready), job.finetuned_model_name || ""].join("|") : null;
    if (seen === servedSeen) return;
    servedSeen = seen;
    announce(MODELS_CHANGED_EVENT);
  }

  // Start Finetuning button
  const startFinetuningBtn = document.getElementById("startFinetuningBtn");
  startFinetuningBtn.addEventListener("click", async function() {
    const selectedModel = picker.selected();
    if (!selectedModel) {
      alert("Please select a model first");
      return;
    }

    // Get corrections path from output path field. Browser mode: the paint
    // is in the browser's store, so there is none to ask for or send.
    const correctionsPath = browser ? "" : document.getElementById("outputPath").value.trim();
    if (!correctionsPath && !browser) {
      alert("Please specify the output path where annotation crops are saved");
      return;
    }

    // A model of several channels trains the one the paint is for. With none
    // picked, the one that matches the paint best, as Rank scores it (the
    // score's channel_index); a score that fails leaves it unpicked, and the
    // submit route's refusal says to pick one.
    if (channelUnpicked(selectedModel.name)) {
      setBusy(startFinetuningBtn, true, "Finding the channel your paint is for...");
      const scored = await postAnswer("/api/finetune/rank/score", { name: selectedModel.name })
        .catch((error) => ({ success: false, error: error.message || String(error) }));
      setBusy(startFinetuningBtn, false);
      if (scored.success && Number.isInteger(scored.channel_index) && fillChannel(selectedModel.name, scored.channel_index)) {
        appendLog(`Channel: ${channelLabel(scored.channel, scored.channel_index)}, the one that matches your paint best (${Math.round(scored.matches * 100)}% match). Pick another in the training form to train that one instead.`);
      } else if (!scored.success) {
        appendLog(`Could not find the channel your paint is for: ${scored.error}`);
      }
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
    };
    // The queues ticked. None: nothing is chosen and LSF has not listed the
    // queues yet, so the picker would show every one ticked, which is the
    // server's default too.
    const queues = form.gpuQueues();
    if (queues.length) params.queue = queues;
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
      // The browser has no MinIO: a slow start there is the paint being read.
      setBusy(startFinetuningBtn, true, browser
        ? "Reading the painted chunks..."
        : "Syncing annotations from MinIO before submit (this can take several minutes)...");
    }, 3000);

    try {
      const result = await postAnswer("/api/finetune/submit", params);

      if (result.success) {
        keyPresses.set(result.job_id, 0);  // a new job: no presses yet
        card.submitted(result.job_id, params.model_name, result.output_type || "binary");
        showFinetunedModel(null);
        jobLog.clear();
        // The answer names the job's output directory, not its log file:
        // the job manager writes that there as training_log.txt, and the
        // status polls name it.
        jobLog.setLogFile(result.output_dir ? `${result.output_dir}/training_log.txt` : null);
        appendLog("Finetuning job submitted successfully!");
        appendLog(`Job ID: ${result.job_id}`);
        // A job in the page has neither an LSF id nor an output directory.
        if (!browser) {
          appendLog(`LSF Job ID: ${result.lsf_job_id || 'N/A'}`);
          appendLog(`Output directory: ${result.output_dir}`);
        }
        if (result.tensorboard_command) appendLog(`TensorBoard: ${result.tensorboard_command}`);
        if (result.note) {
          appendLog(`Note: ${result.note}`);
        }
        // The paint was on another model's grid and is carried onto this
        // one's, into a new volume (the note says so): the volume buttons
        // act on it, and the model lists' "(painted for)" follows it.
        if (result.carried) {
          sessionVolumeChanged(result.carried.volume_id);
          announce(MODELS_CHANGED_EVENT);
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
        showFinetunedModel(data);
        noteServed(data);
        if (data.log_file) jobLog.setLogFile(data.log_file);
        // Server mode's Shift+R over the viewer: the card has just taken
        // this answer, so the key goes by the state the press was counted in.
        if (keyPressed(jobId, data.restart_requests)) restartFromKey();

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
  // Restart, confirms. The button's work, which Shift+R shares without the
  // confirm (restartFromKey), so the two cannot drift. report(text) says why
  // nothing was restarted: the button alerts; the key writes to the logs,
  // since the viewer may have the focus. onSend(jobId), if given, runs as
  // the request goes, once the form has been read. Resolves to the route's
  // answer once the job has taken the restart, else null.
  const restartJobBtn = document.getElementById('restartJobBtn');
  async function restartWithForm({ confirmFirst, report, onSend }) {
    const jobId = card.jobId();
    if (!jobId || jobId === "-") {
      report("No job to restart.");
      return null;
    }
    let requestBody;
    try {
      requestBody = form.trainingParams();
    } catch (error) {
      report(error.message);
      return null;
    }
    if (confirmFirst && !confirm(restartSummary(requestBody, browser))) return null;
    if (onSend) onSend(jobId);

    restartInFlight = true;
    setBusy(restartJobBtn, true);
    restartJobBtn.textContent = 'Restarting...';

    try {
      const data = await postAnswer(`/api/finetune/job/${jobId}/restart`, requestBody);

      if (data.success) {
        jobLog.clear();
        appendLog('Restart request sent - training will restart on same GPU...\n');
        card.restarted();
        showFinetunedModel(null);
        // A card restored after a reload has no poller running, so start
        // one (this replaces the poller if one is already running).
        startStatusPolling(jobId);
        // Likewise the log: reopen it where it stopped if it is not
        // streaming, so the restarted run's lines show up.
        jobLog.resume(jobId);
        return data;
      }
      report('Failed to restart training: ' + data.error);
    } catch (error) {
      console.error('Error restarting training:', error);
      report('Failed to restart training');
    } finally {
      restartInFlight = false;
      setBusy(restartJobBtn, false);
    }
    return null;
  }
  restartJobBtn.addEventListener('click', () => restartWithForm({ confirmFirst: true, report: (text) => alert(text) }));

  // Shift+R, however it arrived: what Restart does, with the form's values
  // and without its confirm (the key is a deliberate act), and only while
  // the card offers Restart, the job waiting for one; otherwise it says why
  // not. A press while a restart is on its way is ignored, saying so, as
  // the button is disabled then. It never starts a new job.
  function restartFromKey() {
    const jobId = card.jobId();
    if (!jobId || jobId === "-") {
      say("Shift+R: no job to restart. Start one on the Finetune tab.");
      return;
    }
    if (restartInFlight) {
      say(`Shift+R: the restart of ${jobId} is already on its way; this press is ignored.`);
      return;
    }
    if (!card.offersRestart()) {
      say(`Shift+R restarts a job that is waiting for a restart; ${jobId} is ${card.status()}.`);
      return;
    }
    restartWithForm({
      confirmFirst: false,
      report: (text) => say(`Shift+R: ${text}`),
      onSend: () => say(`Shift+R: restarting ${jobId} with the Finetune tab's settings...`),
    }).then((data) => {
      if (data) say(`Shift+R: ${data.message || `${jobId} restarted.`}`);
    });
  }
  listen(RESTART_EVENT, restartFromKey);
  document.addEventListener("keydown", (event) => {
    if (!isRestartKey(event)) return;
    event.preventDefault();
    restartFromKey();
  });

  // After a page reload, the most recent live job's card, log and plot, and
  // its status poll and log stream again.
  async function restoreActiveJob() {
    try {
      const data = await getAnswer('/api/finetune/jobs');
      if (!data.success || !data.jobs || data.jobs.length === 0) return;

      // Find the most recent live job: queued, running, or waiting for a
      // restart. A queued job needs its card too, or it cannot be cancelled
      // after a reload. A finished job is not one, even if its last status
      // said its server was ready: it has exited, and cannot be restarted.
      // The list comes back oldest first.
      const activeJob = data.jobs
        .slice()
        .sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))
        .find(j => !TERMINAL.includes(j.status));
      if (!activeJob) return;

      const jobId = activeJob.job_id;
      keyPressed(jobId, activeJob.restart_requests);  // where its count starts
      card.restored(activeJob);
      showFinetunedModel(activeJob);
      noteServed(activeJob);
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

      // Follow it again: its log from where the restored one ends, and its
      // status.
      jobLog.follow(jobId, restoredLogOffset);
      startStatusPolling(jobId);

    } catch (e) {
      console.warn('Could not restore active finetuning job:', e);
    }
  }

  restoreActiveJob();

  // Initialize empty loss plot on page load
  lossPlot.render();
}
