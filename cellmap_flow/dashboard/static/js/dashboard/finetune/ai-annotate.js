// AI-assisted annotation (routes/finetune/ai_annotate.py): a hosted model is
// sent one plane of raw EM, the plane the viewer shows, around the view's
// centre or around the cursor on Shift+G, and asked to paint a structure in.
// The server turns its answer into a mask and stages it; this panel shows the
// preview, and Accept writes it into the annotation volume (undone by the
// patch tools' Undo) or Reject throws it away.
//
// The server decides everything that matters for security: whether the
// feature is on, which providers and models may be picked, and whether the
// user has acknowledged where the data goes. This page only picks among what
// it is offered. Everything the server or the model says is shown as text,
// never parsed as markup, and the previews are only ever the server's base64
// PNGs.
import { setBusy } from "../../lib/dom.js";
import { poll } from "../../lib/poll.js";
import { getAnswer, offerNewVolume, postAnswer } from "./requests.js";

const BASE = "/api/finetune/ai-annotate";
// The label picker's entry for a structure that is not in the catalog.
const OTHER = "__other__";
// A job runs for tens of seconds: its stage is worth following closely. Idle,
// the poll still runs, since Shift+G starts a job on the server, not here.
const RUNNING_POLL_MS = 1500;
const IDLE_POLL_MS = 5000;
// Settings are saved this long after the last change, not on every keystroke.
const SAVE_DELAY_MS = 600;
const PLANES = { 0: "XY", 1: "XZ", 2: "YZ" };

// The server's base64 PNG as an image URL; anything else (absent, or not
// base64) is no image at all, so nothing but a PNG data URL is ever loaded.
function pngUrl(base64) {
  return typeof base64 === "string" && /^[A-Za-z0-9+/]+={0,2}$/.test(base64)
    ? `data:image/png;base64,${base64}`
    : null;
}

// log: the Annotation Crops panel's log.
export function initAiAnnotate({ log }) {
  const panel = document.getElementById("aiAnnotatePanel");
  const summaryStatus = document.getElementById("aiAnnotateSummaryStatus");
  const disabledBox = document.getElementById("aiAnnotateDisabled");
  const disabledReason = document.getElementById("aiAnnotateDisabledReason");
  const controls = document.getElementById("aiAnnotateControls");
  const providerSelect = document.getElementById("aiAnnotateProvider");
  const modelSelect = document.getElementById("aiAnnotateModel");
  const labelSelect = document.getElementById("aiAnnotateLabel");
  const labelName = document.getElementById("aiAnnotateLabelName");
  const promptBox = document.getElementById("aiAnnotatePrompt");
  const promptReset = document.getElementById("aiAnnotatePromptReset");
  const ackBox = document.getElementById("aiAnnotateAck");
  const ackHint = document.getElementById("aiAnnotateAckHint");
  const destination = document.getElementById("aiAnnotateDestination");
  const runBtn = document.getElementById("aiAnnotateRunBtn");
  const keyHint = document.getElementById("aiAnnotateKeyHint");
  const usage = document.getElementById("aiAnnotateUsage");
  const statusLine = document.getElementById("aiAnnotateStatus");
  const errorLine = document.getElementById("aiAnnotateError");
  const review = document.getElementById("aiAnnotateReview");
  const reviewInfo = document.getElementById("aiAnnotateReviewInfo");
  const reviewPrompt = document.getElementById("aiAnnotateReviewPrompt");
  const overwriteBox = document.getElementById("aiAnnotateOverwrite");
  const resendBtn = document.getElementById("aiAnnotateResendBtn");
  const rejectBtn = document.getElementById("aiAnnotateRejectBtn");
  const acceptBtn = document.getElementById("aiAnnotateAcceptBtn");
  const undoViewLabelsBtn = document.getElementById("undoViewLabelsBtn");
  const previews = [
    [document.getElementById("aiAnnotateInputImg"), "input_png"],
    [document.getElementById("aiAnnotateModelImg"), "model_png"],
    [document.getElementById("aiAnnotateOverlayImg"), "overlay_png"],
  ];

  // The configuration's providers by id, and the catalog's structures by key.
  let providers = {};
  let organelles = {};
  // The providers acknowledged for the current dataset (the server's list).
  let acknowledged = [];
  let dailyLimit = null;
  // The last status seen, to tell when a job starts, finishes or changes.
  let last = { status: null, annotate_id: null };
  let poller = null;
  let pollMs = null;
  let saveTimer = null;

  function showError(text) {
    errorLine.textContent = text || "";
    errorLine.hidden = !text;
  }

  // The limit is the server's own cap (daily_call_limit in its AI-annotate
  // config), not the provider's quota; the text says so.
  function showUsage(callsToday) {
    if (callsToday === undefined || callsToday === null) return;
    usage.textContent = dailyLimit
      ? `${callsToday} of ${dailyLimit} calls used today (the daily_call_limit in ai_annotate.yaml)`
      : `${callsToday} model calls today`;
  }

  // ---- Settings --------------------------------------------------------

  // The prompt the label starts from: the catalog's for a known structure;
  // for one named under Other, the server writes it from the name.
  function defaultPrompt() {
    const organelle = organelles[labelSelect.value];
    return organelle ? organelle.prompt || "" : "";
  }

  // Unchanged from the default (or blank) is sent as null, so the server
  // builds the prompt itself and an edit is never needed.
  function promptEdited() {
    const text = promptBox.value.trim();
    return text !== "" && text !== defaultPrompt().trim();
  }

  function showPromptReset() {
    promptReset.hidden = !promptEdited();
  }

  function fillModels(chosen) {
    const provider = providers[providerSelect.value];
    const models = provider ? provider.models || [] : [];
    modelSelect.replaceChildren(...models.map((m) => new Option(m, m)));
    modelSelect.value = models.includes(chosen) ? chosen : models[0] || "";
  }

  // Where the selected provider sends the plane, and whether the user has
  // agreed to that for this dataset. An acknowledgement cannot be taken
  // back from here, so a ticked box stays ticked.
  function showAcknowledgement() {
    const provider = providers[providerSelect.value];
    destination.textContent = provider ? provider.destination || provider.id : "";
    const done = acknowledged.includes(providerSelect.value);
    ackBox.checked = done;
    ackBox.disabled = false;
    ackHint.textContent = done
      ? "Acknowledged for this dataset. Untick to be asked again before the next run."
      : "Tick to agree before the first run; asked once per provider and dataset.";
  }

  function showLabelName() {
    labelName.hidden = labelSelect.value !== OTHER;
  }

  function settingsBody(acknowledge) {
    const body = { provider: providerSelect.value, model: modelSelect.value };
    if (labelSelect.value === OTHER) body.label_name = labelName.value.trim();
    else body.label_key = labelSelect.value;
    body.prompt = promptEdited() ? promptBox.value : null;
    // The destination shown is sent with the acknowledgement: the server
    // refuses it if the config has since moved the provider elsewhere.
    if (acknowledge === true) {
      body.acknowledge = true;
      body.destination = destination.textContent;
    } else if (acknowledge === false) {
      body.acknowledge = false;
    }
    return body;
  }

  // Saves the settings now; the answer's acknowledgements replace ours.
  // Resolves to the answer, or null when it could not be saved (and says why).
  // acknowledge: true agrees to the shown destination, false withdraws that,
  // left out keeps it as it is.
  function saveSettings(acknowledge) {
    clearTimeout(saveTimer);
    saveTimer = null;
    if (labelSelect.value === OTHER && !labelName.value.trim()) {
      showError("Name the structure to annotate.");
      return Promise.resolve(null);
    }
    return postAnswer(`${BASE}/settings`, settingsBody(acknowledge))
      .then((d) => {
        if (Array.isArray(d.acknowledged)) acknowledged = d.acknowledged;
        showAcknowledgement();
        if (!d.success) {
          showError(`Could not save the settings: ${d.error}`);
          return null;
        }
        showError("");
        return d;
      })
      .catch((e) => {
        showError(`Could not save the settings: ${e}`);
        return null;
      });
  }

  function saveSoon() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveSettings, SAVE_DELAY_MS);
  }

  providerSelect.addEventListener("change", () => {
    fillModels(null);
    showAcknowledgement();
    saveSoon();
  });
  modelSelect.addEventListener("change", saveSoon);
  // A prompt edited for one structure asks for the wrong one (and the wrong
  // colour) for another, so a new label starts from its own default.
  labelSelect.addEventListener("change", () => {
    promptBox.value = defaultPrompt();
    showLabelName();
    showPromptReset();
    saveSoon();
  });
  labelName.addEventListener("input", saveSoon);
  promptBox.addEventListener("input", () => {
    showPromptReset();
    saveSoon();
  });
  promptReset.addEventListener("click", () => {
    promptBox.value = defaultPrompt();
    showPromptReset();
    saveSoon();
  });
  ackBox.addEventListener("change", () => {
    const agree = ackBox.checked;
    ackBox.disabled = true;
    saveSettings(agree).then((d) => {
      if (d) {
        log.add(agree
          ? `AI annotation: agreed to send this dataset's planes to ${destination.textContent}`
          : "AI annotation: no longer agreed to send this dataset's planes; the next run asks again");
      }
      // The server's list decides what the box shows, saved or not.
      showAcknowledgement();
      log.showEnd();
    });
  });

  // ---- The job ---------------------------------------------------------

  function setPollInterval(ms) {
    if (poller && pollMs === ms) return;
    if (poller) poller.stop();
    pollMs = ms;
    // The first call waits a full interval: whoever changes the interval has
    // just shown the status it knows.
    poller = poll(refreshStatus, { intervalMs: ms, pauseWhenHidden: true, immediate: false });
  }

  function refreshStatus({ stale } = { stale: () => false }) {
    return getAnswer(`${BASE}/status`)
      .then((d) => {
        if (stale()) return;
        if (!d.success) {
          statusLine.textContent = `Could not read the status: ${d.error}`;
          return;
        }
        showStatus(d);
      })
      .catch(() => {
        // Not fatal: the next poll tries again.
      });
  }

  // When the plane was sent and how much of it the mask covers.
  function describeResult(d) {
    const plane = d.plane || PLANES[d.depth_axis] || "?";
    const parts = [`${plane} plane`];
    if (d.label_name) parts.push(d.label_name);
    if (typeof d.mask_fraction === "number") parts.push(`${(d.mask_fraction * 100).toFixed(1)}% marked`);
    if (d.model) parts.push(d.model);
    return parts.join(", ");
  }

  function showPreview(d) {
    const images = d.preview || {};
    for (const [img, field] of previews) {
      const url = pngUrl(images[field]);
      img.hidden = !url;
      if (url) img.src = url;
      else img.removeAttribute("src");
    }
    reviewInfo.textContent = describeResult(d);
    reviewPrompt.value = d.prompt || "";
  }

  function showStatus(d) {
    const was = last;
    last = { status: d.status, annotate_id: d.annotate_id };
    const isNew = was.status !== d.status || was.annotate_id !== d.annotate_id;

    statusLine.classList.toggle("text-danger", d.status === "failed");
    statusLine.classList.toggle("text-muted", d.status !== "failed");
    // One job at a time: a result waits to be accepted or rejected first.
    runBtn.disabled = d.status === "running" || d.status === "ready";

    if (d.status === "running") {
      const plane = d.plane ? ` (${d.plane} plane)` : "";
      statusLine.textContent = `Running${plane}: ${d.stage_label || d.stage || "starting"}...`;
      summaryStatus.textContent = "running";
      review.hidden = true;
    } else if (d.status === "ready") {
      statusLine.textContent = "Ready: review the result below, then accept or reject it.";
      summaryStatus.textContent = "ready to review";
      if (isNew) {
        // The polls leave the preview's images out; they are asked for once
        // here, when the result (or a resend's) becomes ready.
        getAnswer(`${BASE}/status?preview=1`)
          .then((p) => {
            if (p.success && p.status === "ready" && p.annotate_id === d.annotate_id) showPreview(p);
          })
          .catch(() => showPreview(d));
        // Shift+G starts a job without the panel open: open it to show the result.
        panel.open = true;
        log.add(`AI annotation ready to review: ${describeResult(d)}`);
        log.showEnd();
      }
      review.hidden = false;
    } else if (d.status === "failed") {
      statusLine.textContent = `Failed: ${d.error || "no reason given"}`;
      // Shift+G with no annotation volume: offer one, once, as it happens
      // (not again for an old refusal found when the page loads).
      if (d.needs_volume && was.status !== null && was.status !== "failed") offerNewVolume(`Could not start AI annotation: ${d.error}`);
      summaryStatus.textContent = "failed";
      review.hidden = true;
      if (isNew && was.status === "running") {
        log.add(`AI annotation failed: ${d.error || "no reason given"}`);
        log.showEnd();
      }
    } else {
      statusLine.textContent = "";
      summaryStatus.textContent = "";
      review.hidden = true;
    }

    // A finished call counts against the day's limit.
    if (was.status === "running" && d.status !== "running") refreshUsage();
    setPollInterval(d.status === "running" ? RUNNING_POLL_MS : IDLE_POLL_MS);
  }

  function refreshUsage() {
    getAnswer(`${BASE}/config`)
      .then((d) => {
        if (!d.success || !d.enabled) return;
        showUsage(d.calls_today);
        if (Array.isArray(d.acknowledged)) {
          acknowledged = d.acknowledged;
          showAcknowledgement();
        }
      })
      .catch(() => {});
  }

  function started(d, what) {
    showError("");
    log.add(what);
    log.showEnd();
    showStatus({ status: "running", annotate_id: d.annotate_id, stage_label: "starting" });
  }

  runBtn.addEventListener("click", async () => {
    setBusy(runBtn, true);
    try {
      // The settings shown are the ones the run uses, and the answer says
      // whether this dataset has been acknowledged.
      if (!(await saveSettings())) return;
      if (!acknowledged.includes(providerSelect.value)) {
        showError("Tick the box above to agree to where the plane is sent first.");
        panel.open = true;
        ackBox.focus();
        return;
      }
      const d = await postAnswer(`${BASE}/run`, {});
      if (d.needs_acknowledgement) {
        acknowledged = acknowledged.filter((id) => id !== providerSelect.value);
        showAcknowledgement();
      }
      if (!d.success) {
        showError(d.error);
        log.add(`Could not start AI annotation: ${d.error}`);
        log.showEnd();
        if (d.needs_volume) offerNewVolume(`Could not start AI annotation: ${d.error}`);
        return;
      }
      started(d, "AI annotation started at the view centre");
    } catch (e) {
      showError(`Could not start AI annotation: ${e}`);
    } finally {
      setBusy(runBtn, false);
      runBtn.disabled = last.status === "running" || last.status === "ready";
    }
  });

  resendBtn.addEventListener("click", () => {
    setBusy(resendBtn, true);
    postAnswer(`${BASE}/resend`, { annotate_id: last.annotate_id, prompt: reviewPrompt.value })
      .then((d) => {
        if (!d.success) {
          showError(`Could not resend: ${d.error}`);
          return;
        }
        started(d, "AI annotation: sent the plane again with the edited prompt");
      })
      .catch((e) => showError(`Could not resend: ${e}`))
      .finally(() => setBusy(resendBtn, false));
  });

  // Neuroglancer keeps the chunks it has read. The server re-reads the paint
  // layer alone (answer's layer_refreshed); only when it could not does the
  // whole viewer reload, getting its state back from the dashboard. Absent
  // when no viewer is connected.
  function reloadViewer() {
    const frame = document.querySelector("#my_iframe");
    if (frame) frame.src = frame.src;
  }

  function describeAccept(d) {
    if (!d.reload_viewer) {
      return "Accepted the AI annotation, but nothing was written: every voxel of the plane is " +
             "labelled already (tick Overwrite existing labels to replace them)";
    }
    const over = d.overwritten ? `, ${d.overwritten} of them over existing labels` : "";
    return `Accepted the AI annotation: ${d.filled_foreground} foreground and ` +
           `${d.filled_background} background voxels${over}`;
  }

  // As the patch tools' label actions: a write too large to make without
  // asking is answered needs_confirmation, and sent again confirmed.
  function accept(confirmed) {
    setBusy(acceptBtn, true);
    const body = { annotate_id: last.annotate_id, overwrite: overwriteBox.checked };
    return postAnswer(`${BASE}/accept`, confirmed ? { ...body, confirm: true } : body)
      .then((d) => {
        if (d.needs_confirmation && !confirmed) {
          return confirm(d.error) ? accept(true) : undefined;
        }
        if (d.can_undo !== undefined) undoViewLabelsBtn.disabled = !d.can_undo;
        if (!d.success) {
          showError(`Could not accept: ${d.error}`);
          log.add(`Could not accept the AI annotation: ${d.error}`);
        } else if (d.reload_viewer) {
          showError("");
          log.add(describeAccept(d) + (d.layer_refreshed ? "" : "; reloading the viewer"));
          if (!d.layer_refreshed) reloadViewer();
        } else {
          showError("");
          log.add(describeAccept(d));
        }
        log.showEnd();
        return refreshStatus();
      })
      .catch((e) => showError(`Could not accept: ${e}`))
      .finally(() => setBusy(acceptBtn, false));
  }
  acceptBtn.addEventListener("click", () => accept(false));

  rejectBtn.addEventListener("click", () => {
    setBusy(rejectBtn, true);
    postAnswer(`${BASE}/reject`, { annotate_id: last.annotate_id })
      .then((d) => {
        if (!d.success) {
          showError(`Could not reject: ${d.error}`);
          return;
        }
        showError("");
        log.add("Rejected the AI annotation: nothing was written");
        log.showEnd();
        return refreshStatus();
      })
      .catch((e) => showError(`Could not reject: ${e}`))
      .finally(() => setBusy(rejectBtn, false));
  });

  // ---- Start -------------------------------------------------------------

  function showDisabled(reason) {
    disabledReason.textContent = reason || "AI-assisted annotation is off.";
    disabledBox.hidden = false;
    controls.hidden = true;
    summaryStatus.textContent = "off";
  }

  // Fills the pickers from the configuration, and from the session's
  // settings when it has some, so a reload shows what Shift+G will use.
  function showConfig(d) {
    providers = Object.fromEntries((d.providers || []).map((p) => [p.id, p]));
    organelles = Object.fromEntries((d.organelles || []).map((o) => [o.key, o]));
    acknowledged = Array.isArray(d.acknowledged) ? d.acknowledged : [];
    dailyLimit = d.daily_call_limit || null;
    const settings = d.settings || {};

    providerSelect.replaceChildren(...Object.keys(providers).map((id) => new Option(id, id)));
    providerSelect.value = providers[settings.provider] ? settings.provider : d.default_provider;
    fillModels(settings.model);

    labelSelect.replaceChildren(
      ...(d.organelles || []).map((o) => new Option(o.name, o.key)),
      new Option("Other…", OTHER));
    if (settings.label_key && organelles[settings.label_key]) {
      labelSelect.value = settings.label_key;
    } else if (settings.label_name) {
      labelSelect.value = OTHER;
      labelName.value = settings.label_name;
    }
    promptBox.value = settings.prompt || defaultPrompt();
    showLabelName();
    showPromptReset();
    showAcknowledgement();
    showUsage(d.calls_today);
    if (d.keybinding) keyHint.textContent = `or hover over the viewer and press ${d.keybinding}`;

    disabledBox.hidden = true;
    controls.hidden = false;
    // Saved at once, so Shift+G works with what is shown without touching
    // a setting first (saving is what registers the key on the viewer).
    saveSettings();
    // The status sets the poll going at the pace it calls for; if it could
    // not be read, the idle pace tries again.
    refreshStatus().then(() => {
      if (!poller) setPollInterval(IDLE_POLL_MS);
    });
  }

  getAnswer(`${BASE}/config`)
    .then((d) => {
      if (!d.success) showDisabled(d.error);
      else if (!d.enabled) showDisabled(d.reason);
      else showConfig(d);
    })
    .catch((e) => showDisabled(`Could not read the AI annotation configuration: ${e}`));
}
