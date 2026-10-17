// The dashboard's Finetune tab (templates/_finetune_tab.html): annotation
// volumes and crops for a served model, then a training job on them.
//
// Each module drives one part of the tab; this one wires them together:
// - training-form.js: the form, kept across reloads, and the GPU queue picker;
// - models.js: the model picker;
// - crops.js: New Volume, Load Crops from YAML, annotated regions, Save;
// - sessions.js: Resume Existing Volume;
// - label-it.js: Seed from Prediction, All Background, Split Objects and
//   Undo, over the patch on screen;
// - good-region-marks.js: Mark as Good, the count, clear, the rehearsal hint;
// - ai-annotate.js: AI-assisted annotation, a hosted model's labels for the
//   plane on screen, reviewed and then accepted (undone by Label it's Undo)
//   or rejected;
// - job-monitor.js: submit, the status poll, Restart (and Shift+R, which
//   restarts from the viewer), Stop Early, Cancel, and the job restored
//   after a reload, with job-card.js (the Training Status card),
//   log-stream.js (the Training Logs text) and loss-plot.js (the loss plot).
// They start in this order, which is the order of the tab's first requests:
// the saved output path, the models, the seed sources, the good-region
// count, the AI annotation's configuration, and the job to restore. (The GPU
// queue picker joins the Models tab's poller.)
//
// The same tab serves the browser mode (web/), where an in-page backend
// answers the requests and the finetune trains the model's last layer in the
// page: the template hides what needs a server (server-only), and each module
// asks lib/page-data's isBrowserMode where it must behave differently (no
// output path to insist on, the paint's download and delete, the finetuned
// model's download, the AI annotation's credentials, the head's batch). The
// patch tools, Label it, AI-assisted annotation and Good regions, all run
// there, their routes answered in the page (web/src/backend/routes/
// view-labels.ts, ai-annotate.ts and good-regions.ts).
import { initAiAnnotate } from "./ai-annotate.js";
import { initCrops } from "./crops.js";
import { initGoodRegionMarks } from "./good-region-marks.js";
import { initJobMonitor } from "./job-monitor.js";
import { initLabelIt } from "./label-it.js";
import { readJSON, writeJSON } from "../../lib/stored.js";
import { initModelPicker } from "./models.js";
import { initSessions } from "./sessions.js";
import { initTrainingForm } from "./training-form.js";

export function initFinetuneTab() {
  const logArea = document.getElementById("finetuneLog");
  // The Annotation Crops panel's log, which every part but the job writes
  // to, and Shift+R's restart of the job.
  const log = {
    add(line) {
      logArea.value += line + "\n";
    },
    showEnd() {
      logArea.scrollTop = logArea.scrollHeight;
    },
  };

  const form = initTrainingForm();
  const picker = initModelPicker({ log, savedModelName: form.saved && form.saved.selectedModelName });
  const crops = initCrops({ log, picker, form });
  initSessions({ log, addToViewer: crops.addToViewer, volumeReady: crops.volumeReady });
  initLabelIt({ log });
  initGoodRegionMarks({ log });
  initAiAnnotate({ log });
  initJobMonitor({ picker, form, log });
  rememberOpenSections();
}

// The patch tools' groups (Label it, AI-assisted annotation, Good regions)
// open and close from their headings; each stays as the user left it across
// reloads. A browser that keeps nothing gets the page's defaults.
const OPEN_SECTIONS_KEY = "finetunePatchToolsOpen";

function rememberOpenSections() {
  const saved = readJSON(OPEN_SECTIONS_KEY, {}) || {};
  for (const section of document.querySelectorAll(".patch-tools-section[id]")) {
    if (typeof saved[section.id] === "boolean") section.open = saved[section.id];
    section.addEventListener("toggle", () => {
      saved[section.id] = section.open;
      writeJSON(OPEN_SECTIONS_KEY, saved);
    });
  }
}
