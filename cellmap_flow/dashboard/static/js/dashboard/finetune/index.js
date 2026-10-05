// The dashboard's Finetune tab (templates/_finetune_tab.html): annotation
// volumes and crops for a served model, then a training job on them.
//
// Each module drives one part of the tab; this one wires them together:
// - training-form.js: the form, kept across reloads, and the GPU queue picker;
// - models.js: the model picker;
// - crops.js: New Volume, Load Crops from YAML, annotated regions, Save;
// - sessions.js: Resume Existing Volume;
// - good-regions.js: Mark This View as Good, Clear, the rehearsal hint, and
//   Seed from Prediction and All Background over the same patch;
// - ai-annotate.js: AI-assisted annotation, a hosted model's labels for the
//   plane on screen, reviewed and then accepted (undone by the same Undo) or
//   rejected;
// - job-monitor.js: submit, the status poll, Restart, Stop Early, Cancel,
//   and the job restored after a reload, with job-card.js (the Training
//   Status card), log-stream.js (the Training Logs text) and loss-plot.js
//   (the loss plot).
// They start in this order, which is the order of the tab's first requests:
// the saved output path, the models, the good-region count, the AI
// annotation's configuration, and the job to restore. (The GPU queue picker joins the Models tab's poller.)
import { initAiAnnotate } from "./ai-annotate.js";
import { initCrops } from "./crops.js";
import { initGoodRegions } from "./good-regions.js";
import { initJobMonitor } from "./job-monitor.js";
import { initModelPicker } from "./models.js";
import { initSessions } from "./sessions.js";
import { initTrainingForm } from "./training-form.js";

export function initFinetuneTab() {
  const logArea = document.getElementById("finetuneLog");
  // The Annotation Crops panel's log, which every part but the job writes to.
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
  initSessions({ log, addToViewer: crops.addToViewer });
  initGoodRegions({ log });
  initAiAnnotate({ log });
  initJobMonitor({ picker, form });
}
