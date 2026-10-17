// The Finetune tab's form: the output path and the training parameters,
// kept in localStorage (the output path on the server too), the advice shown
// next to some of them, and the GPU queue picker.
import { isBrowserMode } from "../../lib/page-data.js";
import { fillQueueSelect, queueHint, subscribeGpuQueues } from "../../shared/gpu-queues.js";
import { getAnswer, postAnswer } from "./requests.js";

const FINETUNE_STATE_KEY = "finetuneFormState";

// The fields kept in localStorage, as [key in the saved state, element id,
// how it is restored]:
// - "value": when the saved value is not empty;
// - "blank": also when it is "", which is a choice of its own (keep the
//   manifest's value), not a missing one;
// - "checkbox": whenever it was saved.
// The model picker's choice is saved with them, as selectedModelName.
const SAVED_FIELDS = [
  ["outputPath", "outputPath", "value"],
  ["checkpointPath", "checkpointPath", "value"],
  ["loraRank", "loraRank", "value"],
  ["numEpochs", "numEpochs", "value"],
  ["batchSize", "batchSize", "value"],
  ["patchesPerEpoch", "patchesPerEpoch", "blank"],
  ["rehearsalFraction", "rehearsalFraction", "blank"],
  ["learningRate", "learningRate", "value"],
  ["lossType", "lossType", "value"],
  ["marginValue", "marginValue", "value"],
  ["labelSmoothing", "labelSmoothing", "blank"],
  ["distillationLambda", "distillationLambda", "value"],
  ["distillationScope", "distillationScope", "value"],
  ["balanceClasses", "balanceClasses", "checkbox"],
  ["augment", "augment", "checkbox"],
  ["autoServe", "autoServeCheck", "checkbox"],
  ["gpuQueue", "gpuQueue", "value"],
];

// The fields saved in this mode. The browser mode keeps the Loss Function
// under a key of its own, headLossType: the field was hidden there before
// the head had a choice of losses, and every browser that saved the form
// saved the server template's default, margin, which must not come back as
// the browser's loss. Its batch, voxels per step, is a field of its own
// (batchPixels, in the template's browser-only block). The whole network's
// fields are browser-only too: the Training choice, the Tile Size and the
// whole network's learning rate; and its batch, the server's Batch Size, is
// kept under a key of its own (fullBatchSize) for the same reason as the
// loss: every browser saved the hidden field's 8.
function savedFields() {
  if (!isBrowserMode()) return SAVED_FIELDS;
  const renamed = { lossType: "headLossType", batchSize: "fullBatchSize" };
  return SAVED_FIELDS
    .map((field) => (renamed[field[1]] ? [renamed[field[1]], field[1], field[2]] : field))
    .concat([
      ["batchPixels", "batchPixels", "value"],
      ["finetuneMode", "finetuneMode", "value"],
      ["tileSize", "tileSize", "value"],
      ["fullLearningRate", "learningRateFull", "value"],
    ]);
}

function saveFinetuneState() {
  const state = {};
  for (const [key, id, kind] of savedFields()) {
    const el = document.getElementById(id);
    state[key] = kind === "checkbox" ? el.checked : el.value;
  }
  state.selectedModelName = document.getElementById("modelSelect").value;
  localStorage.setItem(FINETUNE_STATE_KEY, JSON.stringify(state));
}

function restoreFinetuneState() {
  const raw = localStorage.getItem(FINETUNE_STATE_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { return null; }
}

function applySavedState(saved) {
  for (const [key, id, kind] of savedFields()) {
    const el = document.getElementById(id);
    const value = saved[key];
    if (kind === "checkbox") {
      if (value !== undefined) el.checked = value;
    } else if (kind === "blank" ? value !== undefined && value !== null : value) {
      el.value = value;
    }
  }
}

function readPatchesPerEpochOverride() {
  const raw = document.getElementById("patchesPerEpoch").value.trim();
  if (!raw) return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error("Patches per epoch must be blank or a non-negative integer.");
  }
  return value;
}

function readRehearsalFractionOverride() {
  const raw = document.getElementById("rehearsalFraction").value.trim();
  // Blank means "leave the manifest alone"; "0" is a real choice (train
  // without rehearsal while keeping the regions), so it must not be
  // collapsed into blank here.
  if (!raw) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("Good-region rehearsal must be blank or between 0 and 1.");
  }
  return value;
}

// The training parameters in the form, as both the submit and the restart
// routes take them. A blank patches-per-epoch or rehearsal field is left out,
// so the manifest keeps its value, and the margin goes only with the margin
// loss. Throws an Error, with a message for the user, for a value the trainer
// would refuse.
function trainingParams() {
  const patchesPerEpoch = readPatchesPerEpochOverride();
  const rehearsalFraction = readRehearsalFractionOverride();
  const params = {
    lora_r: parseInt(document.getElementById("loraRank").value),
    num_epochs: parseInt(document.getElementById("numEpochs").value),
    batch_size: parseInt(document.getElementById("batchSize").value),
    learning_rate: parseFloat(document.getElementById("learningRate").value),
    loss_type: document.getElementById("lossType").value,
    distillation_lambda: parseFloat(document.getElementById("distillationLambda").value),
    distillation_scope: document.getElementById("distillationScope").value,
    balance_classes: document.getElementById("balanceClasses").checked,
    augment: document.getElementById("augment").checked,
    label_smoothing: parseFloat(document.getElementById("labelSmoothing").value) || 0,
  };
  if (patchesPerEpoch !== undefined) {
    params.patches_per_epoch = patchesPerEpoch;
  }
  if (rehearsalFraction !== undefined) {
    params.rehearsal_fraction = rehearsalFraction;
  }
  // Browser mode: what trains (the Training field); for the head, its
  // batch in voxels per step, its own field; for the whole network, its
  // learning rate, its tile side and the batch above, in tiles. The
  // server-only fields above go too and the page's routes ignore them
  // (web/src/backend/routes/finetune.ts readTraining). The Loss Function
  // offers the head's losses there (BCE, MSE, margin) and is read as on
  // the server; the whole network ignores it.
  if (isBrowserMode()) {
    const mode = document.getElementById("finetuneMode").value;
    params.finetune_mode = mode;
    if (mode === "full") {
      params.learning_rate = parseFloat(document.getElementById("learningRateFull").value);
      const tile = parseInt(document.getElementById("tileSize").value, 10);
      if (Number.isInteger(tile)) params.tile_size = tile;
    } else {
      params.batch_pixels = parseInt(document.getElementById("batchPixels").value, 10);
    }
  }
  if (params.loss_type === "margin") {
    const margin = parseFloat(document.getElementById("marginValue").value);
    if (Number.isFinite(margin)) params.margin = margin;
  }
  return params;
}

// Whether augmentation is worth enabling depends on how many times the run
// revisits the same patches, which is epochs x batches-per-epoch. Static
// advice cannot say that, so compute it from the form as it is edited.
// The cellmap distance models say what they are only by name, which is how
// submit tells them too (autodetect_output_type).
export function isDistanceModel(name) {
  return /distance/i.test(name || "");
}

// The template's hint on what a distance model trains with on scribbles,
// whatever the loss picked above: submit replaces it (training_settings).
function showDistanceModelHint(name) {
  const hint = document.getElementById("distanceModelHint");
  hint.hidden = !isDistanceModel(name);
}

// The distillation a distance model trains with on scribbles, set in the form
// when one is picked so the form shows what the job will get: submit applies
// the same (training_settings), and a form left at 0.01 read as if that would
// be used. Only on picking a model, so values edited afterwards stand. Not in
// the browser mode, whose head trains a distance model on the loss picked
// (it has no interval loss, so the distillation that goes with it does not
// apply; the hint is server-only there).
export function applyModelDefaults(name) {
  showDistanceModelHint(name);
  if (!isDistanceModel(name) || isBrowserMode()) return;
  const distillation = document.getElementById("distillationLambda");
  if (!(parseFloat(distillation.value) >= 0.5)) distillation.value = "0.5";
  saveFinetuneState();
}

function updateAugmentAdvice() {
  const advice = document.getElementById("augmentAdvice");
  const epochs = parseInt(document.getElementById("numEpochs").value, 10);
  const batch = parseInt(document.getElementById("batchSize").value, 10);
  const patchesRaw = document.getElementById("patchesPerEpoch").value;
  const patches = patchesRaw === "" || patchesRaw == null
    ? null
    : parseInt(patchesRaw, 10);

  if (!epochs || !batch || patches === null || Number.isNaN(patches) || patches <= 0) {
    // Blank/auto patches-per-epoch resolves inside the dataset from the
    // populated-chunk count, so the step total is not knowable here.
    advice.textContent = "Off by default; set Patches per Epoch for advice.";
    advice.className = "d-block text-muted";
    return;
  }

  const steps = epochs * Math.ceil(patches / batch);
  const views = epochs; // each patch is revisited once per epoch
  if (steps < 200) {
    advice.textContent = `~${steps} steps: too few for it to help.`;
    advice.className = "d-block text-warning";
  } else {
    advice.textContent = `~${steps} steps, each patch seen ~${views}x: worth enabling.`;
    advice.className = "d-block text-success";
  }
}

// The GPU queue picker shows what the queues look like right now. A training
// job is not cycled onto another queue if the one picked is busy -- it just
// waits -- so the only thing that helps is seeing the wait before submitting
// rather than after. The answers come from the poller the Models tab's
// pickers share.
function showGpuQueues(data) {
  // An answer with no queues leaves the options (at first the template's)
  // and the note as they are, rather than emptying the picker.
  if (!((data && data.queues) || []).length) return;
  const select = document.getElementById("gpuQueue");
  // The queue picked stays, even if LSF no longer lists it.
  fillQueueSelect(select, data, { preferred: select.value, missingSuffix: " (selected)" });
  document.getElementById("finetuneGpuQueueHint").textContent = queueHint(data);
}

// --- The browser's Training field ---------------------------------------
//
// What a model offers comes from its manifest (the page's GET
// /api/finetune/models: finetune_modes, and for the whole network
// full_training's tile sides with what a step costs); models.js hands each
// picked model to setBrowserTraining. The Training field shows only when
// both are offered; the Training pane's data-training says which is picked,
// and web/src/browser.css shows that one's fields.
let browserModel = null;
let headNote = null;  // the template's words for the head, kept to put back

function trainingMode() {
  return document.getElementById("finetuneMode").value;
}

// The model's tile sides as the Tile Size's options; the side saved, or the
// one picked, when the model has it, else its default.
function fillTileSizes(full) {
  const select = document.getElementById("tileSize");
  const keep = select.value || (restoreFinetuneState() || {}).tileSize;
  select.innerHTML = "";
  for (const tile of (full && full.tiles) || []) {
    const option = document.createElement("option");
    option.value = String(tile.side);
    option.textContent = `${tile.side}² (output ${tile.output}²)${tile.side === full.default_tile ? " (Default)" : ""}`;
    select.appendChild(option);
  }
  const sides = Array.from(select.options).map((option) => option.value);
  if (sides.includes(String(keep))) select.value = String(keep);
  else if (full && full.default_tile) select.value = String(full.default_tile);
}

// What a step of the tile picked is expected to cost at the batch picked,
// from the manifest's measurement (scaled by area for a side it did not
// measure), and how much of the tile is output.
function updateTileHint() {
  const hint = document.getElementById("tileSizeHint");
  const full = browserModel && browserModel.full_training;
  if (!full || !(full.tiles || []).length) {
    hint.textContent = (full && full.error) || "";
    hint.title = "";
    return;
  }
  const tile = full.tiles.find((t) => String(t.side) === document.getElementById("tileSize").value);
  if (!tile) return;
  const batch = parseInt(document.getElementById("batchSize").value, 10) || 1;
  const useful = `${Math.round(tile.useful * 100)}% of each tile is output, the rest context`;
  const cost = tile.step_ms === null
    ? ""
    : `About ${formatMs(tile.step_ms * batch)} and ${formatMb(tile.peak_mb * batch)} a step` +
      `${tile.measured ? " where it was measured" : ", scaled from the sides measured"}; `;
  hint.textContent = `${cost}${useful}. Smaller tiles train faster in less memory, but spend more of each tile on context.`;
  hint.title = full.measured_on ? `Measured on ${full.measured_on}` : "";
}

function formatMs(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function formatMb(mb) {
  return mb < 1000 ? `${Math.round(mb)} MB` : `${(mb / 1000).toFixed(1)} GB`;
}

// The pane's data-training and the notes, from the Training field.
function applyTrainingMode() {
  const mode = trainingMode();
  document.getElementById("training").dataset.training = mode;
  const note = document.getElementById("browserFinetuneNote");
  if (note) {
    headNote = headNote ?? note.textContent;
    note.textContent = mode === "full"
      ? "In the browser, this model can train its whole network: every weight, on the voxels you paint, with TensorFlow.js on this machine's GPU, in tiles cut in the plane you paint in. It takes minutes where the last layer takes seconds, and can learn what the model's features miss. Paint some background and some objects on a few slices, then train."
      : headNote;
  }
  updateTileHint();
}

// Browser mode: what `model` (an answer of /api/finetune/models) offers.
// Only the modes its manifest declares can be picked; the Training field
// shows when there are two to pick from; no mode at all says why below it.
export function setBrowserTraining(model) {
  browserModel = model;
  const modes = (model && model.finetune_modes) || ["head"];
  // The whole network needs a GPU (WebGPU); without one it shows, disabled, with why.
  const noGpu = (model && model.full_training && model.full_training.unavailable) || null;
  const usable = noGpu ? modes.filter((mode) => mode !== "full") : modes;
  const select = document.getElementById("finetuneMode");
  Array.from(select.options).forEach((option) => {
    option.disabled = !usable.includes(option.value);
  });
  if (usable.length && !usable.includes(select.value)) select.value = usable[0];
  const both = modes.includes("head") && modes.includes("full");
  document.getElementById("finetuneModeGroup").hidden = !both;
  const note = document.getElementById("finetuneModeNote");
  note.textContent = !both ? "" : noGpu
    || "The last layer trains in seconds; the whole network in minutes, on this machine's GPU.";
  fillTileSizes(model && model.full_training);
  applyTrainingMode();
  if (!modes.length) {
    const browserNote = document.getElementById("browserFinetuneNote");
    if (browserNote) browserNote.textContent = "This model cannot be finetuned in the browser: its manifest declares no training for the file served (an int8 file's weights are quantized and cannot be written).";
  }
}

// Returns { saved, save(), trainingParams() }: saved is the state restored
// at load (null if none); save() stores the form's state now.
export function initTrainingForm() {
  const outputPathInput = document.getElementById("outputPath");

  // Show the margin field only for the margin loss.
  const lossSel = document.getElementById("lossType");
  const marginGroup = document.getElementById("marginGroup");
  const updateMarginVisibility = () => {
    marginGroup.style.display = lossSel.value === "margin" ? "" : "none";
  };
  lossSel.addEventListener("change", updateMarginVisibility);
  updateMarginVisibility();

  // localStorage is per origin and the dashboard takes a random port each
  // time it starts, so the form state restored from it below is gone after a
  // restart. The output path alone is also kept server-side. This reply
  // arrives after that restore has run, and fills the field only if it is
  // still empty -- so localStorage wins when it has a value, and the server
  // copy covers a fresh origin.
  getAnswer("/api/finetune/user-prefs")
    .then(d => {
      if (d && d.success && d.prefs && d.prefs.outputPath && !outputPathInput.value) {
        outputPathInput.value = d.prefs.outputPath;
      }
    })
    .catch(() => {});

  // Persist outputPath server-side on change so it sticks across restarts.
  let outputPathSaveTimer = null;
  outputPathInput.addEventListener("input", () => {
    clearTimeout(outputPathSaveTimer);
    outputPathSaveTimer = setTimeout(() => {
      const val = outputPathInput.value.trim();
      if (val) {
        postAnswer("/api/finetune/user-prefs", { outputPath: val }).catch(() => {});
      }
    }, 500);
  });

  const saved = restoreFinetuneState();
  if (saved) applySavedState(saved);
  // A restored loss sets the field without a change event.
  updateMarginVisibility();

  // Save the state on every change, and on every keystroke in text fields.
  for (const id of [...savedFields().map(([, fieldId]) => fieldId), "modelSelect"]) {
    const el = document.getElementById(id);
    el.addEventListener("change", saveFinetuneState);
    if (el.type === "text" || el.type === "number") {
      el.addEventListener("input", saveFinetuneState);
    }
  }

  subscribeGpuQueues(showGpuQueues);

  const modelSelect = document.getElementById("modelSelect");
  modelSelect.addEventListener("change", () => applyModelDefaults(modelSelect.value));
  showDistanceModelHint(modelSelect.value);

  ["numEpochs", "batchSize", "patchesPerEpoch"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", updateAugmentAdvice);
  });
  updateAugmentAdvice();

  if (isBrowserMode()) {
    document.getElementById("finetuneMode").addEventListener("change", applyTrainingMode);
    ["tileSize", "batchSize"].forEach((id) => document.getElementById(id).addEventListener("change", updateTileHint));
    applyTrainingMode();
  }

  return {
    saved,
    save: saveFinetuneState,
    trainingParams,
  };
}
