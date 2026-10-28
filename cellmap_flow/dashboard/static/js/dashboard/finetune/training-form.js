// The Finetune tab's form: the output path and the training parameters,
// kept in localStorage (the output path on the server too), the advice shown
// next to some of them, the Learning Rates typed or picked from their
// presets and eased down over the run or not, the Channel a model of
// several channels trains, and the GPU queue picker.
import { isBrowserMode } from "../../lib/page-data.js";
import { readJSON, writeJSON } from "../../lib/stored.js";
import { createQueuePicker, queueHint, subscribeGpuQueues } from "../../shared/gpu-queues.js";
import { getAnswer, postAnswer } from "./requests.js";

const FINETUNE_STATE_KEY = "finetuneFormState";

// The fields kept in localStorage, as [key in the saved state, element id,
// how it is restored]:
// - "value": when the saved value is not empty;
// - "blank": also when it is "", which is a choice of its own (keep the
//   manifest's value), not a missing one;
// - "checkbox": whenever it was saved.
// The model picker's choice is saved with them, as selectedModelName, the
// Channel picked for each model, as channelByModel (see setChannelChoice),
// and the GPU queues ticked, as gpuQueues (none before a choice is made:
// every queue).
const SAVED_FIELDS = [
  ["outputPath", "outputPath", "value"],
  ["checkpointPath", "checkpointPath", "value"],
  ["loraRank", "loraRank", "value"],
  ["numEpochs", "numEpochs", "value"],
  ["batchSize", "batchSize", "value"],
  ["patchesPerEpoch", "patchesPerEpoch", "blank"],
  ["rehearsalFraction", "rehearsalFraction", "blank"],
  ["learningRate", "learningRate", "value"],
  ["learningRateEase", "learningRateEase", "checkbox"],
  ["lossType", "lossType", "value"],
  ["marginValue", "marginValue", "value"],
  ["labelSmoothing", "labelSmoothing", "blank"],
  ["distillationLambda", "distillationLambda", "value"],
  ["distillationScope", "distillationScope", "value"],
  ["balanceClasses", "balanceClasses", "checkbox"],
  ["augment", "augment", "checkbox"],
  ["autoServe", "autoServeCheck", "checkbox"],
];

// The fields saved in this mode. The browser mode keeps the Loss Function
// under a key of its own, headLossType: the field was hidden there before
// the head had a choice of losses, and every browser that saved the form
// saved the server template's default, margin, which must not come back as
// the browser's loss. Its batch, voxels per step, is a field of its own
// (batchPixels, in the template's browser-only block). The whole network's
// fields are browser-only too: the Training choice, the Tile Size (a 3D
// model's Block Size) and the whole network's learning rate and its Ease
// down; and its batch, the server's Batch Size, is kept under a key of its
// own (fullBatchSize) for the same reason as the loss: every browser saved
// the hidden field's 8.
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
      ["fullLearningRateEase", "learningRateFullEase", "checkbox"],
    ]);
}

function saveFinetuneState() {
  const state = {};
  for (const [key, id, kind] of savedFields()) {
    const el = document.getElementById(id);
    // A field the form holds (a 3D model's Batch Size, at 1) saves the value it will put back.
    const held = el.disabled && el.dataset.kept !== undefined;
    state[key] = kind === "checkbox" ? el.checked : held ? el.dataset.kept : el.value;
  }
  state.selectedModelName = document.getElementById("modelSelect").value;
  const channel = document.getElementById("trainChannel");
  state.channelByModel = keptChannels(channel.dataset.model, channel.value);
  state.gpuQueues = queuePicker.value();
  writeJSON(FINETUNE_STATE_KEY, state);
}

function restoreFinetuneState() {
  return readJSON(FINETUNE_STATE_KEY, null);
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
  queuePicker.setValue(saved.gpuQueues);
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

// --- The Learning Rates ---------------------------------------------------
//
// A Learning Rate (learningRate: the server's, and the browser's last
// layer's; learningRateFull: the browser's whole network's) takes any number
// above 0, typed as 1e-4 or 0.0003, and has its presets beside it
// (<id>Presets). Picking a preset types it in; the presets show the preset
// typed, or Typed for any other rate. The field is what is saved and sent,
// so a typed rate comes back as typed. Its Ease down box (<id>Ease) is the
// schedule sent with it: ticked, the rate eases down to a tenth of it over
// the run (cosine); unticked, every step takes it (constant).
function rateFields() {
  return isBrowserMode() ? ["learningRate", "learningRateFull"] : ["learningRate"];
}

// Field `id`'s presets show the rate typed in it, or Typed.
function showRatePreset(id) {
  const presets = document.getElementById(`${id}Presets`);
  const typed = Number(document.getElementById(id).value);
  const preset = Array.from(presets.options).find((option) => option.value !== "" && Number(option.value) === typed);
  presets.value = preset ? preset.value : "";
}

// The rate in field `id`. Throws an Error, saying why, for anything but a
// number above 0 (empty, 0 or less, not a number): the routes refuse it
// too, but say less. Number, not parseFloat, which reads "1e-4x" as 1e-4.
function readRate(id) {
  const typed = document.getElementById(id).value.trim();
  const rate = Number(typed);
  if (!typed || !Number.isFinite(rate) || rate <= 0) {
    const why = !typed ? "it is empty" : Number.isFinite(rate) ? `${typed} is not above 0` : `"${typed}" is not a number`;
    throw new Error(`Learning Rate: ${why}. Type a number above 0, such as 1e-4 or 0.0003, or pick one of its presets.`);
  }
  return rate;
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
  // The browser's whole network has a Learning Rate and an Ease down of its own.
  const rate = isBrowserMode() && document.getElementById("finetuneMode").value === "full" ? "learningRateFull" : "learningRate";
  const params = {
    lora_r: parseInt(document.getElementById("loraRank").value),
    num_epochs: parseInt(document.getElementById("numEpochs").value),
    batch_size: parseInt(document.getElementById("batchSize").value),
    learning_rate: readRate(rate),
    learning_rate_schedule: document.getElementById(`${rate}Ease`).checked ? "cosine" : "constant",
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
  // learning rate and schedule (read above), its tile side and the batch
  // above, in tiles, and augmentation. The server-only fields above go too
  // and the page's routes ignore them (web/src/backend/routes/finetune.ts
  // readTraining).
  // The Loss Function offers the head's losses there (BCE, MSE, margin) and
  // is read as on the server, by both.
  if (isBrowserMode()) {
    const mode = document.getElementById("finetuneMode").value;
    params.finetune_mode = mode;
    if (mode === "full") {
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
  // Nothing when the Channel is hidden (the model trains its channels
  // together), and nothing when it is shown but not picked: the submit
  // route refuses that, saying why, in both modes.
  const channel = pickedChannel();
  if (channel !== null) params.select_channel = channel;
  return params;
}

// --- The Channel ----------------------------------------------------------
//
// A model of several output channels whose target is not affinities or flows
// (its /api/finetune/models entry's channel_choice) trains one channel at a
// time: the one picked learns the paint, the others are held to what the
// model predicted (select_channel). Its indices mean nothing for another
// model, so the pick is kept per model (channelByModel in the saved state).
// The select remembers the model it is filled for (data-model, absent while
// hidden), and a pick is saved under that model: when the model picker
// changes, saveFinetuneState runs before models.js fills the select for the
// new model, and must not save the old model's pick as the new one's.
let channelPlaceholder = null;  // the template's "Pick the channel..." option, kept to put first

// The Channel kept for each model, with `value` as model `name`'s (when
// `name` is given).
function keptChannels(name, value) {
  const kept = { ...((restoreFinetuneState() || {}).channelByModel || {}) };
  if (name) kept[name] = value;
  return kept;
}

// The Channel picked, as an index; null while it is hidden or nothing is
// picked. What training sends as select_channel, and Seed from Prediction
// too (label-it.js), so the seed reads the channel the paint is for.
export function pickedChannel() {
  const select = document.getElementById("trainChannel");
  return select.dataset.model && select.value !== "" ? Number(select.value) : null;
}

// Shows the Channel for `model` (an entry of /api/finetune/models, or null)
// when it trains one channel at a time, its options the channels by name
// ("channel <i>" where the names are not known) after the template's
// placeholder, and its pick the one kept for the model; hides it otherwise.
// models.js calls it on every load of the list: the options are rebuilt only
// for another model or other channels, so an open select is not reset under
// the cursor.
export function setChannelChoice(model) {
  const group = document.getElementById("trainChannelGroup");
  const select = document.getElementById("trainChannel");
  const choice = Boolean(model && model.channel_choice);
  group.hidden = !choice;
  if (!choice) {
    delete select.dataset.model;
    return;
  }
  channelPlaceholder = channelPlaceholder ?? select.options[0];
  const names = model.channels ? model.channels.map((name, i) => channelLabel(name, i, false))
    : Array.from({ length: model.output_channels }, (_, i) => `channel ${i}`);
  const options = [channelPlaceholder, ...names.map((name, i) => new Option(name, String(i)))];
  const key = (opts) => opts.map((o) => `${o.value}|${o.text}`).join(",");
  if (select.dataset.model === model.name && key(Array.from(select.options)) === key(options)) return;
  select.replaceChildren(...options);
  select.dataset.model = model.name;
  const kept = String(keptChannels()[model.name] ?? "");
  select.value = options.some((option) => option.value === kept) ? kept : "";
  if (select.value !== "") showChannel(model.name, Number(select.value));
}

// A channel by name and number, the number the viewer's channel slider
// (the layer's c') shows: "organelle (channel 1)", or "organelle (1)" short.
export function channelLabel(name, index, long = true) {
  return Number.isInteger(index) ? `${name} (${long ? "channel " : ""}${index})` : name;
}

// Model `name`'s layer in the viewer shows channel `index` (POST
// /api/finetune/channel), now and whenever it is drawn again, and so do its
// finetunes': the channel picked is the one to look at. Nothing is lost if
// it fails: the viewer just shows the channel it showed.
function showChannel(name, index) {
  postAnswer("/api/finetune/channel", { model_name: name, select_channel: index }).catch(() => {});
}

// Makes channel `index` model `name`'s Channel: kept for it, and shown at
// once when the select is filled for it, else when it is picked. For rank.js's
// Use this model, the explicit "use what the ranking found": the channel that
// matched the paint best, for the model it makes the one to finetune, before
// the model picker has it.
export function pickChannel(name, index) {
  writeJSON(FINETUNE_STATE_KEY, { ...(restoreFinetuneState() || {}), channelByModel: keptChannels(name, String(index)) });
  const select = document.getElementById("trainChannel");
  if (select.dataset.model === name) select.value = String(index);
  showChannel(name, index);
}

// Whether the Channel is shown for model `name` with nothing picked.
// Training it then first picks the channel that matches the paint best
// (job-monitor.js), as a score of it does here (fillChannel).
export function channelUnpicked(name) {
  const select = document.getElementById("trainChannel");
  return select.dataset.model === name && select.value === "";
}

// pickChannel, only when the Channel is shown for model `name` with nothing
// picked and has channel `index`: what a score of the model does with the
// channel that matched best (Fit to Paint's, Rank's for the model being
// trained, Train's), which never overrides a channel the user picked.
// Returns whether it set it.
export function fillChannel(name, index) {
  const select = document.getElementById("trainChannel");
  const fills = channelUnpicked(name) && Array.from(select.options).some((option) => option.value === String(index));
  if (fills) pickChannel(name, index);
  return fills;
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
  // The browser's whole network trains on the same tiles every epoch, as
  // many as the paint makes, which the form cannot know: the advice is the
  // epochs alone.
  if (isBrowserMode()) {
    advice.textContent = epochs > 0
      ? `Each tile or block is trained on ${epochs} time${epochs === 1 ? "" : "s"}, once an epoch. With augmentation, plan on 3 to 5 times the epochs; with paint on only one or two objects, leave it off.`
      : "";
    advice.className = "d-block text-muted";
    return;
  }
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

// The GPU queue picker shows what the queues look like right now; the job
// goes to the freest of those ticked, and moves on to the next while it has
// not started (submit.QueueFallback). The answers come from the poller the
// Models tab's pickers share. Its own choice, not the Models tab's: a
// training job may want other GPUs than a model's server.
let queuePicker;

function showGpuQueues(data) {
  // An answer with no queues leaves the picker and the note as they are,
  // rather than emptying them.
  if (!((data && data.queues) || []).length) return;
  queuePicker.render(data);
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
//
// A 3D model's whole network (full_training.dims 3) trains on cubes, one a
// step: the Tile Size field becomes Block Size, its sides N³, and the Batch
// Size is held at 1 (the batch picked for a 2D model is put back after).
let browserModel = null;
let headNote = null;  // the template's words for the head, kept to put back
let tileWords = null;  // the template's Tile Size label and its title, likewise
let batchHint = null;  // and its Batch Size hint

function trainingMode() {
  return document.getElementById("finetuneMode").value;
}

// Whether the model picked trains its whole network on 3D blocks.
function trainsBlocks() {
  return !!(browserModel && browserModel.full_training && browserModel.full_training.dims === 3);
}

// The model's tile sides as the Tile Size's options (a 3D model's block
// sides as Block Size's); the side saved, or the one picked, when the model
// has it, else its default.
function fillTileSizes(full) {
  const select = document.getElementById("tileSize");
  const keep = select.value || (restoreFinetuneState() || {}).tileSize;
  const power = full && full.dims === 3 ? "³" : "²";
  select.innerHTML = "";
  for (const tile of (full && full.tiles) || []) {
    const option = document.createElement("option");
    option.value = String(tile.side);
    option.textContent = `${tile.side}${power} (output ${tile.output}${power})${tile.side === full.default_tile ? " (Default)" : ""}`;
    select.appendChild(option);
  }
  const sides = Array.from(select.options).map((option) => option.value);
  if (sides.includes(String(keep))) select.value = String(keep);
  else if (full && full.default_tile) select.value = String(full.default_tile);
  const label = document.getElementById("tileSizeLabel");
  tileWords = tileWords ?? { text: label.textContent, title: label.title };
  label.textContent = power === "³" ? "Block Size" : tileWords.text;
  label.title = power === "³"
    ? "The side of the cubes the whole network trains on, in voxels of its input, around the voxels you painted. The model reads 88 voxels of context around what it predicts, so a block's output is 88 voxels smaller on each axis. A smaller block trains a step faster in far less memory (it grows with the cube of the side), but more of it is context. The sides offered are ones the network's pools and crops divide exactly."
    : tileWords.title;
}

// The Batch Size for a 3D model's whole network: one block a step, the
// field held at 1 and the batch picked before kept to put back.
function holdBatchForBlocks() {
  const select = document.getElementById("batchSize");
  const hint = document.getElementById("batchSizeHint");
  batchHint = batchHint ?? hint.textContent;
  if (trainsBlocks()) {
    if (!select.disabled) select.dataset.kept = select.value;
    select.value = "1";
    select.disabled = true;
    hint.textContent = "A 3D model's whole network trains one block a step: a block is a batch's memory already.";
  } else if (select.disabled) {
    select.disabled = false;
    if (select.dataset.kept) select.value = select.dataset.kept;
    delete select.dataset.kept;
    hint.textContent = batchHint;
  }
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
  const noun = full.dims === 3 ? "block" : "tile";
  const percent = tile.useful * 100;
  const useful = `${percent < 10 ? percent.toFixed(1) : Math.round(percent)}% of each ${noun} is output, the rest context`;
  // A 3D model's costs were measured on one GPU, which the hint names.
  const where = full.measured_gpu
    ? `${tile.measured ? "" : ", scaled from the sides"} measured on ${full.measured_gpu}`
    : tile.measured ? " where it was measured" : ", scaled from the sides measured";
  const cost = tile.step_ms === null
    ? ""
    : `About ${formatMs(tile.step_ms * batch)} and ${formatMb(tile.peak_mb * batch)} a step${where}; `;
  hint.textContent = `${cost}${useful}. Smaller ${noun}s train faster in less memory, but spend more of each ${noun} on context.`;
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
    const pieces = trainsBlocks() ? "in blocks around the voxels you paint" : "in tiles cut in the plane you paint in";
    note.textContent = mode === "full"
      ? `In the browser, this model can train its whole network: every weight, on the voxels you paint, with TensorFlow.js on this machine's GPU, ${pieces}. It takes minutes where the last layer takes seconds, and can learn what the model's features miss. Paint some background and some objects on a few slices, then train.`
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
  holdBatchForBlocks();
  applyTrainingMode();
  if (!modes.length) {
    const browserNote = document.getElementById("browserFinetuneNote");
    if (browserNote) browserNote.textContent = "This model cannot be finetuned in the browser: its manifest declares no training for the file served (an int8 file's weights are quantized and cannot be written).";
  }
}

// Returns { saved, save(), trainingParams(), gpuQueues() }: saved is the
// state restored at load (null if none); save() stores the form's state
// now; gpuQueues() is the queues ticked ([] until the queues are listed,
// while none has been chosen).
export function initTrainingForm() {
  const outputPathInput = document.getElementById("outputPath");
  queuePicker = createQueuePicker(document.getElementById("gpuQueue"), {
    missingSuffix: " (selected)",
    onChange: () => saveFinetuneState(),
  });

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

  // A preset picked is typed into its Learning Rate; the presets follow
  // what is typed, and what was restored.
  for (const id of rateFields()) {
    const presets = document.getElementById(`${id}Presets`);
    presets.addEventListener("change", () => {
      const field = document.getElementById(id);
      if (presets.value === "") return field.focus();  // Typed: type one
      field.value = presets.value;
      saveFinetuneState();
    });
    document.getElementById(id).addEventListener("input", () => showRatePreset(id));
    showRatePreset(id);
  }

  // A Channel picked by hand is the one the viewer shows too.
  const channel = document.getElementById("trainChannel");
  channel.addEventListener("change", () => {
    if (channel.dataset.model && channel.value !== "") showChannel(channel.dataset.model, Number(channel.value));
  });

  // Save the state on every change, and on every keystroke in text fields.
  for (const id of [...savedFields().map(([, fieldId]) => fieldId), "modelSelect", "trainChannel"]) {
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
    gpuQueues: () => queuePicker.value(),
  };
}
