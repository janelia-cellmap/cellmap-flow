// The Finetune tab's form: the output path and the training parameters,
// kept in localStorage (the output path on the server too), the advice shown
// next to some of them, and the GPU queue picker.
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

function saveFinetuneState() {
  const state = {};
  for (const [key, id, kind] of SAVED_FIELDS) {
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
  for (const [key, id, kind] of SAVED_FIELDS) {
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

// Whether augmentation is worth enabling depends on how many times the run
// revisits the same patches, which is epochs x batches-per-epoch. Static
// advice cannot say that, so compute it from the form as it is edited.
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
    advice.textContent =
      "Set Patches per Epoch to estimate whether augmentation is worth it.";
    advice.className = "d-block mt-1 text-muted";
    return;
  }

  const steps = epochs * Math.ceil(patches / batch);
  const views = epochs; // each patch is revisited once per epoch
  if (steps < 200) {
    advice.textContent =
      `~${steps} gradient steps: too few for augmentation to help. ` +
      `Raise epochs or patches per epoch first.`;
    advice.className = "d-block mt-1 text-warning";
  } else {
    advice.textContent =
      `~${steps} gradient steps, each patch seen ~${views}x: ` +
      `augmentation is worth enabling.`;
    advice.className = "d-block mt-1 text-success";
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

// Returns { saved, save(), readPatchesPerEpochOverride(),
// readRehearsalFractionOverride() }: saved is the state restored at load
// (null if none); save() stores the form's state now; the two readers
// return a blank field as undefined, and throw an Error, with a message for
// the user, for a value the trainer would refuse.
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

  // Save the state on every change, and on every keystroke in text fields.
  for (const id of [...SAVED_FIELDS.map(([, fieldId]) => fieldId), "modelSelect"]) {
    const el = document.getElementById(id);
    el.addEventListener("change", saveFinetuneState);
    if (el.type === "text" || el.type === "number") {
      el.addEventListener("input", saveFinetuneState);
    }
  }

  subscribeGpuQueues(showGpuQueues);

  ["numEpochs", "batchSize", "patchesPerEpoch"].forEach(function (id) {
    document.getElementById(id).addEventListener("change", updateAugmentAdvice);
  });
  updateAugmentAdvice();

  return {
    saved,
    save: saveFinetuneState,
    readPatchesPerEpochOverride,
    readRehearsalFractionOverride,
  };
}
