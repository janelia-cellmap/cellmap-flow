// The model picker at the top of the Annotation Crops panel: the served
// models that can be finetuned (/api/finetune/models), and the one picked.
import { esc } from "../../lib/dom.js";
import { poll } from "../../lib/poll.js";
import { getAnswer } from "./requests.js";
import { applyModelDefaults } from "./training-form.js";

// Models only appear in g.models_config once a pipeline has been submitted
// on the main tab. The finetune tab is usually opened first, so a single
// load on page load almost always finds nothing: the list is polled quietly
// in the background instead, and polling stops as soon as models show up.
const MODEL_POLL_INTERVAL_MS = 2000;
const MODEL_POLL_MAX_ATTEMPTS = 150;  // ~5 minutes, enough for a queued job

function displayModelInfo(model) {
  if (!model) return;

  const voxelShape = model.write_shape.map((s, i) =>
    Math.round(s / model.output_voxel_size[i])
  );

  document.getElementById("modelInfoContent").innerHTML = `
      <small>
        <strong>Name:</strong> ${esc(model.name)}<br>
        <strong>Output Size (nm):</strong> [${esc(model.write_shape.join(', '))}]<br>
        <strong>Voxel Size (nm):</strong> [${esc(model.output_voxel_size.join(', '))}]<br>
        <strong>Crop Shape (voxels):</strong> [${esc(voxelShape.join(', '))}]<br>
        <strong>Channels:</strong> ${esc(model.output_channels)}
      </small>
    `;
  document.getElementById("modelInfo").style.display = "block";
}

// log: the Annotation Crops panel's log. savedModelName: the pick saved with
// the form, preferred on the first load.
// Returns { selected() }: the picked model's entry, or null if none.
export function initModelPicker({ log, savedModelName }) {
  const modelSelect = document.getElementById("modelSelect");
  const modelSelectionDiv = document.getElementById("modelSelectionDiv");
  let models = [];
  let selectedModel = null;

  let loggedNoModels = false;
  let lastLoadedCount = null;

  // One load of the list. The first load, and a Refresh, say what they
  // found; the polls in between are quiet unless the number of models
  // changes. Resolves to false once models are listed, which stops the poll.
  function loadModels({ tick, stale }) {
    const quiet = tick > 1;
    return getAnswer("/api/finetune/models")
      .then(data => {
        if (stale()) return;
        if (data.error) {
          if (!quiet) log.add(`Error: ${data.error}`);
          return;
        }

        models = data.models;

        if (models.length === 0) {
          if (!loggedNoModels) {
            loggedNoModels = true;
            log.add("No models available for finetuning yet");
            log.add("  → Submit a model from the main tab; this list updates automatically");
          }
          modelSelect.innerHTML = "";
          lastLoadedCount = 0;
          return;
        }
        // Rebuild the options from this response alone.
        const previousName = modelSelect.value;
        modelSelect.innerHTML = "";
        models.forEach(model => {
          const option = document.createElement("option");
          option.value = model.name;
          option.textContent = model.name;
          modelSelect.appendChild(option);
        });

        // Keep the current pick when the list is reloaded (Refresh); on the
        // first load prefer the saved state, then the server's choice.
        const has = (name) => name && models.some(m => m.name === name);
        if (has(previousName)) {
          modelSelect.value = previousName;
        } else if (has(savedModelName)) {
          modelSelect.value = savedModelName;
        } else if (has(data.selected_model)) {
          modelSelect.value = data.selected_model;
        }

        // Only one model: nothing to choose, so hide the picker. Show it
        // again when a later load finds more.
        modelSelectionDiv.style.display = models.length === 1 ? "none" : "";
        selectedModel = models.find(m => m.name === modelSelect.value) || models[0];
        displayModelInfo(selectedModel);
        // A new pick, as choosing it would: the defaults of its kind.
        if (modelSelect.value !== previousName) applyModelDefaults(modelSelect.value);

        loggedNoModels = false;
        if (!quiet || models.length !== lastLoadedCount) {
          log.add(`Loaded ${models.length} model(s)`);
        }
        lastLoadedCount = models.length;
        return false;
      })
      .catch(err => {
        if (!quiet) log.add(`Error loading models: ${err}`);
        console.error(err);
      });
  }

  // Bound once, here: loadModels runs on every refresh and poll.
  modelSelect.addEventListener("change", function() {
    selectedModel = models.find(m => m.name === this.value);
    displayModelInfo(selectedModel);
  });

  // The first load, then the polls: MODEL_POLL_MAX_ATTEMPTS after it. They
  // wait while the page is hidden, and one runs as soon as it is shown.
  const modelPoll = poll(loadModels, {
    intervalMs: MODEL_POLL_INTERVAL_MS,
    maxTicks: 1 + MODEL_POLL_MAX_ATTEMPTS,
    pauseWhenHidden: true,
  });

  document.getElementById("refreshModelsBtn").addEventListener("click", function() {
    log.add("Refreshing model list...");
    // The options are left in place so loadModels can keep the current
    // pick; it rebuilds them from the response.
    models = [];
    selectedModel = null;
    modelPoll.restart();
  });

  return { selected: () => selectedModel };
}
