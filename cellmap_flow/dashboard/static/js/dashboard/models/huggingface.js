// The Hugging Face list: cellmap's models on the Hub (GET
// /api/huggingface-models, cached; POST .../refresh), in a section of its own
// (shared/model-list.js, as every model list is made), loaded when it opens.
// Submit sends the ticked repos as selected_hf_models.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { h } from "../../lib/dom.js";
import { pageData } from "../../lib/page-data.js";
import { dimsWord, searchText } from "../../lib/search.js";
import { listControls, modelRow, modelSection, notice, untickBox } from "../../shared/model-list.js";

// The repos of the Hugging Face models already running, ticked on load; one
// removed from the Loaded models list is taken out.
let running = new Set();
let loaded = false;
// The section, its search bar and Refresh (shown once there is a list), the
// line it shows until it is loaded, and its rows.
let section = null;
let controls = null;
let placeholder = null;
let list = null;

// One row; repo ids and model card descriptions come from Hugging Face, not
// from us, and modelRow builds it from nodes rather than an HTML string.
function row(modelId, metadata) {
  const displayName = modelId.split("/").pop();
  return modelRow({
    id: "chk_hf_" + displayName, value: modelId, name: displayName, note: metadata.description,
    rowClass: "form-check mb-2 hf-model-item", boxClass: "hf-model-checkbox", boxName: "selected_hf_models",
    // Ticked when the page started with it, or when the listing says it is
    // being served (the browser mode's, after a reload or from a link).
    checked: running.has(modelId) || metadata.selected === true,
    // What lib/search.js matches: the repo, its metadata, and the words the
    // ranking's filter takes too, its voxel size ("16nm") and "2d" or "3d".
    search: searchText(modelId, JSON.stringify(metadata), ...(metadata.output_voxel_size || []).map((size) => `${size}nm`),
      dimsWord(metadata.input_shape)),
  });
}

function render(data) {
  placeholder.remove();
  if (data.error) {
    list.replaceChildren(notice("Error: " + data.error, true));
    return;
  }
  const modelIds = Object.keys(data);
  if (modelIds.length === 0) {
    list.replaceChildren(notice("No models found."));
    return;
  }
  controls.element.hidden = false;
  list.replaceChildren(...modelIds.map((modelId) => row(modelId, data[modelId])));
}

function load(url) {
  section.loading(true);
  (url.includes("refresh") ? postJSON(url) : getJSON(url))
    .then((data) => {
      section.loading(false);
      render(data);
      loaded = true;
    })
    .catch((err) => {
      section.loading(false);
      if (err instanceof ApiError) {
        // The routes answer a failed listing with {"error": ...} and a
        // 500; show it in the list like any other answer.
        render({ error: err.message });
        loaded = true;
        return;
      }
      placeholder.textContent = "Error loading models: " + err;
      loaded = false;
    });
}

// The search: a row shows when it has every word typed.
function filter() {
  list.querySelectorAll(".hf-model-item").forEach((item) => {
    item.style.display = controls.matches(item.dataset) ? "" : "none";
  });
}

export function init() {
  running = new Set(pageData().default_hf_repos || []);
  section = modelSection({
    id: "hf", title: "cellmap", bodyId: "hfModelsContainer",
    onOpen: () => {
      if (!loaded) load("/api/huggingface-models");
    },
  });
  controls = listControls({
    id: "hf", placeholder: "Search: mito aff, 16 nm, 2d",
    title: "Every word typed, in the repo, its metadata, its voxel size or its 2D/3D",
    refreshTitle: "Refresh from Hugging Face", onRefresh: () => load("/api/huggingface-models/refresh"), onChange: filter,
  });
  controls.element.hidden = true;
  placeholder = h("p", { class: "text-muted" }, "Click to load Hugging Face models...");
  list = h("div");
  section.body.append(controls.element, placeholder, list);
  document.getElementById("hfModelAccordion").append(section.item);
}

// POST /api/models' part: the repos ticked.
export function payload() {
  const boxes = document.querySelectorAll("#modelSelectionForm input.hf-model-checkbox:checked");
  return { selected_hf_models: [...boxes].map((box) => box.value) };
}

// Untick a loaded model's row here, if it has one; a list not loaded yet
// will not tick it either.
export function untick(model) {
  if (!model.hf_id) return;
  running.delete(model.hf_id);
  document.querySelectorAll("#modelSelectionForm input.hf-model-checkbox").forEach((box) => {
    if (box.value !== model.hf_id) return;
    box.defaultChecked = false;
    untickBox(box);
  });
}
