// The Hugging Face list: cellmap's models on the Hub (GET
// /api/huggingface-models, cached; POST .../refresh), loaded when its
// accordion opens. Submit sends the ticked repos as selected_hf_models.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { h } from "../../lib/dom.js";
import { pageData } from "../../lib/page-data.js";
import { notice, untickBox } from "./row.js";

// The repos of the Hugging Face models already running, ticked on load; one
// removed from the Loaded models list is taken out.
let running = new Set();
let loaded = false;

// One row, built from nodes rather than an HTML string: repo ids and model
// card descriptions come from Hugging Face, not from us.
function row(modelId, metadata) {
  const displayName = modelId.split("/").pop();
  const id = "chk_hf_" + displayName;
  return h("div", {
    class: "form-check mb-2 hf-model-item",
    "data-search": (modelId + " " + JSON.stringify(metadata)).toLowerCase(),
  },
  h("input", {
    class: "form-check-input hf-model-checkbox", type: "checkbox", name: "selected_hf_models", id, value: modelId,
    // Ticked when the page started with it, or when the listing says it is
    // being served (the browser mode's, after a reload or from a link).
    checked: running.has(modelId) || metadata.selected === true,
  }),
  h("label", { class: "form-check-label", for: id }, displayName,
    metadata.description ? [" ", h("small", { style: { color: "#aaa" } }, "- " + metadata.description)] : null));
}

function render(data) {
  const container = document.getElementById("hfModelsContainer");
  container.querySelectorAll(".hf-model-item").forEach((el) => el.remove());
  const placeholder = document.getElementById("hfPlaceholder");
  if (placeholder) placeholder.remove();

  if (data.error) {
    container.append(notice("Error: " + data.error, true));
    return;
  }
  const modelIds = Object.keys(data);
  if (modelIds.length === 0) {
    container.append(notice("No models found."));
    return;
  }
  document.getElementById("hfSearchBar").style.display = "";
  document.getElementById("hfRefreshBtn").style.display = "";
  modelIds.forEach((modelId) => container.append(row(modelId, data[modelId])));
}

function load(url) {
  const spinner = document.getElementById("hfLoadingSpinner");
  spinner.classList.remove("d-none");
  (url.includes("refresh") ? postJSON(url) : getJSON(url))
    .then((data) => {
      spinner.classList.add("d-none");
      render(data);
      loaded = true;
    })
    .catch((err) => {
      spinner.classList.add("d-none");
      if (err instanceof ApiError) {
        // The routes answer a failed listing with {"error": ...} and a
        // 500; show it in the list like any other answer.
        render({ error: err.message });
        loaded = true;
        return;
      }
      const placeholder = document.getElementById("hfPlaceholder");
      if (placeholder) placeholder.textContent = "Error loading models: " + err;
      loaded = false;
    });
}

export function init() {
  running = new Set(pageData().default_hf_repos || []);
  document.getElementById("collapse_hf").addEventListener("show.bs.collapse", () => {
    if (!loaded) load("/api/huggingface-models");
  });
  document.getElementById("hfRefreshBtn").addEventListener("click", () => load("/api/huggingface-models/refresh"));
  document.getElementById("hfSearchBar").addEventListener("input", function () {
    const query = this.value.toLowerCase();
    document.querySelectorAll(".hf-model-item").forEach((item) => {
      item.style.display = item.getAttribute("data-search").includes(query) ? "" : "none";
    });
  });
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
