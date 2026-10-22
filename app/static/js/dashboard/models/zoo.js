// The BioImage Model Zoo list: listed like the Hugging Face models, from the
// zoo's index (models/bioimage_catalog.py; GET /api/bioimage-models, POST
// .../refresh), loaded when its accordion opens and filtered to EM by
// default. A ticked row takes a voxel size. Submit sends the ticked models
// as selected_bioimage_models, each {id, voxel_size}.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { h } from "../../lib/dom.js";
import { pageData } from "../../lib/page-data.js";
import { notice, tag, untickBox } from "./row.js";

// The zoo models already running, ticked on load, with the voxel size each
// was given; one removed from the Loaded models list is taken out.
let running = new Map();
let loaded = false;
const VOXEL_TITLE =
  "Voxel size in nm, z,y,x or one number: the scale the model reads the data at. Filled in with what it "
  + "was trained at when cellmap-flow knows it; blank, the model's own if its description declares one.";
// The zoo changes under the page: a list cached over an hour ago comes back
// "stale" and is fetched again behind it, and an open page fetches it again
// every hour. Quietly: a failure leaves the list shown as it was.
const REFRESH_MS = 60 * 60 * 1000;
let refreshTimer = null;

// One row, built from nodes rather than an HTML string: every text in it
// comes from the zoo's uploaders. voxelSize is the value to start with (an
// array, a string or undefined); a row starts ticked when it is given.
function row(model, voxelSize) {
  const id = "chk_zoo_" + model.key.replace(/\W+/g, "_");
  const input = h("input", {
    class: "form-check-input zoo-model-checkbox", type: "checkbox", id, value: model.key,
    checked: voxelSize !== undefined,
  });
  // What it was trained at, from cellmap-flow's own table: no zoo model
  // says it in its description.
  const trained = model.trained || null;
  const trainedNm = trained && trained.voxel_size ? trained.voxel_size.join("×") + " nm" : "";
  if (voxelSize === undefined && trained && trained.voxel_size) voxelSize = trained.voxel_size;

  // Shown only while the row is ticked.
  const voxelRow = h("div", { class: "zoo-voxel-row d-flex align-items-center gap-2 mt-1" },
    h("label", { for: id + "_voxel" }, "Voxel (nm)"),
    h("input", {
      type: "text", class: "form-control form-control-sm zoo-voxel", id: id + "_voxel", placeholder: "from model",
      title: VOXEL_TITLE, value: Array.isArray(voxelSize) ? voxelSize.join(",") : (voxelSize || ""),
    }));
  const shown = [voxelRow];
  if (trained) {
    shown.push(h("div", { class: "zoo-trained", title: trained.confidence + " confidence: " + trained.source },
      (trainedNm ? "Trained at " + trainedNm + " (z,y,x) on " : "Trained on ")
      + trained.trained_on + "." + (trained.note ? " " + trained.note : "")
      + (trainedNm ? "" : " Enter the voxel size of the data to run it on.")));
  }
  const showTicked = () => shown.forEach((el) => { el.style.display = input.checked ? "" : "none"; });
  showTicked();
  input.addEventListener("change", showTicked);

  return h("div", {
    class: "form-check zoo-model-item",
    dataset: {
      search: [model.name, model.key, model.id, model.description, ...model.tags].join(" ").toLowerCase(),
      em: model.em ? "1" : "",
      dims: model.dims || "",
    },
  },
  input,
  h("label", {
    class: "form-check-label", for: id, title: [model.description, model.key, model.license].filter(Boolean).join("\n"),
  }, model.name),
  model.dims ? tag(model.dims.toUpperCase()) : null,
  trainedNm ? tag(trainedNm, "Trained at (z×y×x) on " + trained.trained_on) : null,
  model.weight_formats.map((format) => tag(format, "Weight format")),
  /^https?:\/\//.test(model.url || "")
    ? h("a", { class: "zoo-link", href: model.url, target: "_blank", rel: "noopener noreferrer", title: "Open on bioimage.io" }, "↗")
    : null,
  model.description ? h("div", { class: "zoo-desc", title: model.description }, model.description) : null,
  shown);
}

// Search words (all must match), EM only, and 2D/3D (neither ticked: any).
// A ticked row stays in view, so nothing is submitted unseen.
function filter() {
  const words = document.getElementById("zooSearchBar").value.toLowerCase().split(/\s+/).filter(Boolean);
  const emOnly = document.getElementById("zooEmOnly").checked;
  const dims = [["zoo2d", "2d"], ["zoo3d", "3d"]]
    .filter(([id]) => document.getElementById(id).checked)
    .map(([, d]) => d);
  const rows = document.querySelectorAll("#zooModelList .zoo-model-item");
  let shown = 0;
  rows.forEach((item) => {
    const matches = words.every((w) => item.dataset.search.includes(w))
      && (!emOnly || item.dataset.em)
      && (!dims.length || dims.includes(item.dataset.dims));
    const show = matches || item.querySelector(".zoo-model-checkbox").checked;
    item.style.display = show ? "" : "none";
    if (show) shown += 1;
  });
  document.getElementById("zooCount").textContent = rows.length ? shown + " / " + rows.length : "";
}

function render(data) {
  const list = document.getElementById("zooModelList");
  // A refresh keeps what is ticked, and the voxel sizes typed.
  const ticked = new Map(running);
  list.querySelectorAll(".zoo-model-item").forEach((item) => {
    const box = item.querySelector(".zoo-model-checkbox");
    if (box.checked) ticked.set(box.value, item.querySelector(".zoo-voxel").value);
    else ticked.delete(box.value);
  });
  list.replaceChildren();
  const placeholder = document.getElementById("zooPlaceholder");
  if (placeholder) placeholder.remove();

  if (data.error) {
    list.append(notice("Error: " + data.error, true));
    return;
  }
  const models = data.models || [];
  if (!models.length) {
    list.append(notice("No models found."));
    return;
  }
  document.getElementById("zooControls").style.display = "";
  document.getElementById("zooRefreshBtn").title =
    "Refresh from bioimage.io" + (data.fetched ? " (listed " + data.fetched + ")" : "");
  models.forEach((model) => list.append(row(model, ticked.has(model.key) ? ticked.get(model.key) : undefined)));
  filter();
}

function refreshQuietly() {
  postJSON("/api/bioimage-models/refresh")
    .then((data) => { if (!data.error) render(data); })
    .catch(() => {});
}

function load(refresh) {
  const spinner = document.getElementById("zooLoadingSpinner");
  spinner.classList.remove("d-none");
  (refresh ? postJSON("/api/bioimage-models/refresh") : getJSON("/api/bioimage-models"))
    .then((data) => {
      render(data);
      loaded = true;
      if (data.stale) refreshQuietly();
      if (!refreshTimer) refreshTimer = setInterval(refreshQuietly, REFRESH_MS);
    })
    .catch((err) => {
      // The routes answer a failed fetch of the zoo's index with
      // {"error": ...}; anything else never reached them.
      render({ error: err instanceof ApiError ? err.message : "Error loading models: " + err });
      loaded = err instanceof ApiError;
    })
    .finally(() => spinner.classList.add("d-none"));
}

export function init() {
  running = new Map((pageData().default_bioimage_models || []).map((m) => [m.id, m.voxel_size]));
  document.getElementById("collapse_zoo").addEventListener("show.bs.collapse", () => {
    if (!loaded) load(false);
  });
  document.getElementById("zooRefreshBtn").addEventListener("click", () => load(true));
  document.getElementById("zooSearchBar").addEventListener("input", filter);
  ["zooEmOnly", "zoo2d", "zoo3d"].forEach((id) => document.getElementById(id).addEventListener("change", filter));
}

// POST /api/models' part: the models ticked, with the voxel size typed
// (blank: the model's own).
export function payload() {
  const selected = [];
  document.querySelectorAll("#zooModelList .zoo-model-item").forEach((item) => {
    const box = item.querySelector(".zoo-model-checkbox");
    if (!box.checked) return;
    const voxel = item.querySelector(".zoo-voxel").value.trim();
    selected.push({ id: box.value, voxel_size: voxel || null });
  });
  return { selected_bioimage_models: selected };
}

// Untick a loaded model's row here, if it has one; a list not loaded yet
// will not tick it either.
export function untick(model) {
  if (!model.zoo_id) return;
  running.delete(model.zoo_id);
  document.querySelectorAll("#zooModelList .zoo-model-checkbox").forEach((box) => {
    if (box.value === model.zoo_id) untickBox(box);
  });
  filter();
}
