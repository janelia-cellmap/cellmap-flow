// The BioImage Model Zoo list: listed like the Hugging Face models, in a
// section of its own (shared/model-list.js), from the zoo's index
// (models/bioimage_catalog.py; GET /api/bioimage-models, POST .../refresh),
// loaded when it opens and filtered to EM by default. A ticked row takes a
// voxel size. Submit sends the ticked models as selected_bioimage_models,
// each {id, voxel_size}.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { h, hint } from "../../lib/dom.js";
import { pageData } from "../../lib/page-data.js";
import { listControls, modelSection, notice, trainedNm, untickBox, zooRow } from "../../shared/model-list.js";

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
// The section, its search bar, Refresh and filters (shown once there is a
// list), the line it shows until it is loaded, and its rows.
let section = null;
let controls = null;
let placeholder = null;
let list = null;

// One row, the zoo's (shared/model-list.js zooRow): every text in it comes
// from the zoo's uploaders, and it is built from nodes rather than an HTML
// string. voxelSize is the value to start with (an array, a string or
// undefined); a row starts ticked when it is given. Under it, while it is
// ticked, the voxel size to run it at, and what it was trained at.
function row(model, voxelSize) {
  const id = "chk_zoo_" + model.key.replace(/\W+/g, "_");
  const checked = voxelSize !== undefined;
  const trained = model.trained || null;
  const trainedAt = trainedNm(model);
  if (voxelSize === undefined && trained && trained.voxel_size) voxelSize = trained.voxel_size;

  const shown = [h("div", { class: "zoo-voxel-row d-flex align-items-center gap-2 mt-1" },
    h("label", { for: id + "_voxel" }, "Voxel (nm)"),
    h("input", {
      type: "text", class: "form-control form-control-sm zoo-voxel", id: id + "_voxel", placeholder: "from model",
      title: VOXEL_TITLE, value: Array.isArray(voxelSize) ? voxelSize.join(",") : (voxelSize || ""),
    }))];
  if (trained) {
    // What else to know before running it (its note) under the "i".
    shown.push(h("div", { class: "zoo-trained", title: trained.confidence + " confidence: " + trained.source },
      hint((trainedAt ? "Trained at " + trainedAt + " (z,y,x) on " : "Trained on ")
        + trained.trained_on + "." + (trainedAt ? "" : " Enter the voxel size of the data to run it on."),
      trained.note)));
  }
  const showTicked = (on) => shown.forEach((el) => { el.style.display = on ? "" : "none"; });
  showTicked(checked);
  return zooRow(model, { id, checked, onTick: showTicked, boxClass: "zoo-model-checkbox", below: shown });
}

// Search words (all must match), EM only, and 2D/3D (neither ticked: any).
// A ticked row stays in view, so nothing is submitted unseen.
function filter() {
  const rows = list.querySelectorAll(".zoo-model-item");
  let shown = 0;
  rows.forEach((item) => {
    const show = controls.matches(item.dataset) || item.querySelector(".zoo-model-checkbox").checked;
    item.style.display = show ? "" : "none";
    if (show) shown += 1;
  });
  controls.count(shown, rows.length);
}

function render(data) {
  // A refresh keeps what is ticked, and the voxel sizes typed.
  const ticked = new Map(running);
  list.querySelectorAll(".zoo-model-item").forEach((item) => {
    const box = item.querySelector(".zoo-model-checkbox");
    if (box.checked) ticked.set(box.value, item.querySelector(".zoo-voxel").value);
    else ticked.delete(box.value);
  });
  placeholder.remove();

  if (data.error) {
    list.replaceChildren(notice("Error: " + data.error, true));
    return;
  }
  const models = data.models || [];
  if (!models.length) {
    list.replaceChildren(notice("No models found."));
    return;
  }
  controls.element.hidden = false;
  controls.refresh.title = "Refresh from bioimage.io" + (data.fetched ? " (listed " + data.fetched + ")" : "");
  list.replaceChildren(...models.map((model) => row(model, ticked.has(model.key) ? ticked.get(model.key) : undefined)));
  filter();
}

function refreshQuietly() {
  postJSON("/api/bioimage-models/refresh")
    .then((data) => { if (!data.error) render(data); })
    .catch(() => {});
}

function load(refresh) {
  section.loading(true);
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
    .finally(() => section.loading(false));
}

export function init() {
  running = new Map((pageData().default_bioimage_models || []).map((m) => [m.id, m.voxel_size]));
  section = modelSection({
    id: "zoo", title: "bioimage.io", bodyId: "zooModelsContainer",
    onOpen: () => {
      if (!loaded) load(false);
    },
  });
  controls = listControls({
    id: "zoo", zoo: true, placeholder: "Search models...", title: "Every word typed, in the name, description, tags or 2D/3D",
    refreshTitle: "Refresh from bioimage.io", onRefresh: () => load(true), onChange: filter,
  });
  controls.element.hidden = true;
  placeholder = h("p", { class: "text-muted" }, "Click to load BioImage Model Zoo models...");
  list = h("div", { class: "zoo-model-list" });
  section.body.append(controls.element, placeholder, list);
  document.getElementById("zooModelAccordion").append(section.item);
}

// POST /api/models' part: the models ticked, with the voxel size typed
// (blank: the model's own).
export function payload() {
  const selected = [];
  list.querySelectorAll(".zoo-model-item").forEach((item) => {
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
  list.querySelectorAll(".zoo-model-checkbox").forEach((box) => {
    if (box.value === model.zoo_id) untickBox(box);
  });
  filter();
}
