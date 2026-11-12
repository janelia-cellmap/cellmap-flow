// The Hugging Face list: the models on the Hub of the organizations listed
// (GET /api/huggingface-models, cached; POST .../refresh), one section for
// each organization (shared/model-list.js, as every model list is made), as
// the Local Catalog has one for each of its groups, loaded when one opens,
// each row tagged with what its listing says the model is (its formats,
// finetune modes, size and browser copy). On a server every model is shown
// to start, and Runs in the browser, ticked, shows only those with a copy
// that runs there; the rows are in order of name, or of size. A ticked row takes the
// voxel size to run it at, filled in with the one it is served at (its
// listing's run_voxel_size). Submit sends the ticked repos as
// selected_hf_models, and the voxel sizes typed as run_voxel_sizes.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { h } from "../../lib/dom.js";
import { listen, MODEL_LIST_CHANGED_EVENT } from "../../lib/events.js";
import { isBrowserMode, pageData } from "../../lib/page-data.js";
import { dimsWord, searchText } from "../../lib/search.js";
import { listControls, modelRow, modelSection, notice, runAtField, runVoxelSizes, untickBox } from "../../shared/model-list.js";

// The repos of the Hugging Face models already running, ticked on load; one
// removed from the Loaded models list is taken out.
let running = new Set();
let loaded = false;
// One group for each organization listed (pageData().hf_orgs: a server's
// hf_catalog.HUGGING_FACE_ORGS, cellmap and ai-hhmi; the browser's page,
// cellmap), in that order: its organization, its section, its search bar and
// Refresh (shown once there is a list), the line it shows until it is
// loaded, and its rows. The first keeps the ids the page has always had
// (hf, hfModelsContainer). One listing fills them all; a repo of an
// organization not listed (one added in the browser) goes in the first.
let groups = [];

function groupOf(modelId) {
  const org = modelId.split("/")[0];
  return groups.find((group) => group.org === org) ?? groups[0];
}

// One row; repo ids and model card descriptions come from Hugging Face, not
// from us, and modelRow builds it from nodes rather than an HTML string.
// Under it while it is ticked, the voxel size to run it at: blank, the one
// its metadata.json declares it reads (input_voxel_size).
function row(modelId, metadata) {
  const displayName = modelId.split("/").pop();
  // Unique across the groups: another organization's repo of the same name has its own id.
  const group = groupOf(modelId);
  const id = "chk_hf_" + (group === groups[0] ? "" : group.id + "_") + displayName;
  // Ticked when the page started with it, or when the listing says it is
  // being served (the browser mode's, after a reload or from a link).
  const checked = running.has(modelId) || metadata.selected === true;
  const runAt = runAtField({ id: `${id}_run`, value: metadata.run_voxel_size, own: metadata.input_voxel_size, checked });
  return modelRow({
    id, value: modelId, name: displayName, note: metadata.description,
    rowClass: "form-check mb-2 hf-model-item", boxClass: "hf-model-checkbox", boxName: "selected_hf_models", facts: metadata,
    checked, onTick: runAt.show, below: runAt.element,
    // What lib/search.js matches: the repo, its metadata, and the words the
    // ranking's filter takes too, its voxel size ("16nm") and "2d" or "3d".
    search: searchText(modelId, JSON.stringify(metadata), ...(metadata.output_voxel_size || []).map((size) => `${size}nm`),
      dimsWord(metadata.input_shape)),
  });
}

// The answer of a listing in every group. One listed with no models hides
// its search bar, Refresh and count again, as before its first list, while
// another group has some (whose Refresh lists every group); when none has
// any, or after an error, each keeps them as they were, a Refresh shown to
// try again, its count cleared.
function render(data) {
  for (const group of groups) group.placeholder.remove();
  if (data.error) {
    for (const group of groups) {
      group.list.replaceChildren(notice("Error: " + data.error, true));
      filter(group);
    }
    return;
  }
  const modelIds = Object.keys(data);
  for (const group of groups) {
    const ids = modelIds.filter((modelId) => groupOf(modelId) === group);
    if (ids.length) group.controls.element.hidden = false;
    else if (modelIds.length) group.controls.element.hidden = true;
    group.list.replaceChildren(...(ids.length ? ids.map((modelId) => row(modelId, data[modelId])) : [notice("No models found.")]));
    filter(group);
  }
}

function loading(on) {
  for (const group of groups) group.section.loading(on);
}

function load(url) {
  loading(true);
  (url.includes("refresh") ? postJSON(url) : getJSON(url))
    .then((data) => {
      loading(false);
      render(data);
      loaded = true;
    })
    .catch((err) => {
      loading(false);
      if (err instanceof ApiError) {
        // The routes answer a failed listing with {"error": ...} and a
        // 500; show it in the list like any other answer.
        render({ error: err.message });
        loaded = true;
        return;
      }
      for (const group of groups) group.placeholder.textContent = "Error loading models: " + err;
      loaded = false;
    });
}

// Rows for the models listed now that the list has none for (the browser
// mode's Kept in this browser brought one into the tab), the rows there left
// as they are, with their ticks and the voxel sizes typed under them.
function addListed() {
  getJSON("/api/huggingface-models")
    .then((data) => {
      const shown = new Set(groups.flatMap((group) => [...group.list.querySelectorAll(".hf-model-checkbox")].map((box) => box.value)));
      for (const modelId of Object.keys(data).filter((modelId) => !shown.has(modelId))) {
        const group = groupOf(modelId);
        group.list.append(row(modelId, data[modelId]));
        group.controls.element.hidden = false;
        filter(group);
      }
    })
    .catch((err) => console.warn("Could not list the Hugging Face models again:", err));
}

// The search and Runs in the browser: a row shows when it has every word
// typed and, while that is ticked, a browser copy. A ticked row stays in
// view, as the zoo's does, so nothing is submitted unseen: the models
// running start ticked, with a copy or not. Then the rows in the order
// chosen, and the count of those shown; a group with no rows keeps the line
// its list shows instead (No models found., an error), its count cleared.
function filter(group) {
  const { controls, list } = group;
  const rows = [...list.querySelectorAll(".hf-model-item")];
  if (!rows.length) {
    controls.count(0, 0);
    return;
  }
  let noCopy = 0;
  const shown = rows.filter((item) => {
    const show = controls.matches(item.dataset) || item.querySelector(".hf-model-checkbox").checked;
    item.style.display = show ? "" : "none";
    if (!show && controls.noCopy(item.dataset)) noCopy += 1;
    return show;
  });
  list.replaceChildren(...controls.sorted(rows));
  controls.count(shown.length, rows.length, noCopy);
}

// An organization's group, appended to the page's Hugging Face accordion.
function addGroup(org, index) {
  const id = index === 0 ? "hf" : "hf_" + org.replace(/[^A-Za-z0-9]/g, "_");
  const group = { org, id };
  group.section = modelSection({
    id, title: org, bodyId: index === 0 ? "hfModelsContainer" : `${id}ModelsContainer`,
    onOpen: () => {
      if (!loaded) load("/api/huggingface-models");
    },
  });
  group.controls = listControls({
    id, placeholder: "Search: mito aff, 16 nm, 2d",
    title: "Every word typed, in the repo, its metadata, its voxel size or its 2D/3D",
    refreshTitle: "Refresh from Hugging Face", onRefresh: () => load("/api/huggingface-models/refresh"), onChange: () => filter(group),
    browser: !isBrowserMode(), sort: true,
  });
  group.controls.element.hidden = true;
  group.placeholder = h("p", { class: "text-muted" }, "Click to load Hugging Face models...");
  group.list = h("div");
  group.section.body.append(group.controls.element, group.placeholder, group.list);
  document.getElementById("hfModelAccordion").append(group.section.item);
  return group;
}

export function init() {
  running = new Set(pageData().default_hf_repos || []);
  groups = (pageData().hf_orgs || ["cellmap"]).map(addGroup);
  // A list not loaded yet lists it when it opens; one with no rows (an
  // error, or none found) is loaded again whole.
  listen(MODEL_LIST_CHANGED_EVENT, () => {
    if (!loaded) return;
    if (groups.some((group) => group.list.querySelector(".hf-model-checkbox"))) addListed();
    else load("/api/huggingface-models");
  });
}

// POST /api/models' part: the repos ticked, and the voxel sizes typed under them.
export function payload() {
  const boxes = [...document.querySelectorAll("#modelSelectionForm input.hf-model-checkbox:checked")];
  return { selected_hf_models: boxes.map((box) => box.value), run_voxel_sizes: runVoxelSizes(boxes) };
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
