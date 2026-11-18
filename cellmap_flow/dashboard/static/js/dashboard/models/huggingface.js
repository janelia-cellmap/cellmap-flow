// The Hugging Face list: the models on the Hub of the organizations listed
// (GET /api/huggingface-models, cached; POST .../refresh), one section for
// each organization (shared/model-list.js, as every model list is made), as
// the Local Catalog has one for each of its groups, loaded when one opens,
// each row tagged with what its listing says the model is (its formats,
// finetune modes, size and browser copy). On a server every model is shown
// to start, and Runs in the browser, ticked, shows only those with a copy
// that runs there; the rows are in order of name, of size, or newest first. A ticked row takes the
// voxel size to run it at, filled in with the one it is served at (its
// listing's run_voxel_size). Submit sends the ticked repos as
// selected_hf_models, and the voxel sizes typed as run_voxel_sizes.
//
// The browser's listing has more than its organizations': the models of the
// repos added beside them (Add a model, or a link that named one, which the
// browser remembers, so they are listed on later visits too) and those
// finetuned in this tab, each with the `group` it comes from (web/src/models
// catalogGroup, as the ranking groups them) and its `repo`. Those are listed
// under sections of their own after the organizations' (OTHER_GROUPS): with
// cellmap's, someone's own repo read as one of cellmap's models. An added
// repo has a Forget (POST /api/huggingface-models/forget). Each
// organization's section has its own search, sort and count; the others are
// short and listed whole. A server's listing has no groups: every model is
// its organization's.
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
// Whether a listing is on its way: a second organization's section opened
// before it answers waits for it, as one listing fills every section, rather
// than starting another (two listed the Hub side by side on a cold cache).
let listing = false;
// One group for each organization listed (pageData().hf_orgs: a server's
// hf_catalog.HUGGING_FACE_ORGS, cellmap and ai-hhmi; the browser's page,
// cellmap), in that order: its organization, its section, its search bar and
// Refresh (shown once there is a list), the line it shows until it is
// loaded, and its rows. The first keeps the ids the page has always had
// (hf, hfModelsContainer). One listing fills them all; a repo of an
// organization not listed goes in the first.
let groups = [];
// The groups listed after the organizations', in their order: each
// section's title, and the line under it saying where its models come from.
const OTHER_GROUPS = {
  Added: {
    title: "Added",
    note: "From other Hugging Face repos, added with Add a model or by a link. This browser remembers them, so they are listed on later visits too.",
  },
  Finetuned: { title: "Finetuned here", note: "Finetuned in this tab, or opened from an .onnx file." },
};
// Their sections, by group, each made when a model is first listed under
// it: { section, list }.
let others = new Map();

function groupOf(modelId) {
  const org = modelId.split("/")[0];
  return groups.find((group) => group.org === org) ?? groups[0];
}

// One row; repo ids and model card descriptions come from Hugging Face, not
// from us, and modelRow builds it from nodes rather than an HTML string. Its
// description is one line under it, as a zoo model's, shown whole on a click:
// beside the name it took several lines a row (a mia-train model's, the first
// paragraph of its card, most of ten).
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
    id, value: modelId, name: displayName, description: metadata.description,
    rowClass: "form-check mb-2 hf-model-item", boxClass: "hf-model-checkbox", boxName: "selected_hf_models", facts: metadata,
    checked, onTick: runAt.show, below: runAt.element,
    // What lib/search.js matches: the repo, its metadata, and the words the
    // ranking's filter takes too, its voxel size ("16nm") and "2d" or "3d".
    search: searchText(modelId, JSON.stringify(metadata), ...(metadata.output_voxel_size || []).map((size) => `${size}nm`),
      dimsWord(metadata.input_shape)),
  });
}

// Whether a listing's model goes in a section after the organizations'.
const otherGroup = (metadata) => (Object.hasOwn(OTHER_GROUPS, metadata.group) ? metadata.group : null);

// The list of `group`'s section, made the first time a model is listed
// under it, after the organizations' and in OTHER_GROUPS' order.
function groupList(group) {
  let made = others.get(group);
  if (!made) {
    const { title, note } = OTHER_GROUPS[group];
    const groupSection = modelSection({ id: `hf_${group.toLowerCase()}`, title, open: true });
    const rows = h("div");
    groupSection.body.append(h("p", { class: "text-muted small mb-2" }, note), rows);
    made = { section: groupSection, list: rows };
    others.set(group, made);
    const accordion = document.getElementById("hfModelAccordion");
    for (const name of Object.keys(OTHER_GROUPS)) if (others.has(name)) accordion.append(others.get(name).section.item);
  }
  return made.list;
}

// A row of an added repo's model goes under its repo's line, which has the
// repo's Forget; every other one at the end of its group's list.
function place(modelId, metadata) {
  const group = otherGroup(metadata);
  const into = groupList(group);
  if (group !== "Added" || !metadata.repo) {
    into.append(row(modelId, metadata));
    return;
  }
  let block = [...into.children].find((el) => el.dataset.repo === metadata.repo);
  if (!block) {
    block = repoBlock(metadata.repo, into);
    into.append(block);
  }
  block.append(row(modelId, metadata));
}

// An added repo's block in the Added list `into`: its name and Forget, then
// its rows. Forget takes the block away, and the section with the last; it
// is refused, saying why under the name, while one of the repo's models runs.
function repoBlock(repo, into) {
  const forget = h("button", {
    type: "button", class: "btn btn-sm btn-outline-secondary py-0",
    title: `Stop listing ${repo}'s models in this browser. Add a model or a link that names it adds it again.`,
  }, "Forget");
  const refused = h("p", { class: "text-danger small mb-1", hidden: true });
  const block = h("div", { class: "mb-2", dataset: { repo } },
    h("div", { class: "d-flex align-items-center gap-2 mb-1" }, h("code", {}, repo), forget), refused);
  forget.addEventListener("click", () => {
    forget.disabled = true;
    refused.hidden = true;
    postJSON("/api/huggingface-models/forget", { repo })
      .then(() => {
        block.remove();
        if (into.children.length) return;
        others.get("Added").section.item.remove();
        others.delete("Added");
      })
      .catch((err) => {
        forget.disabled = false;
        refused.textContent = err.message || String(err);
        refused.hidden = false;
      });
  });
  return block;
}

// The answer of a listing in every group. An organization listed with no
// models hides its search bar, Refresh and count again, as before its first
// list, while another organization has some (whose Refresh lists every
// group); when none has any, or after an error, each keeps them as they
// were, a Refresh shown to try again, its count cleared. The models of
// OTHER_GROUPS go in their sections after them, made again.
function render(data) {
  for (const group of groups) group.placeholder.remove();
  for (const { section: s } of others.values()) s.item.remove();
  others = new Map();
  if (data.error) {
    for (const group of groups) {
      group.list.replaceChildren(notice("Error: " + data.error, true));
      filter(group);
    }
    return;
  }
  const modelIds = Object.keys(data);
  const own = modelIds.filter((modelId) => !otherGroup(data[modelId]));
  for (const group of groups) {
    const ids = own.filter((modelId) => groupOf(modelId) === group);
    if (ids.length) group.controls.element.hidden = false;
    else if (own.length) group.controls.element.hidden = true;
    group.list.replaceChildren(...(ids.length ? ids.map((modelId) => row(modelId, data[modelId])) : [notice("No models found.")]));
    filter(group);
  }
  for (const modelId of modelIds) if (otherGroup(data[modelId])) place(modelId, data[modelId]);
}

function loading(on) {
  for (const group of groups) group.section.loading(on);
}

function load(url) {
  listing = true;
  loading(true);
  (url.includes("refresh") ? postJSON(url) : getJSON(url))
    .finally(() => {
      listing = false;
    })
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
      const shown = new Set([...document.querySelectorAll("#hfModelAccordion .hf-model-checkbox")].map((box) => box.value));
      for (const modelId of Object.keys(data).filter((modelId) => !shown.has(modelId))) {
        if (otherGroup(data[modelId])) {
          place(modelId, data[modelId]);
          continue;
        }
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
      if (!loaded && !listing) load("/api/huggingface-models");
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
    if (document.querySelector("#hfModelAccordion .hf-model-checkbox")) addListed();
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
