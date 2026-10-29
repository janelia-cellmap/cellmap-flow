// The Models tab's lists of models to tick, as the pieces every such list is
// made of: the tab's own (dashboard/models/: the Local Catalog's groups,
// Hugging Face, the BioImage Model Zoo, Add a model, and the tags of the
// Cellpose and Loaded models lists) and the Finetune tab's Which model to
// start from (dashboard/finetune/rank.js), which lists the same models to
// score rather than to run. So the two look alike and find a model alike: a
// collapsible section per source (modelSection), a row per model (modelRow,
// and zooRow for a zoo model's) with the tags of what the model is
// (factTags), the Hugging Face and zoo sections' search bar, Refresh,
// filters and order (listControls), and the Add a model field
// (addModelField). Under a ticked row of the Models tab, and not of the
// ranking's, which scores each model as it is, the voxel size to run it at
// (voxelField: the zoo's Voxel (nm), and the catalog's and Hugging Face's
// Run at (nm), runAtField).
//
// Each piece is made from what it is given and handed back, for its place
// to put where it goes: none looks up an element of the page, or reads a
// form. A row's tick is its place's own: the row starts as it is told
// (checked) and says when it changes (onTick), and what a tick means, run
// on Submit or score in the ranking, is the place's. So are the ids (a
// row's, and the prefix of a section's and of a control's), so the same
// model listed on both tabs is two boxes, and each label ticks its own.
import { h, hint, setHint } from "../lib/dom.js";
import { matchesWords, searchText } from "../lib/search.js";

// A small tag: a model's dims, weight format, Cellpose name or source.
export function tag(text, title) {
  return h("span", { class: "zoo-tag", title: title || null }, text);
}

// What the training form calls each way to finetune (finetune_modes).
const MODE_WORDS = { lora: "LoRA", full: "full", head: "head" };

// A size in MB (1e6 bytes) as a tag says it: "3.2 GB", "56 MB", "7.6 MB".
function mbText(mb) {
  if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`;
  return mb >= 10 ? `${Math.round(mb)} MB` : `${mb.toFixed(1)} MB`;
}

// What a model is, as its row's tags, from the fields every listing gives a
// model (a server's models/hf_catalog.py FACTS, the browser's
// web/src/models/metadata.ts, under the same names), so it is known without
// downloading it: its weight formats ("TorchScript", "torch.export",
// "ONNX"), the ways it can be finetuned ("LoRA + full", "head + full"), its
// size (its largest weights file, or its parameters when that is not known)
// and its copy that runs in the browser, by the precisions that copy comes in
// ("browser: fp32/fp16/int8w"). Nothing for a field not given.
export function factTags(facts) {
  const modes = facts.finetune_modes || [];
  const variants = facts.variants || [];
  const params = facts.params > 0 ? `${(facts.params / 1e6).toFixed(1)}M parameters` : null;
  const size = Number.isFinite(facts.size_mb)
    ? tag(mbText(facts.size_mb), `Its largest weights file${params ? `; ${params}` : ""}`)
    : params && tag(params, "Its parameters");
  return [
    (facts.formats || []).map((format) => tag(format, "Weight format")),
    modes.length ? tag(modes.map((mode) => MODE_WORDS[mode] || mode).join(" + "), "How it can be finetuned (the training form's Finetune mode)") : null,
    size,
    facts.browser_copy
      ? tag(variants.length ? `browser: ${variants.map((v) => v.variant).join("/")}` : "browser",
        `Runs in the browser, as ${facts.browser_copy}${variants.length ? `: ${variants.map((v) => `${v.variant} ${mbText(v.file_mb)}`).join(", ")}` : ""}`)
      : null,
  ];
}

// Untick ``box`` as a click would: the row is told of the change, so a zoo or
// Cellpose row hides its settings. Nothing happens to a box not ticked.
export function untickBox(box) {
  if (!box.checked) return;
  box.checked = false;
  box.dispatchEvent(new Event("change", { bubbles: true }));
}

// An error or "No models found." in place of a list.
export function notice(text, danger) {
  return h("p", { class: danger ? "text-danger" : "text-muted" }, text);
}

// One collapsible section of models, an accordion item as the Models tab's
// are: its header `title`, with a spinner while it loads (loading) and a
// quiet count beside it (count), over its body, which the place fills with
// the section's controls and rows. `id` names its parts (heading_<id>, and
// collapse_<id>, which the header's toggle targets) and `bodyId` its body,
// for a place that is looked up by it. It starts closed unless `open`.
// onOpen runs each time it starts to open (Bootstrap's show.bs.collapse):
// where a list not loaded yet loads. isOpen says whether it is open now.
export function modelSection({ id, title, bodyId, open = false, onOpen }) {
  const spinner = h("span", { class: "spinner-border spinner-border-sm ms-2 d-none", role: "status" });
  const count = h("span", { class: "zoo-count ms-2" });
  const body = h("div", { class: "accordion-body", id: bodyId || null });
  const collapse = h("div", {
    id: `collapse_${id}`, class: `accordion-collapse collapse${open ? " show" : ""}`, "aria-labelledby": `heading_${id}`,
  }, body);
  if (onOpen) collapse.addEventListener("show.bs.collapse", onOpen);
  const item = h("div", { class: "accordion-item" },
    h("h2", { class: "accordion-header", id: `heading_${id}` },
      h("button", {
        class: `accordion-button${open ? "" : " collapsed"}`, type: "button", "data-bs-toggle": "collapse",
        "data-bs-target": `#collapse_${id}`, "aria-expanded": String(open), "aria-controls": `collapse_${id}`,
      }, title, spinner, count)),
    collapse);
  return {
    item,
    body,
    loading: (on) => spinner.classList.toggle("d-none", !on),
    count: (text) => { count.textContent = text; },
    isOpen: () => collapse.classList.contains("show"),
  };
}

// One model's row, as the Models tab's are: its box, its name, a quiet note
// beside the name of what the model is (a Hugging Face model's description,
// the ranking's task and voxel size), its tags after (tag), a link, a
// one-line description under them (the zoo's), and whatever the place adds
// under that (`below`: a zoo row's voxel size, say). `value` is the box's
// value, and `boxClass` and `boxName` what a place's Submit finds the ticked
// boxes by; `rowClass` is the row's, for a list's own spacing; `title` the
// name's tooltip. `facts` are what the model's listing says it is
// (factTags), tagged after the place's own tags. `search`, `em` and `dims`
// are the row's data (data-search, data-em, data-dims), which a section's
// controls match (listControls), with its name, size and whether it has a
// browser copy ("yes", "no", or none when that is not known), from `facts`,
// which they order and filter by.
export function modelRow({
  id, value, name, note, tags, facts, link, description, below, title, checked = false, disabled = false, onTick,
  rowClass = "form-check mb-2", boxClass, boxName, search, em, dims,
}) {
  const browser = facts && "browser_copy" in facts ? (facts.browser_copy ? "yes" : "no") : undefined;
  const size = facts && Number.isFinite(facts.size_mb) ? String(facts.size_mb) : undefined;
  const data = Object.fromEntries(Object.entries({ search, em, dims, browser, size, name: typeof name === "string" ? name : undefined })
    .filter(([, v]) => v !== undefined));
  return h("div", { class: rowClass, dataset: data },
    h("input", {
      class: boxClass ? `form-check-input ${boxClass}` : "form-check-input", type: "checkbox", name: boxName, id, value,
      checked, disabled, onChange: onTick && ((e) => onTick(e.target.checked)),
    }),
    h("label", { class: "form-check-label", for: id, title },
      name, note ? [" ", h("small", { style: { color: "#aaa" } }, "- " + note)] : null),
    tags, facts && factTags(facts), link, description ? h("div", { class: "zoo-desc", title: description }, description) : null, below);
}

// The voxel size a zoo model was trained at, from cellmap-flow's own table
// (models/bioimage_catalog.py trained_at), as "40×4×4 nm"; "" when not known.
// No zoo model says it in its description.
export function trainedNm(model) {
  const trained = model.trained;
  return trained && trained.voxel_size ? trained.voxel_size.join("×") + " nm" : "";
}

// A BioImage Model Zoo model's row (`model` as the zoo's index lists it,
// models/bioimage_catalog.py normalise, with what it was trained at): its
// name, its dims, the voxel size it was trained at and its weight formats
// as tags, its page on bioimage.io, and its description under them; matched
// by its name, key, id, description, tags and dims, and by its EM tag.
// `options` are modelRow's, for what is the place's: id, checked, onTick,
// boxClass and below, or another name.
export function zooRow(model, options) {
  const trained = trainedNm(model);
  return modelRow({
    value: model.key, name: model.name, rowClass: "form-check zoo-model-item",
    title: [model.description, model.key, model.license].filter(Boolean).join("\n"),
    tags: [
      model.dims ? tag(model.dims.toUpperCase()) : null,
      trained ? tag(trained, "Trained at (z×y×x) on " + model.trained.trained_on) : null,
      (model.weight_formats || []).map((format) => tag(format, "Weight format")),
    ],
    link: /^https?:\/\//.test(model.url || "")
      ? h("a", { class: "zoo-link", href: model.url, target: "_blank", rel: "noopener noreferrer", title: "Open on bioimage.io" }, "↗")
      : null,
    description: model.description,
    search: searchText(model.name, model.key, model.id, model.description, ...(model.tags || []), model.dims),
    em: model.em ? "1" : "",
    dims: model.dims || "",
    ...options,
  });
}

// A voxel size to give a ticked model, under its row (modelRow's `below`): a
// field labelled `label`, its id `id`, starting at `value` (z, y, x as an
// array, or as typed; none: blank), with `placeholder` and `title`, and
// `note` under it (an element: hint()'s line, say). It is shown only while
// the row is ticked: as `checked` says to start, then as show(on), the
// row's onTick, says. `inputClass` is the field's, for what a place's
// Submit finds it by.
export function voxelField({ id, label, value, placeholder, title, note, checked, inputClass }) {
  const input = h("input", {
    type: "text", class: `form-control form-control-sm zoo-voxel${inputClass ? " " + inputClass : ""}`, id,
    placeholder, title, value: Array.isArray(value) ? value.join(",") : (value || ""),
  });
  const element = h("div", {},
    h("div", { class: "zoo-voxel-row d-flex align-items-center gap-2 mt-1" }, h("label", { for: id }, label), input),
    note);
  const show = (on) => { element.style.display = on ? "" : "none"; };
  show(checked);
  return { element, input, show };
}

const RUN_AT_TITLE = "The voxel size to run it at, in nm: z,y,x or one number for all three.";
const RUN_AT_MORE =
  "It then reads the raw at that size (the level of that size, when the data has one); its output, and "
  + "everything trained or painted for it, follow. A model that reads its input itself refuses it.";

// The voxel size to run a ticked model of the Models tab at (the Local
// Catalog's, or Hugging Face's), a voxelField: blank, its own, which `own`
// ([z, y, x]) shows when its listing says it; `value` is the one it runs at
// already, as the page or the listing says. Submit sends what is typed
// (runVoxelSizes) for the server or the browser to parse.
export function runAtField({ id, value, own, checked }) {
  return voxelField({
    id, label: "Run at (nm)", value, checked, inputClass: "run-voxel", title: RUN_AT_TITLE,
    placeholder: Array.isArray(own) ? own.join(",") : "its own",
    note: h("div", { class: "zoo-trained" }, hint("Blank: its own voxel size.", RUN_AT_MORE)),
  });
}

// POST /api/models' run_voxel_sizes from ticked `boxes` (modelRow's), by each
// box's value: what is typed in its row's runAtField, trimmed. A blank one,
// or a row with none, gives nothing: that model runs at its own.
export function runVoxelSizes(boxes) {
  return Object.fromEntries(boxes.flatMap((box) => {
    const field = box.parentNode.querySelector("input.run-voxel");
    const typed = field ? field.value.trim() : "";
    return typed ? [[box.value, typed]] : [];
  }));
}

// A section's ways to find a model, as the Models tab's Hugging Face and zoo
// sections have them: a search bar (every word typed, lib/search.js) with a
// Refresh beside it; for the zoo (`zoo`) EM only, ticked to start, 2D and
// 3D; for a list of Hugging Face models, Runs in the browser (`browser`, a
// server's: in the browser every model does), ticked to start, and the
// order of its rows (`sort`: by name, or by size, smallest first); and with
// any of those the count of the rows shown (count). `id` prefixes their ids
// (<id>SearchBar, <id>RefreshBtn, <id>EmOnly, <id>2d, <id>3d, <id>Browser,
// <id>Sort, <id>Count). onChange runs on every change to them, onRefresh on
// Refresh.
// matches(data) is whether a row's data (modelRow's search, em, dims and
// browser) passes them: every word typed is in its search text, it is tagged
// EM when EM only is ticked, its dims are one of those ticked (2D and 3D
// neither: any), and it has a browser copy when Runs in the browser is
// ticked (one not known to have none passes: its listing did not say).
// noCopy(data) is whether that box alone hides it, for the count. sorted(rows)
// gives the rows in the order chosen, by their data's name and size (one of
// no size last); as they came without `sort`. What a row that does not match
// does, and whether a ticked one stays shown regardless, is the place's.
export function listControls({ id, placeholder, title, refreshTitle, onRefresh, onChange, zoo = false, browser = false, sort = false }) {
  const search = h("input", { type: "text", class: "form-control", id: `${id}SearchBar`, placeholder, title, onInput: onChange });
  const refresh = h("button", {
    type: "button", class: "btn btn-sm btn-outline-secondary", id: `${id}RefreshBtn`, style: { whiteSpace: "nowrap" },
    title: refreshTitle, onClick: onRefresh,
  }, "Refresh");
  const box = (suffix, label, boxTitle, checked) => {
    const input = h("input", { class: "form-check-input", type: "checkbox", id: `${id}${suffix}`, checked, onChange });
    return [input, h("div", { class: "form-check mb-0", title: boxTitle }, input, h("label", { class: "form-check-label", for: input.id }, label))];
  };
  const [emOnly, emBox] = zoo ? box("EmOnly", "EM only", "Only models whose tags, name or description say electron microscopy", true) : [];
  const [only2d, box2d] = zoo ? box("2d", "2D", "Models tagged 2D", false) : [];
  const [only3d, box3d] = zoo ? box("3d", "3D", "Models tagged 3D", false) : [];
  const [browserOnly, browserBox] = browser
    ? box("Browser", "Runs in the browser", "Only models with a copy that runs in the browser (an ONNX export, cellmap/onnx-web-models' or a folder of the repo's own). Ticked models are always shown.", true)
    : [];
  const order = sort && h("select", {
    class: "form-select form-select-sm w-auto", id: `${id}Sort`, title: "The order of the models: by name, or by size (the largest weights file), smallest first", onChange,
  }, h("option", { value: "name" }, "Sort by name"), h("option", { value: "size" }, "Sort by size"));
  const shown = h("span", { class: "zoo-count ms-auto", id: `${id}Count`, title: "Shown / listed. Ticked models are always shown." });
  const element = h("div", {},
    h("div", { class: "d-flex gap-2 mb-2" }, search, refresh),
    (zoo || browser || sort) && h("div", { class: "d-flex align-items-center gap-3 mb-2 flex-wrap" }, emBox, box2d, box3d, browserBox, order, shown));
  const noCopy = (data) => Boolean(browserOnly && browserOnly.checked && data.browser === "no");
  const byName = (a, b) => (a.dataset.name || "").localeCompare(b.dataset.name || "");
  const size = (row) => (row.dataset.size ? Number(row.dataset.size) : Infinity);
  return {
    element,
    refresh,
    matches(data) {
      if (!matchesWords(data.search || "", search.value)) return false;
      if (noCopy(data)) return false;
      if (!zoo) return true;
      const dims = [[only2d, "2d"], [only3d, "3d"]].filter(([input]) => input.checked).map(([, d]) => d);
      return (!emOnly.checked || Boolean(data.em)) && (!dims.length || dims.includes(data.dims));
    },
    noCopy,
    sorted(rows) {
      if (!order) return [...rows];
      return [...rows].sort(order.value === "size" ? (a, b) => size(a) - size(b) || byName(a, b) : byName);
    },
    count: (n, listed, hiddenNoCopy = 0) => {
      shown.textContent = listed ? `${n} / ${listed}${hiddenNoCopy ? `, ${hiddenNoCopy} hidden: no browser copy` : ""}` : "";
    },
  };
}

// How a value a model still needs is asked for (models/resolve.py's
// ``needs``): its label and an example, as a field of addModelField's show.
const NEEDED = {
  voxel_size: ["Voxel (nm)", "8, or 40,4,4"],
  input_voxel_size: ["Input voxel (nm)", "8, or 40,4,4"],
  channels: ["Channels", "mito, er"],
  base_model: ["Base model", '{"type": ...}'],
};

export function neededField(key) {
  const [label, example] = NEEDED[key] || [key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, " "), ""];
  return { key, label, placeholder: example, title: "Not known from the reference: give it here (numbers as 8 or 40,4,4)." };
}

// Add a model, as the Models tab has it: a field for anything that names a
// model and a button beside it (`button`, what it does: Resolve, or Add),
// which Enter in the field presses too, and which hands what was typed to
// onSubmit. Then, when the place shows them (show), what the reference was
// found to be (summary), its notes under an "i" beside that, a field for each
// value given or still needed ({key, label, value, title, placeholder};
// neededField says how a needed one is asked for), and the button that adds
// the model (`action`), which hands onAction the fields' values by key. `id`
// is the field's, and prefixes the value fields' (<id>_<key>).
export function addModelField({ id, button, placeholder, title, onSubmit }) {
  const input = h("input", {
    type: "text", class: "form-control form-control-sm", id, placeholder, title, autocomplete: "off", spellcheck: "false",
  });
  const go = h("button", { type: "button", class: "btn btn-sm btn-outline-secondary" }, button);
  const summary = h("div", { class: "small" });
  const fields = h("div", { class: "add-model-fields mt-2" });
  const action = h("button", { type: "button", class: "btn btn-sm btn-outline-secondary mt-2" });
  const result = h("div", { class: "add-model-result mt-2", hidden: true }, summary, fields, action);
  let inputs = [];
  let onAction = null;

  const submit = () => {
    const ref = input.value.trim();
    if (ref) onSubmit(ref);
  };
  go.addEventListener("click", submit);
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    submit();
  });
  const values = () => Object.fromEntries(inputs.map((field) => [field.dataset.key, field.value]));
  action.addEventListener("click", () => onAction && onAction(values()));

  return {
    element: h("div", {},
      h("label", { class: "form-label fw-bold mb-1", for: id }, "Add a model"),
      h("div", { class: "d-flex gap-2" }, input, go),
      result),
    input,
    button: go,
    show(found) {
      result.hidden = false;
      setHint(summary, found.summary || "", found.notes);
      const wanted = found.fields || [];
      inputs = wanted.map((field) => h("input", {
        class: "form-control form-control-sm", id: `${id}_${field.key}`, dataset: { key: field.key }, value: field.value || "",
        title: field.title || null, placeholder: field.placeholder || null,
      }));
      fields.replaceChildren(...wanted.flatMap((field, i) => [h("label", { for: inputs[i].id }, field.label), inputs[i]]));
      action.textContent = found.action || "";
      action.title = found.actionTitle || "";
      action.hidden = !found.onAction;
      onAction = found.onAction || null;
    },
    hide() {
      result.hidden = true;
    },
  };
}
