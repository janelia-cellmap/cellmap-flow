// A node's element on the canvas, and what its controls do: its parameters
// (edited in place), ↻ (back to the palette's defaults) and 🗑 (delete), and
// the INPUT and OUTPUT nodes' own controls (bounding boxes, the output
// channels).
//
// Node ids, names, parameter keys and values come from imported YAML and
// model configs, so none of them is written into HTML: every string goes in
// as text or as an attribute value (lib/dom.h), and each control gets its
// listener directly. The ports (.node-input, .node-output) and the header's
// drag are the canvas's to wire.
import { h } from "../lib/dom.js";
import { loadBoxesFromFile, openBBXGeneratorModal, showBBXJsonHelp } from "./bbx.js";
import { checkMaxBlockSizeCache, computeMaxBlockSize, fetchOutputDtype } from "./blockwise.js";
import { removeNode, renderCanvas } from "./canvas.js";
import { showMessage } from "./messages.js";
import { openOutputChannelsModal } from "./output-channels.js";
import { BLOCKWISE_FIELDS, GPU_QUEUE_NAMES, edited, findNode, nodeDefaults, pipeline } from "./state.js";

// `node` is a copy of a pipeline node, with its type added.
export function createNodeElement(node) {
  const isIo = node.type === "input" || node.type === "output";
  const classes = ["node-box"];
  if (isIo) classes.push("io-node", `${node.type}-node`);
  else if (node.type === "blockwise-config") classes.push("config-node");
  const div = h("div", { class: classes.join(" "), id: `node-${node.id}`, dataset: { nodeid: node.id, nodetype: node.type } });

  const pos = node.position || { x: 20, y: 20 };
  div.style.left = pos.x + "px";
  div.style.top = pos.y + "px";
  if (isIo) {
    // Wide enough for its path, as the param input's font draws it.
    const ctx = document.createElement("canvas").getContext("2d");
    ctx.font = "11px Menlo, Monaco, monospace";
    const textWidth = ctx.measureText(node.params?.dataset_path || "").width + 150;
    div.style.width = Math.min(Math.max(textWidth, 400), 1600) + "px";
  }

  div.append(...(CONTENT.get(node.type) || opContent)(node));
  return div;
}

// Each node type's children, in order. Normalizers, models and
// postprocessors (and a node of a type the builder does not know) are laid
// out by opContent.
const CONTENT = new Map([["input", ioContent], ["output", ioContent], ["blockwise-config", blockwiseContent]]);

const ports = () => [
  h("div", { class: "node-input", title: "Connect from here" }),
  h("div", { class: "node-output", title: "Connect to here" }),
];

const header = (...children) => h("div", { class: "node-header" }, children);
const title = (text) => h("div", {}, h("div", { class: "node-title" }, String(text)));
const actions = (...buttons) => h("div", { class: "node-actions" }, buttons);
const actionButton = (className, text, onClick) => h("button", { class: `icon-btn ${className}`, onClick }, text);

const paramGroup = (label, control, style) =>
  h("div", { class: "param-group", style }, h("label", { class: "param-label" }, label), control);
const onParamInput = (node, nodeType) => (event) => handleParamChange(event.currentTarget, node.id, nodeType);

function ioContent(node) {
  const params = node.params || {};
  const deleteButton = actionButton("delete", "🗑️", () => removeNode(node.id, node.type));
  const content = [paramGroup(
    node.type === "input" ? "Dataset Path" : "Output Path",
    h("input", {
      type: "text", class: "param-input", "data-key": "dataset_path", "data-type": "text",
      value: params.dataset_path || "", onInput: onParamInput(node, node.type),
    }),
    "margin-top: 8px;",
  )];

  if (node.type === "input") {
    const bboxes = params.bounding_boxes || [];
    const bboxDisplay = bboxes.length > 0 ? `${bboxes.length} bbox(es)` : "None";
    const separateId = `separate-zarrs-${node.id}`;
    const fileInput = h("input", {
      type: "file", id: `bbx-file-input-${node.id}`, accept: ".json", style: "display: none;",
      onChange: (event) => loadBoxesFromFile(event.currentTarget, node.id),
    });
    content.push(h("div", { class: "param-group" },
      h("label", { class: "param-label" }, "Bounding Boxes"),
      h("div", { style: "display: flex; gap: 8px; align-items: center; margin-bottom: 8px;" },
        h("button", { class: "configure-btn", onClick: () => openBBXGeneratorModal(node.id) }, `📦 Generate (${bboxDisplay})`),
        h("button", { class: "configure-btn", style: "background: var(--pb-accent-blue); flex: 0;", onClick: () => fileInput.click() }, "📂 Load"),
        h("button", { class: "configure-btn", style: "background: var(--accent); flex: 0; padding: 6px 10px;", onClick: showBBXJsonHelp }, "?"),
      ),
      fileInput,
      h("div", { class: "selected-channels", style: "font-size: 11px; color: var(--pb-text-secondary); margin-top: 4px;" }, bboxDisplay),
      h("div", { style: "margin-top: 10px; display: flex; align-items: center; gap: 8px;" },
        h("input", {
          type: "checkbox", id: separateId, checked: !!params.separate_bounding_boxes_zarrs, style: "cursor: pointer;",
          onChange: (event) => setInputParam(node.id, "separate_bounding_boxes_zarrs", event.currentTarget.checked),
        }),
        h("label", { for: separateId, style: "cursor: pointer; margin: 0; font-size: 13px;" }, "Save each bbox as separate zarr"),
      ),
    ));
  } else {
    const selectedChannels = params.output_channels || [];
    const selectedDisplay = Array.isArray(selectedChannels) ? selectedChannels.join(", ") : "";
    content.push(h("div", { class: "param-group" },
      h("label", { class: "param-label" }, "Output Channels"),
      h("button", { class: "configure-btn", onClick: () => openOutputChannelsModal(node.id) },
        `🔧 Configure (${selectedChannels.length || 0} selected)`),
      h("div", { class: "selected-channels", style: "font-size: 11px; color: var(--pb-text-secondary); margin-top: 4px;" },
        selectedDisplay || "Click configure to select"),
    ));
    content.push(outputDtypeControl(node));
  }
  return [...ports(), header(title(node.name), actions(deleteButton)), ...content];
}

// The dtypes an output can be written in; "auto" (blank) leaves it to the
// task, which takes the last postprocessor's that declares one, else the
// model's own -- what /api/blockwise/output-dtype answers, shown beside.
const OUTPUT_DTYPES = ["uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16", "float32", "float64"];

// The propagated dtype per pipeline composition (its models and
// postprocessors, as JSON), so re-rendering the canvas on every edit does
// not ask the dashboard again for the same answer.
const outputDtypeDefaults = new Map();

function describeOutputDtype() {
  const key = JSON.stringify({ models: pipeline.models, postprocessors: pipeline.postprocessors });
  if (!outputDtypeDefaults.has(key)) {
    outputDtypeDefaults.set(key, fetchOutputDtype().then((answer) => {
      if (!answer.success) return `could not tell: ${answer.error}`;
      if (answer.dtype) return `${answer.dtype}, from ${answer.source}`;
      return "the model's own output dtype (float32 for most models)";
    }).catch((error) => `could not tell: ${error.message}`));
  }
  return outputDtypeDefaults.get(key);
}

function outputDtypeControl(node) {
  const current = String(node.params?.dtype || "");
  const autoOption = h("option", { value: "", selected: current === "" }, "auto");
  const select = h("select", {
    class: "param-input", "data-key": "dtype", "data-type": "text",
    onChange: onParamInput(node, node.type),
  }, autoOption, ...OUTPUT_DTYPES.map((name) => h("option", { value: name, selected: name === current }, name)));
  const hint = h("div", { class: "selected-channels", style: "font-size: 11px; color: var(--pb-text-secondary); margin-top: 4px;" },
    "auto: …");
  describeOutputDtype().then((description) => {
    autoOption.textContent = `auto (${description})`;
    hint.textContent = current ? `auto would be: ${description}` : `auto: ${description}`;
  });
  return paramGroup("Output dtype", h("div", {}, select, hint));
}

// Formats a GiB figure and, when the probe measured it, the % of the GPU's
// total memory it used.
function formatPeak(peakBytes, totalMemoryBytes) {
  const gib = peakBytes / 1024 ** 3;
  const pct = totalMemoryBytes ? ` (${(100 * peakBytes / totalMemoryBytes).toFixed(0)}%)` : "";
  return `${gib.toFixed(2)} GiB${pct}`;
}

const sameShape = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => Number(v) === Number(b[i]));

// A param as a list of numbers: an array, or the JSON text a field holds
// before handleParamChange parses it; null for anything else.
function asNumbers(value) {
  if (typeof value === "string") {
    try { value = JSON.parse(value); } catch { return null; }
  }
  if (!Array.isArray(value) || !value.length || !value.every((v) => Number.isFinite(Number(v)))) return null;
  return value.map(Number);
}

// A voxel size param, one number per axis: a list, or one number for all.
function asVoxelSize(value, axes) {
  const list = asNumbers(value);
  if (list) return list;
  const one = Number(value);
  return value !== null && value !== undefined && value !== "" && Number.isFinite(one) ? Array(axes).fill(one) : null;
}

// The model nodes whose entries set their tile (input_size/output_size:
// fly, onnx), as [node, params] pairs; a script's or a CellMap export's
// tile is the model's own and cannot be set here.
function sizedModels() {
  return (pipeline.models || []).filter((model) => asNumbers(model.params?.input_size) && asNumbers(model.params?.output_size));
}

// A tile picked from the probe is the models' output size: each sized
// model node takes it, its input grown by the same amount so the context
// is kept -- in nm, as the voxel sizes in and out may differ. The task's
// Block Size (the output's chunking) is left alone: blank, it is the tile;
// set, validate checks the tile is a multiple of it. How many nodes took it.
function applyTileToModels(tile) {
  let applied = 0;
  for (const model of sizedModels()) {
    const params = model.params;
    const input = asNumbers(params.input_size);
    const output = asNumbers(params.output_size);
    if (output.length !== tile.length || input.length !== tile.length) continue;
    const inVs = asVoxelSize(params.input_voxel_size, tile.length) || asVoxelSize(params.output_voxel_size, tile.length) || tile.map(() => 1);
    const outVs = asVoxelSize(params.output_voxel_size, tile.length) || inVs;
    const newInput = tile.map((t, i) => (t * outVs[i] + (input[i] * inVs[i] - output[i] * outVs[i])) / inVs[i]);
    if (!newInput.every((v) => Math.abs(v - Math.round(v)) < 1e-6)) continue;
    params.input_size = newInput.map(Math.round);
    params.output_size = tile.slice();
    applied += 1;
  }
  return applied;
}

// One "x2 168×168×168 2.1 GiB (38%)" button per multiple the probe found
// to fit, the one the models' output_size holds marked; picking one writes
// its *spatial* shape (not the trailing channel count) into the sized model
// nodes' output_size, their input_size following (applyTileToModels).
// Nothing fitting: why not, as the probe recorded it -- for a torch build
// without this GPU's kernels that is the whole story, and nothing a smaller
// tile would change.
function maxBlockSizeOptions(node, result) {
  const fitting = result.candidates.filter((c) => c.fits);
  if (!fitting.length) {
    const first = result.candidates[0];
    const tile = first ? ` (${first.block_shape.slice(0, 3).join("×")})` : "";
    // The first line only: a result probed before the probe trimmed CUDA's
    // errors itself still carries three lines of debugging boilerplate.
    const why = (first?.error || "").split("\n")[0];
    return h("div", { class: "max-block-size-error" },
      h("div", {}, `Even the model's own block size${tile} did not fit on ${result.gpu_name}.`),
      why ? h("div", { class: "max-block-size-why" }, why) : null,
    );
  }
  const sized = sizedModels();
  const current = sized.length ? asNumbers(sized[0].params.output_size) : null;
  // Every tile tried, one row each: the ones that fit are buttons, with how
  // their answer compared with the model's own tile's; the one the search
  // stopped at stays in the list, disabled, with why -- out of memory, or
  // a tile the GPU ran but answered wrong at, which is the one to know of.
  const rows = result.candidates.map((c) => {
    const spatial = c.block_shape.slice(0, 3);
    const verdict = tileVerdict(c);
    const peak = c.peak_bytes != null ? formatPeak(c.peak_bytes, result.total_memory_bytes) : "";
    const label = `x${c.multiple}  ${spatial.join("×")}  ${peak}  ${verdict.text}`;
    if (!c.fits) {
      return h("button", { class: "configure-btn refused", disabled: true, title: c.error || "does not fit" }, label);
    }
    return h("button", {
      class: `configure-btn ${verdict.kind}${sameShape(current, spatial) ? " selected" : ""}`,
      title: `${verdict.title} Sets the model's output_size = [${spatial.join(", ")}], input_size following (same context).`,
      onClick: () => {
        if (!applyTileToModels(spatial)) {
          showMessage("No model node here has input_size/output_size to set: a script's tile is the script's own. "
            + "The probe still says what fits.", "info");
          return;
        }
        renderCanvas();
        edited();
      },
    }, label);
  });
  return h("div", { class: "max-block-size-options" },
    rows,
    h("div", { class: "max-block-size-why" },
      "Each tile ran on the same synthetic image, and its answer over the model's own tile's voxels was compared "
      + "with that tile's: \"same answer\" is safe, \"slightly off\" is a tile off the network's pooling grid, and "
      + "\"wrong answer\" is a tensor the GPU kernels cannot index (the output is a constant): never use it."),
    h("div", { class: "max-block-size-why" },
      "A pick sets the model's output size (its tile). Block Size above is the output's chunking: blank is the tile, "
      + "or a size the tile is a multiple of."),
  );
}

// How a probed tile's answer compared with the model's own tile's
// (BlockSizeCandidate.agreement, the correlation of their outputs over the
// voxels they share): {kind, text, title}. 1.00 is the same answer; a UNet
// tile off its pooling grid gives about 0.94; a tensor past the GPU kernels'
// indexing range comes back a constant, which the probe refuses.
function tileVerdict(c) {
  if (!c.fits) {
    const why = (c.error || "does not fit").split("\n")[0];
    if (/out of memory/i.test(why)) return { kind: "refused", text: "out of memory", title: why };
    if (/answers differently/i.test(why)) return { kind: "refused", text: "wrong answer (not usable)", title: why };
    return { kind: "refused", text: "did not run", title: why };
  }
  if (c.multiple === 1) {
    return { kind: "native", text: "the model's own tile", title: "The tile the model was declared with; every other row is compared with it." };
  }
  if (c.agreement == null) {
    return { kind: "same", text: "same answer", title: "Answers as the model's own tile does over the voxels they share." };
  }
  const agreement = Number(c.agreement).toFixed(2);
  if (c.agreement >= 0.99) {
    return { kind: "same", text: `same answer (${agreement})`, title: `Correlation ${agreement} with the model's own tile over the voxels they share.` };
  }
  return { kind: "offset", text: `slightly off (${agreement})`,
           title: `Correlation ${agreement} with the model's own tile: this size is off the network's pooling grid, so the answer differs a little.` };
}

function renderMaxBlockSizeResult(resultBox, node, data) {
  resultBox.replaceChildren();
  if (data.error) {
    resultBox.append(h("div", { class: "max-block-size-error" }, data.error));
  } else if (data.note) {
    resultBox.append(h("div", { class: "max-block-size-note" }, data.note));
  } else if (data.result) {
    const total = data.result.total_memory_bytes ? `, ${(data.result.total_memory_bytes / 1024 ** 3).toFixed(0)} GiB` : "";
    resultBox.append(
      h("div", { class: "max-block-size-note" },
        `${data.result.gpu_name}${total}${data.cached ? " (cached)" : ""}: pick the model's output size`),
      maxBlockSizeOptions(node, data.result),
    );
  }
}

// The probe's state per Blockwise Config node, by id, kept out of the DOM:
// renderCanvas rebuilds every node's element on each edit (picking a block
// size is one), so a probe in flight cannot hold on to the button and result
// box it started from, and its answer would otherwise be gone the moment it
// was used. {probing: true, message, cancel?} while one runs (cancel, once
// the job is submitted, bkills it); {data: {result, cached?} | {error} |
// {note}} once it has answered. A second click while one runs is ignored
// (it once restored the button to the first run's "Probing..." as its
// "original" text, for good).
const maxBlockSizeState = new Map();

function maxBlockSizeSection(node) {
  const state = maxBlockSizeState.get(node.id) || {};
  const probing = Boolean(state.probing);
  const buttons = [h("button", {
    class: "configure-btn primary", disabled: probing,
    onClick: () => runMaxBlockSize(node, false),
  }, probing ? "⏳ Probing (several minutes)…" : "📏 Compute Max GPU Block Size")];
  if (probing && state.cancel) {
    buttons.push(h("button", {
      class: "configure-btn cancel", title: "Stop waiting and kill the probe job",
      onClick: (event) => {
        event.currentTarget.disabled = true;
        state.cancel();
      },
    }, "✕ Cancel"));
  }
  if (state.data?.result && !probing) {
    buttons.push(h("button", {
      class: "configure-btn secondary", title: "Probe again on this queue, ignoring the cached answer",
      onClick: () => runMaxBlockSize(node, true),
    }, "🔄 Recompute"));
  }
  const resultBox = h("div", { class: "max-block-size-result" });
  if (probing) resultBox.append(h("div", { class: "max-block-size-note" }, state.message || "Submitting…"));
  else if (state.data) renderMaxBlockSizeResult(resultBox, node, state.data);
  return h("div", { class: "max-block-size" }, h("div", { class: "max-block-size-buttons" }, buttons), resultBox);
}

// Redraws just this node's probe section, in place, from maxBlockSizeState:
// a whole renderCanvas would take the focus from whatever field is being
// typed in.
function refreshMaxBlockSize(node) {
  const live = findNode("blockwise-config", node.id) || node;
  const section = document.getElementById(`node-${node.id}`)?.querySelector(".max-block-size");
  if (section) section.replaceWith(maxBlockSizeSection({ ...live, type: "blockwise-config" }));
}

async function runMaxBlockSize(node, forceRecompute) {
  if (maxBlockSizeState.get(node.id)?.probing) return;
  const set = (state) => {
    maxBlockSizeState.set(node.id, state);
    refreshMaxBlockSize(node);
  };
  set({ probing: true, message: forceRecompute ? "Submitting the probe job…" : "Checking the cache…" });
  try {
    if (!forceRecompute) {
      const cached = await checkMaxBlockSizeCache();
      if (cached.cached) return set({ data: { result: cached.result, cached: true } });
      if (cached.error) return set({ data: { error: cached.error } });
    }
    const probe = computeMaxBlockSize((message) => set({ probing: true, message, cancel }));
    // Offered from the moment the job is being submitted: a click before
    // bsub has answered still kills the job once its id is known.
    function cancel() {
      set({ probing: true, message: "Cancelling, killing the probe job…" });
      probe.cancel().catch((error) => set({ data: { error: error.message } }));
    }
    set({ probing: true, message: "Submitting the probe job…", cancel });
    const status = await probe.done;
    if (status.cancelled) return set({ data: { note: `Probe cancelled (job ${status.job_id} killed).` } });
    set({ data: status });
  } catch (error) {
    set({ data: { error: error.message } });
  }
}

const QUEUE_OTHER = "__other__";

// The Queue field: a select of the known single-GPU queues, with "Other…"
// revealing a text box for one not listed (a newly opened queue, or a
// deliberately different one). A value already set that isn't in the list
// (an older saved pipeline, or /api/blockwise-config's multi-queue default)
// opens on "Other…" with that value showing, rather than silently dropping
// it; a list is reduced to its first entry (see state.js's GPU_QUEUE_NAMES:
// only one queue makes sense for a single GPU job).
function queueFieldControl(node) {
  const params = node.params || {};
  const raw = params.queue;
  const current = Array.isArray(raw) ? (raw[0] || "") : (raw || "");
  const isKnown = GPU_QUEUE_NAMES.includes(current);

  const otherInput = h("input", {
    type: "text", class: "param-input", "data-key": "queue", "data-type": "text",
    placeholder: "Queue name", value: current,
    style: `margin-top: 4px; display: ${isKnown ? "none" : ""};`,
    onInput: onParamInput(node, "blockwise-config"),
  });

  const select = h("select", { class: "param-input", "data-key": "queue", "data-type": "text" },
    ...GPU_QUEUE_NAMES.map((name) => h("option", { value: name, selected: name === current }, name)),
    h("option", { value: QUEUE_OTHER, selected: current !== "" && !isKnown }, "Other…"),
  );
  select.addEventListener("change", (event) => {
    if (event.currentTarget.value === QUEUE_OTHER) {
      otherInput.style.display = "";
      otherInput.focus();
      return;  // Not yet a real queue; params.queue keeps its previous value.
    }
    otherInput.style.display = "none";
    handleParamChange(event.currentTarget, node.id, "blockwise-config");
  });

  return paramGroup("Queue", h("div", {}, select, otherInput));
}

function blockwiseContent(node) {
  const params = node.params || {};
  return [
    header(title("⚙️ Blockwise Config"), actions(actionButton("delete", "🗑️", () => removeNode(node.id, "blockwise-config")))),
    h("div", { class: "node-params" }, BLOCKWISE_FIELDS.map((field) => {
      if (field.key === "queue") return queueFieldControl(node);
      const inputType = field.key.startsWith("nb_") ? "number" : "text";
      // JSON, not a bare "168, 168, 168": handleParamChange tries JSON.parse
      // first on a re-edit, and only bracketed text round-trips through it
      // (and through _list_from_text, on the Python side) back into a list
      // rather than silently becoming one unparseable string.
      const text = Array.isArray(params[field.key]) ? JSON.stringify(params[field.key]) : (params[field.key] || "");
      return paramGroup(field.label, h("input", {
        type: inputType, class: "param-input", "data-key": field.key, "data-type": inputType,
        value: text, onInput: onParamInput(node, "blockwise-config"),
      }));
    })),
    maxBlockSizeSection(node),
  ];
}

function opContent(node) {
  const paramGroups = Object.entries(node.params || {}).map(([key, value]) =>
    paramGroup(key, paramControl(key, value, onParamInput(node, node.type))));
  return [
    ...ports(),
    header(h("div", {}, h("div", { class: "node-title" }, String(node.name)), h("div", { class: "node-type" }, node.type)), actions(
      actionButton("reset", "↻", () => resetNode(node.id, node.type)),
      actionButton("delete", "🗑️", () => removeNode(node.id, node.type)),
    )),
    h("div", { class: "node-params" }, paramGroups.length ? paramGroups : [
      h("div", { class: "param-group" }, h("div", { class: "param-label" }, "No parameters")),
    ]),
  ];
}

// The input for one parameter: a true/false select for a boolean, a number
// box for a number, and otherwise a text box showing the value, objects as
// JSON. data-type says how handleParamChange reads it back.
function paramControl(key, value, onInput) {
  if (typeof value === "boolean") {
    return h("select", { class: "param-input", "data-key": key, "data-type": "boolean", onInput },
      h("option", { value: "true", selected: value === true }, "true"),
      h("option", { value: "false", selected: value === false }, "false"),
    );
  }
  let text = "";
  if (value !== null && value !== undefined) {
    text = typeof value === "object" ? JSON.stringify(value) : String(value);
  }
  const type = typeof value === "number" ? "number" : "text";
  return h("input", { type, class: "param-input", "data-key": key, "data-type": type, value: text, onInput });
}

// A parameter edited: a number box that does not hold a number is ignored,
// and text that parses as JSON is taken as that JSON. A model's config gets
// the value too.
function handleParamChange(input, nodeId, nodeType) {
  const key = input.dataset.key;
  const type = input.dataset.type || "text";
  let value;
  if (type === "boolean") {
    value = input.value === "true";
  } else if (type === "number") {
    value = parseFloat(input.value);
    if (isNaN(value)) return;
  } else {
    try { value = JSON.parse(input.value); } catch { value = input.value; }
  }
  const node = findNode(nodeType, nodeId);
  if (node) {
    node.params[key] = value;
    if (nodeType === "model" && node.config) {
      node.config[key] = value;
    }
  }
  edited();
}

// ↻ on a normalizer, model or postprocessor node: put its parameters back to
// what a freshly added node of the same name gets.
function resetNode(nodeId, nodeType) {
  const node = findNode(nodeType, nodeId);
  if (!node) return;
  const defaults = nodeDefaults(nodeType, node.name);
  // A model with a config the palette does not list (one imported from a
  // YAML, say) has no defaults to go back to.
  if (!defaults || (node.config && !defaults.config)) {
    showMessage(`No defaults known for ${node.name}; left unchanged`, "error");
    return;
  }
  node.params = defaults.params;
  if (defaults.config) node.config = defaults.config;
  renderCanvas();
  edited();
  showMessage(`${node.name} reset to defaults`, "success");
}

// One of an INPUT node's own parameters (its boxes, the separate-zarrs box).
export function setInputParam(inputNodeId, paramName, paramValue) {
  const inputNode = findNode("input", inputNodeId);
  if (inputNode) {
    inputNode.params = inputNode.params || {};
    inputNode.params[paramName] = paramValue;
    renderCanvas();
    edited({ apply: false });
  }
}
