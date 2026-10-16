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
import { removeNode, renderCanvas } from "./canvas.js";
import { showMessage } from "./messages.js";
import { openOutputChannelsModal } from "./output-channels.js";
import { BLOCKWISE_FIELDS, edited, findNode, nodeDefaults } from "./state.js";

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
  }
  return [...ports(), header(title(node.name), actions(deleteButton)), ...content];
}

function blockwiseContent(node) {
  const params = node.params || {};
  return [
    header(title("⚙️ Blockwise Config"), actions(actionButton("delete", "🗑️", () => removeNode(node.id, "blockwise-config")))),
    h("div", { class: "node-params" }, BLOCKWISE_FIELDS.map((field) => {
      const inputType = field.key.startsWith("nb_") ? "number" : "text";
      return paramGroup(field.label, h("input", {
        type: inputType, class: "param-input", "data-key": field.key, "data-type": inputType,
        value: params[field.key] || "", onInput: onParamInput(node, "blockwise-config"),
      }));
    })),
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
