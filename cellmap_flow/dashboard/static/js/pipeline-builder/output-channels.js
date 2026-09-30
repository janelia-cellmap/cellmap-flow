// An OUTPUT node's channels: which of its models' output channels it writes,
// picked in a dialog from the channels of the models that feed it.
import { h } from "../lib/dom.js";
import { renderCanvas } from "./canvas.js";
import { closeDialog, openDialog, registerDialog } from "./dialogs.js";
import { showMessage } from "./messages.js";
import { edited, findNode, modelChannels, pipeline } from "./state.js";

const emptyState = () => ({ outputNodeId: null, availableChannels: [], selectedChannels: [] });
let modalState = emptyState();

export function openOutputChannelsModal(outputNodeId) {
  const outputNode = findNode("output", outputNodeId);
  if (!outputNode) return;
  modalState.outputNodeId = outputNodeId;
  modalState.availableChannels = availableChannels(outputNodeId);

  // Nothing picked yet means every channel.
  const existingChannels = outputNode.params?.output_channels || [];
  modalState.selectedChannels = existingChannels.length > 0 ? existingChannels : [...modalState.availableChannels];

  populateOutputChannelsModal();
  openDialog("output-channels-modal");
}

// Every channel of the models upstream of the OUTPUT node, in first-seen
// order. The usual path is model -> postprocessors -> OUTPUT, so this walks
// the edges back from it rather than only looking at the nodes wired
// straight into it. An OUTPUT not wired to any model yet is offered every
// model's channels.
function availableChannels(outputNodeId) {
  const modelIds = new Set(pipeline.models.map((m) => m.id));
  const upstreamModels = new Set();
  const seen = new Set([outputNodeId]);
  const queue = [outputNodeId];
  while (queue.length > 0) {
    const nodeId = queue.shift();
    pipeline.edges.forEach((edge) => {
      if (edge.to !== nodeId || seen.has(edge.from)) return;
      seen.add(edge.from);
      queue.push(edge.from);
      if (modelIds.has(edge.from)) upstreamModels.add(edge.from);
    });
  }
  const models = upstreamModels.size > 0 ? pipeline.models.filter((m) => upstreamModels.has(m.id)) : pipeline.models;
  const channels = [];
  models.forEach((model) => {
    modelChannels(model).forEach((channel) => {
      if (!channels.includes(channel)) channels.push(channel);
    });
  });
  if (channels.length === 0) console.warn("No model channels found");
  return channels;
}

function populateOutputChannelsModal() {
  const body = document.getElementById("output-channels-body");
  const available = modalState.availableChannels;
  const selected = modalState.selectedChannels;
  if (available.length === 0) {
    body.innerHTML = '<p style="color: var(--text-secondary);">No channels detected from models</p>';
    return;
  }
  // Channel names come from model configs, which an imported YAML sets:
  // they go in as text and attribute values, never as HTML.
  body.replaceChildren(
    h("div", { style: "margin-bottom: 16px;" }, h("strong", {}, "Select output channels:")),
    ...available.map((channel) => {
      const checkboxId = `channel-${channel}-${Date.now()}`;
      return h("div", { class: "channel-checkbox-group" },
        h("input", { type: "checkbox", id: checkboxId, "data-channel": channel, checked: selected.includes(channel) }),
        h("label", { for: checkboxId }, String(channel)),
      );
    }),
  );
}

function saveOutputChannels() {
  const selectedChannels = [];
  document.querySelectorAll('#output-channels-body input[type="checkbox"]').forEach((checkbox) => {
    if (checkbox.checked) {
      selectedChannels.push(checkbox.dataset.channel);
    }
  });
  const outputNode = findNode("output", modalState.outputNodeId);
  if (outputNode) {
    outputNode.params = outputNode.params || {};
    outputNode.params.output_channels = selectedChannels;
  }
  closeDialog("output-channels-modal");
  renderCanvas();
  edited();
  showMessage(`✓ Output channels configured: ${selectedChannels.join(", ")}`, "success");
}

export function initOutputChannels() {
  registerDialog("output-channels-modal", { onClose: () => { modalState = emptyState(); } });
  document.getElementById("save-output-channels-btn").addEventListener("click", saveOutputChannels);
}
