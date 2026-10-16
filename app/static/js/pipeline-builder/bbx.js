// An INPUT node's bounding boxes: drawn in a Neuroglancer viewer of its
// dataset (the server's /api/bbx-generator, shown in a dialog), or loaded
// from a JSON file, and the help dialog on that file's format.
import { h } from "../lib/dom.js";
import { renderCanvas } from "./canvas.js";
import { closeDialog, openDialog, registerDialog } from "./dialogs.js";
import { showMessage } from "./messages.js";
import { setInputParam } from "./nodes.js";
import { edited, findNode } from "./state.js";

// There is no limit on the boxes one can draw; the count shows as "n/999".
const MAX_BOXES = 999;

let bbxGeneratorState = { inputNodeId: null, numBoxes: 1 };

export function openBBXGeneratorModal(inputNodeId) {
  bbxGeneratorState.inputNodeId = inputNodeId;
  startBBXGeneration();
}

async function startBBXGeneration() {
  bbxGeneratorState.numBoxes = MAX_BOXES;

  const inputNode = findNode("input", bbxGeneratorState.inputNodeId);
  if (!inputNode) {
    showMessage("Input node not found", "error");
    return;
  }
  const datasetPath = inputNode.params?.dataset_path || "";
  if (!datasetPath) {
    showMessage("Please set the dataset path on the INPUT node first", "error");
    return;
  }
  const existingBoundingBoxes = inputNode.params?.bounding_boxes || [];

  openDialog("bbx-viewer-modal");
  // Closing the dialog replaces the state (onViewerClosed): a dialog closed
  // while the viewer was being made must not be filled, nor its poll
  // started, which then ran for the life of the page.
  const opened = bbxGeneratorState;
  const closed = () => bbxGeneratorState !== opened;
  try {
    const response = await fetch("/api/bbx-generator", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_path: datasetPath,
        num_boxes: MAX_BOXES,
        existing_bounding_boxes: existingBoundingBoxes,
      }),
    });
    const result = await response.json();
    if (closed()) return;
    if (response.ok) {
      // The viewer; the server has already put the existing boxes in its
      // annotation layer.
      const iframe = document.createElement("iframe");
      iframe.id = "bbx-iframe";
      iframe.src = result.viewer_url;
      iframe.style.width = "100%";
      iframe.style.height = "100%";
      iframe.style.border = "none";
      document.getElementById("bbx-viewer-content").replaceChildren(iframe);

      const boxCount = result.existing_count || 0;
      document.getElementById("bbx-status").textContent = boxCount > 0
        ? `🎯 Showing ${boxCount} existing box(es). Add more or delete existing ones. Click "Done - Save Boxes" when finished.`
        : `🎯 Draw bounding boxes in Neuroglancer. Click "Done - Save Boxes" when finished.`;

      // The boxes are the INPUT node's parameter, which a loaded JSON file
      // sets: text only.
      if (result.existing_bounding_boxes && result.existing_bounding_boxes.length > 0) {
        document.getElementById("bbx-existing-boxes").style.display = "block";
        document.getElementById("bbx-boxes-list").replaceChildren(...result.existing_bounding_boxes.map((bbox, idx) => h("div",
          { style: "background: var(--pb-bg-tertiary); padding: 6px 10px; border-radius: 4px; font-size: 10px; border-left: 3px solid var(--pb-accent-blue);" },
          h("div", {}, h("strong", {}, `Box ${idx + 1}`)),
          h("div", {}, `Offset: [${(bbox.offset || []).join(", ")}]`),
          h("div", {}, `Shape: [${(bbox.shape || []).join(", ")}]`),
        )));
      }
      pollBBXGeneration();
    } else {
      showMessage("Error: " + (result.error || "Failed to start BBX generator"), "error");
      closeDialog("bbx-viewer-modal");
    }
  } catch (err) {
    if (closed()) return;
    showMessage("Failed to connect to BBX generator: " + err.message, "error");
    closeDialog("bbx-viewer-modal");
  }
}

// The count of boxes drawn so far, every 2 s while the viewer is open.
function pollBBXGeneration() {
  bbxGeneratorState.pollInterval = setInterval(async () => {
    try {
      const response = await fetch("/api/bbx-generator/status", { method: "GET" });
      if (!response.ok) return;
      const result = await response.json();
      if (result.bounding_boxes && result.bounding_boxes.length > 0) {
        document.getElementById("bbx-status").textContent =
          `📦 ${result.bounding_boxes.length}/${bbxGeneratorState.numBoxes} box(es) created`;
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }, 2000);
}

// Closing the viewer (Cancel, ✕, or after saving) stops the poll.
function onViewerClosed() {
  if (bbxGeneratorState.pollInterval) {
    clearInterval(bbxGeneratorState.pollInterval);
  }
  bbxGeneratorState = { inputNodeId: null, numBoxes: 1 };
}

// "Done - Save Boxes": the boxes drawn become the INPUT node's.
async function finalizeBBXGeneration() {
  try {
    const response = await fetch("/api/bbx-generator/finalize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const result = await response.json();
    if (response.ok) {
      const bboxes = result.bounding_boxes || [];
      const inputNode = findNode("input", bbxGeneratorState.inputNodeId);
      if (inputNode) {
        inputNode.params = inputNode.params || {};
        inputNode.params.bounding_boxes = bboxes;
        edited({ apply: false });
        showMessage(`✓ Saved ${bboxes.length} bounding box(es)`, "success");
      }
      closeDialog("bbx-viewer-modal");
      renderCanvas();
    } else {
      showMessage("Error: " + (result.error || "Failed to finalize bounding boxes"), "error");
    }
  } catch (err) {
    showMessage("Failed to finalize bounding boxes: " + err.message, "error");
  }
}

// The node's Load button: a JSON array of {offset: [z, y, x], shape: [z, y, x]}.
export function loadBoxesFromFile(fileInput, inputNodeId) {
  const file = fileInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    try {
      const bboxes = JSON.parse(e.target.result);
      if (!Array.isArray(bboxes)) {
        showMessage("Error: JSON must be an array of bounding boxes", "error");
        return;
      }
      for (const bbox of bboxes) {
        if (!bbox.offset || !bbox.shape || !Array.isArray(bbox.offset) || !Array.isArray(bbox.shape)) {
          showMessage('Error: Each bbox must have "offset" and "shape" arrays', "error");
          return;
        }
        if (bbox.offset.length !== 3 || bbox.shape.length !== 3) {
          showMessage("Error: offset and shape must have 3 elements each [z, y, x]", "error");
          return;
        }
      }
      if (findNode("input", inputNodeId)) {
        setInputParam(inputNodeId, "bounding_boxes", bboxes);
        showMessage(`✓ Loaded ${bboxes.length} bounding box(es) from file`, "success");
      }
    } catch (err) {
      showMessage(`Error parsing JSON: ${err.message}`, "error");
    }
    // So the same file can be loaded again.
    fileInput.value = "";
  };
  reader.readAsText(file);
}

// The node's ? button: the file format, from the server's template.
export function showBBXJsonHelp() {
  const content = document.getElementById("bbx-json-template-content");
  fetch("/api/templates/bbox-json")
    .then((response) => response.text())
    .then((html) => {
      content.innerHTML = html;
    })
    .catch((err) => {
      console.error("Error loading bbox template:", err);
      content.innerHTML = '<p style="color: red;">Error loading template</p>';
    });
  openDialog("bbx-json-help-modal");
}

export function initBbx() {
  registerDialog("bbx-viewer-modal", { onClose: onViewerClosed, backdropCloses: false });
  registerDialog("bbx-json-help-modal");
  document.getElementById("bbx-done-btn").addEventListener("click", finalizeBBXGeneration);
}
