// The Blockwise bar: running the pipeline as a blockwise job on LSF, in four
// steps, each unlocked by the one before -- 1 Validate (can the pipeline run
// blockwise?), 2 Generate (the task YAMLs, named after a job name the user
// gives), 3 Precheck (those YAMLs), 4 Submit (the prechecked YAMLs, under
// Generate's name). With several models, the Model Merge mode says how their
// outputs combine; it is sent with the pipeline.
//
// Submit sends the files Generate wrote and Precheck checked, so the steps
// only hold while the pipeline they ran on is unchanged: any edit -- a node,
// a parameter, an edge, the boxes, the output channels, the blockwise node,
// the merge mode -- sends the user back to Validate.
import { pageData } from "../lib/page-data.js";
import { addNode } from "./canvas.js";
import { showMessage } from "./messages.js";
import { showSection } from "./palette.js";
import { onEdit, pipeline } from "./state.js";

let blockwiseState = {
  validated: false,
  generated: false,
  prechecked: false,
  task_name: null,  // what Generate named the task; Submit names the master job after it
  model_mode: "",   // the Model Merge selection
};

// Counts edits, so a step that was waiting on the server during one can tell
// that its answer is about a pipeline that no longer exists.
let blockwiseEdits = 0;

const statusEl = () => document.getElementById("blockwise-status");

// The status line; the colour is left as it is unless given.
function setStatus(text, color) {
  statusEl().textContent = text;
  if (color !== undefined) statusEl().style.color = color;
}

const STEP_BUTTONS = { generate: "btn-generate", precheck: "btn-precheck", submit: "btn-submit" };

// Disable (true) or enable (false) step buttons; those not named are left as they are.
function setDisabled(states) {
  Object.entries(states).forEach(([step, disabled]) => {
    document.getElementById(STEP_BUTTONS[step]).disabled = disabled;
  });
}

function invalidateBlockwiseSteps() {
  blockwiseEdits += 1;
  const started = blockwiseState.validated || blockwiseState.generated || blockwiseState.prechecked;
  blockwiseState.validated = false;
  blockwiseState.generated = false;
  blockwiseState.prechecked = false;
  blockwiseState.yaml_paths = null;
  blockwiseState.task_name = null;
  setDisabled({ generate: true, precheck: true, submit: true });
  if (started) setStatus("Pipeline changed: validate again", "var(--text-secondary)");
}

// True, after telling the user, when the pipeline was edited while a step
// started at edit count `editsAtStart` was waiting on the server.
function blockwiseEditedSince(editsAtStart) {
  if (editsAtStart === blockwiseEdits) return false;
  setStatus("Pipeline changed: validate again", "var(--text-secondary)");
  showMessage("Pipeline changed during that step; validate again", "info");
  return true;
}

// Show or hide the bar. Showing it means the user is doing blockwise work,
// which always needs exactly one config node: one is placed if there is
// none, and the palette section it comes from is opened so it is visible
// there too. (The node's palette section starts collapsed.)
export async function toggleBlockwiseBar() {
  const blockwiseBar = document.getElementById("blockwise-bar");
  const isHidden = blockwiseBar.style.display === "none";
  blockwiseBar.style.display = isHidden ? "flex" : "none";
  if (!isHidden) return;
  if (!pipeline.blockwise_config.length) {
    await addNode("blockwise-config", "Blockwise Configuration", null);
  }
  showSection("config");
}

function initModelMergerDropdown() {
  const dropdown = document.getElementById("model-merge-mode");
  while (dropdown.options.length > 1) {
    dropdown.remove(1);
  }
  (pageData().model_mergers || []).forEach((merger) => {
    const option = document.createElement("option");
    option.value = merger.class_name;  // e.g. "AndModelMerger"
    option.textContent = `${merger.name} - ${merger.description}`;
    dropdown.appendChild(option);
  });
  dropdown.addEventListener("change", (e) => {
    blockwiseState.model_mode = e.target.value;
    console.log("Selected model merge mode:", blockwiseState.model_mode);
    invalidateBlockwiseSteps();
  });
}

const withMergeMode = () => ({ ...pipeline, model_mode: blockwiseState.model_mode });

function post(url, body) {
  return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

async function validateBlockwise() {
  setStatus("Validating...");
  const editsAtStart = blockwiseEdits;
  try {
    const response = await post("/api/blockwise/validate", { pipeline: withMergeMode() });
    const data = await response.json();
    if (blockwiseEditedSince(editsAtStart)) return;
    if (data.valid) {
      showMessage("✓ Pipeline valid for blockwise processing", "success");
      setStatus("✓ Step 1: Validated", "var(--accent-green)");
      blockwiseState.validated = true;
      blockwiseState.generated = false;
      blockwiseState.prechecked = false;
      setDisabled({ generate: false, precheck: true, submit: true });
    } else {
      showMessage("✗ " + data.error, "error");
      setStatus("✗ Validation failed", "var(--accent-red)");
      blockwiseState.validated = false;
      setDisabled({ generate: true, precheck: true, submit: true });
    }
  } catch (error) {
    showMessage("Error validating: " + error.message, "error");
    setStatus("✗ Error");
    blockwiseState.validated = false;
    setDisabled({ generate: true, precheck: true, submit: true });
  }
}

async function generateBlockwiseTask() {
  if (!blockwiseState.validated) {
    showMessage("⚠️ Please validate first", "info");
    return;
  }
  // The server appends a timestamp; the result names the task YAML, the
  // master LSF job, the daisy task (so the worker jobs) and their logs.
  const jobName = prompt("Enter job name (a timestamp is appended; it names the task YAML and the LSF jobs):", "cellmap_flow");
  if (jobName === null) return;
  setStatus("Generating...");
  const editsAtStart = blockwiseEdits;
  try {
    const response = await post("/api/blockwise/generate", { pipeline: withMergeMode(), job_name: jobName });
    const data = await response.json();
    if (blockwiseEditedSince(editsAtStart)) return;
    if (data.success) {
      const taskPath = (data.task_paths && data.task_paths[0]) || data.task_name || "Task";
      showMessage(`✓ Task generated: ${taskPath}`, "success");
      setStatus("✓ Step 2: Generated", "var(--accent-green)");
      blockwiseState.generated = true;
      blockwiseState.prechecked = false;
      blockwiseState.yaml_paths = data.task_paths;  // what Precheck checks and Submit runs
      blockwiseState.task_name = data.task_name;
      setDisabled({ precheck: false, submit: true });
      console.log("Task name:", data.task_name);
      console.log("Task paths:", data.task_paths);
      console.log("Task YAML:", data.task_yaml);
    } else {
      showMessage("✗ " + data.error, "error");
      setStatus("✗ Generation failed", "var(--accent-red)");
      blockwiseState.generated = false;
      setDisabled({ precheck: true, submit: true });
    }
  } catch (error) {
    showMessage("Error generating task: " + error.message, "error");
    setStatus("✗ Error");
    blockwiseState.generated = false;
    setDisabled({ precheck: true, submit: true });
  }
}

async function precheckBlockwiseTask() {
  if (!blockwiseState.generated) {
    showMessage("⚠️ Please generate task first", "info");
    return;
  }
  setStatus("Prechecking...");
  const editsAtStart = blockwiseEdits;
  try {
    const response = await post("/api/blockwise/precheck", { yaml_paths: blockwiseState.yaml_paths });
    const data = await response.json();
    if (blockwiseEditedSince(editsAtStart)) return;
    if (data.success) {
      showMessage(`✓ Precheck passed: ${data.message}`, "success");
      setStatus("✓ Step 3: Prechecked", "var(--accent-green)");
      blockwiseState.prechecked = true;
      setDisabled({ submit: false });
      console.log("Precheck result:", data.message);
    } else {
      showMessage(`✗ Precheck failed: ${data.error}`, "error");
      setStatus("✗ Precheck failed", "var(--accent-red)");
      blockwiseState.prechecked = false;
      setDisabled({ submit: true });
      console.error("Precheck error:", data.error);
    }
  } catch (error) {
    showMessage("Error prechecking task: " + error.message, "error");
    setStatus("✗ Error");
    blockwiseState.prechecked = false;
    setDisabled({ submit: true });
  }
}

async function submitBlockwiseTask() {
  if (!blockwiseState.prechecked || !(blockwiseState.yaml_paths || []).length) {
    showMessage("⚠️ Please precheck first", "info");
    return;
  }
  // The files Precheck passed, and the name Generate gave them; the server
  // runs these rather than generating new ones.
  const yamlPaths = blockwiseState.yaml_paths.slice();
  const taskName = blockwiseState.task_name;
  setStatus("Submitting...");
  try {
    const response = await post("/api/blockwise/submit", { pipeline: withMergeMode(), task_name: taskName, yaml_paths: yamlPaths });
    const data = await response.json();
    if (data.success) {
      showMessage("✓ Task submitted: " + data.job_id + " (" + data.task_name + ")", "success");
      setStatus("✓ Submitted - Job: " + data.job_id + " - " + data.task_name, "var(--accent-green)");
    } else {
      showMessage("✗ " + data.error, "error");
      setStatus("✗ Submission failed", "var(--accent-red)");
    }
  } catch (error) {
    showMessage("Error submitting task: " + error.message, "error");
    setStatus("✗ Error");
  }
}

export function initBlockwise() {
  onEdit(invalidateBlockwiseSteps);
  initModelMergerDropdown();
  document.getElementById("blockwise-toggle-btn").addEventListener("click", toggleBlockwiseBar);
  document.getElementById("btn-validate").addEventListener("click", validateBlockwise);
  document.getElementById("btn-generate").addEventListener("click", generateBlockwiseTask);
  document.getElementById("btn-precheck").addEventListener("click", precheckBlockwiseTask);
  document.getElementById("btn-submit").addEventListener("click", submitBlockwiseTask);
}
