// "➕ Create Custom": define a model from one of the ModelConfig classes the
// server lists (/api/model-config-types), filling in its parameters; the
// server registers it (/api/create-model-config) and it joins the palette.
import { closeDialog, openDialog, registerDialog } from "./dialogs.js";
import { addPaletteItem } from "./palette.js";

// {class name: {display_name, description, parameters: {name: {required,
// description, input_type, default}}}}, fetched the first time the dialog opens.
let availableModelConfigTypes = {};

async function openCreateModelConfigModal() {
  openDialog("create-model-config-modal");
  if (Object.keys(availableModelConfigTypes).length === 0) {
    try {
      const response = await fetch("/api/model-config-types");
      availableModelConfigTypes = await response.json();
      populateModelConfigTypeSelect();
    } catch (error) {
      console.error("Error loading model config types:", error);
      alert("Failed to load model types");
    }
  }
}

function populateModelConfigTypeSelect() {
  const select = document.getElementById("model-config-type-select");
  select.innerHTML = '<option value="">-- Select Model Type --</option>';
  Object.entries(availableModelConfigTypes).forEach(([className, config]) => {
    const option = document.createElement("option");
    option.value = className;
    option.textContent = `${config.display_name} - ${config.description}`;
    select.appendChild(option);
  });
}

// The form for one class: its required parameters, then its optional ones.
function updateModelConfigForm(className) {
  const container = document.getElementById("model-config-form-container");
  const formDiv = document.getElementById("model-config-form");
  if (!className) {
    container.style.display = "none";
    return;
  }
  const config = availableModelConfigTypes[className];
  formDiv.innerHTML = "";

  const required = Object.entries(config.parameters).filter(([_, info]) => info.required);
  const optional = Object.entries(config.parameters).filter(([_, info]) => !info.required);
  if (required.length > 0) {
    const requiredDiv = document.createElement("div");
    requiredDiv.innerHTML = '<h3 style="margin-top: 0; margin-bottom: 12px; color: var(--text-primary);">Required Parameters</h3>';
    required.forEach(([paramName, paramInfo]) => requiredDiv.appendChild(createParameterField(paramName, paramInfo)));
    formDiv.appendChild(requiredDiv);
  }
  if (optional.length > 0) {
    const optionalDiv = document.createElement("div");
    optionalDiv.innerHTML = '<h3 style="margin-top: 16px; margin-bottom: 12px; color: var(--text-secondary);">Optional Parameters</h3>';
    optional.forEach(([paramName, paramInfo]) => optionalDiv.appendChild(createParameterField(paramName, paramInfo)));
    formDiv.appendChild(optionalDiv);
  }
  container.style.display = "block";
}

// One parameter's field: a number box, a path, a textarea (lists, tuples) or text.
function createParameterField(paramName, paramInfo) {
  const fieldDiv = document.createElement("div");
  fieldDiv.style.marginBottom = "12px";

  const label = document.createElement("label");
  label.className = "param-label";
  label.textContent = paramInfo.description;
  if (paramInfo.required) {
    label.textContent += " *";
  }
  fieldDiv.appendChild(label);

  if (paramInfo.input_type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.id = `param-${paramName}`;
    textarea.className = "param-input";
    textarea.placeholder = `e.g., ["ch0", "ch1"] or (16, 16, 16)`;
    textarea.style.width = "100%";
    textarea.style.padding = "8px";
    textarea.style.minHeight = "60px";
    textarea.style.marginTop = "4px";
    textarea.style.fontFamily = "monospace";
    fieldDiv.appendChild(textarea);
    return fieldDiv;
  }

  const input = document.createElement("input");
  input.id = `param-${paramName}`;
  input.className = "param-input";
  input.placeholder = paramInfo.description;
  input.style.width = "100%";
  input.style.padding = "8px";
  input.style.marginTop = "4px";
  if (paramInfo.input_type === "number") {
    input.type = "number";
    input.step = "any";
  } else if (paramInfo.input_type === "file") {
    input.type = "text";
    input.placeholder = `e.g., /path/to/${paramName}`;
  } else {
    input.type = "text";
  }
  if (paramInfo.default !== undefined) {
    input.placeholder += ` (default: ${paramInfo.default})`;
  }
  fieldDiv.appendChild(input);
  return fieldDiv;
}

async function submitCreateModelConfig() {
  const className = document.getElementById("model-config-type-select").value;
  if (!className) {
    alert("Please select a model type");
    return;
  }
  // Every parameter's text; a blank one is null.
  const params = {};
  Object.keys(availableModelConfigTypes[className].parameters).forEach((paramName) => {
    const input = document.getElementById(`param-${paramName}`);
    if (input) {
      params[paramName] = input.value.trim() || null;
    }
  });

  try {
    const response = await fetch("/api/create-model-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_name: className, params }),
    });
    if (!response.ok) {
      const error = await response.json();
      alert(`Error: ${error.error}`);
      return;
    }
    const result = await response.json();
    console.log("Model config created:", result);
    addPaletteItem("models", "model", result.model_name);
    alert(`✓ Created ${result.model_name}`);
    closeDialog("create-model-config-modal");
  } catch (error) {
    console.error("Error creating model config:", error);
    alert("Failed to create model config");
  }
}

export function initModelConfigModal() {
  registerDialog("create-model-config-modal");
  document.getElementById("create-model-config-open-btn").addEventListener("click", openCreateModelConfigModal);
  const select = document.getElementById("model-config-type-select");
  select.addEventListener("change", () => updateModelConfigForm(select.value));
  document.getElementById("create-model-config-btn").addEventListener("click", submitCreateModelConfig);
}
