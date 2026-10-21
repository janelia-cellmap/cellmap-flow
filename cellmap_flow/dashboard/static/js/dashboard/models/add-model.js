// Add a model: resolve what was pasted (POST /api/models/resolve), ask for
// what it still needs, and run it (POST /api/models/add). The model then
// gets a ticked box among the catalog's (catalog.addedBox), so Submit keeps
// it running.
import { ApiError, postJSON } from "../../lib/api.js";
import { h } from "../../lib/dom.js";
import { announce, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { addedBox } from "./catalog.js";

// How a needed value is asked for: its label and an example.
const NEEDED = {
  voxel_size: ["Voxel (nm)", "8, or 40,4,4"],
  input_voxel_size: ["Input voxel (nm)", "8, or 40,4,4"],
  channels: ["Channels", "mito, er"],
  base_model: ["Base model", '{"type": ...}'],
};

// A typed value as the entry wants it: numbers and lists of numbers as
// such, a JSON object (a finetune's base_model) parsed, else the text.
function typed(text) {
  const value = text.trim();
  if (value.startsWith("{")) return JSON.parse(value);
  const parts = value.split(/[\s,]+/).filter(Boolean);
  if (parts.length && parts.every((p) => /^-?\d+(\.\d+)?$/.test(p))) {
    const numbers = parts.map(Number);
    return numbers.length === 1 ? numbers[0] : numbers;
  }
  return value;
}

// ``log(line)`` writes to the tab's submission log.
export function init({ log }) {
  const addRef = document.getElementById("addModelRef");
  const addResult = document.getElementById("addModelResult");
  const addFields = document.getElementById("addModelFields");
  let resolved = null;

  function field(label, key, value, title, placeholder) {
    const id = "addModel_" + key;
    addFields.append(
      h("label", { for: id }, label),
      h("input", {
        class: "form-control form-control-sm", id, dataset: { key }, value: value || "",
        title: title || null, placeholder: placeholder || null,
      }),
    );
  }

  function showResolved(d) {
    resolved = d;
    addResult.hidden = false;
    addFields.replaceChildren();
    const env = d.env ? `runs in ${d.env}` : "runs in this environment";
    document.getElementById("addModelSummary").textContent = `${d.type}: ${d.how} (${env})`;
    document.getElementById("addModelNotes").textContent = (d.notes || []).join(" ");
    field("Name", "name", d.name, "The model's name: its layer and job are called so.");
    (d.needs || []).forEach((key) => {
      const [label, example] = NEEDED[key]
        || [key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, " "), ""];
      field(label, key, "", "Not known from the reference: give it here (numbers as 8 or 40,4,4).", example);
    });
  }

  document.getElementById("addModelResolveBtn").addEventListener("click", () => {
    const ref = addRef.value.trim();
    if (!ref) return;
    postJSON("/api/models/resolve", { ref })
      .then(showResolved)
      .catch((err) => {
        addResult.hidden = true;
        alert("Could not resolve that model: " + (err instanceof ApiError ? err.message : err));
      });
  });

  document.getElementById("addModelRunBtn").addEventListener("click", () => {
    if (!resolved) return;
    const entry = { type: resolved.type, ...resolved.params, name: resolved.name };
    try {
      addFields.querySelectorAll("input").forEach((input) => {
        if (input.value.trim() !== "") entry[input.dataset.key] = typed(input.value);
      });
    } catch (e) {
      alert("Could not read a value: " + e.message);
      return;
    }
    const missing = (resolved.needs || []).filter((key) => entry[key] === undefined);
    if (missing.length) {
      alert("Still needed: " + missing.join(", "));
      return;
    }
    postJSON("/api/models/add", { entry })
      .then((d) => {
        addedBox(d.name);
        addResult.hidden = true;
        addRef.value = "";
        log(`Added ${d.name}: starting its server`);
        announce(MODELS_CHANGED_EVENT);
      })
      .catch((err) => alert("Could not add that model: " + (err instanceof ApiError ? err.message : err)));
  });
}
