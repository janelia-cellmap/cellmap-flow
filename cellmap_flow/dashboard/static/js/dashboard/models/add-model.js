// Add a model: resolve what was pasted (POST /api/models/resolve), ask for
// what it still needs, and run it (POST /api/models/add), in the field every
// model list's Add a model is (shared/model-list.js addModelField). The
// model then gets a ticked box among the catalog's (catalog.addedBox), so
// Submit keeps it running.
import { ApiError, postJSON } from "../../lib/api.js";
import { announce, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { addModelField, neededField } from "../../shared/model-list.js";
import { addedBox } from "./catalog.js";

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
  let resolved = null;
  const field = addModelField({
    id: "addModelRef", button: "Resolve",
    placeholder: "Hugging Face repo, zoo model, Cellpose model, path...",
    title: "Anything that names a model: a Hugging Face repo (org/repo), a BioImage Model Zoo id, nickname or URL, a Cellpose model (cpsam_v2) or weights, an exported folder, a fly_organelles checkpoint, a script (.py), or dacapo:run@iteration. Prefixes (hf:, bioimageio:, cellpose:, fly:, script:) settle a doubt.",
    onSubmit: resolve,
  });
  // Before the boxes of the models added.
  document.getElementById("addModel").prepend(field.element);

  function resolve(ref) {
    postJSON("/api/models/resolve", { ref })
      .then((d) => {
        resolved = d;
        const env = d.env ? `runs in ${d.env}` : "runs in this environment";
        field.show({
          summary: `${d.type}: ${d.how} (${env})`,
          notes: (d.notes || []).join(" "),
          fields: [
            { key: "name", label: "Name", value: d.name, title: "The model's name: its layer and job are called so." },
            ...(d.needs || []).map(neededField),
          ],
          action: "Run",
          actionTitle: "Start its server now; it joins the models below, ticked, until unticked",
          onAction: run,
        });
      })
      .catch((err) => {
        field.hide();
        alert("Could not resolve that model: " + (err instanceof ApiError ? err.message : err));
      });
  }

  // ``values``: the fields' text by key, the name and what it needs.
  function run(values) {
    if (!resolved) return;
    const entry = { type: resolved.type, ...resolved.params, name: resolved.name };
    try {
      for (const [key, text] of Object.entries(values)) {
        if (text.trim() !== "") entry[key] = typed(text);
      }
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
        field.hide();
        field.input.value = "";
        log(`Added ${d.name}: starting its server`);
        announce(MODELS_CHANGED_EVENT);
      })
      .catch((err) => alert("Could not add that model: " + (err instanceof ApiError ? err.message : err)));
  }
}
