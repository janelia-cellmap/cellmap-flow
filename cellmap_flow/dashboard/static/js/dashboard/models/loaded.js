// Loaded models: every model running or starting, wherever on this tab (or
// in the YAML, or by a finetune) it was picked, with where it came from. GET
// /api/models/loaded lists them (services/launch.loaded_models; in the
// browser web/src/backend/routes/models.ts). Each ✕ stops that one model
// (POST /api/models/remove) and unticks its rows on the tab, so a Submit
// does not start it again; nothing else is sent, so the other models, and
// whatever is ticked or unticked below and not submitted, stay as they are.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { h } from "../../lib/dom.js";
import { announce, listen, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { tag } from "./row.js";

// The sources services/launch.SOURCES names, as the list labels them.
const SOURCE_LABELS = {
  catalog: "catalog",
  huggingface: "Hugging Face",
  bioimage: "zoo",
  cellpose: "Cellpose",
  added: "added",
  finetuned: "finetuned",
};
const SOURCE_TITLES = {
  catalog: "From the Local Catalog",
  huggingface: "From the Hugging Face list",
  bioimage: "From the BioImage Model Zoo list",
  cellpose: "From the Cellpose list",
  added: "Added by reference (Add a model), or by the YAML the dashboard was started with",
  finetuned: "Made by a finetune on the Finetune tab",
};

// ``untick(model)`` unticks the model's rows wherever they are on the tab;
// ``log(line)`` writes to the tab's submission log. Returns ``refresh()``,
// which asks for the list again, and ``starting()``, whether a listed model
// is still starting (minutes in a queue, or loading), so the list is worth
// asking again.
export function init({ untick, log }) {
  const list = document.getElementById("loadedModelsList");
  // Bumped by each refresh and each remove: an answer to an older refresh
  // (the poll's, in flight while a ✕ was answered) is not drawn over a
  // newer list.
  let seq = 0;
  let starting = false;

  function row(model) {
    // A name has no spaces: a line may break after each _, - or ., so a long
    // one wraps between its words.
    const name = h("span", { class: "loaded-model-name", title: model.name },
      model.name.split(/(?<=[_.-])/).flatMap((part, i) => (i ? [h("wbr"), part] : [part])));
    // The browser mode's "waiting": no volume is open to run it on yet,
    // said as what to do.
    const status = model.status && model.status !== "running"
      ? (model.status === "waiting" ? tag("needs a volume", "It runs on the volume you open in the viewer.") : tag(model.status))
      : null;
    if (status) status.classList.add("loaded-model-status");
    const remove = h("button", {
      type: "button", class: "loaded-model-remove", "aria-label": "Remove " + model.name, disabled: !model.removable,
    }, "✕");
    remove.addEventListener("click", () => removeLoaded(model, remove));
    return h("li", { class: "loaded-model-item", dataset: { name: model.name } },
      h("div", { class: "loaded-model-text" },
        h("div", { class: "loaded-model-head" },
          name, tag(SOURCE_LABELS[model.source] || model.source, SOURCE_TITLES[model.source]), status),
        model.detail ? h("div", { class: "loaded-model-detail", title: model.detail }, model.detail) : null,
        model.note ? h("div", { class: "loaded-model-note" }, model.note) : null),
      // The title is on the span around the button: a disabled button gets
      // no mouse events, so its own would never show.
      h("span", {
        class: "loaded-model-remove-wrap",
        title: model.removable
          ? "Stop " + model.name + " and remove its layer; the other models stay as they are. "
            + "Tick it again and Submit to bring it back."
          : (model.note || "This model cannot be removed here."),
      }, remove));
  }

  function render(models) {
    list.replaceChildren(...models.map(row));
    document.getElementById("loadedModelsEmpty").hidden = models.length > 0;
    const startingCount = models.filter((m) => m.status === "starting").length;
    const waitingCount = models.filter((m) => m.status === "waiting").length;
    const counts = [];
    if (models.length - startingCount - waitingCount) counts.push((models.length - startingCount - waitingCount) + " running");
    if (startingCount) counts.push(startingCount + " starting");
    if (waitingCount) counts.push(waitingCount + " waiting for a volume");
    document.getElementById("loadedModelsCount").textContent = counts.length ? "(" + counts.join(", ") + ")" : "";
    // The Cellpose panel's "N running", after a remove or a Submit as on load.
    const cellposeRunning = models.filter((m) => m.source === "cellpose").length;
    document.getElementById("cellposeRunningCount").textContent = cellposeRunning ? cellposeRunning + " running" : "";
    // Asked again while one waits too: opening a volume serves it.
    starting = startingCount + waitingCount > 0;
  }

  // Quietly: a failed refresh leaves the list as it was.
  function refresh() {
    const mine = ++seq;
    return getJSON("/api/models/loaded")
      .then((data) => { if (mine === seq) render(data.models || []); })
      .catch((err) => console.warn("Could not list the loaded models:", err));
  }

  function removeLoaded(model, button) {
    button.disabled = true;
    postJSON("/api/models/remove", { name: model.name })
      .then((data) => {
        untick(model);
        // A finetune's model whose job has ended had no server left to stop.
        log(`Removed ${model.name}: ${model.status === "ended" ? "its" : "stopped, and its"} layer taken away`);
        seq += 1;
        render(data.models || []);
        announce(MODELS_CHANGED_EVENT);
      })
      .catch((err) => {
        const message = err instanceof ApiError ? err.message : String(err);
        // Gone already (stopped elsewhere, or the list was old): show the list as it is.
        if (err instanceof ApiError && err.status === 404) {
          refresh();
          return;
        }
        button.disabled = !model.removable;
        log(`Not removed: ${message}`);
        alert("Could not remove " + model.name + ": " + message);
      });
  }

  // When the models served may have changed (this tab's Submit, Run and ✕
  // say so too, rather than ask themselves); and each time the tab is opened.
  listen(MODELS_CHANGED_EVENT, () => refresh());
  const modelTabButton = document.getElementById("model-tab");
  if (modelTabButton) modelTabButton.addEventListener("shown.bs.tab", refresh);
  refresh();
  return { refresh, starting: () => starting };
}
