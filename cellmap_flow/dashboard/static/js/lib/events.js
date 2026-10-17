// The events one part of the dashboard page sends the others, on document,
// so that neither has to import the other.
//
// MODELS_CHANGED_EVENT: the models served may have changed. The Models tab
// announces it on Submit, on Add a model's Run and on a Loaded models ✕
// (dashboard/models-tab.js); the Finetune tab when its job's model is
// served, replaced or ends (finetune/job-monitor.js). Each part that shows
// the models asks the dashboard again when it hears it: the Models tab's
// Loaded models list, the model-advice banner (dashboard/model-advice.js)
// and the Finetune tab's model picker (finetune/models.js).
export const MODELS_CHANGED_EVENT = "cellmap-flow:models-changed";

// RESTART_EVENT: Shift+R, pressed over the viewer in the browser mode.
// web/src/backend/routes/finetune.ts sends it, under its own copy of this
// name, so the string must stay as it is; finetune/job-monitor.js restarts
// the job when it hears it.
export const RESTART_EVENT = "cellmap-flow:restart-finetune";

// Tell the rest of the page that `name` happened.
export function announce(name) {
  document.dispatchEvent(new CustomEvent(name));
}

// Run listener(event) each time `name` is announced. Returns a function that
// stops listening.
export function listen(name, listener) {
  document.addEventListener(name, listener);
  return () => document.removeEventListener(name, listener);
}
