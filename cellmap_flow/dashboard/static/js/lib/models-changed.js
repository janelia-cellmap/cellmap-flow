// "The models served may have changed": one tab tells the others without
// importing them. The Finetune tab announces it when its job's model is
// served, replaced or ends (finetune/job-monitor.js); the Models tab's
// Loaded models list asks the dashboard again when it hears it
// (dashboard/models-tab.js).
const EVENT = "cellmap-flow:models-changed";

export function announceModelsChanged() {
  document.dispatchEvent(new CustomEvent(EVENT));
}

// Returns a function that stops listening.
export function onModelsChanged(listener) {
  document.addEventListener(EVENT, listener);
  return () => document.removeEventListener(EVENT, listener);
}
