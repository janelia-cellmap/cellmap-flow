// Events one part of the dashboard page sends the others, on document.
export const MODELS_CHANGED_EVENT = "cellmap-flow:models-changed";  // the models served may have changed
// The backend changed the chain it runs without a model change (Fit to
// Paint's window): the Processing tab's lists read it again.
export const CHAIN_CHANGED_EVENT = "cellmap-flow:chain-changed";
// Shift+R over the viewer in browser mode: web/src/backend/routes/finetune.ts sends this string too.
export const RESTART_EVENT = "cellmap-flow:restart-finetune";
// The Sign in with Hugging Face popup has answered, in browser mode (detail:
// {signed_in, error?}): web/src/backend/routes/publish.ts sends this string too.
export const HUGGINGFACE_SIGN_IN_EVENT = "cellmap-flow:huggingface-sign-in";

export function announce(name) {
  document.dispatchEvent(new CustomEvent(name));
}

export function listen(name, listener) {
  document.addEventListener(name, listener);
}
