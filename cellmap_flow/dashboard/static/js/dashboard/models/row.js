// What the Models tab's rows share: a tag beside a label, and unticking a
// box so that its row hears of it.
import { h } from "../../lib/dom.js";

// A small tag: a model's dims, weight format, Cellpose name or source.
export function tag(text, title) {
  return h("span", { class: "zoo-tag", title: title || null }, text);
}

// Untick ``box`` as a click would: the row is told of the change, so a zoo or
// Cellpose row hides its settings. Nothing happens to a box not ticked.
export function untickBox(box) {
  if (!box.checked) return;
  box.checked = false;
  box.dispatchEvent(new Event("change", { bubbles: true }));
}

// An error or "No models found." in place of a list.
export function notice(text, danger) {
  return h("p", { class: danger ? "text-danger" : "text-muted" }, text);
}
