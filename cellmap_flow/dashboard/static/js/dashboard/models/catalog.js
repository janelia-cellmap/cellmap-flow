// The Local Catalog's boxes (the template renders them from models.yaml) and
// the boxes of the models added by reference (add-model.js): each is an
// `input.model-checkbox` in #modelSelectionForm whose value is the model's
// name. Submit sends the ticked names as selected_models.
import { h } from "../../lib/dom.js";
import { untickBox } from "./row.js";

// POST /api/models' part: the names ticked.
export function payload() {
  const boxes = document.querySelectorAll("#modelSelectionForm input.model-checkbox:checked");
  return { selected_models: [...boxes].map((box) => box.value) };
}

// Untick a loaded model's box here, if it has one, so a Submit leaves it stopped.
export function untick(model) {
  document.querySelectorAll("#modelSelectionForm input.model-checkbox").forEach((box) => {
    if (box.value === model.name) untickBox(box);
  });
}

// A ticked box for a model added by reference: Submit keeps it running until
// it is unticked.
export function addedBox(name) {
  const id = "chk_added_" + name;
  document.getElementById("addedModels").append(
    h("div", { class: "form-check mb-1" },
      h("input", { class: "form-check-input model-checkbox", type: "checkbox", value: name, id, checked: true }),
      h("label", { class: "form-check-label", for: id }, name)),
  );
}
