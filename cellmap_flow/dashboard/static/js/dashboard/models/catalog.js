// The Local Catalog: a section for each of its groups (models.yaml's, and
// "User" for the session's running models), from the page's
// #model-catalog-data (routes/index_page.py), with the rows every model list
// has (shared/model-list.js); and the boxes of the models added by reference
// (add-model.js). Each box is an `input.model-checkbox` in
// #modelSelectionForm whose value is the model's name, ticked to start when
// that model is running. Submit sends the ticked names as selected_models.
import { pageData } from "../../lib/page-data.js";
import { modelRow, modelSection, untickBox } from "../../shared/model-list.js";

export function init() {
  const { groups, ticked } = pageData("model-catalog-data");
  const accordion = document.getElementById("modelAccordion");
  groups.forEach(([group, names], i) => {
    const section = modelSection({ id: String(i + 1), title: group });
    section.body.append(...names.map((name) => modelRow({
      id: `chk_${group}_${name}`, value: name, name, checked: ticked.includes(name),
      boxClass: "model-checkbox", boxName: "selected_models",
    })));
    accordion.append(section.item);
  });
}

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
  document.getElementById("addedModels").append(
    modelRow({ id: "chk_added_" + name, value: name, name, checked: true, boxClass: "model-checkbox", rowClass: "form-check mb-1" }));
}
