// The Local Catalog: a section for each of its groups (models.yaml's, and
// "User" for the session's running models), from the page's
// #model-catalog-data (routes/index_page.py), with the rows every model list
// has (shared/model-list.js), tagged with what each export's folder holds
// (its group's facts, by name: formats, finetune modes and size); and the
// boxes of the models added by reference (add-model.js). Each box is an
// `input.model-checkbox` in #modelSelectionForm whose value is the model's
// name, ticked to start when that model is running. A ticked catalog row
// takes the voxel size to run it at, filled in with the one it runs at (the
// page data's run_voxel_sizes, by name). Submit sends the ticked names as
// selected_models, and the voxel sizes typed as run_voxel_sizes.
import { pageData } from "../../lib/page-data.js";
import { modelRow, modelSection, runAtField, runVoxelSizes, untickBox } from "../../shared/model-list.js";

export function init() {
  const { groups, ticked, run_voxel_sizes: runningAt = {} } = pageData("model-catalog-data");
  const accordion = document.getElementById("modelAccordion");
  // A model listed twice, in its group and in "User" while it runs, is run
  // at one voxel size: what is typed under one of its rows is typed under
  // the others, or the row not looked at would send the size it had.
  const fieldsOf = new Map();
  groups.forEach(([group, names, facts = {}], i) => {
    const section = modelSection({ id: String(i + 1), title: group });
    section.body.append(...names.map((name) => {
      const id = `chk_${group}_${name}`;
      const checked = ticked.includes(name);
      const runAt = runAtField({ id: `${id}_run`, value: runningAt[name], checked });
      fieldsOf.set(name, [...(fieldsOf.get(name) || []), runAt.input]);
      runAt.input.addEventListener("input", () => fieldsOf.get(name).forEach((input) => { input.value = runAt.input.value; }));
      return modelRow({
        id, value: name, name, checked, onTick: runAt.show, below: runAt.element,
        boxClass: "model-checkbox", boxName: "selected_models", facts: facts[name],
      });
    }));
    accordion.append(section.item);
  });
}

// POST /api/models' part: the names ticked, and the voxel sizes typed under them.
export function payload() {
  const boxes = [...document.querySelectorAll("#modelSelectionForm input.model-checkbox:checked")];
  return { selected_models: boxes.map((box) => box.value), run_voxel_sizes: runVoxelSizes(boxes) };
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
