// New Volume's voxel size, for a volume painted before any model (a
// "paint-first" volume, portable/volume.py plan_paint_volume): with no model
// in the picker, the template's #volumeVoxelRow is shown beside New Volume,
// its field offering the raw's levels (the models answer's raw_voxel_sizes,
// finest first) and taking any size typed. Blank asks for the level the
// viewer shows when New Volume is clicked, which the route works out. With a
// model picked the row is hidden: that volume lies on the model's grid.
//
// The picker (models.js) fills #modelSelect with an option per model and
// empties it when there are none, so the row follows the select's options;
// the levels are asked for each time the row is shown, as the dataset may
// have changed since.
import { getAnswer } from "./requests.js";

// Returns { voxelSize() }: the size typed, as New Volume sends it (crops.js).
export function initVolumeGrid() {
  const row = document.getElementById("volumeVoxelRow");
  const field = document.getElementById("volumeVoxelSize");
  const choices = document.getElementById("volumeVoxelSizes");
  const modelSelect = document.getElementById("modelSelect");

  // The raw's levels as the field's choices: "8" for an isotropic level,
  // "40, 8, 8" (z, y, x) for another.
  function offer(sizes) {
    choices.replaceChildren(...sizes.map((size) => {
      const option = document.createElement("option");
      option.value = size.every((v) => v === size[0]) ? String(size[0]) : size.join(", ");
      return option;
    }));
  }

  // Shown while the picker has no model, once the models answer agrees
  // (the select is empty too before the picker's first load): then with
  // the levels that answer lists.
  let asked = 0;
  function follow() {
    if (modelSelect.options.length) {
      row.hidden = true;
      return;
    }
    const ask = ++asked;
    getAnswer("/api/finetune/models")
      .then((data) => {
        if (ask !== asked || modelSelect.options.length) return;
        const none = !(data.models && data.models.length);
        row.hidden = !none;
        if (none) offer(data.raw_voxel_sizes || []);
      })
      .catch(() => {});
  }
  new MutationObserver(follow).observe(modelSelect, { childList: true });
  follow();

  return {
    // [z, y, x] in nm from "8" or "40, 8, 8" (commas or spaces between),
    // null when blank; an Error saying what to type otherwise. Whether the
    // raw can be planned at it is the route's to say.
    voxelSize() {
      const text = field.value.trim();
      if (!text) return null;
      const parts = text.split(/[\s,]+/).filter(Boolean).map(Number);
      if ((parts.length !== 1 && parts.length !== 3) || parts.some((v) => !Number.isFinite(v) || v <= 0)) {
        throw new Error(`Voxel size: type one positive number of nm for all three axes, or three (z, y, x), not "${text}".`);
      }
      return parts.length === 1 ? [parts[0], parts[0], parts[0]] : parts;
    },
  };
}
