// The Cellpose panel: Cellpose 4's models, one row each
// (routes/index_page.cellpose_panel_data, in the page's #cellpose-data).
// Ticked, a row asks for the voxel size, which Cellpose cannot know (it sees
// any scale, and segments well only where objects are about 30 voxels
// across), and the output. The models already running start ticked with
// their settings, one row per output running; "+ output" adds a row to run
// another output of the same model beside. Submit sends the ticked rows as
// selected_cellpose_models, each {model, voxel_size, output, do_3d,
// stitch_threshold}.
import { h, hint } from "../../lib/dom.js";
import { pageData } from "../../lib/page-data.js";
import { tag, untickBox } from "../../shared/model-list.js";

const OUTPUTS = [
  ["flows", "Flows + probability", "Three channels: flow_y and flow_x (the flows towards each cell's centre) and cell (the cell probability, 0 to 1). Not masks: for those, add the CellposeMasksPostprocessor (it makes masks of these on the server, with thresholds you can change live) or choose Masks. The layer shows the three together: each cell coloured by its flows' direction, dimmed by the probability (untick the shader's flows box for the probability alone)."],
  ["probability", "Probability only", "The cell probability, 0 to 1: one channel."],
  ["masks", "Masks", "Instance masks, made chunk by chunk: an object crossing a chunk's edge gets an id on each side."],
];
const VOXEL_TITLE =
  "Voxel size in nm, z,y,x or one number: the scale Cellpose reads the data at. Required: Cellpose-SAM "
  + "segments objects about 30 voxels across best, so pick the scale at which yours are about that.";
const STITCH_TITLE =
  "Cellpose's stitch_threshold: a mask takes the id of the mask in the slice before that it overlaps "
  + "by at least this IoU (0 to 1), within a chunk. Blank or 0: each slice's masks stay apart.";
const THREE_D_TITLE =
  "Segment in 3D (Cellpose's do_3D): the network runs on every YX, ZY and ZX plane of a chunk and the "
  + "flows are followed in 3D, so an object keeps one id through z. Flows get a flow_z channel. About 3x "
  + "slower than slice by slice; runs beside a 2D row of the same model as its own layer (_3d).";
let rowCount = 0;

function rows() {
  return [...document.querySelectorAll("#cellposeModelList .cellpose-model-item")];
}

// One row for ``model`` ({model, label, description}); ``settings``, when
// given, are a running model's ({output, do_3d, voxel_size,
// stitch_threshold}) and tick it. ``extra`` rows (from "+ output") can be
// removed.
function row(model, settings, extra) {
  rowCount += 1;
  const id = "chk_cellpose_" + rowCount;
  const input = h("input", {
    class: "form-check-input cellpose-model-checkbox", type: "checkbox", id, value: model.model,
    checked: !!settings || !!extra,
  });

  const another = h("button", {
    type: "button", class: "cellpose-row-btn",
    title: extra
      ? "Remove this row (Submit then stops what it ran)"
      : "Another row for this model, to run another output beside this one",
  }, extra ? "×" : "+ output");

  // Shown only while the row is ticked.
  const voxelSize = settings && settings.voxel_size;
  const voxel = h("input", {
    type: "text", class: "form-control form-control-sm zoo-voxel cellpose-voxel", id: id + "_voxel",
    placeholder: "required", title: VOXEL_TITLE,
    value: Array.isArray(voxelSize) ? voxelSize.join(",") : (voxelSize || ""),
  });
  const output = h("select", { class: "form-select form-select-sm cellpose-output", id: id + "_output" },
    OUTPUTS.map(([value, text, title]) => h("option", { value, title }, text)));
  output.value = (settings && settings.output) || "flows";
  const threeD = h("input", {
    type: "checkbox", class: "form-check-input cellpose-3d", id: id + "_3d", title: THREE_D_TITLE,
    checked: !!(settings && settings.do_3d),
  });
  // 2D masks only: linking each slice's masks to the slice before's.
  const stitchLabel = h("label", { for: id + "_stitch", title: STITCH_TITLE }, "Link slices (IoU)");
  const stitch = h("input", {
    type: "number", min: "0", max: "1", step: "0.05", class: "form-control form-control-sm cellpose-stitch",
    id: id + "_stitch", placeholder: "off", title: STITCH_TITLE,
    value: settings && settings.stitch_threshold ? settings.stitch_threshold : null,
  });
  const settingsRow = h("div", { class: "zoo-voxel-row d-flex align-items-center gap-2 mt-1 flex-wrap" },
    h("label", { for: id + "_voxel" }, "Voxel (nm)"), voxel,
    h("label", { for: id + "_output" }, "Output"), output,
    // In a box of its own (.cellpose-3d-box): inside the row's form-check,
    // Bootstrap floats a checkbox into the row's left margin, over the
    // Output list.
    h("div", { class: "cellpose-3d-box" }, threeD, h("label", { for: id + "_3d", title: THREE_D_TITLE }, "3D")),
    stitchLabel, stitch);
  const note = h("div", { class: "zoo-trained" },
    hint("Enter a voxel size at which your objects are about 30 voxels across.",
      "Objects about 30 voxels across work best. Runs in the cellpose4 environment."));

  const div = h("div", { class: "form-check cellpose-model-item", dataset: { model: model.model } },
    input,
    h("label", { class: "form-check-label", for: id, title: model.model }, model.label),
    tag(model.model, "Cellpose's name for it (pretrained_model)"),
    another,
    h("div", { class: "zoo-desc", title: model.description }, model.description),
    settingsRow, note);

  another.addEventListener("click", () => {
    if (extra) {
      div.remove();
      return;
    }
    // After this model's last row, with the first output none of them has.
    const mine = rows().filter((item) => item.dataset.model === model.model);
    const used = mine.map((item) => item.querySelector(".cellpose-output").value);
    const free = OUTPUTS.map(([value]) => value).find((value) => !used.includes(value));
    const added = row(model, null, true);
    added.querySelector(".cellpose-voxel").value = voxel.value;
    added.querySelector(".cellpose-3d").checked = threeD.checked;
    const select = added.querySelector(".cellpose-output");
    select.value = free || "flows";
    select.dispatchEvent(new Event("change"));
    mine[mine.length - 1].after(added);
  });

  const showTicked = () => {
    [settingsRow, note].forEach((el) => { el.style.display = input.checked ? "" : "none"; });
    // 3D masks are 3D objects already: nothing to link.
    const linkable = output.value === "masks" && !threeD.checked;
    stitchLabel.style.display = linkable ? "" : "none";
    stitch.style.display = linkable ? "" : "none";
    output.title = (OUTPUTS.find(([value]) => value === output.value) || [])[2] || "";
  };
  showTicked();
  input.addEventListener("change", showTicked);
  output.addEventListener("change", showTicked);
  threeD.addEventListener("change", showTicked);
  return div;
}

export function init() {
  const data = pageData("cellpose-data");
  const list = document.getElementById("cellposeModelList");
  const running = data.running || [];
  (data.models || []).forEach((model) => {
    const mine = running.filter((r) => r.model === model.model);
    if (!mine.length) list.append(row(model, null, false));
    // The first running output on the model's own row, any others on
    // extra rows below it.
    mine.forEach((settings, i) => list.append(row(model, settings, i > 0)));
  });
}

// POST /api/models' part: the ticked rows. A blank voxel size is sent as
// null, and Submit refuses it with a message naming the model, as it does a
// zoo model's.
export function payload() {
  const selected = [];
  rows().forEach((item) => {
    if (!item.querySelector(".cellpose-model-checkbox").checked) return;
    const output = item.querySelector(".cellpose-output").value;
    const do3d = item.querySelector(".cellpose-3d").checked;
    const entry = {
      model: item.dataset.model,
      voxel_size: item.querySelector(".cellpose-voxel").value.trim() || null,
      output,
      do_3d: do3d,
    };
    const stitch = item.querySelector(".cellpose-stitch").value.trim();
    if (output === "masks" && !do3d && stitch !== "") entry.stitch_threshold = stitch;
    selected.push(entry);
  });
  return { selected_cellpose_models: selected };
}

// Untick a loaded model's row here: the one with its model, output and 3D.
export function untick(model) {
  if (!model.cellpose) return;
  const { model: cpModel, output, do_3d: do3d } = model.cellpose;
  rows().forEach((item) => {
    if (item.dataset.model === cpModel && item.querySelector(".cellpose-output").value === output
        && item.querySelector(".cellpose-3d").checked === !!do3d) {
      untickBox(item.querySelector(".cellpose-model-checkbox"));
    }
  });
}
