// Good regions: views the user marks as ones the model already gets right.
// Training rehearses them (holds the model to what it predicts there), and
// the rehearsal setting's hint says how much that will weigh. Beside them,
// three buttons act on the same patch at once (routes/finetune/view_labels.py):
// label it from the model's prediction or all background, or relabel its
// objects by connected component.
import { setBusy } from "../../lib/dom.js";
import { getAnswer, postAnswer } from "./requests.js";

// log: the Annotation Crops panel's log.
export function initGoodRegions({ log }) {
  const markGoodRegionBtn = document.getElementById("markGoodRegionBtn");
  const goodRegionCount = document.getElementById("goodRegionCount");
  const rehearsalFraction = document.getElementById("rehearsalFraction");

  // Last count seen, so the rehearsal hint can say whether the setting will
  // do anything without re-fetching every time the dropdown moves.
  let lastGoodRegionCount = 0;

  function renderGoodRegionCount(count) {
    lastGoodRegionCount = count || 0;
    goodRegionCount.textContent = count
      ? `${count} good region${count === 1 ? "" : "s"}`
      : "none yet";
    updateRehearsalHint();
  }

  function updateRehearsalHint() {
    const hint = document.getElementById("rehearsalFractionHint");

    // The label's tooltip says what the setting is; this says what it does now.
    if (lastGoodRegionCount === 0) {
      hint.textContent = "No good regions marked: no effect.";
      return;
    }
    const raw = rehearsalFraction.value.trim();
    // Blank means the manifest decides, which is 25% unless it says otherwise.
    const fraction = raw === "" ? 0.25 : Number(raw);
    if (!Number.isFinite(fraction) || fraction <= 0) {
      hint.textContent =
        `0: the ${lastGoodRegionCount} good region` +
        `${lastGoodRegionCount === 1 ? " is" : "s are"} ignored.`;
      return;
    }
    hint.textContent =
      `~${Math.round(fraction * 100)}% of patches from ` +
      `${lastGoodRegionCount} good region${lastGoodRegionCount === 1 ? "" : "s"}.`;
  }

  markGoodRegionBtn.addEventListener("click", function () {
    setBusy(markGoodRegionBtn, true);
    postAnswer("/api/finetune/good-regions/mark-view", {})
      .then((d) => {
        if (d.success) {
          renderGoodRegionCount(d.count);
          const at = d.region.offset_nm.map((v) => Math.round(v)).join(", ");
          log.add(`Marked ${d.region.label} as good at [${at}] nm (${d.count} total)`);
        } else {
          // A mark that could not be saved must be loud: it would otherwise
          // look drawn, and be gone.
          log.add(`Could not mark region: ${d.error}`);
          alert(`Could not mark region as good:\n\n${d.error}`);
        }
        log.showEnd();
      })
      .catch((e) => {
        log.add(`Could not mark region: ${e}`);
      })
      .finally(() => setBusy(markGoodRegionBtn, false));
  });

  document.getElementById("clearGoodRegionsBtn").addEventListener("click", function () {
    // Deletes every region in the session, and there is no undo; the button
    // sits right next to "Mark This View as Good".
    const what = lastGoodRegionCount
      ? `all ${lastGoodRegionCount} good region${lastGoodRegionCount === 1 ? "" : "s"}`
      : "all good regions";
    if (!confirm(`Remove ${what} from this session? This cannot be undone.`)) return;
    postAnswer("/api/finetune/good-regions/delete", {})
      .then((d) => {
        renderGoodRegionCount(d.count || 0);
        log.add("Cleared good regions");
      })
      .catch(() => {});
  });

  // Neuroglancer keeps the chunks it has read, so new labels show only once
  // the paint layer is re-read. The server re-adds that layer (answer's
  // layer_refreshed); when it could not, the whole viewer reloads instead,
  // getting its state back from the dashboard. Absent when no viewer is connected.
  function reloadViewer() {
    const frame = document.querySelector("#my_iframe");
    if (frame) frame.src = frame.src;
  }

  // what: "seed the view", say, for the log; describe(d): the log line for an
  // answer that changed labels. A box too large to label without asking is
  // answered needs_confirmation, and sent again confirmed; the button stays
  // busy until that answer too.
  function labelView(button, url, what, describe, confirmed) {
    setBusy(button, true);
    return postAnswer(url, confirmed ? { confirm: true } : {})
      .then((d) => {
        if (d.needs_confirmation && !confirmed) {
          return confirm(d.error) ? labelView(button, url, what, describe, true) : undefined;
        }
        if (!d.success) {
          log.add(`Could not ${what}: ${d.error}`);
          alert(`Could not ${what}:\n\n${d.error}`);
        } else if (d.reload_viewer) {
          log.add(describe(d) + (d.layer_refreshed ? "" : "; reloading the viewer"));
          if (!d.layer_refreshed) reloadViewer();
        } else {
          log.add(describe(d));
        }
        log.showEnd();
      })
      .catch((e) => log.add(`Could not ${what}: ${e}`))
      .finally(() => setBusy(button, false));
  }

  function describeFill(d) {
    if (!d.reload_viewer) return "Nothing to label: every voxel of the view is labelled already.";
    const from = d.model ? ` from ${d.model}` : "";
    return `Labelled the view${from}: ${d.filled_foreground} foreground and ${d.filled_background} background voxels`;
  }

  function describeSplit(d) {
    const objects = `${d.objects} object${d.objects === 1 ? "" : "s"}`;
    if (!d.reload_viewer) {
      return `Nothing to relabel: ${objects}, each already one id. A cut has to go through ` +
             "every slice the object spans: paint the wall in the slices above and below too.";
    }
    return `Relabelled the view: ${objects}, ${d.split} split off, ${d.merged} merged`;
  }

  const seedViewBtn = document.getElementById("seedViewBtn");
  seedViewBtn.addEventListener("click", () =>
    labelView(seedViewBtn, "/api/finetune/view-labels/seed", "seed the view", describeFill));
  const backgroundViewBtn = document.getElementById("backgroundViewBtn");
  backgroundViewBtn.addEventListener("click", () =>
    labelView(backgroundViewBtn, "/api/finetune/view-labels/background", "label the view background", describeFill));
  const splitObjectsBtn = document.getElementById("splitObjectsBtn");
  splitObjectsBtn.addEventListener("click", () =>
    labelView(splitObjectsBtn, "/api/finetune/view-labels/split", "split the view's objects", describeSplit));

  rehearsalFraction.addEventListener("change", updateRehearsalHint);

  getAnswer("/api/finetune/good-regions")
    .then((d) => renderGoodRegionCount(d.count || 0))
    .catch(() => renderGoodRegionCount(0));
}
