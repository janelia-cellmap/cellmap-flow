// Good regions: views the user marks as ones the model already gets right.
// Training rehearses them (holds the model to what it predicts there), and
// the rehearsal setting's hint says how much that will weigh. Beside them,
// two buttons label the same patch at once (routes/finetune/view_labels.py):
// from the model's prediction, or all background.
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

  // Neuroglancer keeps the chunks it has read, so new labels show only after
  // it reloads. The iframe gets its state back from the dashboard: the same
  // view, layers and position. Absent when no viewer is connected.
  function reloadViewer() {
    const frame = document.querySelector("#my_iframe");
    if (frame) frame.src = frame.src;
  }

  // what: "seed the view", say, for the log. A box too large to label
  // without asking is answered needs_confirmation, and sent again confirmed;
  // the button stays busy until that answer too.
  function labelView(button, url, what, confirmed) {
    setBusy(button, true);
    return postAnswer(url, confirmed ? { confirm: true } : {})
      .then((d) => {
        if (d.needs_confirmation && !confirmed) {
          return confirm(d.error) ? labelView(button, url, what, true) : undefined;
        }
        if (!d.success) {
          log.add(`Could not ${what}: ${d.error}`);
          alert(`Could not ${what}:\n\n${d.error}`);
        } else if (d.reload_viewer) {
          const from = d.model ? ` from ${d.model}` : "";
          log.add(`Labelled the view${from}: ${d.filled_foreground} foreground and ` +
                  `${d.filled_background} background voxels; reloading the viewer`);
          reloadViewer();
        } else {
          log.add("Nothing to label: every voxel of the view is labelled already.");
        }
        log.showEnd();
      })
      .catch((e) => log.add(`Could not ${what}: ${e}`))
      .finally(() => setBusy(button, false));
  }

  const seedViewBtn = document.getElementById("seedViewBtn");
  seedViewBtn.addEventListener("click", () =>
    labelView(seedViewBtn, "/api/finetune/view-labels/seed", "seed the view"));
  const backgroundViewBtn = document.getElementById("backgroundViewBtn");
  backgroundViewBtn.addEventListener("click", () =>
    labelView(backgroundViewBtn, "/api/finetune/view-labels/background", "label the view background"));

  rehearsalFraction.addEventListener("change", updateRehearsalHint);

  getAnswer("/api/finetune/good-regions")
    .then((d) => renderGoodRegionCount(d.count || 0))
    .catch(() => renderGoodRegionCount(0));
}
