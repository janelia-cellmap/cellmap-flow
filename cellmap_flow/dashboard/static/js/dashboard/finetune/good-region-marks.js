// Good regions: views the user marks as ones the model already gets right
// (Mark as Good), their count, and clear. Training holds the model to what it
// predicts there, and on the server the rehearsal setting's hint says how
// much that will weigh. The routes are the server's good_regions.py, or in
// the browser mode the page's own (web/src/backend/routes/good-regions.ts,
// where the head's distillation holds the regions), with the same answers,
// so this runs in both modes. The Label it buttons over the same patch are
// good-regions.js's, which the browser mode does not start.
import { setBusy } from "../../lib/dom.js";
import { getAnswer, offerNewVolume, postAnswer } from "./requests.js";

// log: the Annotation Crops panel's log.
export function initGoodRegionMarks({ log }) {
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

  // The hint sits under the server-only Good-Region Rehearsal field, so the
  // browser mode never shows it.
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
          if (d.needs_volume) offerNewVolume(`Could not mark region as good: ${d.error}`);
          else alert(`Could not mark region as good:\n\n${d.error}`);
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

  rehearsalFraction.addEventListener("change", updateRehearsalHint);

  getAnswer("/api/finetune/good-regions")
    .then((d) => renderGoodRegionCount(d.count || 0))
    .catch(() => renderGoodRegionCount(0));
}
