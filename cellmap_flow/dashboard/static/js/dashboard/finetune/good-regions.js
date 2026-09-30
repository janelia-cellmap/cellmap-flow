// Good regions: views the user marks as ones the model already gets right.
// Training rehearses them (holds the model to what it predicts there), and
// the rehearsal setting's hint says how much that will weigh.
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

    const base =
      "Share of training patches drawn from regions you marked good, " +
      "where the model is held to what it already predicts.";

    if (lastGoodRegionCount === 0) {
      hint.textContent =
        base + " No good regions marked yet, so this has no effect.";
      return;
    }
    const raw = rehearsalFraction.value.trim();
    // Blank means the manifest decides, which is 25% unless it says otherwise.
    const fraction = raw === "" ? 0.25 : Number(raw);
    if (!Number.isFinite(fraction) || fraction <= 0) {
      hint.textContent =
        base +
        ` Set to 0, so the ${lastGoodRegionCount} marked region` +
        `${lastGoodRegionCount === 1 ? "" : "s"} will be ignored this run.`;
      return;
    }
    hint.textContent =
      base +
      ` About ${Math.round(fraction * 100)}% of patches will come from the ` +
      `${lastGoodRegionCount} region${lastGoodRegionCount === 1 ? "" : "s"} ` +
      "you marked.";
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

  rehearsalFraction.addEventListener("change", updateRehearsalHint);

  getAnswer("/api/finetune/good-regions")
    .then((d) => renderGoodRegionCount(d.count || 0))
    .catch(() => renderGoodRegionCount(0));
}
