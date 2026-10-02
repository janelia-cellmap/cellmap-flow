// The Review tab's Progress card, and the queue list (the "order" picker),
// which is part of the same answer: /api/review/progress, for the index
// that is open.
import { poll } from "../../lib/poll.js";

const PROGRESS_POLL_MS = 5000;

export function createProgress() {
  const $ = (id) => document.getElementById(id);
  let poller = null;  // the refresh every PROGRESS_POLL_MS, once an index is open

  async function refresh() {
    try {
      const r = await fetch("/api/review/progress");
      if (!r.ok) {
        $("reviewProgressSummary").textContent =
          "(no review db open — open one above)";
        return;
      }
      const p = await r.json();
      const bs = p.by_state || {};
      const total = p.total || 0;
      const blessed = bs.blessed || 0;
      const edited = bs.edited || 0;
      const erased = bs.erased || 0;
      const unreviewed = bs.unreviewed || (total - blessed - edited - erased);
      const pctB = total ? (100 * blessed / total) : 0;
      const pctE = total ? (100 * edited / total) : 0;
      const pctX = total ? (100 * erased / total) : 0;
      $("reviewProgressBarBlessed").style.width = pctB + "%";
      $("reviewProgressBarBlessed").textContent = blessed;
      $("reviewProgressBarEdited").style.width = pctE + "%";
      $("reviewProgressBarEdited").textContent = edited;
      $("reviewProgressBarErased").style.width = pctX + "%";
      $("reviewProgressBarErased").textContent = erased;
      $("reviewProgressCounts").textContent =
        `${blessed} / ${edited} / ${erased} / ${unreviewed} (of ${total})`;
      const q = p.queues || {};
      const qSummary = Object.entries(q).map(
        ([k, v]) => `${k}: ${v.reviewed}/${v.total}`
      ).join(", ");
      $("reviewProgressSummary").textContent =
        `queues — ${qSummary}`;
      // The queues are the index's own (its rank_<queue> columns), so list
      // exactly those, keeping the current choice when it is still there.
      // Rebuilt only when they change: this runs every 5 s, and replacing
      // the options closes the dropdown if it is open.
      const sel = $("reviewOrder");
      const options = Object.entries(q).map(
        ([name, v]) => [name, v.label ? `${name} (${v.label})` : name]
      );
      if (sel.dataset.queues !== JSON.stringify(options)) {
        const chosen = sel.value;
        sel.replaceChildren(...options.map(([name, text]) => new Option(text, name)));
        if (Object.prototype.hasOwnProperty.call(q, chosen)) sel.value = chosen;
        sel.dataset.queues = JSON.stringify(options);
      }
    } catch (e) {
      $("reviewProgressSummary").textContent = "progress error: " + e;
    }
  }

  // Every 5 s, the first 5 s from now (the callers have just refreshed). A
  // tick that comes while the last refresh is unanswered is skipped, so a
  // slow server gets one request at a time. While the page is hidden nobody
  // sees the card, so the refreshes wait, and one runs as soon as it is
  // shown again.
  function startPolling() {
    return poll(refresh, { intervalMs: PROGRESS_POLL_MS, immediate: false, pauseWhenHidden: true });
  }

  return {
    // Once, now: after a verdict or an undo, say.
    refresh,
    // Every 5 s from now on, for an index just opened; this replaces the
    // refresh running before, if any.
    restartPolling() {
      if (poller) poller.stop();
      poller = startPolling();
    },
    // Every 5 s, unless that is already happening.
    keepPolling() {
      if (!poller) poller = startPolling();
    },
  };
}
