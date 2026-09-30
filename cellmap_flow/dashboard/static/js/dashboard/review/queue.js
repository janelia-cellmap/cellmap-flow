// The Review tab's Queue card: Next and Go to ID, which bring up an
// instance; the verdicts on it (Bless, Edit with its dialog, Erase); and
// Undo. A verdict or an undo refreshes the progress, and with Auto-advance
// ticked a verdict moves on to the next instance.
//
// current: the Current instance block (current.js); progress: the Progress
// card (progress.js).
export function initQueue({ current, progress }) {
  const $ = (id) => document.getElementById(id);

  async function gotoInstanceId() {
    const idStr = $("reviewGotoId").value.trim();
    if (!idStr) return;
    const id = parseInt(idStr, 10);
    if (!Number.isFinite(id) || id < 1) {
      alert("Go to ID: please enter a positive integer");
      return;
    }
    try {
      const r = await fetch(`/api/review/show/${id}`);
      const j = await r.json();
      if (!r.ok) { alert("go-to failed: " + (j.error || r.status)); return; }
      current.show(j, "show");
    } catch (e) { alert("go-to error: " + e); }
  }
  $("reviewGotoBtn").addEventListener("click", gotoInstanceId);
  $("reviewGotoId").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); gotoInstanceId(); }
  });

  $("reviewNextBtn").addEventListener("click", async () => {
    const order = $("reviewOrder").value;
    const minVox = $("reviewMinVox").value;
    // skip_rank = rank of currently-shown instance; ensures Next advances
    // past what's already on screen even when not verdicted.
    const shown = current.instance();
    const skipRank = (shown && shown.rank !== undefined) ? shown.rank : null;
    const url = `/api/review/next?order=${encodeURIComponent(order)}`
              + (minVox ? `&min_vox=${encodeURIComponent(minVox)}` : "")
              + (skipRank != null ? `&skip_rank=${encodeURIComponent(skipRank)}` : "");
    try {
      const r = await fetch(url);
      const j = await r.json();
      if (!r.ok) { alert("next failed: " + (j.error || r.status)); return; }
      if (j.done) { current.none(); return; }
      current.show(j, "next");
    } catch (e) { alert("next error: " + e); }
  });

  async function postVerdict(verdict, editDetails) {
    const shown = current.instance();
    if (!shown) return;
    const body = {id: shown.id, verdict};
    if (editDetails != null) body.edit_details = editDetails;
    if (current.method() != null) body.entry_method = current.method();
    try {
      const r = await fetch("/api/review/verdict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (!r.ok || !j.success) { alert("verdict failed: " + (j.error || r.status)); return; }
      progress.refresh();
      if ($("reviewAutoAdvanceChk").checked) {
        $("reviewNextBtn").click();  // auto-advance
      } else {
        // Stay at current instance; reflect new verdict in UI
        current.setState(verdict);
      }
    } catch (e) { alert("verdict error: " + e); }
  }

  $("reviewBlessBtn").addEventListener("click", () => postVerdict("blessed", null));
  $("reviewEraseBtn").addEventListener("click", () => postVerdict("erased", null));

  $("reviewEditBtn").addEventListener("click", () => {
    $("reviewEditDetails").value = "";
    new bootstrap.Modal($("reviewEditModal")).show();
  });

  $("reviewEditSaveBtn").addEventListener("click", () => {
    const raw = $("reviewEditDetails").value.trim();
    let parsed = null;
    if (raw) {
      try { parsed = JSON.parse(raw); }
      catch (e) { alert("edit_details must be valid JSON: " + e); return; }
      if (typeof parsed !== "object" || Array.isArray(parsed)) {
        alert("edit_details must be a JSON object (not array or primitive)"); return;
      }
    }
    bootstrap.Modal.getInstance($("reviewEditModal")).hide();
    postVerdict("edited", parsed);
  });

  $("reviewUndoBtn").addEventListener("click", async () => {
    const shown = current.instance();
    if (!shown) return;
    try {
      const r = await fetch("/api/review/undo", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({id: shown.id}),
      });
      const j = await r.json();
      if (!r.ok || !j.success) { alert("undo failed: " + (j.error || r.status)); return; }
      progress.refresh();
    } catch (e) { alert("undo error: " + e); }
  });
}
