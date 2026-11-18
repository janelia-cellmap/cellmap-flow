// The Review tab's Current instance block: the instance on screen, how it
// got there, and the verdict buttons, which are enabled only while there is
// one.
//
// How it got there is the entry method a verdict on it records: "next"
// (Next, or auto-advance), "show" (Go to ID) or "pick" (t pressed over it in
// the viewer).
export function createCurrentInstance() {
  const $ = (id) => document.getElementById(id);
  let instance = null;  // the server's row for it, as /next, /show or a pick gave it
  let method = null;

  function setControlsEnabled(hasInstance) {
    $("reviewBlessBtn").disabled = !hasInstance;
    $("reviewEditBtn").disabled = !hasInstance;
    $("reviewEraseBtn").disabled = !hasInstance;
    $("reviewUndoBtn").disabled = !hasInstance;
  }

  return {
    instance: () => instance,
    method: () => method,

    // inst, reached by `how` (its entry method).
    show(inst, how) {
      method = how;
      instance = inst;
      $("reviewNoneBlock").style.display = "none";
      $("reviewCurrentBlock").style.display = "";
      $("reviewCurrentId").textContent = `id=${inst.id}`;
      $("reviewCurrentState").textContent = inst.review_state || "unreviewed";
      $("reviewCurrentEntryMethod").textContent = how;
      $("reviewCurrentRank").textContent =
        (inst.rank !== undefined && inst.rank !== null) ? inst.rank : "—";
      $("reviewCurrentVox").textContent = inst.vox.toLocaleString();
      // Optional columns: an index has them only if its builder wrote them.
      $("reviewCurrentEmMean").textContent =
        inst.em_mean != null ? Number(inst.em_mean).toFixed(1) : "—";
      $("reviewCurrentSphericity").textContent =
        inst.sphericity != null ? Number(inst.sphericity).toFixed(3) : "—";
      $("reviewCurrentFmScore").textContent =
        inst.fm_score != null ? Number(inst.fm_score).toExponential(3) : "—";
      $("reviewCurrentCentroid").textContent =
        `z=${inst.cz_nm.toFixed(0)}\ny=${inst.cy_nm.toFixed(0)}\nx=${inst.cx_nm.toFixed(0)}`;
      $("reviewCurrentCentroid").style.whiteSpace = "pre-line";
      $("reviewCurrentBbox").textContent =
        `z=[${inst.bz0}, ${inst.bz1})\ny=[${inst.by0}, ${inst.by1})\nx=[${inst.bx0}, ${inst.bx1})`;
      $("reviewCurrentBbox").style.whiteSpace = "pre-line";
      setControlsEnabled(true);
    },

    // The queue has nothing left for the current filter.
    none() {
      instance = null;
      method = null;
      $("reviewCurrentBlock").style.display = "none";
      $("reviewNoneBlock").style.display = "";
      setControlsEnabled(false);
    },

    // A verdict recorded on the instance shown, which stays on screen.
    setState(verdict) {
      if (!instance) return;
      instance.review_state = verdict;
      $("reviewCurrentState").textContent = verdict;
    },
  };
}
