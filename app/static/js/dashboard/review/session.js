// The Review tab's Session card: opening an index (Open), taking up the one
// already open when the tab is shown, and the viewer's pick stream, which
// follows the open session.
//
// current: the Current instance block (current.js); progress: the Progress
// card (progress.js).
export function initSession({ current, progress }) {
  const $ = (id) => document.getElementById(id);
  const picks = createPickStream((pick) => current.show(pick, "pick"));

  $("reviewOpenBtn").addEventListener("click", async () => {
    const body = {
      db_path: $("reviewDbPath").value.trim(),
      reviewer: $("reviewReviewer").value.trim() || null,
      segmentation_layer: $("reviewSegLayer").value.trim() || null,
    };
    if (!body.db_path) { alert("db path is required"); return; }
    try {
      const r = await fetch("/api/review/open", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (!r.ok || !j.success) {
        $("reviewSessionStatus").textContent = "open failed: " + (j.error || r.status);
        $("reviewSessionStatus").className = "small text-danger mt-2";
        return;
      }
      // The reviewer and layer name are the request's own text: build the
      // status from nodes so neither is ever parsed as markup.
      const status = $("reviewSessionStatus");
      const opened = document.createElement("strong");
      opened.textContent = "opened";
      status.replaceChildren(
        opened,
        document.createElement("br"),
        `${j.n_instances} instances`,
        document.createElement("br"),
        `reviewer: ${j.reviewer}`
          + (j.segmentation_layer ? `, pick layer=${j.segmentation_layer}` : ""),
      );
      status.className = "small text-success mt-2";
      progress.refresh();
      progress.restartPolling();
      picks.restart();
    } catch (e) {
      $("reviewSessionStatus").textContent = "open error: " + e;
      $("reviewSessionStatus").className = "small text-danger mt-2";
    }
  });

  // On tab activation, refresh status + progress if a db is already open
  document.addEventListener("shown.bs.tab", (e) => {
    if (e.target && e.target.id === "review-tab") {
      fetch("/api/review/status").then(r => r.json()).then(s => {
        if (s.db_path) {
          $("reviewDbPath").value = s.db_path;
          $("reviewReviewer").value = s.reviewer || "";
          $("reviewSegLayer").value = s.segmentation_layer || "";
          $("reviewSessionStatus").textContent =
            `open (viewer=${s.viewer_attached ? "attached" : "none"})`;
          $("reviewSessionStatus").className = "small text-success mt-2";
          progress.refresh();
          progress.keepPolling();
          picks.ensureOpen();
        }
      }).catch(() => {});
    }
  });
}

// Subscribe to /api/review/pick_stream — Server-Sent Events one-way push
// from Flask. The action handler (Tornado side, NG-Python) numbers each
// pick when 't' lands; the SSE generator emits one event per new pick.
// Single long-lived HTTP connection — does not re-acquire a browser
// HTTP/1.1 slot per update, so it doesn't compete with NG chunk fetches.
// Polling every 700ms instead queued for 14s on average under chunk load.
//
// onPick(instance) gets each new pick's instance.
function createPickStream(onPick) {
  let pickStream = null;
  let lastPickSeq = 0;

  function open() {
    if (pickStream) { try { pickStream.close(); } catch (e) {} }
    pickStream = new EventSource("/api/review/pick_stream");
    pickStream.onmessage = (ev) => {
      try {
        const j = JSON.parse(ev.data);
        if (j && j.seq !== undefined && j.seq !== lastPickSeq && j.pick) {
          lastPickSeq = j.seq;
          onPick(j.pick);
        }
      } catch (e) { /* ignore malformed events */ }
    };
    pickStream.onerror = () => {
      // EventSource auto-reconnects after a small back-off; nothing to do
      // unless we want to surface a status. Keep silent.
    };
  }

  return {
    // A session just opened, whose picks are numbered from the start again;
    // the stream open before, if any, is closed.
    restart() {
      lastPickSeq = 0;
      open();
    },
    // Open the stream unless it already is (it reconnects by itself).
    ensureOpen() {
      if (!pickStream) open();
    },
  };
}
