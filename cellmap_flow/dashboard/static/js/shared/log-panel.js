// The dashboard's log, at the foot of the page (templates/_log_panel.html,
// on the dashboard page and the pipeline builder's): a bar with the latest
// line, which opens to the lines before it. Each page starts it as its
// template says (data-open) and then remembers how it was left.
//
// The lines are what the dashboard logs (routes/logging_routes.py), from
// /api/logs/stream by the dashboard's log protocol (lib/sse.js): every
// record's number is its event id, so a stream that reconnects is sent only
// the records not shown yet. The browser reconnects by itself after a
// dropped connection but gives up after an error answer (a proxy's 502
// while the dashboard restarts, say); then this opens a new stream, from the
// last record shown, every RETRY_MS. One stream per page: it holds one of
// the six connections a browser opens to the dashboard for as long as the
// page is open.
//
// The panel keeps the last MAX_LINES records, as the server does, so a long
// session does not grow the page without end.
//
// _log_panel.html loads this module by itself; it mounts the page's panel.
import { openLogStream } from "../lib/sse.js";
import { readJSON, writeJSON } from "../lib/stored.js";

const MAX_LINES = 1000;  // the server's buffer (dashboard/state.py log_buffer)
const RETRY_MS = 5000;
// "2026-10-27 21:03:05 WARNING cellmap_flow.jobs.launch: ..." (logging_setup.LOG_FORMAT)
const LEVEL = /^\S+ \S+ (WARNING|ERROR|CRITICAL) /;

export function mountLogPanel(panel) {
  const header = panel.querySelector(".log-header");
  const latest = panel.querySelector(".log-latest");
  const status = panel.querySelector(".log-status");
  const content = panel.querySelector(".log-content");
  const output = panel.querySelector(".log-output");
  const openKey = `cellmap-flow:log-panel-open:${location.pathname}`;

  function setOpen(open) {
    content.hidden = !open;
    panel.classList.toggle("collapsed", !open);
    header.setAttribute("aria-expanded", String(open));
    if (open) content.scrollTop = content.scrollHeight;
  }
  setOpen(readJSON(openKey, panel.dataset.open === "true"));
  header.addEventListener("click", () => {
    writeJSON(openKey, content.hidden);
    setOpen(content.hidden);
  });

  // New lines keep the end in view, unless someone has scrolled up to read.
  // Once a frame: a scroll lays the lines out again, and one after every line
  // made the 1,000 a stream sends as it connects take half a second.
  let following = true;
  content.addEventListener("scroll", () => {
    following = content.scrollTop + content.clientHeight >= content.scrollHeight - 4;
  });
  let endQueued = false;
  function keepEnd() {
    if (endQueued) return;
    endQueued = true;
    requestAnimationFrame(() => {
      endQueued = false;
      if (following && !content.hidden) content.scrollTop = content.scrollHeight;
    });
  }

  // One record; a traceback is one record of several lines.
  function add(record) {
    const line = document.createElement("div");
    line.textContent = record;
    const level = LEVEL.exec(record);
    if (level) line.className = level[1] === "WARNING" ? "log-warning" : "log-error";
    output.appendChild(line);
    while (output.childElementCount > MAX_LINES) output.firstElementChild.remove();
    latest.textContent = record.split("\n", 1)[0];
    latest.className = `log-latest ${line.className}`;
    keepEnd();
  }

  let stream = null;
  function connect(offset) {
    stream = openLogStream("/api/logs/stream", {
      offset,
      onOpen() {
        status.textContent = "";
      },
      onLine: add,
      onError() {
        status.textContent = "reconnecting…";
        if (stream.closed) setTimeout(() => connect(stream.offset), RETRY_MS);
      },
    });
  }
  connect(0);
}

const panel = document.getElementById("log-panel");
if (panel) mountLogPanel(panel);
