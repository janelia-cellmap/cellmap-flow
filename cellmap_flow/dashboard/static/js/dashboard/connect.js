// The connect panel: point the dashboard at a dataset through /api/set-data,
// then reload the page, which then has the viewer. It is shown instead of
// Neuroglancer when no viewer is set ("No Viewer Connected"), and in the
// viewer's place when the header's Open Dataset asks for it, to open another
// one; the running models carry over to it (routes/index_page.set_data, in
// the browser web/src/backend/routes/data.ts).
//
// The datasets connected to before are offered in the path field, newest
// first (its datalist, #recentDatasets). The field had only the browser's
// own form history, which shows a few entries, and only those that start
// with what is typed. They are kept in this browser's localStorage: a
// convenience of this browser alone, gone with its site data, and the
// field works as before when the storage cannot be read. A folder picked
// on this machine (the browser mode's `local/<key>/` URLs) is left out: it
// opens through the Browse button, which asks for its permission again.
import { ApiError, postJSON } from "../lib/api.js";
import { readJSON, writeJSON } from "../lib/stored.js";

const RECENT_KEY = "cellmap-flow:recent-datasets";
// Enough to find last week's datasets; the datalist scrolls past a dozen.
const MAX_RECENT = 20;
const LOCAL_FOLDER = /\/local\/[^/]+\//;

// The datasets connected to before, newest first; none when the storage
// cannot be read (a private window, blocked site data).
function recentDatasets() {
  const stored = readJSON(RECENT_KEY, []);
  return Array.isArray(stored) ? stored.filter((path) => typeof path === "string" && path) : [];
}

// `path` first among the recent datasets, once, the oldest dropped past
// MAX_RECENT; a picked folder is not kept.
function rememberDataset(path) {
  if (LOCAL_FOLDER.test(path)) return;
  const recent = [path, ...recentDatasets().filter((other) => other !== path)].slice(0, MAX_RECENT);
  writeJSON(RECENT_KEY, recent);
}

// The field's datalist filled with the recent datasets.
function showRecentDatasets() {
  const list = document.getElementById("recentDatasets");
  if (!list) return;
  list.replaceChildren(...recentDatasets().map((path) => {
    const option = document.createElement("option");
    option.value = path;
    return option;
  }));
}

// The phone layout of templates/index.html (its max-width: 767.98px block),
// where the dashboard, toggled, covers the viewer's column.
const PHONE = window.matchMedia("(max-width: 767.98px)");

// The header's Open Dataset (Open Volume in the browser mode): with a viewer
// open, the panel in the viewer's place, its field focused; the button
// again, or Cancel, puts the viewer back. templates/index.html shows the
// one or the other from the column's choosing-dataset class. Without a
// viewer the panel is showing already, and the button only focuses the
// field. The browser mode's viewer comes after the page is drawn
// (web/src/viewer), so it is looked for on each click.
function initOpenDataset() {
  const open = document.getElementById("openDatasetBtn");
  const column = document.getElementById("iframe-column");
  const input = document.getElementById("datasetPathInput");
  if (!open || !column || !input) return;
  function choose(on) {
    column.classList.toggle("choosing-dataset", on);
    open.setAttribute("aria-pressed", String(on));
    if (!on) return;
    if (PHONE.matches) document.body.classList.remove("dashboard-toggled");
    input.focus();
  }
  open.addEventListener("click", () => {
    if (!document.getElementById("my_iframe")) {
      input.focus();
      return;
    }
    choose(!column.classList.contains("choosing-dataset"));
  });
  const cancel = document.getElementById("cancelOpenDatasetBtn");
  if (cancel) cancel.addEventListener("click", () => choose(false));
}

export function initConnect() {
  const btn = document.getElementById("setDataBtn");
  if (!btn) return;  // not on this page
  initOpenDataset();
  const statusEl = document.getElementById("setDataStatus");
  showRecentDatasets();

  btn.addEventListener("click", function () {
    const datasetPath = document.getElementById("datasetPathInput").value.trim();
    if (!datasetPath) {
      statusEl.textContent = "Please enter a dataset path.";
      return;
    }
    const label = btn.textContent;  // restored if this fails
    btn.disabled = true;
    btn.textContent = "Loading...";
    statusEl.textContent = "Setting up neuroglancer viewer...";

    function fail(message) {
      statusEl.style.color = "#f87171";
      statusEl.textContent = "Error: " + message;
      btn.disabled = false;
      btn.textContent = label;
    }

    postJSON("/api/set-data", { dataset_path: datasetPath })
      .then((data) => {
        if (data.error) {
          fail(data.error);
          return;
        }
        // Only a dataset that opened is offered again.
        rememberDataset(datasetPath);
        statusEl.style.color = "#4ade80";
        statusEl.textContent = "Success! Reloading...";
        setTimeout(function () { window.location.reload(); }, 1000);
      })
      // The route answers a bad path with {"error": ...} and a 400 or 500.
      .catch((err) => fail(err instanceof ApiError ? err.message : err));
  });
}
