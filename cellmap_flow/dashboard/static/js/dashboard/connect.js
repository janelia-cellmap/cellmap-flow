// The "No Viewer Connected" panel, shown instead of Neuroglancer when no
// viewer is set: point the dashboard at a dataset through /api/set-data,
// then reload the page, which then has the viewer.
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

export function initConnect() {
  const btn = document.getElementById("setDataBtn");
  if (!btn) return;  // a viewer is connected, so the panel isn't on the page
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
