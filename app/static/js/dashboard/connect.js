// The "No Viewer Connected" panel, shown instead of Neuroglancer when no
// viewer is set: point the dashboard at a dataset through /api/set-data,
// then reload the page, which then has the viewer.
import { ApiError, postJSON } from "../lib/api.js";

export function initConnect() {
  const btn = document.getElementById("setDataBtn");
  if (!btn) return;  // a viewer is connected, so the panel isn't on the page
  const statusEl = document.getElementById("setDataStatus");

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
        statusEl.style.color = "#4ade80";
        statusEl.textContent = "Success! Reloading...";
        setTimeout(function () { window.location.reload(); }, 1000);
      })
      // The route answers a bad path with {"error": ...} and a 400 or 500.
      .catch((err) => fail(err instanceof ApiError ? err.message : err));
  });
}
