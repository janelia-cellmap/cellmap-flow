// Resume Existing Volume: find the sessions under a directory, and copy one
// into a new session to carry on annotating (the original stays as it was).
import { esc, setBusy } from "../../lib/dom.js";
import { postAnswer, watchProgress } from "./requests.js";

// A session copy's progress, as the status line shows it.
function describeCopy(p) {
  if (p.phase === "starting" || p.phase === "setup") {
    return p.message || "Starting...";
  } else if (p.phase === "copying") {
    const filePart = p.files_total
      ? ` — ${p.files_done}/${p.files_total} files`
      : "";
    return `Copying ${p.parent_done + 1}/${p.parent_total}: ${p.current}${filePart}`;
  } else if (p.phase === "copying_minio") {
    return `Copying MinIO storage (${p.files_done || 0}/${p.files_total || "?"} files)`;
  } else if (p.phase === "mirroring_minio") {
    return `Mirroring volume to MinIO (${p.current})`;
  } else if (p.phase === "done") {
    return `Finalizing... ${p.copied_count || 0} entries copied`;
  } else if (p.phase === "error") {
    return `Error: ${p.error || "load failed"}`;
  }
  return p.message || JSON.stringify(p);
}

// log: the Annotation Crops panel's log; addToViewer: shows the copied
// volume in the viewer (crops.js).
export function initSessions({ log, addToViewer }) {
  const outputPathInput = document.getElementById("outputPath");
  const existingSessionsList = document.getElementById("existingSessionsList");
  const existingSessionsEmpty = document.getElementById("existingSessionsEmpty");
  const existingBrowseRoot = document.getElementById("existingBrowseRoot");
  const loadExistingConfirmBtn = document.getElementById("loadExistingConfirmBtn");
  let loadExistingModal = null;

  function scanExistingSessions() {
    const rootPath = existingBrowseRoot.value.trim();
    if (!rootPath) {
      existingSessionsList.innerHTML = '<option disabled>Enter a directory above and click Scan</option>';
      existingSessionsEmpty.style.display = "none";
      return;
    }
    try { localStorage.setItem("existingBrowseRoot", rootPath); } catch (_) {}
    existingSessionsList.innerHTML = '<option>Loading...</option>';
    existingSessionsEmpty.style.display = "none";
    loadExistingConfirmBtn.disabled = true;

    postAnswer("/api/finetune/list-existing-sessions", { output_path: rootPath })
      .then(data => {
        existingSessionsList.innerHTML = "";
        if (!data.success) {
          existingSessionsList.innerHTML = `<option disabled>Error: ${esc(data.error)}</option>`;
          return;
        }
        if (!data.sessions || data.sessions.length === 0) {
          existingSessionsEmpty.style.display = "block";
          return;
        }
        data.sessions.forEach(s => {
          const volNames = s.volumes.map(v => v.volume_id).join(", ") || "(no volume)";
          const opt = document.createElement("option");
          opt.value = s.session_path;
          opt.textContent = `${s.session_id} — ${s.chunk_count} chunks — ${volNames}`;
          opt.disabled = s.volumes.length === 0;
          existingSessionsList.appendChild(opt);
        });
      })
      .catch(err => {
        existingSessionsList.innerHTML = `<option disabled>Error: ${esc(err)}</option>`;
      });
  }

  existingSessionsList.addEventListener("change", () => {
    loadExistingConfirmBtn.disabled = !existingSessionsList.value;
  });
  document.getElementById("existingBrowseRefreshBtn").addEventListener("click", scanExistingSessions);
  existingBrowseRoot.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); scanExistingSessions(); }
  });

  document.getElementById("loadExistingVolumeBtn").addEventListener("click", function() {
    if (!loadExistingModal) {
      loadExistingModal = new bootstrap.Modal(document.getElementById("loadExistingModal"));
    }
    // Seed the root: prefer outputPath field, then last used, then blank
    const outputPath = outputPathInput.value.trim();
    const saved = (() => { try { return localStorage.getItem("existingBrowseRoot"); } catch (_) { return null; } })();
    existingBrowseRoot.value = outputPath || saved || "";
    loadExistingModal.show();
    if (existingBrowseRoot.value) scanExistingSessions();
    else {
      existingSessionsList.innerHTML = '<option disabled>Enter a directory above and click Scan</option>';
      existingSessionsEmpty.style.display = "none";
      loadExistingConfirmBtn.disabled = true;
    }
  });

  loadExistingConfirmBtn.addEventListener("click", function() {
    const sessionPath = existingSessionsList.value;
    // Target output path: use the outputPath field if set, otherwise the
    // parent of the source session (so we save back into the same root).
    let outputPath = outputPathInput.value.trim();
    if (!outputPath) {
      outputPath = existingBrowseRoot.value.trim();
      if (outputPath) {
        outputPathInput.value = outputPath;
      }
    }
    if (!sessionPath || !outputPath) return;

    const status = document.getElementById("loadExistingStatus");
    setBusy(loadExistingConfirmBtn, true, "Loading...");
    log.add(`Loading existing volume from: ${sessionPath}`);
    status.textContent = "Starting...";

    const progress = watchProgress("/api/finetune/load-existing-volume-progress", (p) => {
      status.textContent = describeCopy(p);
    });

    postAnswer("/api/finetune/load-existing-volume", {
      source_session_path: sessionPath,
      output_path: outputPath,
      load_id: progress.loadId,
    })
      .then(data => {
        progress.stop();
        setBusy(loadExistingConfirmBtn, false);
        if (!data.success) {
          status.textContent = `Error: ${data.error}`;
          log.add(`✗ Error: ${data.error}`);
          return;
        }
        status.textContent =
          `Loaded volume ${data.volume_id} (${data.copied_count} entries, ${data.painted_chunk_count || 0} painted chunks).`;
        log.add(`✓ Loaded volume ${data.volume_id} into ${data.new_session_path}`);
        log.add(`  Copied ${data.copied_count} files. ${data.painted_chunk_count || 0} painted chunks served.`);
        if (data.copied_minio) {
          log.add(`  Migrated MinIO storage from source session.`);
        }
        log.add(`  MinIO URL: ${data.minio_url}`);
        if (loadExistingModal) loadExistingModal.hide();
        addToViewer(
          data.volume_id,
          data.neuroglancer_url,
          `sparse_annotation_${data.volume_id}`,
        );
      })
      .catch(err => {
        progress.stop();
        setBusy(loadExistingConfirmBtn, false);
        status.textContent = `Request failed: ${err}`;
        log.add(`✗ Error: ${err}`);
      });
  });
}
