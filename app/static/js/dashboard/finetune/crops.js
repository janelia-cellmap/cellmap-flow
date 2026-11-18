// The Annotation Crops panel's work on a session's annotations: a new
// annotation volume, crops loaded from a YAML manifest, the annotated-region
// boxes, and saving the annotations to disk. In the browser mode (lib/page-data
// isBrowserMode) the volume needs no output path, and in place of the save
// the paint can be downloaded as a zarr or deleted.
import { esc, setBusy } from "../../lib/dom.js";
import { isBrowserMode } from "../../lib/page-data.js";
import { readText, writeText } from "../../lib/stored.js";
import { VOLUME_CREATED, getAnswer, postAnswer, watchProgress } from "./requests.js";

// A crop import's progress, as the status line shows it.
function describeCropImport(p) {
  if (p.phase === "starting" || p.phase === "setup") {
    return p.message || "Starting...";
  } else if (p.phase === "crop_start") {
    return `Crop ${p.crop_index + 1}/${p.n_crops}: opening ${p.current_path}`;
  } else if (p.phase === "tile") {
    const pct = p.tile_total ? ((p.tile_done / p.tile_total) * 100).toFixed(0) : 0;
    return `Crop ${p.crop_index + 1}/${p.n_crops}: writing slab ${p.tile_done}/${p.tile_total} (${pct}%) — ${p.current_path}`;
  } else if (p.phase === "done") {
    return `Finishing... ${p.n_crops_imported || 0} crop(s) imported`;
  }
  return p.message || JSON.stringify(p);
}

// sessionVolumeChanged's way to volumeReady, once initCrops has run.
let volumeReadyHook = null;

// Another part of the tab made `volumeId` the session's volume (rank.js's
// Use this model): Download paint and Delete volume act on it from now on.
export function sessionVolumeChanged(volumeId) {
  if (volumeReadyHook) volumeReadyHook(volumeId);
}

// log: the panel's log; picker: the model picker; form: the training form
// (its state is saved once a volume exists); grid: New Volume's voxel size
// for a volume painted before any model (volume-grid.js).
// Returns { addToViewer(cropId, neuroglancerUrl), volumeReady(volumeId) }.
// addToViewer shows an annotation volume in the viewer, as the layer
// annotation_<cropId>: the server names it, so that Load Crops, which looks
// for that layer before adding its own, finds this one. The tab used to name
// it sparse_annotation_<id>, and an import into the volume then added a
// second writable layer over the same data. volumeReady says which volume the
// browser mode's Download paint and Delete volume act on (nothing elsewhere).
export function initCrops({ log, picker, form, grid }) {
  const outputPathInput = document.getElementById("outputPath");
  const browser = isBrowserMode();

  // Browser mode: the paint lives in this browser, so the template's
  // browser-only block offers it as a download (a zip of the zarr; the page
  // fetches the link on a click, web/src/rebase.ts, as it does the Export
  // Config link, and the service worker relays the GET to the in-page
  // backend) and a delete. Both act on the volume last created or
  // resumed; with none (and again after a delete) the link has no target and
  // the button is disabled.
  function showVolumeActions(volumeId) {
    const download = document.getElementById("downloadVolumeBtn");
    const remove = document.getElementById("deleteVolumeBtn");
    if (volumeId) {
      download.href = `/api/finetune/volume/${encodeURIComponent(volumeId)}/download`;
      download.classList.remove("disabled");
      download.removeAttribute("aria-disabled");
      download.removeAttribute("tabindex");
      remove.dataset.volumeId = volumeId;
      remove.disabled = false;
    } else {
      download.removeAttribute("href");
      download.classList.add("disabled");
      download.setAttribute("aria-disabled", "true");
      download.setAttribute("tabindex", "-1");
      delete remove.dataset.volumeId;
      remove.disabled = true;
    }
  }

  function volumeReady(volumeId) {
    if (browser) showVolumeActions(volumeId);
  }
  volumeReadyHook = volumeReady;

  // After a reload the session still has its volume (the backend keeps it
  // for the tab), so the buttons learn which one from the browser's own
  // route and work without a New Volume or Resume first.
  if (browser) {
    getAnswer("/api/finetune/session-volume")
      .then((data) => {
        if (data.success && data.volume_id) volumeReady(data.volume_id);
      })
      .catch(() => {});
  }

  if (browser) {
    const deleteVolumeBtn = document.getElementById("deleteVolumeBtn");
    deleteVolumeBtn.addEventListener("click", function () {
      const volumeId = deleteVolumeBtn.dataset.volumeId;
      if (!volumeId) return;
      if (!confirm(`Delete the paint of volume ${volumeId} from this browser?\n\n` +
        "This cannot be undone; download it first to keep it.")) {
        return;
      }
      setBusy(deleteVolumeBtn, true, "Deleting...");
      postAnswer("/api/finetune/delete-volume", { volume_id: volumeId })
        .then(data => {
          if (data.success) {
            log.add(`✓ Deleted volume ${volumeId} from this browser`);
            delete deleteVolumeBtn.dataset.volumeId;
          } else {
            log.add(`✗ Error deleting volume: ${data.error}`);
          }
        })
        .catch(err => {
          log.add(`✗ Error: ${err}`);
          console.error(err);
        })
        .finally(() => {
          // setBusy enables the button again; a deleted volume leaves nothing to act on.
          setBusy(deleteVolumeBtn, false);
          if (!deleteVolumeBtn.dataset.volumeId) showVolumeActions(null);
        });
    });
  }

  function addToViewer(cropId, neuroglancerUrl) {
    log.add(`Adding layer to viewer...`);

    postAnswer("/api/finetune/add-to-viewer", { crop_id: cropId, minio_url: neuroglancerUrl })
      .then(data => {
        if (data.success) {
          log.add(`✓ Layer added: ${data.layer_name}`);
        } else {
          log.add(`✗ Error adding layer: ${data.error}`);
        }
      })
      .catch(err => {
        log.add(`✗ Error: ${err}`);
        console.error(err);
      });
  }

  // Create annotation volume (sparse, full dataset): on the picked model's
  // grid, or with no model, on the raw's grid at the voxel size typed (blank:
  // the level on screen), a volume Which model to start from's Use this
  // model later copies onto a model's grid to train.
  const createVolumeBtn = document.getElementById("createVolumeBtn");
  // What the click came to, for what offered it (requests.js offerNewVolume).
  const made = (ok) => document.dispatchEvent(new CustomEvent(VOLUME_CREATED, { detail: { ok } }));
  createVolumeBtn.addEventListener("click", function() {
    const selectedModel = picker.selected();
    let voxelSize = null;
    if (!selectedModel) {
      try {
        voxelSize = grid.voxelSize();
      } catch (err) {
        alert(err.message);
        made(false);
        return;
      }
    }

    // Browser mode: the volume goes to the browser's own store, so there is
    // no path to ask for (the field is hidden), and none is sent.
    const outputPath = browser ? "" : outputPathInput.value.trim();
    if (!outputPath && !browser) {
      alert("Please specify an output path for the zarr files");
      made(false);
      return;
    }

    setBusy(createVolumeBtn, true, "Creating...");
    log.add(selectedModel
      ? `Creating annotation volume for full dataset...`
      : `Creating annotation volume for full dataset, for no model yet, at ${voxelSize ? `${voxelSize.join(" × ")} nm` : "the level on screen"}...`);
    if (!browser) log.add(`Output path: ${outputPath}`);

    const payload = selectedModel
      ? { model_name: selectedModel.name, output_path: outputPath }
      : { voxel_size: voxelSize, output_path: outputPath };

    postAnswer("/api/finetune/create-volume", payload)
      .then(data => {
        if (data.success) {
          form.save();
          log.add(`✓ Created annotation volume: ${data.volume_id}`);
          log.add(`  Dataset shape (voxels): [${data.metadata.dataset_shape_voxels.join(', ')}]`);
          log.add(`  Chunk size (voxels): [${data.metadata.chunk_size.join(', ')}]`);
          log.add(`  Voxel size (nm): [${data.metadata.output_voxel_size.join(', ')}]`);
          // The browser has no zarr on a disk and no MinIO: its paint is in
          // the page's store, and the buttons below take it out.
          if (!browser) {
            log.add(`  Zarr path: ${data.zarr_path}`);
            log.add(`  MinIO URL: ${data.minio_url}`);
          }
          log.add(`  Label scheme: paint 1=background, 2=foreground, 0=unannotated`);
          if (data.model_name === null) {
            log.add(`  For no model yet: to train, rank the models under Which model to start from and click Use this model, which copies this paint onto its grid`);
          }

          if (!browser) {
            document.getElementById("minioStatusContent").innerHTML = `
              <small>
                <strong>Status:</strong> <span style="color: var(--success);">Running</span><br>
                <strong>URL:</strong> ${esc(data.minio_url)}
              </small>
            `;
            document.getElementById("minioStatus").style.display = "block";
          }

          volumeReady(data.volume_id);
          addToViewer(data.volume_id, data.neuroglancer_url);
        } else {
          log.add(`✗ Error: ${data.error}`);
        }
        made(data.success);
      })
      .catch(err => {
        log.add(`✗ Error: ${err}`);
        console.error(err);
        made(false);
      })
      .finally(() => setBusy(createVolumeBtn, false));
  });

  // Load crops from a YAML manifest (modal-driven, mirrors Resume Existing Volume)
  const loadCropsYamlPath = document.getElementById("loadCropsYamlPath");
  const loadCropsYaml = document.getElementById("loadCropsYaml");
  const loadCropsStatus = document.getElementById("loadCropsStatus");
  let loadCropsModal = null;
  document.getElementById("openLoadCropsBtn").addEventListener("click", function() {
    if (!picker.selected()) {
      alert("No model selected");
      return;
    }
    if (!loadCropsModal) {
      loadCropsModal = new bootstrap.Modal(document.getElementById("loadCropsModal"));
    }
    // Restore the last-used path so users don't retype it.
    const saved = readText("loadCropsYamlPath", "");
    if (saved) loadCropsYamlPath.value = saved;
    loadCropsStatus.textContent = "";
    loadCropsModal.show();
  });

  // "Load File" reads the file server-side and fills the textarea so the user
  // can preview/edit before submitting.
  document.getElementById("loadCropsYamlReadBtn").addEventListener("click", async function() {
    const path = loadCropsYamlPath.value.trim();
    const status = loadCropsStatus;
    if (!path) {
      status.textContent = "Enter a path first.";
      return;
    }
    status.textContent = "Reading...";
    try {
      const data = await getAnswer(`/api/finetune/read-yaml?path=${encodeURIComponent(path)}`);
      if (data.success) {
        loadCropsYaml.value = data.text;
        status.textContent = `Loaded ${data.text.length} chars from ${path}`;
        writeText("loadCropsYamlPath", path);
      } else {
        status.textContent = `Read failed: ${data.error}`;
      }
    } catch (err) {
      status.textContent = `Read failed: ${err}`;
    }
  });

  // Submit: send pasted YAML if present, else send the path (server parses
  // both). The model picked, if any, is sent too: the route needs one only to
  // lay a new volume on its grid when the session has none (with none
  // picked, it makes the volume painted before any model, as New Volume
  // does).
  const loadCropsSubmitBtn = document.getElementById("loadCropsSubmitBtn");
  loadCropsSubmitBtn.addEventListener("click", function() {
    const selectedModel = picker.selected();
    const yamlText = loadCropsYaml.value.trim();
    const yamlPath = loadCropsYamlPath.value.trim();
    const yamlPayload = yamlText || yamlPath;
    if (!yamlPayload) {
      loadCropsStatus.textContent =
        "Paste YAML or enter a file path first.";
      return;
    }
    if (yamlPath) writeText("loadCropsYamlPath", yamlPath);
    const outputPath = outputPathInput.value.trim();
    const status = loadCropsStatus;
    status.textContent = "Starting...";
    setBusy(loadCropsSubmitBtn, true);

    const progress = watchProgress("/api/finetune/load-crops-progress", (p) => {
      status.textContent = describeCropImport(p);
    });

    postAnswer("/api/finetune/load-crops", {
      ...(selectedModel ? { model_name: selectedModel.name } : {}),
      output_path: outputPath,
      yaml: yamlPayload,
      load_id: progress.loadId,
    })
      .then(data => {
        progress.stop();
        setBusy(loadCropsSubmitBtn, false);
        if (!data.success) {
          status.textContent = `Error: ${data.error || "load failed"}`;
          log.add(`✗ Load crops error: ${data.error}`);
          if (data.details) {
            log.add(`  details: ${JSON.stringify(data.details)}`);
          }
          return;
        }
        // Crops are written into one sparse annotation volume, not into
        // per-crop chunk zarrs, so the server reports crops and voxels.
        const fgVoxels = Number(data.fg_voxels_written || 0).toLocaleString();
        status.textContent =
          `Imported ${data.n_crops_imported} of ${data.n_crops_requested} crop(s) ` +
          `(${fgVoxels} foreground voxels); ${data.n_errors} error(s).`;
        log.add(
          `✓ Imported ${data.n_crops_imported} of ${data.n_crops_requested} crop(s) into ` +
          `volume ${data.volume_id}: ${fgVoxels} foreground voxels written, ${data.n_errors} error(s)`);
        if (data.errors && data.errors.length) {
          for (const err of data.errors) {
            log.add(`  ✗ ${err.path}: ${err.error}`);
          }
        }
        if (loadCropsModal) loadCropsModal.hide();
      })
      .catch(err => {
        progress.stop();
        setBusy(loadCropsSubmitBtn, false);
        status.textContent = `Request failed: ${err}`;
        log.add(`✗ Load crops request failed: ${err}`);
      });
  });

  // Redraw the annotated-region boxes, on request only (the button's note
  // says why).
  const showAnnotatedRegionsBtn = document.getElementById("showAnnotatedRegionsBtn");
  const annotatedRegionCount = document.getElementById("annotatedRegionCount");
  showAnnotatedRegionsBtn.addEventListener("click", function () {
    setBusy(showAnnotatedRegionsBtn, true);
    annotatedRegionCount.textContent = "updating...";
    postAnswer("/api/finetune/refresh-annotated-regions", {})
      .then((d) => {
        if (d.success) {
          const n = d.count || 0;
          annotatedRegionCount.textContent =
            n === 0 ? "none yet" : `${n} region${n === 1 ? "" : "s"}`;
        } else {
          annotatedRegionCount.textContent = "failed";
          log.add(`Could not refresh annotated regions: ${d.error}`);
          log.showEnd();
        }
      })
      .catch((e) => {
        annotatedRegionCount.textContent = "failed";
        log.add(`Could not refresh annotated regions: ${e}`);
      })
      .finally(() => setBusy(showAnnotatedRegionsBtn, false));
  });

  // Save annotations to disk
  const saveAnnotationsBtn = document.getElementById("saveAnnotationsBtn");
  saveAnnotationsBtn.addEventListener("click", function() {
    setBusy(saveAnnotationsBtn, true);
    saveAnnotationsBtn.textContent = "💾 Syncing...";
    log.add(`Syncing all annotations from MinIO to local disk...`);

    postAnswer("/api/finetune/sync-annotations", { force: true })
      .then(data => {
        if (data.success) {
          // The message already carries the count ("Synced N annotations");
          // the server has no total to put it against.
          log.add(`✓ ${data.message}`);
        } else {
          log.add(`✗ Error: ${data.error || data.message}`);
        }
      })
      .catch(err => {
        log.add(`✗ Error: ${err}`);
        console.error(err);
      })
      .finally(() => setBusy(saveAnnotationsBtn, false));
  });

  return { addToViewer, volumeReady };
}
