// The Annotation Crops panel's work on a session's annotations: a new
// annotation volume, crops loaded from a YAML manifest, the annotated-region
// boxes, and saving the annotations to disk.
import { esc } from "../../lib/dom.js";
import { getAnswer, postAnswer, watchProgress } from "./requests.js";

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

// log: the panel's log; picker: the model picker; form: the training form
// (its state is saved once a volume exists).
// Returns { addToViewer(cropId, neuroglancerUrl, layerName) }, which shows
// an annotation volume in the viewer.
export function initCrops({ log, picker, form }) {
  const outputPathInput = document.getElementById("outputPath");

  function addToViewer(cropId, neuroglancerUrl, layerName) {
    log.add(`Adding layer to viewer...`);

    const payload = {
      crop_id: cropId,
      minio_url: neuroglancerUrl
    };
    if (layerName) {
      payload.layer_name = layerName;
    }

    postAnswer("/api/finetune/add-to-viewer", payload)
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

  // Create annotation volume (sparse, full dataset)
  const createVolumeBtn = document.getElementById("createVolumeBtn");
  // Restored after each attempt.
  const createVolumeBtnLabel = createVolumeBtn.textContent.trim();
  createVolumeBtn.addEventListener("click", function() {
    const selectedModel = picker.selected();
    if (!selectedModel) {
      alert("No model selected");
      return;
    }

    const outputPath = outputPathInput.value.trim();
    if (!outputPath) {
      alert("Please specify an output path for the zarr files");
      return;
    }

    createVolumeBtn.disabled = true;
    createVolumeBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Creating...';
    log.add(`Creating annotation volume for full dataset...`);
    log.add(`Output path: ${outputPath}`);

    const payload = {
      model_name: selectedModel.name,
      output_path: outputPath
    };

    postAnswer("/api/finetune/create-volume", payload)
      .then(data => {
        if (data.success) {
          form.save();
          log.add(`✓ Created annotation volume: ${data.volume_id}`);
          log.add(`  Dataset shape (voxels): [${data.metadata.dataset_shape_voxels.join(', ')}]`);
          log.add(`  Chunk size (voxels): [${data.metadata.chunk_size.join(', ')}]`);
          log.add(`  Voxel size (nm): [${data.metadata.output_voxel_size.join(', ')}]`);
          log.add(`  Zarr path: ${data.zarr_path}`);
          log.add(`  MinIO URL: ${data.minio_url}`);
          log.add(`  Label scheme: paint 1=background, 2=foreground, 0=unannotated`);

          document.getElementById("minioStatusContent").innerHTML = `
            <small>
              <strong>Status:</strong> <span style="color: var(--success);">Running</span><br>
              <strong>URL:</strong> ${esc(data.minio_url)}
            </small>
          `;
          document.getElementById("minioStatus").style.display = "block";

          addToViewer(data.volume_id, data.neuroglancer_url, `sparse_annotation_${data.volume_id}`);
        } else {
          log.add(`✗ Error: ${data.error}`);
        }
        createVolumeBtn.disabled = false;
        createVolumeBtn.textContent = createVolumeBtnLabel;
      })
      .catch(err => {
        log.add(`✗ Error: ${err}`);
        console.error(err);
        createVolumeBtn.disabled = false;
        createVolumeBtn.textContent = createVolumeBtnLabel;
      });
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
    try {
      const saved = localStorage.getItem("loadCropsYamlPath");
      if (saved) loadCropsYamlPath.value = saved;
    } catch (_) {}
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
        try { localStorage.setItem("loadCropsYamlPath", path); } catch (_) {}
      } else {
        status.textContent = `Read failed: ${data.error}`;
      }
    } catch (err) {
      status.textContent = `Read failed: ${err}`;
    }
  });

  // Submit: send pasted YAML if present, else send the path (server parses both).
  const loadCropsSubmitBtn = document.getElementById("loadCropsSubmitBtn");
  loadCropsSubmitBtn.addEventListener("click", function() {
    const selectedModel = picker.selected();
    if (!selectedModel) {
      alert("No model selected");
      return;
    }
    const yamlText = loadCropsYaml.value.trim();
    const yamlPath = loadCropsYamlPath.value.trim();
    const yamlPayload = yamlText || yamlPath;
    if (!yamlPayload) {
      loadCropsStatus.textContent =
        "Paste YAML or enter a file path first.";
      return;
    }
    try { if (yamlPath) localStorage.setItem("loadCropsYamlPath", yamlPath); } catch (_) {}
    const outputPath = outputPathInput.value.trim();
    const status = loadCropsStatus;
    status.textContent = "Starting...";
    loadCropsSubmitBtn.disabled = true;

    const progress = watchProgress("/api/finetune/load-crops-progress", (p) => {
      status.textContent = describeCropImport(p);
    });

    postAnswer("/api/finetune/load-crops", {
      model_name: selectedModel.name,
      output_path: outputPath,
      yaml: yamlPayload,
      load_id: progress.loadId,
    })
      .then(data => {
        progress.stop();
        loadCropsSubmitBtn.disabled = false;
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
        loadCropsSubmitBtn.disabled = false;
        status.textContent = `Request failed: ${err}`;
        log.add(`✗ Load crops request failed: ${err}`);
      });
  });

  // Redraw the annotated-region boxes, on request only (the button's note
  // says why).
  const showAnnotatedRegionsBtn = document.getElementById("showAnnotatedRegionsBtn");
  const annotatedRegionCount = document.getElementById("annotatedRegionCount");
  showAnnotatedRegionsBtn.addEventListener("click", function () {
    showAnnotatedRegionsBtn.disabled = true;
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
      .finally(() => {
        showAnnotatedRegionsBtn.disabled = false;
      });
  });

  // Save annotations to disk
  const saveAnnotationsBtn = document.getElementById("saveAnnotationsBtn");
  saveAnnotationsBtn.addEventListener("click", function() {
    saveAnnotationsBtn.disabled = true;
    const originalText = saveAnnotationsBtn.textContent;
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
        saveAnnotationsBtn.disabled = false;
        saveAnnotationsBtn.textContent = originalText;
      })
      .catch(err => {
        log.add(`✗ Error: ${err}`);
        console.error(err);
        saveAnnotationsBtn.disabled = false;
        saveAnnotationsBtn.textContent = originalText;
      });
  });

  return { addToViewer };
}
