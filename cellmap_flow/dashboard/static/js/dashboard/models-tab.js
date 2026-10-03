// The Models tab (templates/_models_tab.html): the local catalog, Hugging
// Face, BioImage Model Zoo and Cellpose models to serve, the LSF server config, the GPU
// queue picker, and the inference jobs' own output.
import { ApiError, getJSON, postJSON } from "../lib/api.js";
import { pageData } from "../lib/page-data.js";
import { poll } from "../lib/poll.js";
import { fillQueueSelect, queueHint, subscribeGpuQueues } from "../shared/gpu-queues.js";
import { readCount, saveServerConfig, WALLTIME_RE } from "../shared/server-config.js";

// onModelsSubmitted runs after the server has taken a new model selection.
export function initModelsTab({ onModelsSubmitted } = {}) {
  const submitBtn = document.getElementById("submitModelsBtn");
  const logArea = document.getElementById("modelSubmissionLogModels");
  // The repos of the Hugging Face models already running, ticked on load.
  const defaultHfRepos = pageData().default_hf_repos || [];
  // The session's Resample setting (a YAML's resample, view --resample).
  const resampleCheckbox = document.getElementById("resampleCheckbox");
  resampleCheckbox.checked = !!pageData().resample;
  let hfModelsLoaded = false;

  function renderHfModels(data) {
    const container = document.getElementById("hfModelsContainer");
    const searchBar = document.getElementById("hfSearchBar");
    const refreshBtn = document.getElementById("hfRefreshBtn");

    // Remove old model items
    container.querySelectorAll(".hf-model-item").forEach(function (el) { el.remove(); });
    const placeholder = document.getElementById("hfPlaceholder");
    if (placeholder) placeholder.remove();

    if (data.error) {
      const p = document.createElement("p");
      p.className = "text-danger";
      p.textContent = "Error: " + data.error;
      container.appendChild(p);
      return;
    }

    const modelIds = Object.keys(data);
    if (modelIds.length === 0) {
      const p = document.createElement("p");
      p.className = "text-muted";
      p.textContent = "No models found.";
      container.appendChild(p);
      return;
    }

    searchBar.style.display = "";
    refreshBtn.style.display = "";

    modelIds.forEach(function (modelId) {
      const metadata = data[modelId];
      const displayName = modelId.split("/").pop();
      const searchStr = (modelId + " " + JSON.stringify(metadata)).toLowerCase();
      const div = document.createElement("div");
      div.className = "form-check mb-2 hf-model-item";
      div.setAttribute("data-search", searchStr);
      // Built from nodes rather than an HTML string: repo ids and model card
      // descriptions come from Hugging Face, not from us.
      const input = document.createElement("input");
      input.className = "form-check-input hf-model-checkbox";
      input.type = "checkbox";
      input.name = "selected_hf_models";
      input.id = "chk_hf_" + displayName;
      input.value = modelId;
      input.defaultChecked = defaultHfRepos.includes(modelId);
      const label = document.createElement("label");
      label.className = "form-check-label";
      label.htmlFor = input.id;
      label.textContent = displayName;
      if (metadata.description) {
        const small = document.createElement("small");
        small.style.color = "#aaa";
        small.textContent = "- " + metadata.description;
        label.append(" ", small);
      }
      div.append(input, label);
      container.appendChild(div);
    });
  }

  function loadHfModels(url) {
    const spinner = document.getElementById("hfLoadingSpinner");
    spinner.classList.remove("d-none");

    (url.includes("refresh") ? postJSON(url) : getJSON(url))
      .then((data) => {
        spinner.classList.add("d-none");
        renderHfModels(data);
        hfModelsLoaded = true;
      })
      .catch((err) => {
        spinner.classList.add("d-none");
        if (err instanceof ApiError) {
          // The routes answer a failed listing with {"error": ...} and a
          // 500; show it in the list like any other answer.
          renderHfModels({ error: err.message });
          hfModelsLoaded = true;
          return;
        }
        const placeholder = document.getElementById("hfPlaceholder");
        if (placeholder) placeholder.textContent = "Error loading models: " + err;
        hfModelsLoaded = false;
      });
  }

  // Load HF models when the accordion is expanded
  document.getElementById("collapse_hf").addEventListener("show.bs.collapse", function () {
    if (hfModelsLoaded) return;
    loadHfModels("/api/huggingface-models");
  });

  // Refresh button
  document.getElementById("hfRefreshBtn").addEventListener("click", function () {
    loadHfModels("/api/huggingface-models/refresh");
  });

  // Search bar filtering
  document.getElementById("hfSearchBar").addEventListener("input", function () {
    const query = this.value.toLowerCase();
    document.querySelectorAll(".hf-model-item").forEach(function (item) {
      const searchData = item.getAttribute("data-search");
      item.style.display = searchData.includes(query) ? "" : "none";
    });
  });

  // The BioImage Model Zoo: listed like the Hugging Face models, from the
  // zoo's index (models/bioimage_catalog.py), and filtered to EM by default.
  // The zoo models already running are ticked on load, with the voxel size
  // each was given.
  const zooRunning = new Map((pageData().default_bioimage_models || []).map((m) => [m.id, m.voxel_size]));
  let zooModelsLoaded = false;
  const ZOO_VOXEL_TITLE =
    "Voxel size in nm, z,y,x or one number: the scale the model reads the data at. Filled in with what it "
    + "was trained at when cellmap-flow knows it; blank, the model's own if its description declares one.";

  function zooTag(text, title) {
    const tag = document.createElement("span");
    tag.className = "zoo-tag";
    tag.textContent = text;
    if (title) tag.title = title;
    return tag;
  }

  // One row, built from nodes rather than an HTML string: every text in it
  // comes from the zoo's uploaders. voxelSize is the value to start with
  // (an array, a string or undefined); a row starts ticked when it is given.
  function zooRow(model, voxelSize) {
    const div = document.createElement("div");
    div.className = "form-check zoo-model-item";
    div.dataset.search = [model.name, model.key, model.id, model.description, ...model.tags].join(" ").toLowerCase();
    div.dataset.em = model.em ? "1" : "";
    div.dataset.dims = model.dims || "";

    const input = document.createElement("input");
    input.className = "form-check-input zoo-model-checkbox";
    input.type = "checkbox";
    input.id = "chk_zoo_" + model.key.replace(/\W+/g, "_");
    input.value = model.key;
    input.checked = voxelSize !== undefined;
    const label = document.createElement("label");
    label.className = "form-check-label";
    label.htmlFor = input.id;
    label.textContent = model.name;
    label.title = [model.description, model.key, model.license].filter(Boolean).join("\n");
    div.append(input, label);

    if (model.dims) div.append(zooTag(model.dims.toUpperCase()));
    // What it was trained at, from cellmap-flow's own table: no zoo model
    // says it in its description.
    const trained = model.trained || null;
    const trainedNm = trained && trained.voxel_size ? trained.voxel_size.join("\u00d7") + " nm" : "";
    if (trainedNm) div.append(zooTag(trainedNm, "Trained at (z\u00d7y\u00d7x) on " + trained.trained_on));
    model.weight_formats.forEach((format) => div.append(zooTag(format, "Weight format")));
    if (/^https?:\/\//.test(model.url || "")) {
      const link = document.createElement("a");
      link.className = "zoo-link";
      link.href = model.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.title = "Open on bioimage.io";
      link.textContent = "\u2197";
      div.append(link);
    }
    if (model.description) {
      const desc = document.createElement("div");
      desc.className = "zoo-desc";
      desc.textContent = model.description;
      desc.title = model.description;
      div.append(desc);
    }

    // Shown only while the row is ticked.
    const voxelRow = document.createElement("div");
    voxelRow.className = "zoo-voxel-row d-flex align-items-center gap-2 mt-1";
    const voxelLabel = document.createElement("label");
    voxelLabel.textContent = "Voxel (nm)";
    voxelLabel.htmlFor = input.id + "_voxel";
    const voxel = document.createElement("input");
    voxel.type = "text";
    voxel.className = "form-control form-control-sm zoo-voxel";
    voxel.id = input.id + "_voxel";
    voxel.placeholder = "from model";
    voxel.title = ZOO_VOXEL_TITLE;
    if (voxelSize === undefined && trained && trained.voxel_size) voxelSize = trained.voxel_size;
    voxel.value = Array.isArray(voxelSize) ? voxelSize.join(",") : (voxelSize || "");
    voxelRow.append(voxelLabel, voxel);
    const shown = [voxelRow];
    if (trained) {
      const hint = document.createElement("div");
      hint.className = "zoo-trained";
      hint.textContent = (trainedNm ? "Trained at " + trainedNm + " (z,y,x) on " : "Trained on ")
        + trained.trained_on + "." + (trained.note ? " " + trained.note : "")
        + (trainedNm ? "" : " Enter the voxel size of the data to run it on.");
      hint.title = trained.confidence + " confidence: " + trained.source;
      shown.push(hint);
    }
    const showTicked = () => shown.forEach((el) => { el.style.display = input.checked ? "" : "none"; });
    showTicked();
    input.addEventListener("change", showTicked);
    div.append(...shown);
    return div;
  }

  // Search words (all must match), EM only, and 2D/3D (neither ticked: any).
  // A ticked row stays in view, so nothing is submitted unseen.
  function filterZooModels() {
    const words = document.getElementById("zooSearchBar").value.toLowerCase().split(/\s+/).filter(Boolean);
    const emOnly = document.getElementById("zooEmOnly").checked;
    const dims = [["zoo2d", "2d"], ["zoo3d", "3d"]]
      .filter(([id]) => document.getElementById(id).checked)
      .map(([, d]) => d);
    const rows = document.querySelectorAll("#zooModelList .zoo-model-item");
    let shown = 0;
    rows.forEach((row) => {
      const matches = words.every((w) => row.dataset.search.includes(w))
        && (!emOnly || row.dataset.em)
        && (!dims.length || dims.includes(row.dataset.dims));
      const show = matches || row.querySelector(".zoo-model-checkbox").checked;
      row.style.display = show ? "" : "none";
      if (show) shown += 1;
    });
    document.getElementById("zooCount").textContent = rows.length ? shown + " / " + rows.length : "";
  }

  function renderZooModels(data) {
    const list = document.getElementById("zooModelList");
    // A refresh keeps what is ticked, and the voxel sizes typed.
    const ticked = new Map(zooRunning);
    list.querySelectorAll(".zoo-model-item").forEach((row) => {
      const box = row.querySelector(".zoo-model-checkbox");
      if (box.checked) ticked.set(box.value, row.querySelector(".zoo-voxel").value);
      else ticked.delete(box.value);
    });
    list.replaceChildren();
    const placeholder = document.getElementById("zooPlaceholder");
    if (placeholder) placeholder.remove();

    if (data.error) {
      const p = document.createElement("p");
      p.className = "text-danger";
      p.textContent = "Error: " + data.error;
      list.appendChild(p);
      return;
    }
    const models = data.models || [];
    if (!models.length) {
      const p = document.createElement("p");
      p.className = "text-muted";
      p.textContent = "No models found.";
      list.appendChild(p);
      return;
    }
    document.getElementById("zooControls").style.display = "";
    document.getElementById("zooRefreshBtn").title =
      "Refresh from bioimage.io" + (data.fetched ? " (listed " + data.fetched + ")" : "");
    models.forEach((model) => {
      list.appendChild(zooRow(model, ticked.has(model.key) ? ticked.get(model.key) : undefined));
    });
    filterZooModels();
  }

  // The zoo changes under the page: a list cached over an hour ago comes
  // back "stale" and is fetched again behind it, and an open page fetches it
  // again every hour. Quietly: a failure leaves the list shown as it was.
  const ZOO_REFRESH_MS = 60 * 60 * 1000;
  let zooRefreshTimer = null;

  function refreshZooQuietly() {
    postJSON("/api/bioimage-models/refresh")
      .then((data) => { if (!data.error) renderZooModels(data); })
      .catch(() => {});
  }

  function loadZooModels(refresh) {
    const spinner = document.getElementById("zooLoadingSpinner");
    spinner.classList.remove("d-none");
    (refresh ? postJSON("/api/bioimage-models/refresh") : getJSON("/api/bioimage-models"))
      .then((data) => {
        renderZooModels(data);
        zooModelsLoaded = true;
        if (data.stale) refreshZooQuietly();
        if (!zooRefreshTimer) zooRefreshTimer = setInterval(refreshZooQuietly, ZOO_REFRESH_MS);
      })
      .catch((err) => {
        // The routes answer a failed fetch of the zoo's index with
        // {"error": ...}; anything else never reached them.
        renderZooModels({ error: err instanceof ApiError ? err.message : "Error loading models: " + err });
        zooModelsLoaded = err instanceof ApiError;
      })
      .finally(() => spinner.classList.add("d-none"));
  }

  document.getElementById("collapse_zoo").addEventListener("show.bs.collapse", function () {
    if (!zooModelsLoaded) loadZooModels(false);
  });
  document.getElementById("zooRefreshBtn").addEventListener("click", () => loadZooModels(true));
  document.getElementById("zooSearchBar").addEventListener("input", filterZooModels);
  ["zooEmOnly", "zoo2d", "zoo3d"].forEach((id) => {
    document.getElementById(id).addEventListener("change", filterZooModels);
  });

  // Cellpose: Cellpose 4's models, one row each (routes/index_page
  // .cellpose_panel_data). Ticked, a row asks for the voxel size, which
  // Cellpose cannot know (it sees any scale, and segments well only where
  // objects are about 30 voxels across), and the output. The models already
  // running start ticked with their settings, one row per output running;
  // "+ output" adds a row to run another output of the same model beside.
  const cellposeData = pageData("cellpose-data");
  const CELLPOSE_OUTPUTS = [
    ["flows", "All channels", "Cellpose's three channels: flow_y and flow_x (the flows towards each cell's centre) and cell (its probability, which the layer opens on; the flows are on its channel slider). The CellposeMasksPostprocessor makes masks of them, with thresholds you can change live."],
    ["probability", "Probability only", "The cell probability, 0 to 1: one channel."],
    ["masks", "Masks", "Instance masks, made chunk by chunk: an object crossing a chunk's edge gets an id on each side."],
  ];
  const CELLPOSE_VOXEL_TITLE =
    "Voxel size in nm, z,y,x or one number: the scale Cellpose reads the data at. Required: Cellpose-SAM "
    + "segments objects about 30 voxels across best, so pick the scale at which yours are about that.";
  const CELLPOSE_STITCH_TITLE =
    "Cellpose's stitch_threshold: a mask takes the id of the mask in the slice before that it overlaps "
    + "by at least this IoU (0 to 1), within a chunk. Blank or 0: each slice's masks stay apart.";
  let cellposeRowCount = 0;

  // One row for ``model`` ({model, label, description}); ``settings``, when
  // given, are a running model's ({output, voxel_size, stitch_threshold})
  // and tick it. ``extra`` rows (from "+ output") can be removed.
  function cellposeRow(model, settings, extra) {
    cellposeRowCount += 1;
    const id = "chk_cellpose_" + cellposeRowCount;
    const div = document.createElement("div");
    div.className = "form-check cellpose-model-item";
    div.dataset.model = model.model;

    const input = document.createElement("input");
    input.className = "form-check-input cellpose-model-checkbox";
    input.type = "checkbox";
    input.id = id;
    input.value = model.model;
    input.checked = !!settings || !!extra;
    const label = document.createElement("label");
    label.className = "form-check-label";
    label.htmlFor = id;
    label.textContent = model.label;
    label.title = model.model;
    div.append(input, label, zooTag(model.model, "Cellpose's name for it (pretrained_model)"));

    const another = document.createElement("button");
    another.type = "button";
    another.className = "cellpose-row-btn";
    another.textContent = extra ? "×" : "+ output";
    another.title = extra
      ? "Remove this row (Submit then stops what it ran)"
      : "Another row for this model, to run another output beside this one";
    another.addEventListener("click", () => {
      if (extra) {
        div.remove();
        return;
      }
      // After this model's last row, with the first output none of them has.
      const rows = [...document.querySelectorAll("#cellposeModelList .cellpose-model-item")]
        .filter((row) => row.dataset.model === model.model);
      const used = rows.map((row) => row.querySelector(".cellpose-output").value);
      const free = CELLPOSE_OUTPUTS.map(([value]) => value).find((value) => !used.includes(value));
      const voxel = div.querySelector(".cellpose-voxel").value;
      const row = cellposeRow(model, null, true);
      row.querySelector(".cellpose-voxel").value = voxel;
      const select = row.querySelector(".cellpose-output");
      select.value = free || "flows";
      select.dispatchEvent(new Event("change"));
      rows[rows.length - 1].after(row);
    });
    div.append(another);

    const desc = document.createElement("div");
    desc.className = "zoo-desc";
    desc.textContent = model.description;
    desc.title = model.description;
    div.append(desc);

    // Shown only while the row is ticked.
    const settingsRow = document.createElement("div");
    settingsRow.className = "zoo-voxel-row d-flex align-items-center gap-2 mt-1 flex-wrap";
    const voxelLabel = document.createElement("label");
    voxelLabel.textContent = "Voxel (nm)";
    voxelLabel.htmlFor = id + "_voxel";
    const voxel = document.createElement("input");
    voxel.type = "text";
    voxel.className = "form-control form-control-sm zoo-voxel cellpose-voxel";
    voxel.id = id + "_voxel";
    voxel.placeholder = "required";
    voxel.title = CELLPOSE_VOXEL_TITLE;
    const voxelSize = settings && settings.voxel_size;
    voxel.value = Array.isArray(voxelSize) ? voxelSize.join(",") : (voxelSize || "");

    const outputLabel = document.createElement("label");
    outputLabel.textContent = "Output";
    outputLabel.htmlFor = id + "_output";
    const output = document.createElement("select");
    output.className = "form-select form-select-sm cellpose-output";
    output.id = id + "_output";
    CELLPOSE_OUTPUTS.forEach(([value, text, title]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      option.title = title;
      output.append(option);
    });
    output.value = (settings && settings.output) || "flows";

    // Masks only: linking each slice's masks to the slice before's.
    const stitchLabel = document.createElement("label");
    stitchLabel.textContent = "Link slices (IoU)";
    stitchLabel.htmlFor = id + "_stitch";
    stitchLabel.title = CELLPOSE_STITCH_TITLE;
    const stitch = document.createElement("input");
    stitch.type = "number";
    stitch.min = "0";
    stitch.max = "1";
    stitch.step = "0.05";
    stitch.className = "form-control form-control-sm cellpose-stitch";
    stitch.id = id + "_stitch";
    stitch.placeholder = "off";
    stitch.title = CELLPOSE_STITCH_TITLE;
    if (settings && settings.stitch_threshold) stitch.value = settings.stitch_threshold;
    settingsRow.append(voxelLabel, voxel, outputLabel, output, stitchLabel, stitch);

    const hint = document.createElement("div");
    hint.className = "zoo-trained";
    hint.textContent = "Objects about 30 voxels across work best: enter the voxel size at which yours are "
      + "about that. Runs in the cellpose4 environment.";

    const showTicked = () => {
      [settingsRow, hint].forEach((el) => { el.style.display = input.checked ? "" : "none"; });
      const masks = output.value === "masks";
      stitchLabel.style.display = masks ? "" : "none";
      stitch.style.display = masks ? "" : "none";
      output.title = (CELLPOSE_OUTPUTS.find(([value]) => value === output.value) || [])[2] || "";
    };
    showTicked();
    input.addEventListener("change", showTicked);
    output.addEventListener("change", showTicked);
    div.append(settingsRow, hint);
    return div;
  }

  function renderCellposeModels() {
    const list = document.getElementById("cellposeModelList");
    const running = cellposeData.running || [];
    (cellposeData.models || []).forEach((model) => {
      const mine = running.filter((r) => r.model === model.model);
      if (!mine.length) list.append(cellposeRow(model, null, false));
      // The first running output on the model's own row, any others on
      // extra rows below it.
      mine.forEach((settings, i) => list.append(cellposeRow(model, settings, i > 0)));
    });
    document.getElementById("cellposeRunningCount").textContent =
      running.length ? running.length + " running" : "";
  }

  renderCellposeModels();

  // The ticked Cellpose rows as POST /api/models takes them. A blank voxel
  // size is sent as null, and Submit refuses it with a message naming the
  // model, as it does a zoo model's.
  function selectedCellposeModels() {
    const selected = [];
    document.querySelectorAll("#cellposeModelList .cellpose-model-item").forEach((row) => {
      if (!row.querySelector(".cellpose-model-checkbox").checked) return;
      const output = row.querySelector(".cellpose-output").value;
      const entry = {
        model: row.dataset.model,
        voxel_size: row.querySelector(".cellpose-voxel").value.trim() || null,
        output,
      };
      const stitch = row.querySelector(".cellpose-stitch").value.trim();
      if (output === "masks" && stitch !== "") entry.stitch_threshold = stitch;
      selected.push(entry);
    });
    return selected;
  }

  // Inference job output.
  //
  // The dashboard's own log stream only carries what this process logs. A
  // server that dies on startup, or 500s on every chunk, writes its traceback
  // to the LSF job's output on whichever node it landed on -- so the only way
  // to read it used to be ssh + bpeek.
  const JOB_LOGS_POLL_MS = 5000;
  let jobLogsFollower = null;

  function renderJobLogs(data) {
    const area = document.getElementById("jobLogsArea");
    const statusEl = document.getElementById("jobLogsStatus");
    const jobs = (data && data.jobs) || [];
    if (!jobs.length) {
      area.value = "No inference jobs have been submitted yet.";
      statusEl.textContent = "";
      return;
    }
    area.value = jobs.map(function (j) {
      const header = [
        j.model_name || "(unnamed)",
        j.job_id ? "job " + j.job_id : null,
        j.status || null,
        j.host || null,
      ].filter(Boolean).join("  |  ");
      // null log means there is no way to read this job's output at all,
      // which is worth saying plainly rather than showing as empty.
      const body = j.log === null || j.log === undefined
        ? "(no output available for this job)"
        : (j.log || "(no output yet)");
      return "=== " + header + " ===\n" + body;
    }).join("\n\n");
    // Keep the newest output in view.
    area.scrollTop = area.scrollHeight;
    statusEl.textContent = "updated " + new Date().toLocaleTimeString();
  }

  // /api/job-logs runs bpeek per job, each with its own timeout, so a
  // refresh can outlast the 5s follow interval. One at a time, whether it
  // comes from Follow or from the button: overlapping requests pile onto a
  // dashboard that is already the slow part.
  let jobLogsBusy = false;

  function refreshJobLogs() {
    if (jobLogsBusy) return Promise.resolve();
    jobLogsBusy = true;
    const statusEl = document.getElementById("jobLogsStatus");
    return getJSON("/api/job-logs")
      .then(renderJobLogs)
      .catch(function (e) { statusEl.textContent = "error: " + e; })
      .then(function () { jobLogsBusy = false; });
  }

  document.getElementById("jobLogsBtn").addEventListener("click", function () {
    const area = document.getElementById("jobLogsArea");
    const showing = area.style.display !== "none";
    area.style.display = showing ? "none" : "block";
    this.textContent = showing ? "Show Job Logs" : "Hide Job Logs";
    if (!showing) refreshJobLogs();
  });

  document.getElementById("jobLogsFollow").addEventListener("change", function () {
    if (jobLogsFollower) {
      jobLogsFollower.stop();
      jobLogsFollower = null;
    }
    if (this.checked) {
      const area = document.getElementById("jobLogsArea");
      if (area.style.display === "none") {
        document.getElementById("jobLogsBtn").click();
      }
      jobLogsFollower = poll(refreshJobLogs, { intervalMs: JOB_LOGS_POLL_MS, immediate: false });
    }
  });

  // After a Submit, show Job Logs and keep it current until no job is still
  // starting (waiting in a queue, or installing its environment), even with
  // Follow off: until then the page showed nothing of what Submit started.
  // A tick resolving false stops the poll.
  let startingWatcher = null;

  function watchStartingJobs() {
    const area = document.getElementById("jobLogsArea");
    if (area.style.display === "none") document.getElementById("jobLogsBtn").click();
    if (startingWatcher) startingWatcher.stop();
    // The first ticks can come before a job is handed to LSF at all, which
    // lists nothing: only a list with no starting job in it ends the watch.
    let ticks = 0;
    startingWatcher = poll(function () {
      if (jobLogsFollower) return Promise.resolve(false);  // Follow keeps it current
      ticks += 1;
      return getJSON("/api/job-logs").then(function (data) {
        renderJobLogs(data);
        const jobs = (data && data.jobs) || [];
        const starting = jobs.some(function (j) { return j.status === "starting"; });
        return starting || (jobs.length === 0 && ticks < 6) ? undefined : false;
      }).catch(function () { return false; });
    }, { intervalMs: JOB_LOGS_POLL_MS, maxTicks: 60 });
  }

  // GPU queue picker: the one next to Submit and the Server Config one show
  // the same configured queue.
  let gpuQueueCurrent = "";

  function gpuQueueSelects() {
    return ["gpuQueueSelect", "cfg_queue"]
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean);
  }

  function renderGpuQueues(data) {
    gpuQueueSelects().forEach(function (sel) {
      fillQueueSelect(sel, data, { preferred: gpuQueueCurrent });
    });
    const hint = document.getElementById("gpuQueueHint");
    if (hint) hint.textContent = queueHint(data);
  }

  function setGpuQueue(value) {
    if (!value) return;
    gpuQueueCurrent = value;
    gpuQueueSelects().forEach(function (sel) {
      if (sel.value !== value) sel.value = value;
    });
    postJSON("/api/server-config", { queue: value }).catch(function () {});
  }

  gpuQueueSelects().forEach(function (sel) {
    sel.addEventListener("change", function () { setGpuQueue(sel.value); });
  });

  // Seed from the configured queue before drawing options, so an out-of-list
  // queue survives the first render.
  getJSON("/api/server-config")
    .then(function (data) { gpuQueueCurrent = (data && data.queue) || ""; })
    .catch(function () {})
    .then(function () { subscribeGpuQueues(renderGpuQueues); });

  // Server Config: load values when accordion opens
  let serverConfigLoaded = false;
  document.getElementById("collapse_server_config").addEventListener("show.bs.collapse", function () {
    if (serverConfigLoaded) return;
    loadServerConfig();
  });

  function loadServerConfig() {
    getJSON("/api/server-config")
      .then((data) => {
        // Go through the shared value so the accordion and the picker next
        // to Submit cannot drift apart.
        gpuQueueCurrent = data.queue || gpuQueueCurrent;
        gpuQueueSelects().forEach(function (sel) {
          if (gpuQueueCurrent) sel.value = gpuQueueCurrent;
        });
        document.getElementById("cfg_charge_group").value = data.charge_group || "";
        document.getElementById("cfg_walltime").value = data.walltime || "";
        // Default to on when the server has not stored a preference.
        document.getElementById("cfg_cycle_gpu_queues").checked =
          data.cycle_gpu_queues !== false;
        document.getElementById("cfg_nb_cores_worker").value = data.nb_cores_worker || "";
        document.getElementById("cfg_nb_workers").value = data.nb_workers || "";
        serverConfigLoaded = true;
      })
      .catch(function (err) {
        const statusEl = document.getElementById("serverConfigStatus");
        statusEl.style.color = "#f87171";
        statusEl.textContent = "Could not load the server config: " + err;
      });
  }

  document.getElementById("updateServerConfigBtn").addEventListener("click", function () {
    const statusEl = document.getElementById("serverConfigStatus");
    function fail(message) {
      statusEl.style.color = "#f87171";
      statusEl.textContent = message;
    }
    // The text fields are sent even when blank (blank walltime means the
    // queue default), so before they hold the server's values an Update
    // would wipe the charge group and time limit.
    if (!serverConfigLoaded) {
      fail("Still loading the current config, try again.");
      loadServerConfig();
      return;
    }
    const walltime = document.getElementById("cfg_walltime").value.trim();
    if (walltime && !WALLTIME_RE.test(walltime)) {
      fail("Time Limit must be HH:MM or a number of minutes.");
      return;
    }
    const payload = {
      charge_group: document.getElementById("cfg_charge_group").value.trim(),
      walltime: walltime,
      // Send a real boolean: the server stores this verbatim, and the
      // string "false" would read as true.
      cycle_gpu_queues: document.getElementById("cfg_cycle_gpu_queues").checked,
    };
    // Empty until LSF has listed the queues; don't send that as a queue.
    const queue = document.getElementById("cfg_queue").value;
    if (queue) payload.queue = queue;
    try {
      const cores = readCount(document.getElementById("cfg_nb_cores_worker").value, "Cores per Worker");
      const workers = readCount(document.getElementById("cfg_nb_workers").value, "Number of Workers");
      if (cores !== undefined) payload.nb_cores_worker = cores;
      if (workers !== undefined) payload.nb_workers = workers;
    } catch (err) {
      fail(err.message);
      return;
    }
    saveServerConfig(payload)
      .then(function () {
        statusEl.style.color = "#4ade80";
        statusEl.textContent = "Saved!";
        setTimeout(function () { statusEl.textContent = ""; }, 3000);
      })
      .catch(function (err) {
        fail("Error: " + (err instanceof ApiError ? err.message : err));
      });
  });

  // Add a model: resolve what was pasted (POST /api/models/resolve), ask for
  // what it still needs, and run it (POST /api/models/add). The model then
  // gets a ticked box in the catalog's list, so Submit keeps it running.
  const addRef = document.getElementById("addModelRef");
  const addResult = document.getElementById("addModelResult");
  const addFields = document.getElementById("addModelFields");
  const addRunBtn = document.getElementById("addModelRunBtn");
  let resolved = null;

  // How a needed value is asked for: its label and an example.
  const NEEDED = {
    voxel_size: ["Voxel (nm)", "8, or 40,4,4"],
    input_voxel_size: ["Input voxel (nm)", "8, or 40,4,4"],
    channels: ["Channels", "mito, er"],
    base_model: ["Base model", '{"type": ...}'],
  };

  function field(label, key, value, title, placeholder) {
    const id = "addModel_" + key;
    const lab = document.createElement("label");
    lab.htmlFor = id;
    lab.textContent = label;
    const input = document.createElement("input");
    input.className = "form-control form-control-sm";
    input.id = id;
    input.dataset.key = key;
    input.value = value || "";
    if (title) input.title = title;
    if (placeholder) input.placeholder = placeholder;
    addFields.append(lab, input);
  }

  // A typed value as the entry wants it: numbers and lists of numbers as
  // such, a JSON object (a finetune's base_model) parsed, else the text.
  function typed(text) {
    const value = text.trim();
    if (value.startsWith("{")) return JSON.parse(value);
    const parts = value.split(/[\s,]+/).filter(Boolean);
    if (parts.length && parts.every((p) => /^-?\d+(\.\d+)?$/.test(p))) {
      const numbers = parts.map(Number);
      return numbers.length === 1 ? numbers[0] : numbers;
    }
    return value;
  }

  function showResolved(d) {
    resolved = d;
    addResult.hidden = false;
    addFields.replaceChildren();
    const env = d.env ? `runs in ${d.env}` : "runs in this environment";
    document.getElementById("addModelSummary").textContent = `${d.type}: ${d.how} (${env})`;
    document.getElementById("addModelNotes").textContent = (d.notes || []).join(" ");
    field("Name", "name", d.name, "The model's name: its layer and job are called so.");
    (d.needs || []).forEach((key) => {
      const [label, example] = NEEDED[key]
        || [key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, " "), ""];
      field(label, key, "", "Not known from the reference: give it here (numbers as 8 or 40,4,4).", example);
    });
  }

  function addedBox(name) {
    const div = document.createElement("div");
    div.className = "form-check mb-1";
    const input = document.createElement("input");
    input.className = "form-check-input model-checkbox";
    input.type = "checkbox";
    input.value = name;
    input.id = "chk_added_" + name;
    input.checked = true;
    const label = document.createElement("label");
    label.className = "form-check-label";
    label.htmlFor = input.id;
    label.textContent = name;
    div.append(input, label);
    document.getElementById("addedModels").appendChild(div);
  }

  document.getElementById("addModelResolveBtn").addEventListener("click", function () {
    const ref = addRef.value.trim();
    if (!ref) return;
    postJSON("/api/models/resolve", { ref })
      .then(showResolved)
      .catch((err) => {
        addResult.hidden = true;
        alert("Could not resolve that model: " + (err instanceof ApiError ? err.message : err));
      });
  });

  addRunBtn.addEventListener("click", function () {
    if (!resolved) return;
    const entry = { type: resolved.type, ...resolved.params, name: resolved.name };
    try {
      addFields.querySelectorAll("input").forEach((input) => {
        if (input.value.trim() !== "") entry[input.dataset.key] = typed(input.value);
      });
    } catch (e) {
      alert("Could not read a value: " + e.message);
      return;
    }
    const missing = (resolved.needs || []).filter((key) => entry[key] === undefined);
    if (missing.length) {
      alert("Still needed: " + missing.join(", "));
      return;
    }
    postJSON("/api/models/add", { entry })
      .then((d) => {
        addedBox(d.name);
        addResult.hidden = true;
        addRef.value = "";
        logArea.value += `Added ${d.name}: starting its server\n`;
        if (onModelsSubmitted) onModelsSubmitted();
      })
      .catch((err) => alert("Could not add that model: " + (err instanceof ApiError ? err.message : err)));
  });

  submitBtn.addEventListener("click", function () {
    // Gather checked local catalog models
    const checkedLocal = document.querySelectorAll("#modelSelectionForm input.model-checkbox:checked");
    const selected = [];
    checkedLocal.forEach((checkbox) => {
      selected.push(checkbox.value);
    });

    // Gather checked HF models
    const checkedHf = document.querySelectorAll("#modelSelectionForm input.hf-model-checkbox:checked");
    const selectedHf = [];
    checkedHf.forEach((checkbox) => {
      selectedHf.push(checkbox.value);
    });

    // Ticked zoo models, with the voxel size typed (blank: the model's own).
    const selectedZoo = [];
    document.querySelectorAll("#zooModelList .zoo-model-item").forEach((row) => {
      const box = row.querySelector(".zoo-model-checkbox");
      if (!box.checked) return;
      const voxel = row.querySelector(".zoo-voxel").value.trim();
      selectedZoo.push({ id: box.value, voxel_size: voxel || null });
    });

    const selectedCellpose = selectedCellposeModels();

    console.log("Selected models:", selected, "HF models:", selectedHf, "zoo models:", selectedZoo,
                "Cellpose models:", selectedCellpose);
    postJSON("/api/models", {
      selected_models: selected,
      selected_hf_models: selectedHf,
      selected_bioimage_models: selectedZoo,
      selected_cellpose_models: selectedCellpose,
      resample: resampleCheckbox.checked,
    })
      .then((data) => {
        console.log("Server response:", data);
        const started = [...(data.models || []), ...(data.hf_models || []),
                         ...(data.bioimage_models || []).map((m) => m.id),
                         ...(data.cellpose_models || []).map((m) => m.model + " (" + m.output + ")")];
        logArea.value += started.length
          ? `Submitted ${started.join(", ")}: starting (see Job Logs)\n`
          : "Submitted: no model selected; any running ones are stopped\n";
        logArea.scrollTop = logArea.scrollHeight;
        if (started.length) watchStartingJobs();
        if (onModelsSubmitted) onModelsSubmitted();
      })
      .catch((err) => {
        console.error("Error:", err);
        const message = err instanceof ApiError ? err.message : String(err);
        logArea.value += `Not submitted: ${message}\n`;
        alert("Could not submit: " + message);
      });
  });
}
