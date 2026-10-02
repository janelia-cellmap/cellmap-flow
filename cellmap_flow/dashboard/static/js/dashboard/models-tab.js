// The Models tab (templates/_models_tab.html): the local catalog and
// Hugging Face models to serve, the LSF server config, the GPU queue picker,
// and the inference jobs' own output.
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

    console.log("Selected models:", selected, "HF models:", selectedHf);
    postJSON("/api/models", {
      selected_models: selected,
      selected_hf_models: selectedHf,
      resample: resampleCheckbox.checked,
    })
      .then((data) => {
        console.log("Server response:", data);
        logArea.value += "Server response:\n" + JSON.stringify(data, null, 2) + "\n";
        if (onModelsSubmitted) onModelsSubmitted();
      })
      .catch((err) => {
        console.error("Error:", err);
        alert("Error submitting model selection" + err);
      });
  });
}
