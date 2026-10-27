// The GPU queue picker next to Submit and the Server Config accordion: both
// show the configured queues (GET/POST /api/server-config), the accordion
// the LSF settings beside them.
import { ApiError, getJSON, postJSON } from "../../lib/api.js";
import { createQueuePicker, queueHint, subscribeGpuQueues } from "../../shared/gpu-queues.js";
import { readCount, saveServerConfig, WALLTIME_RE } from "../../shared/server-config.js";

export function init() {
  let loaded = false;

  // The two pickers are one setting: a box clicked in either is saved at
  // once, and shown in the other.
  function saveQueues(queues) {
    showQueues(queues);
    postJSON("/api/server-config", { queue: queues }).catch(() => {});
  }
  const pickers = ["gpuQueuePicker", "cfg_queue"]
    .map((id) => document.getElementById(id))
    .filter(Boolean)
    .map((root) => createQueuePicker(root, { small: true, onChange: saveQueues }));

  function showQueues(queues) {
    pickers.forEach((picker) => picker.setValue(queues));
  }

  function renderQueues(data) {
    pickers.forEach((picker) => picker.render(data));
    const hint = document.getElementById("gpuQueueHint");
    if (hint) hint.textContent = queueHint(data);
  }

  // The configured queues before the first answer is drawn, so a queue LSF
  // does not list survives the first render.
  getJSON("/api/server-config")
    .then((data) => showQueues(data && data.queue))
    .catch(() => {})
    .then(() => subscribeGpuQueues(renderQueues));

  function load() {
    getJSON("/api/server-config")
      .then((data) => {
        // Both pickers, so the accordion and the picker next to Submit
        // cannot drift apart.
        if (data.queue) showQueues(data.queue);
        document.getElementById("cfg_charge_group").value = data.charge_group || "";
        document.getElementById("cfg_walltime").value = data.walltime || "";
        // Default to on when the server has not stored a preference.
        document.getElementById("cfg_cycle_gpu_queues").checked = data.cycle_gpu_queues !== false;
        document.getElementById("cfg_nb_cores_worker").value = data.nb_cores_worker || "";
        document.getElementById("cfg_nb_workers").value = data.nb_workers || "";
        loaded = true;
      })
      .catch((err) => {
        const statusEl = document.getElementById("serverConfigStatus");
        statusEl.style.color = "#f87171";
        statusEl.textContent = "Could not load the server config: " + err;
      });
  }

  document.getElementById("collapse_server_config").addEventListener("show.bs.collapse", () => {
    if (!loaded) load();
  });

  document.getElementById("updateServerConfigBtn").addEventListener("click", () => {
    const statusEl = document.getElementById("serverConfigStatus");
    function fail(message) {
      statusEl.style.color = "#f87171";
      statusEl.textContent = message;
    }
    // The text fields are sent even when blank (blank walltime means the
    // queue default), so before they hold the server's values an Update
    // would wipe the charge group and time limit.
    if (!loaded) {
      fail("Still loading the current config, try again.");
      load();
      return;
    }
    const walltime = document.getElementById("cfg_walltime").value.trim();
    if (walltime && !WALLTIME_RE.test(walltime)) {
      fail("Time Limit must be HH:MM or a number of minutes.");
      return;
    }
    const payload = {
      charge_group: document.getElementById("cfg_charge_group").value.trim(),
      walltime,
      // Send a real boolean: the server stores this verbatim, and the
      // string "false" would read as true.
      cycle_gpu_queues: document.getElementById("cfg_cycle_gpu_queues").checked,
    };
    // Empty until LSF has listed the queues; don't send that as the queues.
    const queues = pickers.length ? pickers[0].value() : [];
    if (queues.length) payload.queue = queues;
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
      .then(() => {
        statusEl.style.color = "#4ade80";
        statusEl.textContent = "Saved!";
        setTimeout(() => { statusEl.textContent = ""; }, 3000);
      })
      .catch((err) => fail("Error: " + (err instanceof ApiError ? err.message : err)));
  });
}
