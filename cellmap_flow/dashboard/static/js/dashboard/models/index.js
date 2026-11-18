// The Models tab (templates/_models_tab.html): the models loaded, each with
// a ✕ to stop it (loaded.js); the lists to pick models from, the local
// catalog (catalog.js), Hugging Face (huggingface.js), the BioImage Model
// Zoo (zoo.js) and Cellpose (cellpose.js); Add a model (add-model.js); the
// LSF server config and GPU queue picker (server-config.js); and the
// inference jobs' own output (job-logs.js). The catalog's, Hugging Face's
// and the zoo's sections, their rows and Add a model are made of the pieces
// every model list is (shared/model-list.js), the Finetune tab's ranking's
// too.
//
// Each list exports payload(), its part of POST /api/models, and
// untick(model), which unticks a loaded model's row in it: Submit is the
// merge of the payloads (submitBody), and a ✕ tells every list.
import { ApiError, postJSON } from "../../lib/api.js";
import { announce, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { pageData } from "../../lib/page-data.js";
import { poll } from "../../lib/poll.js";
import * as addModel from "./add-model.js";
import * as catalog from "./catalog.js";
import * as cellpose from "./cellpose.js";
import * as huggingface from "./huggingface.js";
import * as jobLogs from "./job-logs.js";
import * as loaded from "./loaded.js";
import * as serverConfig from "./server-config.js";
import * as zoo from "./zoo.js";

const LISTS = [catalog, huggingface, zoo, cellpose];

// POST /api/models' body: the lists' `payloads` in one, with Resample. The
// catalog's and Hugging Face's each have their own run_voxel_sizes, merged
// here into one rather than the second taking the first's place.
export function submitBody(payloads, resample) {
  const runVoxelSizes = Object.assign({}, ...payloads.map((payload) => payload.run_voxel_sizes));
  return Object.assign({ resample }, ...payloads, { run_voxel_sizes: runVoxelSizes });
}

export function initModelsTab() {
  const logArea = document.getElementById("modelSubmissionLogModels");
  const log = (line) => {
    logArea.value += line + "\n";
    logArea.scrollTop = logArea.scrollHeight;
  };
  // The session's Resample setting (a YAML's resample, view --resample).
  const resampleCheckbox = document.getElementById("resampleCheckbox");
  resampleCheckbox.checked = !!pageData().resample;

  catalog.init();
  huggingface.init();
  zoo.init();
  cellpose.init();
  addModel.init({ log });
  serverConfig.init();
  const logs = jobLogs.init();
  const loadedModels = loaded.init({ untick: (model) => LISTS.forEach((list) => list.untick(model)), log });

  // One poll for both: the loaded models always, as a model can stop or
  // start with nothing on this page doing it (its layer deleted in the
  // viewer, its server up after minutes in a queue), the job logs while one
  // is starting.
  poll(() => Promise.all([loadedModels.refresh(), logs.watching() && logs.look()]),
       { intervalMs: jobLogs.POLL_MS, immediate: false, pauseWhenHidden: true });

  const submit = (body) => postJSON("/api/models", body)
    .then((data) => {
      console.log("Server response:", data);
      const started = [...(data.models || []), ...(data.hf_models || []),
                       ...(data.bioimage_models || []).map((m) => m.id),
                       ...(data.cellpose_models || []).map((m) => m.model + " (" + m.output + ")")];
      const installing = (data.installing || []).map((env) => `"${env}"`);
      log(!started.length
        ? "Submitted: no model selected; any running ones are stopped"
        : installing.length
          ? `Submitted ${started.join(", ")}: installing the pixi environment ${installing.join(", ")} first ` +
            "(once; several minutes), then starting (see Job Logs)"
          : `Submitted ${started.join(", ")}: starting (see Job Logs)`);
      if (started.length) logs.watchStartingJobs();
      announce(MODELS_CHANGED_EVENT);
    })
    .catch((err) => {
      console.error("Error:", err);
      const message = err instanceof ApiError ? err.message : String(err);
      // A model whose pixi environment is not installed yet: nothing was
      // started, and the same Submit, sent again agreeing to install it,
      // installs it before that model's job.
      const needs = err instanceof ApiError && err.status === 409 ? err.body?.needs_install : null;
      if (needs?.length) {
        if (confirm(installQuestion(needs))) return submit({ ...body, install_envs: needs.map((n) => n.env) });
        log(`Not submitted: ${message}`);
        return;
      }
      log(`Not submitted: ${message}`);
      alert("Could not submit: " + message);
    });

  document.getElementById("submitModelsBtn").addEventListener("click", () => {
    const body = submitBody(LISTS.map((list) => list.payload()), resampleCheckbox.checked);
    console.log("Selected models:", body);
    submit(body);
  });
}

// The question a Submit asks before installing the pixi environments its
// models need: `needs` is the 409's needs_install, [{env, models}].
export function installQuestion(needs) {
  const lines = needs.map(({ env, models }) =>
    `${models.join(", ")} ${models.length === 1 ? "runs" : "run"} in the pixi environment "${env}", ` +
    "which is not installed yet.");
  const it = needs.length === 1 ? "it" : "them";
  return [...lines, "",
    `Install ${it} now? It takes several minutes, and only once: ${it === "it" ? "it is" : "they are"} ` +
    "installed beside this cellmap-flow, for everyone who runs it. Job Logs shows the progress, and the " +
    "model starts when it is done."]
    .join("\n");
}
