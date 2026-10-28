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
// merge of the payloads, and a ✕ tells every list.
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

  // One poll for both, asking only while either has something starting.
  poll(() => Promise.all([loadedModels.starting() && loadedModels.refresh(), logs.watching() && logs.look()]),
       { intervalMs: jobLogs.POLL_MS, immediate: false, pauseWhenHidden: true });

  document.getElementById("submitModelsBtn").addEventListener("click", () => {
    const body = Object.assign({ resample: resampleCheckbox.checked }, ...LISTS.map((list) => list.payload()));
    console.log("Selected models:", body);
    postJSON("/api/models", body)
      .then((data) => {
        console.log("Server response:", data);
        const started = [...(data.models || []), ...(data.hf_models || []),
                         ...(data.bioimage_models || []).map((m) => m.id),
                         ...(data.cellpose_models || []).map((m) => m.model + " (" + m.output + ")")];
        log(started.length
          ? `Submitted ${started.join(", ")}: starting (see Job Logs)`
          : "Submitted: no model selected; any running ones are stopped");
        if (started.length) logs.watchStartingJobs();
        announce(MODELS_CHANGED_EVENT);
      })
      .catch((err) => {
        console.error("Error:", err);
        const message = err instanceof ApiError ? err.message : String(err);
        log(`Not submitted: ${message}`);
        alert("Could not submit: " + message);
      });
  });
}
