// The dashboard page (templates/index.html and the tab partials it
// includes).
//
// A module runs once the page is parsed and before DOMContentLoaded, so the
// page's elements are all there.
import { ApiError, postJSON } from "../lib/api.js";
import { mountOpChain } from "../shared/op-chain.js";
import { readCount, saveServerConfig } from "../shared/server-config.js";
import { initConnect } from "./connect.js";
import { initFinetuneTab } from "./finetune/index.js";
import { initModelAdvice, refreshModelAdvice } from "./model-advice.js";
import { initModelsTab } from "./models-tab.js";
import { initReviewTab } from "./review/index.js";

// The header's "Toggle Dashboard" button: the dashboard column the other way
// from how the page's layout starts it. templates/index.html does the rest
// from the class: beside the viewer the column starts shown and this hides
// it (the viewer's column fills the width it leaves); on a phone it starts
// hidden and this shows it over the viewer.
function toggleDashboard() {
  document.body.classList.toggle("dashboard-toggled");
}

// The browser mode calls this once it has loaded this script (web/src/main.ts),
// when the chain its backend serves is not the one this page was rendered
// with: after a reload of the tab, or from a shared link. `chain` is
// {input_norm, postprocess}, each [{name, ...params}] with the values as the
// form sends them; the lists then show it, ticked and in order, as the Flask
// dashboard shows the session's chain by rendering it into the page.
export function showChain(chain) {
  [[inputChain, chain.input_norm], [postChain, chain.postprocess]].forEach(([opChain, steps]) => {
    if (!Array.isArray(steps)) return;
    const params = {};
    steps.forEach((step) => { params[step.name] = step; });
    opChain.apply(steps.map((step) => step.name), params);
  });
}

// Enter in a parameter field of the Input or Postprocess list submits the
// pipeline, like Submit All. Only there: every submit rebuilds all the
// viewer's layers, so Enter anywhere else -- a newline in a textarea, the
// output path, a search box, a modal -- must not trigger one, and Enter on a
// focused button already clicks it.
const SUBMIT_ON_ENTER_SKIP_TYPES = ["checkbox", "radio", "button", "submit", "reset", "file"];

function submitOnEnter(event) {
  if (event.key !== "Enter" || event.isComposing) return;
  const target = event.target;
  if (!target || target.tagName !== "INPUT") return;
  if (SUBMIT_ON_ENTER_SKIP_TYPES.indexOf(target.type) !== -1) return;
  const form = target.closest("#inputNormForm, #postProcessForm");
  if (!form) return;
  event.preventDefault();
  const button = form.querySelector("#submitAll");
  if (button) button.click();
}

// Submit All, in each of the Input and Postprocess tabs: PUT both chains to
// /api/pipeline, which sets them and redraws the viewer's layers.
function initSubmitAll(inputChain, postChain) {
  function handleSubmitAll() {
    const finalPayload = {
      input_norm: inputChain.getChain(),
      postprocess: postChain.getChain(),
    };
    console.log("Combined Payload:", finalPayload);
    postJSON("/api/pipeline", finalPayload, { method: "PUT" })
      .then((data) => {
        console.log("Server response:", data);
        ["submissionLog_inputNorm", "submissionLog_postProcess"].forEach((id) => {
          const logArea = document.getElementById(id);
          if (logArea) logArea.value += "Server response:\n" + JSON.stringify(data, null, 2) + "\n";
        });
        refreshModelAdvice();
      })
      .catch((err) => {
        console.error("Error:", err);
        // A chain the server refused comes back with why (a parameter its
        // op's class does not take, say); no answer, or an error page, does not.
        const why = err instanceof ApiError && err.body && err.body.error ? ": " + err.body.error : "";
        alert("Error submitting combined data" + why);
      });
  }
  // Both partials' buttons have id="submitAll".
  document.querySelectorAll("#submitAll").forEach((btn) => {
    btn.addEventListener("click", handleSubmitAll);
  });
}

// The first-run dialog, on the page only until a server config is saved.
function initServerConfigModal() {
  const modalEl = document.getElementById("serverConfigModal");
  if (!modalEl) return;
  const modal = new bootstrap.Modal(modalEl);
  modal.show();

  document.getElementById("modalSaveConfigBtn").addEventListener("click", function () {
    const statusEl = document.getElementById("modalConfigStatus");
    function fail(message) {
      statusEl.style.color = "#f87171";
      statusEl.textContent = message;
    }
    // A blank count or queue is left out so the server keeps its default.
    const payload = {
      charge_group: document.getElementById("modal_charge_group").value.trim(),
    };
    const queue = document.getElementById("modal_queue").value.trim();
    if (queue) payload.queue = queue;
    const counts = [
      ["modal_nb_cores_worker", "nb_cores_worker", "Cores per Worker"],
      ["modal_nb_workers", "nb_workers", "Number of Workers"],
    ];
    try {
      for (const [id, key, label] of counts) {
        const n = readCount(document.getElementById(id).value, label);
        if (n !== undefined) payload[key] = n;
      }
    } catch (err) {
      fail(err.message);
      return;
    }
    saveServerConfig(payload)
      .then(function () {
        statusEl.style.color = "#4ade80";
        statusEl.textContent = "Saved!";
        setTimeout(function () { modal.hide(); }, 500);
      })
      .catch(function (err) {
        fail(err instanceof ApiError ? "Error saving config: " + err.message : "Error: " + err);
      });
  });
}

initConnect();
// Loading a model is the point at which advice is most useful, and that
// path never goes through Submit. The jobs start on background
// threads, so refreshModelAdvice polls rather than assuming the server is
// already up.
initModelsTab({ onModelsSubmitted: refreshModelAdvice });
document.getElementById("toggleDashboardBtn").addEventListener("click", toggleDashboard);
const inputChain = mountOpChain(document.getElementById("inputNormList"), { kind: "input" });
const postChain = mountOpChain(document.getElementById("postProcessList"), { kind: "postprocess" });
initModelAdvice({ input: inputChain, postprocess: postChain });
document.addEventListener("keydown", submitOnEnter);
initSubmitAll(inputChain, postChain);
initServerConfigModal();
initReviewTab();
// The Finetune tab starts last, so its first requests go after the other
// tabs' (the Models tab asks for the GPU queues only once the server config
// has answered): on DOMContentLoaded while the page is still being parsed,
// as its inline script did, else at once. The browser mode loads this
// script long after parsing (web/src/main.ts imports it once its backend
// listens, while the viewer's iframe may still be loading), when that event
// has fired and would never come; "complete" is not the test, since the
// page's load event waits for the iframe.
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => initFinetuneTab());
else initFinetuneTab();
