// The dashboard page (templates/index.html and the tab partials it
// includes).
//
// A module runs once the page is parsed and before DOMContentLoaded, so the
// page's elements are all there.
import { ApiError, getJSON, postJSON } from "../lib/api.js";
import { CHAIN_CHANGED_EVENT, listen, MODELS_CHANGED_EVENT } from "../lib/events.js";
import { createQueuePicker, subscribeGpuQueues } from "../shared/gpu-queues.js";
import { chainChanges, mountOpChain } from "../shared/op-chain.js";
import { readCount, saveServerConfig } from "../shared/server-config.js";
import { initConnect } from "./connect.js";
import { initFinetuneTab } from "./finetune/index.js";
import { initModelAdvice, refreshModelAdvice } from "./model-advice.js";
import { initModelsTab } from "./models/index.js";
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
// dashboard shows the session's chain by rendering it into the page. It is
// the chain the backend runs, so nothing is left to apply.
export function showChain(chain) {
  applyBar.show(chain);
}

// Enter in a parameter field of the Processing tab's lists applies the chain,
// like the Apply button. Only there: every apply rebuilds all the viewer's
// layers, so Enter anywhere else -- a newline in a textarea, the output path,
// a search box, a modal -- must not trigger one, and Enter on a focused
// button already clicks it.
const APPLY_ON_ENTER_SKIP_TYPES = ["checkbox", "radio", "button", "submit", "reset", "file"];

function applyOnEnter(event, apply) {
  if (event.key !== "Enter" || event.isComposing) return;
  const target = event.target;
  if (!target || target.tagName !== "INPUT") return;
  if (APPLY_ON_ENTER_SKIP_TYPES.indexOf(target.type) !== -1) return;
  if (!target.closest("#inputNormForm, #postProcessForm")) return;
  event.preventDefault();
  apply();
}

// Postprocessing steps in a few words: "Sigmoid → Threshold 0.076".
function postText(steps) {
  return steps.map((step) => step.name.replace(/Postprocessor$/, "")
    + (step.name === "ThresholdPostprocessor" && step.threshold !== undefined ? ` ${step.threshold}` : "")).join(" → ");
}

// A notice of the Processing tab (templates/_processing_tab.html): `text`,
// or nothing (null hides it).
function say(notice, text) {
  notice.querySelector(".chain-notice-text").textContent = text || "";
  notice.hidden = !text;
}

// The Apply bar at the top of the Processing tab
// (templates/_processing_tab.html), and the chain the backend runs.
//
// The bar counts the edits of the tab's two lists that the backend does not
// run (op-chain.js pendingChanges, both lists together) and lights its Apply
// while there are any, and each list's sub-tab carries its own count, so an
// edit in the list not shown is not out of sight; Apply sends both lists
// with PUT /api/pipeline, which sets them and redraws the viewer's layers.
// Nothing else here sends a chain: a suggestion's Apply (model-advice.js)
// fills a list in and calls this Apply.
//
// The backend can change its chain itself: a model's start sets the input
// normalization the model declares, or its kind of model's convention, when
// the chain had none (routes/pipeline.py set_declared_input_norm,
// web/src/backend/serving.ts setDeclaredInputNorm), and the postprocessing
// its output needs, a sigmoid for logits (set_needed_postprocess: once its
// server answers, which the model advice's poll sees as the chain's digest
// moving, and says with CHAIN_CHANGED_EVENT); and Use this model sets what
// the ranking fitted. So after every model change (MODELS_CHANGED_EVENT) the
// chain is read again, from GET /api/models/loaded's `pipeline`, and a list
// whose chain the backend changed shows the new one, as applied, with a
// notice above it saying so.
function initApply(inputChain, postChain) {
  const button = document.getElementById("applyButton");
  const pending = document.getElementById("applyPending");
  const refused = document.getElementById("applyNotice");
  const parts = [
    { key: "input_norm", chain: inputChain, name: "input normalization",
      notice: document.getElementById("inputNormNotice"), badge: document.getElementById("inputNormPending") },
    { key: "postprocess", chain: postChain, name: "postprocessing",
      notice: document.getElementById("postProcessNotice"), badge: document.getElementById("postProcessPending") },
  ];
  let applying = false;
  // The chain the backend runs, as its answers give it ({input_norm,
  // postprocess}): PUT /api/pipeline's, and GET /api/models/loaded's, read
  // once when the page loads and after every model change. The next answer
  // is compared with it to tell a chain the backend changed itself from one
  // this page applied; null until the first answer.
  let live = null;
  // The requests that read or set the chain, one after another, so an
  // answer is never compared with one sent after it.
  let queue = Promise.resolve();
  function inOrder(request) {
    const answer = queue.then(request);
    queue = answer.catch(() => {});
    return answer;
  }

  function render() {
    const counts = parts.map((part) => part.chain.pendingChanges());
    const total = counts[0] + counts[1];
    // Each sub-tab's badge: its list's count, none at 0 (Bootstrap hides an
    // empty badge).
    parts.forEach((part, i) => { part.badge.textContent = counts[i] || ""; });
    button.disabled = applying;
    button.textContent = applying ? "Applying…" : "Apply";
    button.classList.toggle("btn-primary", total > 0);
    button.classList.toggle("btn-outline-secondary", total === 0);
    pending.textContent = `${total} change${total === 1 ? "" : "s"} not applied`;
    pending.classList.toggle("text-warning", total > 0);
    pending.hidden = total === 0;
  }

  function apply() {
    if (applying) return;
    const sent = { input_norm: inputChain.getChain(), postprocess: postChain.getChain() };
    applying = true;
    render();
    console.log("Applying the chain:", sent);
    inOrder(() => postJSON("/api/pipeline", sent, { method: "PUT" }))
      .then((data) => {
        console.log("Server response:", data);
        // What was sent, not the lists as they are now: an edit made while
        // the request was out is not applied.
        parts.forEach((part) => {
          part.chain.markApplied(sent[part.key]);
          say(part.notice, null);
        });
        say(refused, null);
        live = data.pipeline || live;
        refreshModelAdvice();
      })
      .catch((err) => {
        console.error("Error:", err);
        // A chain the server refused comes back with why (a parameter its
        // op's class does not take, say); no answer, or an error page, does not.
        const why = err instanceof ApiError && err.body && err.body.error ? ": " + err.body.error : "";
        say(refused, "Could not apply the chain" + why);
      })
      .finally(() => {
        applying = false;
        render();
      });
  }

  // `pipeline` is the backend's chain as an answer gives it, `setBy` the
  // model whose start set its input normalization (or null), and `declared`
  // whether that model declares it (false: the convention for its kind of
  // model, which the Flask dashboard sets for a CellMap model that declares
  // none); `postBy` the model whose settings its postprocessing is (or
  // null), and `fitted` whether the ranking fitted them to the paint (false:
  // the model's start set the sigmoid its logits need). A list whose chain
  // changed shows the new one, as applied, and its notice says why; an edit
  // in it not applied yet is replaced, while the other list keeps its edits.
  function showLive(pipeline, setBy, declared, postBy, fitted) {
    if (!pipeline) return;
    const before = live;
    live = pipeline;
    // The first answer: the page was rendered with this chain.
    if (before === null) return;
    parts.forEach((part) => {
      const steps = pipeline[part.key];
      if (!Array.isArray(steps) || chainChanges(before[part.key], steps) === 0) return;
      part.chain.show(steps);
      part.chain.markApplied();
      let why = `The ${part.name} was changed outside this tab (a model's start, Use this model or Fit to Paint); the list shows it as it runs now.`;
      if (part.key === "input_norm" && setBy) {
        why = declared
          ? `Input normalization set from ${setBy}'s declaration when it started.`
          : `Input normalization set to the CellMap convention, [-1, 1], when ${setBy} started: the model does not declare one. Change it here if the model was trained on another range.`;
      }
      if (part.key === "postprocess" && postBy) {
        why = fitted
          ? `Postprocessing fitted to your paint by the ranking for ${postBy}: ${postText(steps)}.`
          : `Postprocessing set when ${postBy} started: its output is logits, so ${postText(steps)} turns it into probabilities.`;
      }
      say(part.notice, why);
    });
  }

  function readLive() {
    return inOrder(() => getJSON("/api/models/loaded"))
      // The browser's backend answers no input_norm_declared: its models all declare theirs.
      .then((data) => showLive(data.pipeline, data.input_norm_from, data.input_norm_declared !== false,
        data.postprocess_from, data.postprocess_fitted === true))
      .catch((err) => console.warn("Could not read the chain the models run through:", err));
  }

  parts.forEach((part) => part.chain.onEdit(render));
  button.addEventListener("click", apply);
  [refused, ...parts.map((part) => part.notice)].forEach((notice) => {
    notice.querySelector(".chain-notice-close").addEventListener("click", () => say(notice, null));
  });
  listen(MODELS_CHANGED_EVENT, readLive);
  // Fit to Paint sets the raw's window with no model change (finetune/rank.js).
  listen(CHAIN_CHANGED_EVENT, readLive);
  render();
  readLive();

  return {
    apply,
    // showChain's: the lists show `chain`, the one the backend runs.
    show(chain) {
      parts.forEach((part) => {
        const steps = chain[part.key];
        if (!Array.isArray(steps)) return;
        part.chain.show(steps);
        part.chain.markApplied();
      });
      live = chain;
    },
  };
}

// The first-run dialog, on the page only until a server config is saved.
function initServerConfigModal() {
  const modalEl = document.getElementById("serverConfigModal");
  if (!modalEl) return;
  const modal = new bootstrap.Modal(modalEl);
  modal.show();
  const queuePicker = createQueuePicker(document.getElementById("modal_queue"), { small: true });
  const stopQueues = subscribeGpuQueues(queuePicker.render);

  document.getElementById("modalSaveConfigBtn").addEventListener("click", function () {
    const statusEl = document.getElementById("modalConfigStatus");
    function fail(message) {
      statusEl.style.color = "#f87171";
      statusEl.textContent = message;
    }
    // A blank count, or no queues (LSF has not listed them yet), is left out
    // so the server keeps its default.
    const payload = {
      charge_group: document.getElementById("modal_charge_group").value.trim(),
    };
    const queues = queuePicker.value();
    if (queues.length) payload.queue = queues;
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
        stopQueues();
        setTimeout(function () { modal.hide(); }, 500);
      })
      .catch(function (err) {
        fail(err instanceof ApiError ? "Error saving config: " + err.message : "Error: " + err);
      });
  });
}

initConnect();
initModelsTab();
document.getElementById("toggleDashboardBtn").addEventListener("click", toggleDashboard);
const inputChain = mountOpChain(document.getElementById("inputNormList"), { kind: "input" });
const postChain = mountOpChain(document.getElementById("postProcessList"), { kind: "postprocess" });
const applyBar = initApply(inputChain, postChain);
initModelAdvice({ input: inputChain, postprocess: postChain, apply: applyBar.apply });
document.addEventListener("keydown", (event) => applyOnEnter(event, applyBar.apply));
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
