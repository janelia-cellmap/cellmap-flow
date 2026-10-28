// The model advice: what the server makes of each model's input and output
// handling. The suggestions for the chain (the input normalization, the
// postprocessing) are in the Processing tab, above its lists; how a model
// reads the data's scale is in the banner above the dashboard tabs, on every
// tab, as it is about the Models tab's Resample.
//
// Asks the server what activation the model's output actually has (see
// cellmap_flow/serving/probe.py) and compares it against the chain
// currently configured. Only renders when there is something to say.
import { getJSON } from "../lib/api.js";
import { esc } from "../lib/dom.js";
import { announce, CHAIN_CHANGED_EVENT, listen, MODELS_CHANGED_EVENT } from "../lib/events.js";
import { poll } from "../lib/poll.js";

// A confirmation needs no decision from anyone, so it should not sit at the
// top of the page indefinitely the way an actionable suggestion does.
const CONFIRM_BANNER_MS = 8000;

// The probe is collected during inference-server startup, and models are
// launched on background threads that the submit request does not wait for,
// so right after a submit there is usually nothing to report yet, and a job
// queued on the cluster can wait many minutes. So the advice is asked for
// until every model's probe has answered, however long that takes: every few
// seconds at first, then less often (lib/poll.js maxIntervalMs).
const ADVICE_POLL_INTERVAL_MS = 3000;
const ADVICE_POLL_MAX_INTERVAL_MS = 30000;

// The Processing tab's two lists (shared/op-chain.js) a suggestion's Apply
// fills in, and its Apply bar's apply (main.js initApply), from
// initModelAdvice.
let processing = null;

export function initModelAdvice({ input, postprocess, apply }) {
  processing = { input, postprocess, apply };
  // Loading a model, not only Submit All, is when advice is most useful.
  listen(MODELS_CHANGED_EVENT, () => refreshModelAdvice());
  // So is a chain the backend set itself with no model change (Fit to
  // Paint, Use these settings: finetune/rank.js): an affinity model's chain
  // into a segmentation is then offered cut at the threshold the ranking
  // fitted, where the line shown, once every probe had answered and the
  // poll stopped, still cut at 0.5.
  listen(CHAIN_CHANGED_EVENT, () => refreshModelAdvice());
  // The poll waits while the page is hidden, and may have stopped: a tab
  // shown again asks at once.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshModelAdvice();
  });
  // The models running when the page loaded: a YAML's, or a reload's.
  refreshModelAdvice();
}

// A suggestion's Apply: make `list` exactly the suggested steps, in the
// suggested order (steps run top to bottom), each with the parameters the
// server gives (a normalizer's values; a postprocessor's where this chain
// needs other than its defaults), and apply the chain as the Apply bar does,
// with any other edit of the two lists not applied yet.
function applySuggestion(list, names, params) {
  list.apply(names, params);
  processing.apply();
}

// The server repeats its advice on every poll, so the advice cannot simply
// be redrawn each time: a suggestion the user dismissed or applied would come
// back a few seconds later, and a confirmation that had timed out reappear.
// So each piece of advice has a key, and once it is hidden (dismissed,
// applied, or a confirmation's time up) that key stays hidden while the page
// is open.
//
// The key is the model, the kind of advice and what the advice is about for
// that model (the steps and values a suggestion proposes, what a warning
// found), and never the chain configured now. A suggestion the user turned
// down stays down when the chain changes, by their Apply, an edit, or the
// backend; it comes back only when the advice is about something new for the
// model, as another suggestion or another finding (a probe that answered,
// other data). A suggestion itself can name the chain ("currently set: ..."),
// so `says` is what it says now, and it is rebuilt, under the same key, when
// that changes.
const hiddenAdvice = new Set();
const shownAdvice = new Map();  // key -> {div, says}: what is on the page now

function adviceKey(kind, m, about) {
  return JSON.stringify([kind, m.model || "", about]);
}

function hideAdvice(key) {
  hiddenAdvice.add(key);
  const shown = shownAdvice.get(key);
  shownAdvice.delete(key);
  if (shown) shown.div.remove();
}

// Put the advice for `key` in `box`, building it the first time and
// again when what it says changes (`says`, by default nothing beyond the key),
// and record it in `wanted` so renderModelAdvice keeps it.
function showAdvice(box, wanted, key, build, says = "") {
  if (hiddenAdvice.has(key)) return;
  wanted.add(key);
  let shown = shownAdvice.get(key);
  if (!shown || shown.says !== says) {
    if (shown) shown.div.remove();
    shown = { div: build(), says };
    shownAdvice.set(key, shown);
  }
  box.appendChild(shown.div);  // (re)appending keeps the box in poll order
}

function adviceButton(className, text, onclick) {
  const btn = document.createElement("button");
  btn.className = className;
  btn.textContent = text;
  btn.onclick = onclick;
  return btn;
}

// Model names reach the advice from HuggingFace repo ids and from user YAML,
// and the HF catalogue is third-party data we render verbatim. None of it is
// trusted markup, so everything interpolated into innerHTML below is escaped.
function renderInputNormAdvice(m, box, wanted) {
  const sug = m.input_norm_suggestion || {};
  if (!sug.input_norm) return;
  // sug.order, not Object.keys: jsonify sorts keys and the order of these
  // steps changes what they compute.
  const names = sug.order || Object.keys(sug.input_norm);
  // "low" means nothing declared how the model was trained and the [-1, 1]
  // range was assumed. Still say so -- the wrong input scale is not visibly
  // wrong, it just degrades the prediction -- but render it muted so it does
  // not read as a finding.
  const assumed = sug.confidence === "low";
  // The steps are the declared ones but a value is not (a LambdaNormalizer
  // expression edited): [{step, param, configured, declared}], from the
  // backend, which compares the values as the chain carries them.
  const differences = m.input_norm_differences || [];
  if (m.input_norm_fitted) {
    // The steps the Finetune tab's ranking fitted to the paint for this
    // model (Use this model, Fit to Paint, Use these settings): a
    // confirmation too, never the declared steps suggested back over them,
    // whose Apply undid the ranking.
    const configured = m.configured_input_norm || [];
    const key = adviceKey("input-fitted", m, configured);
    showAdvice(box, wanted, key, function () {
      const div = document.createElement("div");
      div.className = "alert alert-success py-1 px-2";
      div.innerHTML = "<small><strong>Input normalization fitted to your paint</strong> for <code>"
                    + esc(m.model || "model") + "</code> by the ranking: " + esc(configured.join(" → ")) + ".</small>";
      setTimeout(function () { hideAdvice(key); }, CONFIRM_BANNER_MS);
      return div;
    });
    return;
  }
  if (m.input_norm_matches) {
    // Confirmation, not a prompt: when the norms came from a yaml there is
    // no other way to tell whether they suit the model. Shown once for these
    // steps and values, as a suggestion is.
    const key = adviceKey("input-matches", m, [names, sug.input_norm]);
    showAdvice(box, wanted, key, function () {
      const div = document.createElement("div");
      div.className = "alert py-1 px-2 " + (assumed ? "alert-secondary" : "alert-success");
      div.innerHTML = "<small><strong>Input normalization "
                    + (assumed ? "matches the assumed range" : "looks right")
                    + "</strong> for <code>" + esc(m.model || "model") + "</code>: "
                    + esc(names.join(" → ")) + ". " + esc(sug.reason || "") + "</small>";
      setTimeout(function () { hideAdvice(key); }, CONFIRM_BANNER_MS);
      return div;
    });
    return;
  }
  // The suggestion is the steps and values to set, whether the chain has
  // other steps or the same ones with other values: one key for both, so a
  // suggestion turned down is not offered again for a chain edited since.
  const key = adviceKey("input", m, [names, sug.input_norm]);
  const says = JSON.stringify([m.configured_input_norm || [], differences]);
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    div.className = "alert " + (assumed ? "alert-secondary" : "alert-info");
    if (differences.length) {
      // The same steps, so naming them would say nothing: say the values.
      div.innerHTML = "<strong>Input normalization differs from what <code>"
                    + esc(m.model || "model") + "</code> declares</strong><br>"
                    + differences.map(function (d) {
                        return esc(d.step) + " " + esc(d.param) + " is " + esc(String(d.configured))
                             + ", declared " + esc(String(d.declared));
                      }).join("<br>")
                    + "<br><small>" + esc(sug.reason || "") + "</small>";
    } else {
      div.innerHTML = "<strong>"
                    + (assumed ? "Assumed input normalization" : "Suggested input normalization")
                    + "</strong> for <code>" + esc(m.model || "model") + "</code><br>"
                    + esc(sug.reason || "")
                    + "<br><small>confidence: " + esc(sug.confidence || "unknown")
                    + "; currently set: "
                    + esc((m.configured_input_norm || []).join(" → ") || "none")
                    + "</small>";
    }
    div.appendChild(document.createElement("br"));
    div.appendChild(adviceButton("btn btn-sm btn-primary mt-2 me-2",
      differences.length ? "Apply the declared values" : "Apply: " + names.join(" → "),
      function () { applySuggestion(processing.input, names, sug.input_norm); hideAdvice(key); }));
    div.appendChild(adviceButton("btn btn-sm btn-link mt-2", "Dismiss",
      function () { hideAdvice(key); }));
    return div;
  }, says);
}

// A chain the model's output shows right without (an affinity model's into
// a segmentation): one quiet line with its Apply, not a suggestion's banner,
// which over the sigmoid a start or the ranking set read as though that
// were wrong.
function renderOptionalPostprocess(m, rev, box, wanted) {
  const suggest = rev.suggest || [];
  if (!suggest.length) return;
  const key = adviceKey("post-optional", m, [rev.message, suggest]);
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    div.className = "small text-muted mb-2";
    div.innerHTML = "Optional for <code>" + esc(m.model || "model") + "</code>: " + esc(rev.message) + " ";
    div.appendChild(adviceButton("btn btn-link btn-sm p-0 align-baseline me-2", "Apply: " + suggest.join(" → "),
      function () { applySuggestion(processing.postprocess, suggest, rev.params); hideAdvice(key); }));
    div.appendChild(adviceButton("btn btn-link btn-sm p-0 align-baseline text-muted", "Dismiss",
      function () { hideAdvice(key); }));
    return div;
  }, JSON.stringify(rev.params || {}));
}

// A served model whose output neither its network's graph nor its probe on
// noise says (a flat probe: model_info's output_class null), so nothing
// sets its postprocessing: one quiet line saying so, and what the probe saw.
function renderUnknownOutput(m, rev, box, wanted) {
  const key = adviceKey("post-unknown", m, [rev.message]);
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    div.className = "small text-muted mb-2";
    div.innerHTML = "<code>" + esc(m.model || "model") + "</code>: " + esc(rev.message) + " ";
    div.appendChild(adviceButton("btn btn-link btn-sm p-0 align-baseline text-muted", "Dismiss",
      function () { hideAdvice(key); }));
    return div;
  });
}

function renderPostprocessAdvice(m, box, wanted) {
  const rev = m.postprocess_review || {};
  if (rev.level === "optional") {
    renderOptionalPostprocess(m, rev, box, wanted);
    return;
  }
  // "unknown" with no answer from the server is a server not up yet.
  if (rev.level === "unknown" && m.probe_available) {
    renderUnknownOutput(m, rev, box, wanted);
    return;
  }
  if (rev.level !== "suggest" && rev.level !== "warn") return;
  const warn = rev.level === "warn";
  // Keyed by the finding (the level and the message, which name nothing of
  // the chain), not by the steps proposed: a warning can propose the chain
  // itself less a step (the second sigmoid), which changes with every edit.
  const key = adviceKey("post", m, [rev.level, rev.message]);
  const says = JSON.stringify([rev.suggest || [], rev.params || {}]);
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    div.className = "alert " + (warn ? "alert-warning" : "alert-info");
    // The message says what the output is and what says so: the network's
    // last op, or the range its output on noise had (a guess).
    let html = "<strong>" + (warn ? "Check postprocessing" : "Suggested postprocessing")
             + "</strong> for <code>" + esc(m.model || "model") + "</code><br>";
    html += esc(rev.message);
    const tuned = Object.keys(rev.params || {});
    if (tuned.length) {
      html += "<br><small>with " + tuned.map(function (name) {
        const v = rev.params[name];
        return "<code>" + esc(name) + "</code> " + Object.keys(v).map(function (k) {
          return esc(k) + "=" + esc(v[k]);
        }).join(", ");
      }).join("; ") + "</small>";
    }
    div.innerHTML = html;
    if ((rev.suggest || []).length) {
      div.appendChild(document.createElement("br"));
      div.appendChild(adviceButton("btn btn-sm btn-primary mt-2 me-2",
        "Apply: " + rev.suggest.join(" → "),
        function () { applySuggestion(processing.postprocess, rev.suggest, rev.params); hideAdvice(key); }));
    }
    div.appendChild(adviceButton("btn btn-sm btn-link mt-2", "Dismiss",
      function () { hideAdvice(key); }));
    return div;
  }, says);
}

function nm(voxelSize) {
  return (voxelSize || []).join("×") + " nm";
}

// How the model's server reads the data, when the data has no level at the
// model's input voxel size. Read as if it were (the default), the model sees
// the wrong scale and nothing else shows it, so that is a warning; resampled,
// a confirmation that goes away.
function renderVoxelSizeAdvice(m, box, wanted) {
  const relabelled = m.input_relabelled_from;
  const resampled = m.input_resampled_from;
  if (!relabelled && !resampled) return;
  const key = adviceKey("voxel", m, [relabelled, resampled, m.input_voxel_size]);
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    const model = "<code>" + esc(m.model || "model") + "</code>";
    if (relabelled) {
      div.className = "alert alert-warning";
      div.innerHTML = "<strong>Wrong scale</strong>: " + model + " expects "
                    + esc(nm(m.input_voxel_size)) + " data, but the dataset has no such level, so it reads "
                    + esc(nm(relabelled)) + " data as if it were " + esc(nm(m.input_voxel_size))
                    + ". Tick <em>Resample if no scale matches the model</em> in the Models tab and submit it again.";
      div.appendChild(adviceButton("btn btn-sm btn-link", "Dismiss",
        function () { hideAdvice(key); }));
    } else {
      div.className = "alert alert-secondary py-1 px-2";
      div.innerHTML = "<small>" + model + " reads the data resampled from "
                    + esc(nm(resampled)) + " to " + esc(nm(m.input_voxel_size)) + ".</small>";
      setTimeout(function () { hideAdvice(key); }, CONFIRM_BANNER_MS);
    }
    return div;
  });
}

function renderModelAdvice(models) {
  const banner = document.getElementById("modelAdviceBanner");
  const chainBox = document.getElementById("processingAdvice");
  if (!banner || !chainBox) return;
  const wanted = new Set();
  (models || []).forEach(function (m) {
    renderVoxelSizeAdvice(m, banner, wanted);
    renderInputNormAdvice(m, chainBox, wanted);
    renderPostprocessAdvice(m, chainBox, wanted);
  });
  // Drop what the server no longer says (the chain or the probe moved on);
  // advice not hidden shows again if the server comes back to it.
  Array.from(shownAdvice.keys()).forEach(function (key) {
    if (wanted.has(key)) return;
    shownAdvice.get(key).div.remove();
    shownAdvice.delete(key);
  });
}

// The chain's digest the last answer of the server's gave (the browser's
// backend answers none): a model's start sets the postprocessing its output
// needs once its server answers (routes/pipeline.py
// set_needed_postprocess), after the Submit that started it, so when the
// digest moves the Processing tab reads the chain again (main.js, on
// CHAIN_CHANGED_EVENT).
let adviceDigest;

// One poll of /api/model_advice. Resolves to false to stop polling once
// every model's probe has answered and no model's start is on its way
// (`starting`); until then, and after a request that failed (a server slow
// to answer, or restarting), it asks again. No models yet is a submit whose
// jobs have not started.
async function checkAdvice({ stale }) {
  let d;
  try {
    d = await getJSON("/api/model_advice");
  } catch (e) {
    console.log("model advice unavailable", e);
    return undefined;
  }
  if (stale()) return undefined;  // a newer refresh replaced this one
  if (!d || !d.success) return undefined;
  renderModelAdvice(d.models);
  if (d.digest !== undefined && d.digest !== adviceDigest) {
    adviceDigest = d.digest;
    announce(CHAIN_CHANGED_EVENT);
  }
  const models = d.models || [];
  const pending = models.length === 0 || d.starting === true || models.some(function (m) {
    return !m.probe_available;
  });
  return pending ? undefined : false;
}

// One poll chain at a time: a new call starts it over, and an answer still
// in flight from the chain it replaced is ignored.
let advicePoller = null;

export function refreshModelAdvice() {
  if (advicePoller) {
    advicePoller.restart();
    return;
  }
  advicePoller = poll(checkAdvice, {
    intervalMs: ADVICE_POLL_INTERVAL_MS,
    maxIntervalMs: ADVICE_POLL_MAX_INTERVAL_MS,
    pauseWhenHidden: true,
  });
}
