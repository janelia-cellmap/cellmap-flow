// The model-advice banner above the dashboard tabs.
//
// Asks the server what activation the model's output actually has (see
// cellmap_flow/serving/probe.py) and compares it against the chain
// currently configured. Only renders when there is something to say.
import { getJSON } from "../lib/api.js";
import { esc } from "../lib/dom.js";
import { poll } from "../lib/poll.js";

// A confirmation needs no decision from anyone, so it should not sit at the
// top of the page indefinitely the way an actionable suggestion does.
const CONFIRM_BANNER_MS = 8000;

// The probe is collected during inference-server startup, and models are
// launched on background threads that the submit request does not wait for,
// so right after a submit there is usually nothing to report yet. Keep
// checking until every job has answered, rather than taking one shot at a
// fixed delay and silently giving up.
const ADVICE_POLL_INTERVAL_MS = 5000;
const ADVICE_POLL_MAX_ATTEMPTS = 60;  // ~5 min, enough for a queued job

// The Input and Postprocess chains (shared/op-chain.js) the Apply buttons
// change, from initModelAdvice.
let chains = null;

export function initModelAdvice({ input, postprocess }) {
  chains = { input, postprocess };
}

// Tick exactly the suggested steps, fill in their parameters, and move them
// to the top in the suggested order: steps are applied top to bottom.
// Defaults assume each step sees the range its usual predecessor produces;
// where this chain differs, the server says what to use.
function applyPostprocessSuggestion(names, params) {
  chains.postprocess.apply(names, params);
}

// Same idea for the input side. Normalizers carry parameter values, not just
// a name, so this fills those in as well as ticking the box.
function applyInputNormSuggestion(norms, order) {
  chains.input.apply(order || Object.keys(norms || {}), norms);
}

// The server repeats the same advice on every poll until the configured
// chain changes, so the banner cannot simply be redrawn each time: a
// suggestion the user dismissed or applied came back five seconds later, and
// a confirmation that had timed out reappeared. Each piece of advice is keyed
// by what it says, including the chain it was about; once hidden, that key
// stays hidden, while advice about a changed chain has a new key and shows.
const hiddenAdvice = new Set();
const shownAdvice = new Map();  // key -> element currently in the banner

function adviceKey(kind, m, what) {
  return JSON.stringify([kind, m.model || "", what]);
}

function hideAdvice(key) {
  hiddenAdvice.add(key);
  const div = shownAdvice.get(key);
  shownAdvice.delete(key);
  if (div) div.remove();
}

// Put the advice for `key` in the banner, building it only the first time,
// and record it in `wanted` so renderModelAdvice keeps it.
function showAdvice(box, wanted, key, build) {
  if (hiddenAdvice.has(key)) return;
  wanted.add(key);
  let div = shownAdvice.get(key);
  if (!div) {
    div = build();
    shownAdvice.set(key, div);
  }
  box.appendChild(div);  // (re)appending keeps the banner in poll order
}

function adviceButton(className, text, onclick) {
  const btn = document.createElement("button");
  btn.className = className;
  btn.textContent = text;
  btn.onclick = onclick;
  return btn;
}

// Model names reach this banner from HuggingFace repo ids and from user YAML,
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
  const key = adviceKey("input", m, [
    !!m.input_norm_matches, names, sug.input_norm, m.configured_input_norm || [],
  ]);
  if (m.input_norm_matches) {
    // Confirmation, not a prompt: when the norms came from a yaml there is
    // no other way to tell whether they suit the model.
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
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    div.className = "alert " + (assumed ? "alert-secondary" : "alert-info");
    div.innerHTML = "<strong>"
                  + (assumed ? "Assumed input normalization" : "Suggested input normalization")
                  + "</strong> for <code>" + esc(m.model || "model") + "</code><br>"
                  + esc(sug.reason || "")
                  + "<br><small>confidence: " + esc(sug.confidence || "unknown")
                  + "; currently set: "
                  + esc((m.configured_input_norm || []).join(" → ") || "none")
                  + "</small>";
    div.appendChild(document.createElement("br"));
    div.appendChild(adviceButton("btn btn-sm btn-primary ms-2 mt-2",
      "Apply: " + names.join(" → "),
      function () { applyInputNormSuggestion(sug.input_norm, names); hideAdvice(key); }));
    div.appendChild(adviceButton("btn btn-sm btn-link", "Dismiss",
      function () { hideAdvice(key); }));
    return div;
  });
}

function renderPostprocessAdvice(m, box, wanted) {
  const rev = m.postprocess_review || {};
  if (rev.level !== "suggest" && rev.level !== "warn") return;
  const warn = rev.level === "warn";
  const key = adviceKey("post", m, [
    rev.level, rev.message, rev.suggest || [], rev.params || {}, m.configured_postprocess || [],
  ]);
  showAdvice(box, wanted, key, function () {
    const div = document.createElement("div");
    div.className = "alert " + (warn ? "alert-warning" : "alert-info");
    let html = "<strong>" + (warn ? "Check postprocessing" : "Suggested postprocessing")
             + "</strong> for <code>" + esc(m.model || "model") + "</code><br>";
    if (m.output_min !== null && m.output_min !== undefined) {
      html += "<small>Observed output range ["
           + Number(m.output_min).toPrecision(4) + ", "
           + Number(m.output_max).toPrecision(4) + "]</small><br>";
    }
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
      div.appendChild(adviceButton("btn btn-sm btn-primary ms-2 mt-2",
        "Apply: " + rev.suggest.join(" → "),
        function () { applyPostprocessSuggestion(rev.suggest, rev.params); hideAdvice(key); }));
    }
    div.appendChild(adviceButton("btn btn-sm btn-link", "Dismiss",
      function () { hideAdvice(key); }));
    return div;
  });
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
  const box = document.getElementById("modelAdviceBanner");
  if (!box) return;
  const wanted = new Set();
  (models || []).forEach(function (m) {
    renderVoxelSizeAdvice(m, box, wanted);
    renderInputNormAdvice(m, box, wanted);
    renderPostprocessAdvice(m, box, wanted);
  });
  // Drop what the server no longer says (the chain or the probe moved on).
  Array.from(shownAdvice.keys()).forEach(function (key) {
    if (wanted.has(key)) return;
    shownAdvice.get(key).remove();
    shownAdvice.delete(key);
  });
}

// One poll of /api/model_advice. Resolves to false to stop polling: every
// job has answered, the server has nothing to say, or it is unreachable.
async function checkAdvice({ stale }) {
  let d;
  try {
    d = await getJSON("/api/model_advice");
  } catch (e) {
    console.log("model advice unavailable", e);
    return false;
  }
  if (stale()) return undefined;  // a newer refresh replaced this one
  if (!d || !d.success) return false;
  renderModelAdvice(d.models);
  const models = d.models || [];
  const pending = models.length === 0 || models.some(function (m) {
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
    maxTicks: ADVICE_POLL_MAX_ATTEMPTS + 1,  // the first check, then the retries
  });
}
