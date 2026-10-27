// Which model to start from: the models the page can run, ranked by how well
// each one's prediction already matches the paint of the session's volume, so
// the one finetuned is the one with the least to learn. GET
// /api/finetune/rank/candidates lists them with the volume, and POST
// /api/finetune/rank/score scores one (web/src/backend/routes/rank.ts in the
// browser, dashboard/routes/finetune/rank_routes.py on a server). The scores
// are asked for one at a time, as each can take up to a minute in the browser
// (a download, then a run over the painted chunks) and longer on a server for
// a model not running (a cluster job started for it), and each is shown as it
// comes. A server's candidate says how to build it (entry), sent back with
// each request, and one not running (cluster) starts unticked. A filter narrows the models listed; Rank scores the ticked ones it
// shows. Each row gives what its score was read at, as the controls it sets
// take it (portable/rank.py score_row): the threshold, a probability, and
// the input normalization it was read through (input_norm: the model's own,
// or the session's when that matched better, maybe stretched to the raw
// under the paint). Show serves a model and adds its layer, as the Models
// tab's Add does; Use this model (POST /api/finetune/rank/use) makes one the
// model to finetune: that input normalization the session's, the model
// served, a new volume made for it with the paint carried onto its grid,
// that volume the session's and its paint layer in the viewer. The model picker (models.js pickModel) and the Annotation
// Crops panel's volume buttons (crops.js sessionVolumeChanged) follow, and
// a binary model's row's channel (channel_index, the channel that matched
// best) becomes its Channel in the training form (training-form.js
// pickChannel), shown once the model is picked if it trains one channel at
// a time. Rank itself fills the Channel of the model the form shows, when
// none is picked (fillChannel): that is the model painted for, which has no
// Use this model.
//
// Fit to Paint, beside Seed from Prediction, does the same for the model
// that seeds the paint (Label it's Prediction from) without a new volume:
// it scores that model with the same route, and sets its threshold and,
// when it is not the session's as it stands, the input normalization its
// score was read through (POST /api/finetune/rank/input-norm); its
// Channel only when the form shows one for that model with nothing picked
// (fillChannel), so a channel the user picked stays. Both it and Use this
// model mark Seed from Prediction's threshold as set by ranking, and say on
// the line under Label it's buttons what they set.
import { h, setBusy } from "../../lib/dom.js";
import { announce, CHAIN_CHANGED_EVENT, listen, MODELS_CHANGED_EVENT } from "../../lib/events.js";
import { readJSON, writeJSON } from "../../lib/stored.js";
import { sessionVolumeChanged } from "./crops.js";
import { pickModel } from "./models.js";
import { getAnswer, offerNewVolume, postAnswer } from "./requests.js";
import { fillChannel, pickChannel } from "./training-form.js";

// A score this close to the best one is marked about as good as it: two
// points of matching are not worth choosing a model by.
export const ABOUT_AS_GOOD = 0.02;
// With fewer painted voxels than this behind the best score, scores a few
// points apart are as good as tied, and the table says so.
export const LITTLE_PAINT = 1000;

// The table's rows from the score answers so far (each with the model's
// name, and success false with an error for one that could not be scored):
// the models scored, the best match of the paint first (equal scores keep
// the order they came in), then the others in the order they came. The
// first is marked best, and each within ABOUT_AS_GOOD of it tied (and a
// hair over: 0.81 - 0.79 is not exactly 0.02 in floating point).
// littlePaint: whether the best was scored on fewer than LITTLE_PAINT voxels.
export function rankResults(answers) {
  const scored = answers.filter(isScored).sort((a, b) => b.matches - a.matches);
  const failed = answers.filter((answer) => !isScored(answer));
  const best = scored[0];
  const rows = scored
    .map((answer, i) => ({ ...answer, best: i === 0, tied: i > 0 && best.matches - answer.matches <= ABOUT_AS_GOOD + 1e-9 }))
    .concat(failed.map((answer) => ({ ...answer, best: false, tied: false })));
  return { rows, littlePaint: Boolean(best) && best.scored_voxels < LITTLE_PAINT };
}

// Whether `candidate` shows under the filter typed: every word of it is in
// the model's name, task, channels or a voxel size ("16 nm" or "16nm"), or
// is its "2d" or "3d", or "finetuned", whatever the case. Spaces, commas,
// hyphens and underscores all separate words, in the filter and the names
// alike, so "mito aff", "mito-aff" and "mito_aff" all find
// mito-aff-unet-setup-16, listed as mito_aff_unet_setup_16. Blank shows all.
export function matchesFilter(candidate, query) {
  const words = query.toLowerCase().replace(/(\d)\s+nm\b/g, "$1nm").split(/[\s,_-]+/).filter(Boolean);
  const text = [
    candidate.name, candidate.task, ...(candidate.channels || []),
    ...(candidate.voxel_size_nm || []).map((size) => `${size}nm`),
    candidate.dims && `${candidate.dims}d`, candidate.finetuned && "finetuned",
  ].filter(Boolean).join(" ").toLowerCase().replace(/[_-]+/g, " ");
  return words.every((word) => text.includes(word));
}

function isScored(answer) {
  return Boolean(answer.success) && Number.isFinite(answer.matches);
}

// Where a score's input normalization came from (portable/rank.py
// norm_tries's `from`), in the row's words.
const NORM_FROM = {
  declared: "its own",
  convention: "CellMap convention",
  assumed: "assumed [−1, 1]",
  session: "yours",
  none: "none",
};

// A scored row's two settings, as columns (each a fixed width, so they line
// up from row to row): the threshold Seed from Prediction takes, 0.5 when
// the model's own boundary fits the paint best, else with the match at 0.5;
// and the input normalization it was read through, where it came from and
// its steps, stretched to the raw under the paint when that scored clearly
// better, then with the match unstretched (cellmap_flow/portable/rank.py
// score_row).
function settingsCells(row) {
  const own = row.cutoff === 0;
  const stretched = Array.isArray(row.window);
  return h("div", { class: "d-flex flex-wrap column-gap-3 small" },
    h("span", { style: { minWidth: "11rem" }, title: "The probability above which the prediction fits your paint best, as Seed from Prediction's threshold takes it; 0.5 is the model's own boundary" },
      "Threshold ", h("strong", {}, String(row.threshold)),
      own ? h("span", { class: "text-muted" }, " (model's own)") : h("span", { class: "text-muted" }, ` (${percent(row.matches_at_zero)} at 0.5)`)),
    h("span", { title: "The input normalization the model was read through: its own (what it declares, or the CellMap convention for a CellMap model), or yours from the Processing tab when that matched better; Use this model sets it" },
      "Input ", h("strong", {}, NORM_FROM[row.input_norm_from] || "as normalized"),
      Array.isArray(row.input_norm) && row.input_norm.length ? h("span", { class: "text-muted" }, ` ${stepsText(row.input_norm)}`) : "",
      stretched ? h("span", { class: "text-muted" }, `, stretched (${percent(row.matches_without_window)} unstretched)`) : ""));
}

// Normalizer steps in a few words each: a MinMaxNormalizer as the raw's
// range it reads as 0 to 1, a LambdaNormalizer as its expression, any other
// by its name.
function stepsText(steps) {
  return steps.map((step) => {
    if (step.name === "MinMaxNormalizer") return `${levels([Number(step.min_value ?? 0), Number(step.max_value ?? 255)])} to 0–1`;
    if (step.name === "LambdaNormalizer") return step.expression;
    return step.name.replace(/Normalizer$/, "");
  }).join(" → ");
}

// A raw window [lo, hi] as grey levels: whole ones for a raw of hundreds of
// levels, three significant digits for a raw of a few (a float raw's 0–1).
function levels(window) {
  return window.map((v) => String(Math.abs(v) >= 100 ? Math.round(v) : Number(v.toPrecision(3)))).join("–");
}

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "–";
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

// A candidate's line beside its name: what it predicts, at what voxel size,
// in 2D or 3D, and whether it is a finetune.
function describe(candidate) {
  const size = candidate.voxel_size_nm || [];
  const nm = size.length && size.every((s) => s === size[0]) ? `${size[0]} nm` : `${size.join("×")} nm`;
  return [candidate.task, size.length && nm, candidate.dims && `${candidate.dims}D`, candidate.finetuned && "finetuned"]
    .filter(Boolean).join(", ");
}

// Whether the card is open, kept in this browser as the patch tools'
// sections are (index.js rememberOpenSections); closed until opened.
const OPEN_KEY = "finetuneRankOpen";
// The threshold Use this model or Fit to Paint last set, and for which
// model: {threshold, model}, kept in this browser for the field's mark.
const MARK_KEY = "cellmap_flow.seed_threshold_by_ranking";

export function initRank() {
  const section = document.getElementById("rankSection");
  const filter = document.getElementById("rankFilter");
  const tickAll = document.getElementById("rankTickAll");
  const tickNone = document.getElementById("rankTickNone");
  const candidateList = document.getElementById("rankCandidates");
  const rankBtn = document.getElementById("rankBtn");
  const stopBtn = document.getElementById("rankStopBtn");
  const status = document.getElementById("rankStatus");
  const results = document.getElementById("rankResults");
  const rows = document.getElementById("rankRows");
  const littlePaint = document.getElementById("rankLittlePaint");
  // Label it's: the model seeding the paint, the threshold and settings a
  // score sets (label-it.js reads and keeps them), the threshold's mark,
  // Fit to Paint and the line saying what was set.
  const seedModel = document.getElementById("seedModel");
  const seedThreshold = document.getElementById("seedThreshold");
  const seedSettings = document.getElementById("seedSettings");
  const thresholdMark = document.getElementById("seedThresholdByRanking");
  const fitBtn = document.getElementById("fitToPaintBtn");
  const fitStatus = document.getElementById("fitToPaintStatus");

  let candidates = [];
  // The models unticked, by hand or as listed (ticked false): one listed
  // later starts as its listing says.
  const unticked = new Set();
  const seen = new Set();
  // The ranking shown: its score answers in the order they came, the models
  // it scores by name (for Show), the volume it was painted on, and its
  // status line ({ text, tone }), which stays while its table does.
  let answers = [];
  let ranked = new Map();
  let rankedOn = null;
  let summary = null;
  let running = false;
  let stopping = false;
  let listed = false;
  // The model Use this model is making the one to finetune, while it does;
  // and the one it made, with its volume, while that is the session's.
  let switching = null;
  let training = null;
  // Whether Fit to Paint is scoring and setting, which Rank and Use wait for
  // (and it for them): one score at a time. And whether the last listing
  // found paint to rank on, for Rank once Fit to Paint is done.
  let fitting = false;
  let ready = false;

  function say(text, tone = "muted") {
    status.textContent = text;
    status.className = `d-block text-${tone}`;
  }

  // The line under Label it's buttons, for what a score set or why it could not.
  function sayFit(text, tone = "muted") {
    fitStatus.hidden = false;
    fitStatus.textContent = text;
    fitStatus.className = `d-block mt-1 text-${tone}`;
  }

  // What Use this model or Fit to Paint set from `row`, the model's score:
  // the threshold, the channel when one was set (`channel`, its name), and
  // the input normalization the score was read through (inputText).
  function fittedText(name, row, channel) {
    return `${name} fitted to your paint (${percent(row.matches)} match): Seed from Prediction's threshold is ${row.threshold}${channel ? `, ${channelSet(channel)}` : ""}, and ${inputText(row)}.`;
  }

  // Whether setting `row`'s input normalization changes the session's: it
  // does unless the score was read through the session's as it stands (or
  // through none, with none set).
  function changesInput(row) {
    return Array.isArray(row.input_norm) && (Array.isArray(row.window) || !["session", "none"].includes(row.input_norm_from));
  }

  // The input normalization a score set, in words, for every model, or that
  // it was left as it was.
  function inputText(row) {
    if (!changesInput(row)) return "the input normalization is left as it was";
    const from = { declared: "its declared one", convention: "the CellMap convention", assumed: "the assumed [−1, 1]", session: "yours" }[row.input_norm_from] || "the one it was read through";
    return `the input normalization is now ${from}${Array.isArray(row.window) ? `, stretched to the raw under your paint` : ""} (${stepsText(row.input_norm)}), for every model`;
  }

  // A channel set from a score, in words: a binary model's score is read on
  // its best channel (channel_index), which Use this model makes the model's
  // Channel (pickChannel) and Fit to Paint fills in when none is picked
  // (fillChannel): the channel it trains and Seed from Prediction reads. A
  // model of one channel trains that one anyway; an affinity model's
  // channels are scored together, and set none.
  function channelSet(channel) {
    return `the channel it trains and seeds from is ${channel}`;
  }

  // The threshold a score fitted, made Seed from Prediction's (label-it.js
  // keeps it with the seed settings on its change) and marked as set by
  // ranking, with the settings opened to show it. The mark is kept in this
  // browser with the value it marks, and shown while the field holds that
  // value: typing another takes it away.
  function installThreshold(name, threshold) {
    if (!Number.isFinite(threshold)) return;
    writeJSON(MARK_KEY, { threshold, model: name });
    seedThreshold.value = threshold;
    seedThreshold.dispatchEvent(new Event("change"));
    seedSettings.open = true;
  }

  function showMark() {
    const mark = readJSON(MARK_KEY, null);
    const marks = Boolean(mark) && seedThreshold.value !== "" && Number(seedThreshold.value) === Number(mark.threshold);
    thresholdMark.hidden = !marks;
    if (marks) thresholdMark.title = `Set by ranking for ${mark.model}: the threshold that fits your paint best. Changing it takes the mark away.`;
  }

  // The candidates and the volume as the page has them now. Resolves to
  // { ready, message, tone }: whether there is paint to rank on, and what
  // the status line says about it. A ranking of another volume's paint is
  // cleared, and the rows shown take the models' served state from the list.
  async function listCandidates() {
    let d;
    try {
      d = await getAnswer("/api/finetune/rank/candidates");
    } catch (e) {
      return { ready: false, message: `Could not list the models: ${e}`, tone: "danger" };
    }
    if (!d.success) return { ready: false, message: `Could not list the models: ${d.error}`, tone: "danger" };
    listed = true;
    candidates = d.candidates || [];
    for (const candidate of candidates) {
      if (!seen.has(candidate.name) && candidate.ticked === false) unticked.add(candidate.name);
      seen.add(candidate.name);
    }
    const volume = d.volume;
    // (Not while Use this model makes the new volume, which keeps the ranking.)
    if (!running && switching === null && answers.length && rankedOn !== (volume && volume.id)) {
      answers = [];
      summary = null;
    }
    if (training && training.volumeId !== (volume && volume.id)) training = null;
    for (const candidate of candidates) if (ranked.has(candidate.name)) ranked.set(candidate.name, candidate);
    showCandidates();
    showResults();
    if (!volume) {
      return { ready: false, message: "No annotation volume yet: make one with New Volume (Annotation Crops tab), paint on it, then rank.", tone: "muted" };
    }
    if (!volume.painted_chunks) {
      return { ready: false, message: `Nothing is painted in ${volume.id} yet: paint background (1) and objects (2 and up), then rank.`, tone: "muted" };
    }
    if (!candidates.length) return { ready: false, message: "No models to rank.", tone: "muted" };
    const paintedFor = volume.model_name ? `painted for ${volume.model_name}` : "painted before a model was chosen";
    return { ready: true, volume, message: `${plural(volume.painted_chunks, "painted chunk")} in ${volume.id}, ${paintedFor}.`, tone: "muted" };
  }

  // The status line and Rank from a listing; a ranking running, or one
  // shown, keeps its own line. Rank waits while Use this model or Fit to
  // Paint runs.
  function showReadiness(state) {
    if (running) return;
    ready = state.ready;
    rankBtn.disabled = !state.ready || switching !== null || fitting;
    if (state.ready && answers.length && summary) say(summary.text, summary.tone);
    else say(state.message, state.tone);
  }

  // A listing for the status line, one at a time.
  let refreshing = null;
  function refresh() {
    refreshing = refreshing || listCandidates().then(showReadiness).finally(() => { refreshing = null; });
    return refreshing;
  }

  // The models the filter shows.
  function shown() {
    return candidates.filter((candidate) => matchesFilter(candidate, filter.value));
  }

  function showCandidates() {
    tickAll.disabled = tickNone.disabled = running;
    const none = candidates.length && !shown().length
      && h("small", { class: "text-muted" }, "No model matches the filter.");
    candidateList.replaceChildren(none || "", ...candidates.map((candidate, i) => {
      const id = `rankCandidate${i}`;
      return h("div", { class: "form-check", hidden: !matchesFilter(candidate, filter.value) },
        h("input", {
          type: "checkbox", class: "form-check-input", id, value: candidate.name,
          checked: !unticked.has(candidate.name), disabled: running,
          onChange: (e) => (e.target.checked ? unticked.delete(candidate.name) : unticked.add(candidate.name)),
        }),
        h("label", { class: "form-check-label", for: id, title: candidate.source || `${candidate.repo}/${candidate.folder}` },
          candidate.name, candidate.volume_model ? " (painted for)" : "", " ",
          h("small", { class: "text-muted" }, describe(candidate))));
    }));
  }

  function showResults() {
    // Fit to Paint waits while a ranking or Use this model runs.
    fitBtn.disabled = running || switching !== null || fitting;
    const { rows: table, littlePaint: little } = rankResults(answers);
    results.hidden = !table.length;
    littlePaint.hidden = !little;
    rows.replaceChildren(...table.map(resultRow));
  }

  // A model's row: its name and badges beside Use this model (a scored model
  // the volume is not for) and Show, its scores on a line under them, its
  // threshold and input window under those, then what they were scored on.
  function resultRow(row) {
    const candidate = ranked.get(row.name);
    const scored = isScored(row);
    const details = [
      row.channel && `${row.channel} channel`,
      scored && `${Number(row.scored_voxels || 0).toLocaleString()} voxels (${row.object_voxels} object, ${row.background_voxels} background)`,
      row.note,
      Number.isFinite(row.seconds) && `${row.seconds.toFixed(1)} s`,
    ].filter(Boolean).join(", ");
    return h("li", {
      class: "py-1 px-2 mb-1 rounded",
      style: row.best ? { background: "var(--accent-muted)" } : { borderBottom: "1px solid var(--border)" },
    },
      h("div", { class: "d-flex flex-wrap align-items-start gap-2" },
        h("div", { class: "flex-grow-1 text-break" },
          row.name, candidate && candidate.volume_model ? " (painted for)" : "",
          row.best && h("span", { class: "badge bg-success ms-1" }, "best"),
          row.tied && h("span", { class: "badge bg-secondary ms-1" }, "about as good"),
          training && training.name === row.name && h("span", { class: "badge bg-info ms-1" }, "training this")),
        // The buttons go under the name when the column is too narrow for both.
        h("div", { class: "d-flex gap-1 flex-shrink-0 ms-auto" },
          candidate && scored && !candidate.volume_model && useButton(candidate, row),
          candidate && showButton(candidate))),
      scored
        ? h("div", {}, h("strong", {}, `${percent(row.matches)} match`),
          ` · objects ${percent(row.objects_found)} · background ${percent(row.background_right)}`)
        : h("div", { class: "small text-danger" }, row.error || "No score in the answer"),
      scored && settingsCells(row),
      details && h("small", { class: "d-block text-muted" }, details));
  }

  // None for a server's model that is not running and has no entry to start
  // it from: one of the session's, ticked on the Models tab to run it.
  function showButton(candidate) {
    if (candidate.cluster && !candidate.served && !candidate.entry) return "";
    const button = h("button", {
      type: "button", class: "btn btn-outline-secondary btn-sm", disabled: candidate.served,
      title: candidate.served ? "Served already: its layer is in the viewer" : "Serve this model and add its layer to the viewer",
    }, candidate.served ? "Shown" : "Show");
    button.addEventListener("click", () => showModel(candidate, button));
    return button;
  }

  function showModel(candidate, button) {
    setBusy(button, true);
    postAnswer("/api/models/add", { entry: candidate.entry || { name: candidate.name, repo: candidate.repo, folder: candidate.folder } })
      .then((d) => {
        if (!d.success) throw new Error(d.error);
        candidate.served = true;
        showResults();
        announce(MODELS_CHANGED_EVENT);
      })
      .catch((e) => {
        setBusy(button, false);
        say(`Could not show ${candidate.name}: ${e.message || e}`, "danger");
      });
  }

  // The best row's is the card's suggestion, filled.
  function useButton(candidate, row) {
    const button = h("button", {
      type: "button", class: `btn btn-sm ${row.best ? "btn-primary" : "btn-outline-secondary"}`, disabled: running || switching !== null || fitting,
      title: "Finetune this model instead: your paint is copied to a new volume made for it, which becomes the session's, and the current volume is kept",
    }, "Use this model");
    if (switching === candidate.name) setBusy(button, true, "Copying paint...");
    button.addEventListener("click", () => useModel(candidate, row));
    return button;
  }

  // Makes `candidate` the model to finetune, as its score (`row`) found it
  // best: the input normalization it was read through as the session's
  // (before the model is served, so it is served through it), and its
  // threshold as Seed from Prediction's, marked as set by ranking.
  // The ranking stays: the new volume holds the same paint, on the model's
  // grid.
  async function useModel(candidate, row) {
    if (running || switching !== null || fitting) return;
    if (!confirm(`Make ${candidate.name} the model to finetune? Your paint is copied to a new volume for it; the current volume is kept.`)) return;
    switching = candidate.name;
    rankBtn.disabled = true;
    showResults();
    say(`Making ${candidate.name} the model to finetune: serving it and copying your paint to a new volume...`);
    let d;
    try {
      d = await postAnswer("/api/finetune/rank/use", { ...request(candidate), ...(changesInput(row) ? { input_norm: row.input_norm } : {}) });
    } catch (e) {
      d = { success: false, error: `Could not make ${candidate.name} the model to finetune: ${e}` };
    }
    switching = null;
    if (!d.success) {
      rankBtn.disabled = false;
      showResults();
      say(d.error || `Could not make ${candidate.name} the model to finetune.`, "danger");
      return;
    }
    training = { name: d.name, volumeId: d.volume_id };
    rankedOn = d.volume_id;
    // As the listing will have them, until it comes.
    for (const c of [...candidates, ...ranked.values()]) c.volume_model = c.name === d.name;
    const dropped = d.conflicts ? ` (${Number(d.conflicts).toLocaleString()} left unpainted where object and background met)` : "";
    installThreshold(d.name, row.threshold);
    const channel = Number.isInteger(row.channel_index) ? row.channel : null;
    // Before the pick below, which shows it.
    if (channel !== null) pickChannel(d.name, row.channel_index);
    sayFit(fittedText(d.name, row, channel), "success");
    // The Processing tab's input normalization shows it as it now runs
    // (main.js reads the chain again), so a later Apply keeps it.
    if (Array.isArray(d.input_norm)) announce(CHAIN_CHANGED_EVENT);
    const set = [
      changesInput(row) && inputText(row),
      Number.isFinite(row.threshold) && `Seed from Prediction's threshold is ${row.threshold}`,
      channel && channelSet(channel),
    ].filter(Boolean);
    summary = {
      text: `${d.name} is the model to finetune now: ${Number(d.painted_voxels || 0).toLocaleString()} painted voxels carried from ${d.from_volume} to ${d.volume_id}${dropped}.${set.length ? ` ${set.join("; ")}, as its score found best.` : ""}`,
      tone: "success",
    };
    rankBtn.disabled = false;
    showCandidates();
    showResults();
    say(summary.text, summary.tone);
    pickModel(d.name);
    sessionVolumeChanged(d.volume_id);
    // Its layer, and its server if it had none: the listing that follows
    // moves "(painted for)" to it too.
    announce(MODELS_CHANGED_EVENT);
  }

  // What names `candidate` to the routes: its name, and how to build it.
  function request(candidate) {
    return { name: candidate.name, ...(candidate.entry ? { entry: candidate.entry } : {}) };
  }

  // One model's score answer, with its name; a request that failed is an
  // answer that says so, for its row.
  function score(candidate) {
    return postAnswer("/api/finetune/rank/score", request(candidate))
      .then((d) => ({ ...d, name: candidate.name, error: d.success ? undefined : d.error || "no score" }))
      .catch((e) => ({ name: candidate.name, success: false, error: String(e) }));
  }

  async function rank() {
    if (running || switching !== null || fitting) return;
    running = true;
    setBusy(rankBtn, true, "Ranking...");
    // A volume made, or a model added, since the list was shown is in this one.
    const state = await listCandidates();
    const chosen = shown().filter((candidate) => !unticked.has(candidate.name));
    if (!state.ready || !chosen.length) {
      running = false;
      setBusy(rankBtn, false);
      showCandidates();
      if (state.ready) say("Tick at least one of the models shown to rank.", "warning");
      else showReadiness(state);
      return;
    }
    answers = [];
    summary = null;
    ranked = new Map(chosen.map((candidate) => [candidate.name, candidate]));
    rankedOn = state.volume.id;
    stopping = false;
    stopBtn.hidden = false;
    stopBtn.disabled = false;
    showCandidates();
    showResults();
    const started = performance.now();
    // The model the training form's Channel is shown for (the one painted
    // for, which has no Use this model) gets the channel that matched best,
    // when none is picked (fillChannel).
    let filled = null;
    for (let i = 0; i < chosen.length && !stopping; i++) {
      const starting = chosen[i].cluster && !chosen[i].served ? ", starting it on the cluster first" : "";
      say(`Scoring ${chosen[i].name} (${i + 1} of ${chosen.length})${starting}...`);
      const answer = await score(chosen[i]);
      answers.push(answer);
      if (isScored(answer) && Number.isInteger(answer.channel_index) && fillChannel(answer.name, answer.channel_index)) {
        filled = ` ${answer.name}'s Channel is now ${answer.channel}, the one that matched best.`;
      }
      showResults();
    }
    running = false;
    setBusy(rankBtn, false);
    stopBtn.hidden = true;
    showCandidates();
    showResults();
    const seconds = Math.round((performance.now() - started) / 1000);
    const done = answers.length < chosen.length
      ? `Stopped after ${answers.length} of ${chosen.length} models`
      : `Ranked ${plural(answers.length, "model")}`;
    const [best, ...rest] = rankResults(answers).rows;
    const tied = rest.filter((row) => row.tied).length;
    summary = best && best.best
      ? {
        text: `${done} in ${seconds} s: ${best.name} matches your paint best, ${percent(best.matches)}` +
              (tied ? `; ${tied} more ${tied === 1 ? "is" : "are"} about as good.` : ".") + (filled || ""),
        tone: "muted",
      }
      : { text: `${done} in ${seconds} s: none could be scored (each row says why).`, tone: "danger" };
    say(summary.text, summary.tone);
  }

  // Fit to Paint: the model seeding the paint scored on it, as a ranking
  // scores one, and what fits it best set as Use this model sets it, with
  // no new volume: the input normalization first (POST
  // /api/finetune/rank/input-norm, when it is not the session's as it
  // stands), then the threshold, which was fitted under it and so is set
  // only once it is.
  async function fitToPaint() {
    if (running || switching !== null || fitting) return;
    const name = seedModel.value;
    if (!name) {
      sayFit("No model to fit: serve one on the Models tab, and pick it under Prediction from.", "warning");
      return;
    }
    fitting = true;
    setBusy(fitBtn, true, "Fitting...");
    rankBtn.disabled = true;
    showResults();
    sayFit(`Scoring ${name} on your paint...`);
    try {
      const d = await score({ name });
      if (!isScored(d)) {
        sayFit(`Could not fit ${name} to your paint: ${d.error}`, "danger");
        if (d.needs_volume) offerNewVolume(`Could not fit ${name} to your paint: ${d.error}`);
        return;
      }
      if (changesInput(d)) {
        const set = await postAnswer("/api/finetune/rank/input-norm", { input_norm: d.input_norm });
        if (!set.success) {
          sayFit(`Could not set the input normalization that fits ${name} to your paint: ${set.error}`, "danger");
          return;
        }
        // The Processing tab's input normalization shows it as it now runs
        // (main.js reads the chain again), so a later Apply keeps it.
        announce(CHAIN_CHANGED_EVENT);
      }
      installThreshold(name, d.threshold);
      const channel = Number.isInteger(d.channel_index) && fillChannel(name, d.channel_index) ? d.channel : null;
      sayFit(fittedText(name, d, channel), "success");
    } catch (e) {
      sayFit(`Could not fit ${name} to your paint: ${e.message || e}`, "danger");
    } finally {
      fitting = false;
      setBusy(fitBtn, false);
      rankBtn.disabled = !ready;
      showResults();
    }
  }

  rankBtn.addEventListener("click", rank);
  fitBtn.addEventListener("click", fitToPaint);
  seedThreshold.addEventListener("input", showMark);
  seedThreshold.addEventListener("change", showMark);
  showMark();
  filter.addEventListener("input", showCandidates);
  // Tick or untick the models shown; the hidden ones keep their ticks.
  tickAll.addEventListener("click", () => {
    for (const candidate of shown()) unticked.delete(candidate.name);
    showCandidates();
  });
  tickNone.addEventListener("click", () => {
    for (const candidate of shown()) unticked.add(candidate.name);
    showCandidates();
  });
  stopBtn.addEventListener("click", () => {
    stopping = true;
    stopBtn.disabled = true;
    say(`${status.textContent} Stopping after this one.`);
  });

  // Listed while the card is open: when the Training tab is shown or the
  // card opened there, and on Rank. While Rank waits for paint, also as the
  // pointer comes onto the card: the paint is made in the viewer beside it.
  // Once listed, again whenever the models served may have changed, for the
  // Show buttons. A closed card lists nothing.
  const trainingPane = document.getElementById("training");
  document.getElementById("training-tab").addEventListener("shown.bs.tab", () => {
    if (section.open) refresh();
  });
  document.getElementById("rankCard").addEventListener("pointerenter", () => {
    if (section.open && rankBtn.disabled && !running && switching === null && !fitting) refresh();
  });
  listen(MODELS_CHANGED_EVENT, () => {
    if (listed && section.open) refresh();
  });

  const open = readJSON(OPEN_KEY, null);
  if (typeof open === "boolean") section.open = open;
  section.addEventListener("toggle", () => {
    writeJSON(OPEN_KEY, section.open);
    if (section.open && trainingPane.classList.contains("active")) refresh();
  });
}
